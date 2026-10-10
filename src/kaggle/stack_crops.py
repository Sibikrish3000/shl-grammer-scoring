"""Crop-trained speaker-grouped stack (idea from the rank-1 public solution).

Test clips are shorter than train clips (47% are ~45 s vs 14% of train) and every model predicts
lower for less audio. Each train clip is therefore also used as N_CROPS test-length crops
(kaggle.crops); models train on full clips + crops and a validation clip is scored by the mean
prediction over its crops, so CV measures test-like lengths. Test clips use the full-clip features.

Usage: python -m kaggle.stack_crops [--models a,b] [--pool 0.5] [--out submissions/x.csv]
"""
import argparse
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold

from kaggle.stack import (EMB, FEAT, batch_of, load, nnls_fit, pool, ridge, rmse, speaker_groups, stack, svr)

LAYERS = {"wavlm": [19, 20, 21], "whisper": [30, 31, 32], "text_verbatim": [21, 22, 23]}


def reps(is_tr):
    """name -> (X_full over all 985 clips, X_crops over the 3076 crop rows)."""
    out = {}
    for n, layers in LAYERS.items():
        f, c = f"{EMB}/{n}_mean.npy", f"{EMB}/{n}_crops_mean.npy"
        if n == "text_verbatim":
            f, c = f"{EMB}/{n}_full_mean.npy", f"{EMB}/{n}_crops_mean.npy"
        if os.path.exists(f) and os.path.exists(c):
            out[n] = tuple(np.load(p).astype(np.float32)[:, layers].mean(1) for p in (f, c))
    v = f"{FEAT}/voxtral"
    if os.path.exists(f"{v}/voxtral_crops.npy"):
        out["voxtral"] = (np.load(f"{v}/voxtral_full.npy").astype(np.float32),
                          np.load(f"{v}/voxtral_crops.npy").astype(np.float32))
    if {"wavlm", "whisper", "voxtral"} <= out.keys():
        out["audio_vox"] = tuple(np.hstack([out[n][i] for n in ("wavlm", "whisper", "voxtral")]) for i in (0, 1))
    return out


def fit_fold(model, Xf, Xc, parent, y, tr, va, Xte, r):
    """Train on full rows + crops of the train-fold clips; predict val clips' crops (mean) and the test clips."""
    in_tr = np.isin(parent, tr)
    m = clone(model).fit(np.vstack([Xf[tr], Xc[in_tr]]), np.concatenate([y[tr], y[parent[in_tr]]]))
    in_va = np.isin(parent, va)
    pv = pd.Series(m.predict(Xc[in_va])).groupby(parent[in_va]).mean().reindex(va).values
    return r, va, pv, m.predict(Xte)


def cross_val(model, Xf, Xc, parent, y, folds, Xte):
    oof, pred = np.zeros((len(folds), len(y))), np.zeros(len(Xte))
    runs = Parallel(n_jobs=3)(delayed(fit_fold)(model, Xf, Xc, parent, y, tr, va, Xte, r)
                              for r, fold in enumerate(folds) for tr, va in fold)
    for r, va, pv, t in runs:
        oof[r, va] = pv
        pred += t / len(runs)
    return np.clip(oof, 1, 5), np.clip(pred, 1, 5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="")
    ap.add_argument("--pool", type=float, default=None)
    ap.add_argument("--out", default="submissions/stack_crops.csv")
    args = ap.parse_args()

    idx, is_tr, y_all, hand, er = load()
    wavlm = np.load(f"{EMB}/wavlm_mean.npy").astype(np.float32)
    g_tr, g_te, _, _ = speaker_groups(wavlm, is_tr)
    gate, zero = er >= 0.99, y_all == 0
    nz = ~zero
    y, groups = y_all[nz], g_tr[nz]
    bins = np.maximum(np.round(y * 2).astype(int), 4)
    folds = [list(StratifiedGroupKFold(5, shuffle=True, random_state=s).split(y, bins, groups)) for s in range(3)]
    b_tr, b_te = batch_of(hand.duration.values)[is_tr][nz], batch_of(hand.duration.values)[~is_tr]

    crops = pd.read_csv(f"{FEAT}/crops.csv")
    pos = pd.Series(np.arange(is_tr.sum()), index=idx.filename[is_tr].values)  # train row -> position
    row_of = pos.loc[crops.filename].values
    keep = nz[row_of]                      # drop crops of label-0 clips
    nz_pos = pd.Series(np.arange(nz.sum()), index=np.where(nz)[0])
    parent = nz_pos.loc[row_of[keep]].values

    R = reps(is_tr)
    cand = {
        "wavlm_ridge": ("wavlm", None), "whisper_ridge": ("whisper", None), "whisper_svr": ("whisper", "svr"),
        "voxtral_ridge": ("voxtral", None), "audio_vox_ridge": ("audio_vox", None),
        "text_vb_ridge": ("text_verbatim", None), "text_vb_svr": ("text_verbatim", "svr"),
    }
    chosen = args.models.split(",") if args.models else [k for k, (r, _) in cand.items() if r in R]

    P_oof, P_te, names = [], [], []
    for name in chosen:
        rep, kind = cand[name]
        Xf_all, Xc_all = R[rep]
        d = Xf_all.shape[1]
        model = ridge(1.0, d) if kind is None else svr(3, d)
        oof, te = cross_val(model, Xf_all[is_tr][nz], Xc_all[keep], parent, y, folds, Xf_all[~is_tr])
        print(f"  {name:16s} crop-val RMSE {np.mean([rmse(o, y) for o in oof]):.4f}", flush=True)
        P_oof.append(oof)
        P_te.append(te)
        names.append(name)
    P_oof, P_te = np.stack(P_oof, -1), np.stack(P_te, -1)

    st = np.zeros((len(folds), len(y)))
    for r, fold in enumerate(folds):
        for tr, va in fold:
            st[r, va] = stack(P_oof[r][tr], y[tr], b_tr[tr], P_oof[r][va], b_tr[va])
    st = np.clip(st, 2, 5)
    print(f"stack nested crop-val RMSE {np.mean([rmse(s, y) for s in st]):.4f}")
    alphas = [0, 0.25, 0.5, 0.75, 1.0]
    pool_cv = {a: np.mean([rmse(pool(s, groups, a), y) for s in st]) for a in alphas}
    print("speaker pooling CV:", {a: round(float(v), 4) for a, v in pool_cv.items()})
    alpha = args.pool if args.pool is not None else min(pool_cv, key=pool_cv.get)

    P_mean = P_oof.mean(0)
    for b, lab in ((0, "45 s"), (1, "other")):
        w, c = nnls_fit(P_mean[b_tr == b], y[b_tr == b])
        print(f"weights {lab:6s}", dict(zip(names, np.round(w, 3))), f"intercept {c:.3f}")
    test = pool(np.clip(stack(P_mean, y, b_tr, P_te, b_te), 2, 5), g_te, alpha)
    test[gate[~is_tr]] = 0.0
    print(f"pool alpha={alpha} crop-val RMSE {pool_cv[alpha]:.4f} | test mean {test.mean():.3f} "
          f"45 s-batch mean {test[b_te == 0].mean():.3f}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.DataFrame({"filename": idx.filename[~is_tr].values, "label": test}).to_csv(args.out, index=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
