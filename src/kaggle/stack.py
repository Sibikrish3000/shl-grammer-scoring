"""Speaker-aware ensemble: frozen embeddings + hand features -> Ridge/SVR/LGB -> NNLS stack.

Ideas taken from top public solutions:
  * Speaker groups: train speakers record ~3 answers with near-identical grades, test speakers are
    unseen -> all CV is StratifiedGroupKFold over voice-similarity groups (random K-fold leaks).
  * Zero gate: label-0 clips are a constantly-loud noise batch; an energy-ratio threshold flags them.
    They are predicted 0 and excluded from regression training. Other predictions clipped to [2, 5].
  * Recording batch: a ~45 s batch is 14% of train but 47% of test; the stacker is fit per batch.

Usage: python -m kaggle.stack [--models a,b,...] [--pool 0.5] [--out submissions/x.csv]
"""
import argparse
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
import soundfile as sf
from joblib import Parallel, delayed
from scipy.optimize import nnls
from scipy.sparse.csgraph import connected_components
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

DATA_DIR = os.environ.get("DATA_DIR", "data")
FEAT = os.environ.get("FEAT_DIR", "features")
EMB = f"{FEAT}/emb"


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def energy_ratio(path):
    """Fraction of 25 ms frames with RMS > 0.02; ~1.0 means constant loud noise (the grade-0 batch)."""
    x, _ = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(1)
    f = x[: len(x) // 400 * 400].reshape(-1, 400)
    return float((np.sqrt((f ** 2).mean(1)) > 0.02).mean())


def load():
    idx = pd.read_csv(f"{EMB}/index.csv")
    labels = pd.read_csv(f"{DATA_DIR}/train.csv").set_index("filename")["label"]
    # filenames repeat across splits (audio_1.wav is in train and test), so align per split
    hand = pd.concat([pd.concat([pd.read_csv(f"{FEAT}/{g}_{s}.csv").set_index("filename") for g in ("text", "lm", "audio")],
                                axis=1).loc[idx.filename[idx.split == s]] for s in ("train", "test")]).reset_index(drop=True)
    assert len(hand) == len(idx)
    path = f"{FEAT}/energy_ratio.npy"
    if not os.path.exists(path):
        np.save(path, np.array([energy_ratio(f"{DATA_DIR}/{s}/{f}") for f, s in zip(idx.filename, idx.split)]))
    er = np.load(path)
    is_tr = (idx.split == "train").values
    y = labels.loc[idx.filename[is_tr]].values
    return idx, is_tr, y, hand, er


def speaker_groups(wavlm, is_tr, layers=(3, 6), pct=99.5):
    """Connected components of train clips whose voice similarity beats the test->train 99.5th pct."""
    v = np.hstack([wavlm[:, l] for l in layers]).astype(np.float32)
    v = (v - v.mean(0)) / (v.std(0) + 1e-6)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    sim = v @ v.T
    np.fill_diagonal(sim, -1)
    tr, te = np.where(is_tr)[0], np.where(~is_tr)[0]
    tau = np.percentile(sim[np.ix_(te, tr)].max(1), pct)
    _, g_tr = connected_components(sim[np.ix_(tr, tr)] > tau, directed=False)
    _, g_te = connected_components(sim[np.ix_(te, te)] > tau, directed=False)
    return g_tr, g_te, tau, sim


def batch_of(duration):
    return np.where(np.abs(duration - 45.06) < 0.1, 0, 1)  # 0 = the ~45 s batch


def ridge(alpha, d):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha * d))


def svr(C, d):
    return make_pipeline(StandardScaler(), SVR(C=C, gamma=1.0 / d, epsilon=0.1))


LGB = lgb.LGBMRegressor(objective="regression", learning_rate=0.02, n_estimators=500, num_leaves=7,
                        min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)


def cross_val(model, X_tr, X_te, y, folds):
    def fit(r, tr, va):
        m = clone(model).fit(X_tr[tr], y[tr])
        return r, va, m.predict(X_tr[va]), m.predict(X_te)

    oof, pred = np.zeros((len(folds), len(y))), np.zeros(len(X_te))
    runs = Parallel(n_jobs=-1)(delayed(fit)(r, tr, va) for r, fold in enumerate(folds) for tr, va in fold)
    for r, va, p, t in runs:
        oof[r, va] = p
        pred += t / len(runs)
    return np.clip(oof, 1, 5), np.clip(pred, 1, 5)


def nnls_fit(P, t):
    pm, tm = P.mean(0), t.mean()
    w, _ = nnls(P - pm, t - tm)
    return w, tm - pm @ w


def stack(P_fit, t_fit, b_fit, P_new, b_new):
    """NNLS with intercept, fit separately for the 45 s batch and for the rest."""
    out = np.zeros(len(P_new))
    for b in (0, 1):
        w, c = nnls_fit(P_fit[b_fit == b], t_fit[b_fit == b])
        out[b_new == b] = P_new[b_new == b] @ w + c
    return out


def pool(pred, groups, alpha):
    """Shrink each prediction toward its speaker-group mean (alpha=1 -> full group mean)."""
    s = pd.Series(pred).groupby(groups).transform("mean").values
    return (1 - alpha) * pred + alpha * s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="")
    ap.add_argument("--pool", type=float, default=None, help="test-time speaker pooling; default: pick on CV")
    ap.add_argument("--out", default="submissions/stack.csv")
    args = ap.parse_args()

    idx, is_tr, y_all, hand, er = load()
    wavlm = np.load(f"{EMB}/wavlm_mean.npy").astype(np.float32)
    g_tr, g_te, tau, _ = speaker_groups(wavlm, is_tr)
    gate = er >= 0.99
    zero = y_all == 0
    print(f"gate: train zeros caught {(gate[is_tr] & zero).sum()}/{zero.sum()}, false pos {(gate[is_tr] & ~zero).sum()}, "
          f"test gated {gate[~is_tr].sum()}")
    sizes = np.bincount(g_tr)
    print(f"speaker groups (tau={tau:.3f}): {len(sizes)} train groups, size 1/2/3/4+: "
          f"{(sizes == 1).sum()}/{(sizes == 2).sum()}/{(sizes == 3).sum()}/{(sizes >= 4).sum()}, "
          f"{len(np.unique(g_te))} test groups for {len(g_te)} clips")

    nz = ~zero  # regress on graded clips only
    y = y_all[nz]
    groups = g_tr[nz]
    bins = np.maximum(np.round(y * 2).astype(int), 4)
    folds = [list(StratifiedGroupKFold(5, shuffle=True, random_state=s).split(y, bins, groups)) for s in range(3)]
    batch = batch_of(hand.duration.values)
    b_tr, b_te = batch[is_tr][nz], batch[~is_tr]

    # Representations: rows = all clips (train then test)
    reps = {"hand": hand.values.astype(np.float32)}
    emb_layers = {"wavlm": [19, 20, 21], "whisper": [30, 31, 32], "text": [21, 22, 23]}  # from kaggle.sweep
    for name, layers in emb_layers.items():
        p = f"{EMB}/{name}_mean.npy"
        if os.path.exists(p):
            E = np.load(p).astype(np.float32)
            reps[name] = E[:, layers].mean(1)
    vp = f"{FEAT}/voxtral/voxtral_full.npy"
    if os.path.exists(vp):  # Voxtral-Mini-3B audio-LLM hidden states + its direct rubric judgement
        reps["voxtral"] = np.load(vp).astype(np.float32)
        reps["voxtral_judge"] = pd.read_csv(f"{FEAT}/voxtral/voxtral_judge_full.csv").values.astype(np.float32)
    if "wavlm" in reps and "whisper" in reps:
        reps["audio_cat"] = np.hstack([reps["wavlm"], reps["whisper"]])
        if "voxtral" in reps:
            reps["audio_vox"] = np.hstack([reps["audio_cat"], reps["voxtral"]])
            reps["hand_judge"] = np.hstack([reps["hand"], reps["voxtral_judge"]])
        if "text" in reps:
            reps["all_cat"] = np.hstack([reps["audio_cat"], reps["text"]])

    candidates = {
        "hand_lgb": ("hand", LGB),
        "hand_svr": ("hand", make_pipeline(StandardScaler(), SVR(C=1))),
        "wavlm_ridge": ("wavlm", None), "wavlm_svr": ("wavlm", "svr"),
        "whisper_ridge": ("whisper", None), "whisper_svr": ("whisper", "svr"),
        "text_ridge": ("text", None), "text_svr": ("text", "svr"),
        "audio_cat_ridge": ("audio_cat", None), "all_cat_ridge": ("all_cat", None),
        "voxtral_ridge": ("voxtral", None), "voxtral_svr": ("voxtral", "svr"),
        "audio_vox_ridge": ("audio_vox", None), "hand_judge_svr": ("hand_judge", make_pipeline(StandardScaler(), SVR(C=1))),
    }
    chosen = args.models.split(",") if args.models else [k for k, (r, _) in candidates.items() if r in reps]

    P_oof, P_te, names = [], [], []
    for name in chosen:
        rep, m = candidates[name]
        X = reps[rep]
        d = X.shape[1]
        model = ridge(1.0, d) if m is None else svr(3, d) if m == "svr" else m
        oof, te = cross_val(model, X[is_tr][nz], X[~is_tr], y, folds)
        print(f"  {name:15s} grouped-CV RMSE {np.mean([rmse(o, y) for o in oof]):.4f}", flush=True)
        P_oof.append(oof)
        P_te.append(te)
        names.append(name)
    P_oof, P_te = np.stack(P_oof, -1), np.stack(P_te, -1)  # (repeats, n, models), (n_te, models)

    # Nested stack estimate: within each repeat, fit NNLS on the other folds' OOF rows
    st = np.zeros((len(folds), len(y)))
    for r, fold in enumerate(folds):
        for tr, va in fold:
            st[r, va] = stack(P_oof[r][tr], y[tr], b_tr[tr], P_oof[r][va], b_tr[va])
    st = np.clip(st, 2, 5)
    print(f"stack nested grouped-CV RMSE {np.mean([rmse(s, y) for s in st]):.4f}")

    # Speaker pooling, evaluated on the OOF (val folds hold whole speaker groups)
    alphas = [0, 0.25, 0.5, 0.75, 1.0]
    pool_cv = {a: np.mean([rmse(pool(s, groups, a), y) for s in st]) for a in alphas}
    print("speaker pooling CV:", {a: round(v, 4) for a, v in pool_cv.items()})
    alpha = args.pool if args.pool is not None else min(pool_cv, key=pool_cv.get)

    # Final: stack weights on all OOF rows (mean over repeats), applied to test predictions
    P_mean = P_oof.mean(0)
    for b, lab in ((0, "45 s"), (1, "other")):
        w, c = nnls_fit(P_mean[b_tr == b], y[b_tr == b])
        print(f"weights {lab:6s}", dict(zip(names, np.round(w, 3))), f"intercept {c:.3f}")
    test = np.clip(stack(P_mean, y, b_tr, P_te, b_te), 2, 5)
    test = pool(test, g_te, alpha)
    test[gate[~is_tr]] = 0.0

    # Training-data RMSE (in-sample): stack applied to refit-free OOF is not in-sample, so report both
    full_oof = np.zeros(len(y_all))
    full_oof[nz] = pool(np.clip(stack(P_mean, y, b_tr, P_mean, b_tr), 2, 5), groups, alpha)
    full_oof[gate[is_tr]] = 0.0
    print(f"pool alpha={alpha}  final grouped-CV RMSE (graded clips) {pool_cv[alpha]:.4f}  "
          f"| all train clips incl. gate {rmse(full_oof, y_all):.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.DataFrame({"filename": idx.filename[~is_tr].values, "label": test}).to_csv(args.out, index=False)
    print("wrote", args.out, "| test pred mean", round(test.mean(), 3), "| batch-45 mean", round(test[b_te == 0].mean(), 3))


if __name__ == "__main__":
    main()
