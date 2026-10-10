"""Layer sweep for an embedding: grouped-CV RMSE of Ridge on single layers and 3-layer windows.

Usage: python -m kaggle.sweep wavlm [--alpha 1.0]
"""
import argparse

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from kaggle.stack import EMB, cross_val, load, ridge, rmse, speaker_groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("emb")
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args()
    idx, is_tr, y_all, hand, er = load()
    g_tr, *_ = speaker_groups(np.load(f"{EMB}/wavlm_mean.npy").astype(np.float32), is_tr)
    nz = y_all > 0
    y, groups = y_all[nz], g_tr[nz]
    bins = np.maximum(np.round(y * 2).astype(int), 4)
    folds = [list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(y, bins, groups))]
    E = np.load(f"{EMB}/{args.emb}_mean.npy").astype(np.float32)
    res = {}
    for l in range(E.shape[1]):
        X = E[:, l]
        oof, _ = cross_val(ridge(args.alpha, X.shape[1]), X[is_tr][nz], X[~is_tr], y, folds)
        res[l] = rmse(oof[0], y)
    best = sorted(res, key=res.get)[:5]
    print(args.emb, "per-layer RMSE:", {l: round(v, 3) for l, v in res.items()})
    print("best layers:", best)
    for c in best[:3]:
        win = [l for l in (c - 1, c, c + 1) if 0 <= l < E.shape[1]]
        X = E[:, win].mean(1)
        oof, _ = cross_val(ridge(args.alpha, X.shape[1]), X[is_tr][nz], X[~is_tr], y, folds)
        print(f"  window {win}: {rmse(oof[0], y):.4f}")


if __name__ == "__main__":
    main()
