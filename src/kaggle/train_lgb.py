"""Approach A: LightGBM on cached features, stratified K-fold CV, tracked in MLflow.

Usage: python -m kaggle.train_lgb [--groups text,lm,audio] [--rounds 400] [--lr 0.03] [--leaves 15]
Reads features/<group>_<split>.csv (see kaggle.features); writes submissions/<run>.csv.
"""
import argparse
import os
import tempfile

import lightgbm as lgb
import matplotlib
import mlflow
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import StratifiedKFold

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DATA_DIR = os.environ.get("DATA_DIR", "data")
FEAT_DIR = os.environ.get("FEAT_DIR", "features")


def load(split, groups):
    df = pd.read_csv(f"{DATA_DIR}/{split}.csv")
    for g in groups:
        df = df.merge(pd.read_csv(f"{FEAT_DIR}/{g}_{split}.csv"), on="filename", how="left")
    return df


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="text,lm,audio")
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--leaves", type=int, default=15)
    ap.add_argument("--min-child", type=int, default=20)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3, help="repeat CV with different splits, average")
    args = ap.parse_args()
    groups = args.groups.split(",")

    train, test = load("train", groups), load("test", groups)
    feats = [c for c in train.columns if c not in ("filename", "label")]
    X, y, Xt = train[feats], train["label"].values, test[feats]
    bins = np.round(y * 2).astype(int)  # half-point labels -> strata (rare 1.0/1.5 merged below)
    bins = np.where(bins < 4, 0, bins)

    params = dict(objective="regression", learning_rate=args.lr, num_leaves=args.leaves,
                  min_child_samples=args.min_child, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, reg_lambda=1.0, n_estimators=args.rounds, verbose=-1)

    oof = np.zeros(len(y))
    test_pred = np.zeros(len(Xt))
    imp = pd.Series(0.0, index=feats)
    for seed in range(args.seeds):
        skf = StratifiedKFold(args.folds, shuffle=True, random_state=seed)
        for tr, va in skf.split(X, bins):
            m = lgb.LGBMRegressor(**params, random_state=seed).fit(X.iloc[tr], y[tr])
            oof[va] += m.predict(X.iloc[va]) / args.seeds
            test_pred += m.predict(Xt) / (args.seeds * args.folds)
            imp += pd.Series(m.booster_.feature_importance("gain"), index=feats)
    oof, test_pred = np.clip(oof, 0, 5), np.clip(test_pred, 0, 5)

    # Training-data RMSE (required by the brief): model refit on all train rows, scored on them
    full = lgb.LGBMRegressor(**params, random_state=0).fit(X, y)
    train_rmse = rmse(y, np.clip(full.predict(X), 0, 5))

    metrics = {"cv_rmse": rmse(y, oof), "cv_pearson": float(pearsonr(y, oof)[0]), "train_rmse": train_rmse}
    print({k: round(v, 4) for k, v in metrics.items()})

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("shl-grammar")
    with mlflow.start_run(run_name=f"lgb_{'+'.join(groups)}") as run:
        mlflow.log_params({**vars(args), "n_features": len(feats)})
        mlflow.log_metrics(metrics)
        with tempfile.TemporaryDirectory() as d:
            imp.sort_values().to_frame("gain").to_csv(f"{d}/importance.csv")
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            ax[0].scatter(y, oof, alpha=0.4)
            ax[0].plot([0, 5], [0, 5], "r--")
            ax[0].set(xlabel="true", ylabel="OOF pred", title=f"OOF  r={metrics['cv_pearson']:.3f}")
            imp.sort_values().tail(20).plot.barh(ax=ax[1], title="gain importance (top 20)")
            fig.tight_layout()
            fig.savefig(f"{d}/oof_importance.png", dpi=100)
            pd.DataFrame({"filename": train["filename"], "label": y, "oof": oof}).to_csv(f"{d}/oof.csv", index=False)
            mlflow.log_artifacts(d)
        os.makedirs("submissions", exist_ok=True)
        sub = f"submissions/{run.info.run_id[:8]}_lgb.csv"
        pd.DataFrame({"filename": test["filename"], "label": test_pred}).to_csv(sub, index=False)
        mlflow.log_artifact(sub)
    print("top features:", imp.sort_values(ascending=False).head(8).index.tolist())
    print("submission:", sub)


if __name__ == "__main__":
    main()
