"""Post-process a submission: constant shift and/or stretch of full-length clips around the mean.

Usage: python -m kaggle.post IN OUT [--shift -0.12] [--stretch-full 1.15]
Gated zeros (label == 0) are left untouched; results clipped to [0, 5].
"""
import argparse

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--shift", type=float, default=0.0)
    ap.add_argument("--stretch-full", type=float, default=1.0, help="scale full-length (>55 s) clips around their mean")
    args = ap.parse_args()
    s = pd.read_csv(args.inp)
    dur = pd.read_csv("features/audio_test.csv").set_index("filename").loc[s.filename, "duration"].values
    p = s.label.values.copy()
    live = p > 0
    full = live & (dur > 55)
    p[full] = p[full].mean() + args.stretch_full * (p[full] - p[full].mean())
    p[live] += args.shift
    s["label"] = np.clip(p, 0, 5)
    s.to_csv(args.out, index=False)
    print(f"mean {s.label[live].mean():.3f} (was {pd.read_csv(args.inp).label[live].mean():.3f}), full clips {full.sum()}")


if __name__ == "__main__":
    main()
