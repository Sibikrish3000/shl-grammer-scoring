"""Average submissions: python -m kaggle.avg OUT A.csv[:w] B.csv[:w] ...   (weights default to 1)."""
import sys

import numpy as np
import pandas as pd

out, parts = sys.argv[1], sys.argv[2:]
subs = [(pd.read_csv(p.split(":")[0]), float(p.split(":")[1]) if ":" in p else 1.0) for p in parts]
base = subs[0][0]
assert all((s.filename.values == base.filename.values).all() for s, _ in subs), "row order differs"
w = np.array([w for _, w in subs]) / sum(w for _, w in subs)
res = base.copy()
res["label"] = sum(wi * s.label.values for (s, _), wi in zip(subs, w))
res.to_csv(out, index=False)
print(out, "mean", round(res.label.mean(), 3), "corr", round(np.corrcoef(subs[0][0].label, subs[-1][0].label)[0, 1], 4))
