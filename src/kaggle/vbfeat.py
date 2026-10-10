"""Hand text features on the verbatim-prompted transcripts (full clips + test-length crops).

-> features/vb_{full,crops}.csv. Same 13 stats as kaggle.features plus filler / restart counts that the
verbatim prompt makes meaningful, and a hallucination-loop flag (Whisper repeating one sentence).
"""
import os
import re

import numpy as np
import pandas as pd

from kaggle.features import text_features

FEAT = os.environ.get("FEAT_DIR", "features")
WORD = re.compile(r"[a-zA-Z']+")


def extra(text):
    w = [x.lower() for x in WORD.findall(text)]
    n = max(len(w), 1)
    sents = [s.strip().lower() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    top = pd.Series(sents).value_counts().iloc[0] if sents else 0  # most repeated sentence
    return {
        "um_rate": sum(x in {"um", "umm", "uh", "uhh", "er", "erm", "hmm"} for x in w) / n,
        "restart_rate": sum(a == b for a, b in zip(w, w[1:])) / n,  # "there are, there are"
        "bigram_restart": sum(a == b for a, b in zip(zip(w, w[1:]), zip(w[2:], w[3:]))) / n,
        "top_sent_repeat": top,
        "loop_flag": float(top >= 4),
        "log_words": np.log1p(len(w)),
    }


def main():
    for name in ("full", "crops"):
        d = pd.read_csv(f"{FEAT}/verbatim/verbatim_{name}.csv")
        t = d["transcript"].fillna("")
        f = pd.DataFrame([{**text_features(x), **extra(x)} for x in t])
        f.to_csv(f"{FEAT}/vb_{name}.csv", index=False)
        print(name, f.shape, "loops flagged:", int(f.loop_flag.sum()))


if __name__ == "__main__":
    main()
