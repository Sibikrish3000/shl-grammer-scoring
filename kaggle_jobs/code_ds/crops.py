"""Test-length crops of the train clips -> features/crops.csv (filename, crop, n_samples).

70% of test clips are shorter than 55 s against ~25% of train, and less audio lowers every
model's prediction. Each train clip gets N_CROPS copies cut at the end to a length drawn from the
test clips' own durations (seed 0), so models can be trained and validated at test-like lengths.
Shared by kaggle.embed --crops and the Kaggle transcription job.
"""
import os

import numpy as np
import pandas as pd
import soundfile as sf

DATA_DIR = os.environ.get("DATA_DIR", "data")
FEAT = os.environ.get("FEAT_DIR", "features")
N_CROPS = 4


def build():
    test_len = np.array([sf.info(f"{DATA_DIR}/test/{f}").frames for f in pd.read_csv(f"{DATA_DIR}/test.csv").filename])
    tr = pd.read_csv(f"{DATA_DIR}/train.csv")
    full = np.array([sf.info(f"{DATA_DIR}/train/{f}").frames for f in tr.filename])
    rng = np.random.default_rng(0)
    rows = [pd.DataFrame({"filename": tr.filename, "crop": i,
                          "n_samples": np.minimum(rng.choice(test_len, len(tr)), full)}) for i in range(N_CROPS)]
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(f"{FEAT}/crops.csv", index=False)
    sr = sf.info(f"{DATA_DIR}/test/{pd.read_csv(f'{DATA_DIR}/test.csv').filename[0]}").samplerate
    print(df.shape, f"sr={sr}", "crop seconds:", (df.n_samples / sr).describe().round(1).to_dict())
    print("test seconds:", pd.Series(test_len / sr).describe().round(1).to_dict())


if __name__ == "__main__":
    build()
