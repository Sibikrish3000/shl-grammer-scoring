"""Frozen pretrained embeddings per clip, mean-pooled over time, for every hidden layer.

  wavlm      microsoft/wavlm-large           -> features/emb/wavlm_mean.npy      (N, 25, 1024)
  whisper    openai/whisper-large-v3 encoder -> features/emb/whisper_mean.npy    (N, 33, 1280)
  text       microsoft/deberta-v3-large      -> features/emb/text_mean.npy       (N, 25, 1024)

Rows follow features/emb/index.csv (train rows then test rows). fp16 storage.
Usage: python -m kaggle.embed [wavlm|whisper|text ...]
"""
import os
import sys

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio.functional as AF

DATA_DIR = os.environ.get("DATA_DIR", "data")
OUT = os.path.join(os.environ.get("FEAT_DIR", "features"), "emb")
DEV = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16 if DEV != "cpu" else torch.float32
SR = 16000


def clips():
    tr = pd.read_csv(f"{DATA_DIR}/train.csv").assign(split="train")
    te = pd.read_csv(f"{DATA_DIR}/test.csv").assign(split="test")
    df = pd.concat([tr, te], ignore_index=True)[["filename", "split"]]
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(f"{OUT}/index.csv", index=False)
    return df


def load_16k(path):
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(wav.mean(1))
    return (AF.resample(wav, sr, SR) if sr != SR else wav).numpy()


def save(name, rows):
    arr = np.stack(rows).astype(np.float16)
    assert np.isfinite(arr).all(), f"{name}: non-finite values (fp16 overflow?)"
    np.save(f"{OUT}/{name}_mean.npy", arr)
    print(name, arr.shape, flush=True)


@torch.inference_mode()
def wavlm(df):
    from transformers import AutoFeatureExtractor, WavLMModel
    fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-large")
    model = WavLMModel.from_pretrained("microsoft/wavlm-large", dtype=DTYPE).to(DEV).eval()
    rows = []
    for i, r in df.iterrows():
        x = fe(load_16k(f"{DATA_DIR}/{r.split}/{r.filename}"), sampling_rate=SR, return_tensors="pt").input_values
        hs = model(x.to(DEV, DTYPE), output_hidden_states=True).hidden_states  # 25 x (1, T, 1024)
        rows.append(torch.stack([h[0].float().mean(0) for h in hs]).cpu().numpy())
        if i % 100 == 0:
            print(f"  wavlm {i}/{len(df)}", flush=True)
    save("wavlm", rows)


@torch.inference_mode()
def whisper(df):
    from transformers import WhisperFeatureExtractor, WhisperModel
    fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")
    enc = WhisperModel.from_pretrained("openai/whisper-large-v3", dtype=DTYPE).encoder.to(DEV).eval()
    win = 30 * SR
    rows = []
    for i, r in df.iterrows():
        x = load_16k(f"{DATA_DIR}/{r.split}/{r.filename}")
        chunks = [x[s:s + win] for s in range(0, len(x), win)]
        if len(chunks) > 1 and len(chunks[-1]) < SR:  # drop a <1 s tail
            chunks = chunks[:-1]
        total, n = 0, 0
        for c in chunks:
            feats = fe(c, sampling_rate=SR, return_tensors="pt").input_features.to(DEV, DTYPE)
            valid = max(1, min(1500, int(np.ceil(len(c) / 320))))  # 20 ms frames; the rest is padding
            hs = enc(feats, output_hidden_states=True).hidden_states  # 33 x (1, 1500, 1280)
            total = total + torch.stack([h[0, :valid].float().sum(0) for h in hs])
            n += valid
        rows.append((total / n).cpu().numpy())
        if i % 100 == 0:
            print(f"  whisper {i}/{len(df)}", flush=True)
    save("whisper", rows)


@torch.inference_mode()
def text(df):
    from transformers import AutoModel, AutoTokenizer
    # filenames repeat across splits, so join on (filename, split)
    tr = pd.concat([pd.read_csv(f"{DATA_DIR}/{s}_transcripts.csv").assign(split=s) for s in ("train", "test")])
    txt = df.merge(tr[["filename", "split", "transcript"]], on=["filename", "split"], how="left")["transcript"]
    assert len(txt) == len(df)
    txt = txt.fillna("").tolist()
    tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
    model = AutoModel.from_pretrained("microsoft/deberta-v3-large", dtype=torch.float32).to(DEV).eval()
    rows = []
    for i in range(0, len(txt), 16):
        enc = tok(txt[i:i + 16], truncation=True, max_length=512, padding=True, return_tensors="pt").to(DEV)
        hs = model(**enc, output_hidden_states=True).hidden_states
        m = enc["attention_mask"].unsqueeze(-1).float()
        rows += list(torch.stack([(h * m).sum(1) / m.sum(1) for h in hs], 1).cpu().numpy())
    save("text", rows)


if __name__ == "__main__":
    df = clips()
    for name in sys.argv[1:] or ["wavlm", "whisper", "text"]:
        {"wavlm": wavlm, "whisper": whisper, "text": text}[name](df)
