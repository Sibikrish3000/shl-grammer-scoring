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


def crops():
    """Test-length crops of the train clips (kaggle.crops); n = samples kept from the start."""
    df = pd.read_csv(f"{os.path.dirname(OUT)}/crops.csv")
    return df.assign(split="train", n=df.n_samples)[["filename", "split", "n"]]


def load_16k(r):
    wav, sr = sf.read(f"{DATA_DIR}/{r.split}/{r.filename}", dtype="float32", always_2d=True)
    wav = wav[: int(r.n)] if "n" in r and pd.notna(r.n) else wav  # crop (all clips are 16 kHz)
    wav = torch.from_numpy(wav.mean(1))
    return (AF.resample(wav, sr, SR) if sr != SR else wav).numpy()


def save(name, rows):
    arr = np.stack(rows).astype(np.float16)
    assert np.isfinite(arr).all(), f"{name}: non-finite values (fp16 overflow?)"
    np.save(f"{OUT}/{name}_mean.npy", arr)
    print(name, arr.shape, flush=True)


@torch.inference_mode()
def wavlm(df, suffix=""):
    from transformers import AutoFeatureExtractor, WavLMModel
    fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-large")
    model = WavLMModel.from_pretrained("microsoft/wavlm-large", dtype=DTYPE).to(DEV).eval()
    rows = []
    for i, r in df.iterrows():
        x = fe(load_16k(r), sampling_rate=SR, return_tensors="pt").input_values
        hs = model(x.to(DEV, DTYPE), output_hidden_states=True).hidden_states  # 25 x (1, T, 1024)
        rows.append(torch.stack([h[0].float().mean(0) for h in hs]).cpu().numpy())
        if i % 100 == 0:
            print(f"  wavlm {i}/{len(df)}", flush=True)
    save("wavlm" + suffix, rows)


@torch.inference_mode()
def whisper(df, suffix=""):
    from transformers import WhisperFeatureExtractor, WhisperModel
    fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")
    enc = WhisperModel.from_pretrained("openai/whisper-large-v3", dtype=DTYPE).encoder.to(DEV).eval()
    win = 30 * SR
    rows = []
    for i, r in df.iterrows():
        x = load_16k(r)
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
    save("whisper" + suffix, rows)


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


@torch.inference_mode()
def text_verbatim():
    """DeBERTa-large over the verbatim-prompted transcripts: full clips and test-length crops."""
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
    model = AutoModel.from_pretrained("microsoft/deberta-v3-large", dtype=torch.float32).to(DEV).eval()
    for name in ("full", "crops"):
        txt = pd.read_csv(f"{os.path.dirname(OUT)}/verbatim/verbatim_{name}.csv")["transcript"].fillna("").tolist()
        rows = []
        for i in range(0, len(txt), 16):
            enc = tok(txt[i:i + 16], truncation=True, max_length=512, padding=True, return_tensors="pt").to(DEV)
            hs = model(**enc, output_hidden_states=True).hidden_states
            m = enc["attention_mask"].unsqueeze(-1).float()
            rows += list(torch.stack([(h * m).sum(1) / m.sum(1) for h in hs], 1).cpu().numpy())
            if i % 320 == 0:
                print(f"  text_verbatim {name} {i}/{len(txt)}", flush=True)
        save(f"text_verbatim_{name}", rows)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["text_verbatim"]:
        text_verbatim()
        sys.exit()
    if "--crops" in args:  # audio encoders on the test-length train crops -> <name>_crops_mean.npy
        args.remove("--crops")
        for name in args or ["wavlm", "whisper"]:
            {"wavlm": wavlm, "whisper": whisper}[name](crops(), "_crops")
    else:
        df = clips()
        for name in args or ["wavlm", "whisper", "text"]:
            {"wavlm": wavlm, "whisper": whisper, "text": text}[name](df)
