"""Feature extraction for the SHL grammar scorer.

Three groups, each cached to features/<group>_<split>.csv so modelling runs don't recompute them:
  text  - surface / structural stats from the Whisper transcript (instant)
  lm    - grammar signal from pretrained LMs: CoLA acceptability + GPT-2 perplexity
  audio - Silero VAD speech/pause stats from the raw wav

Usage: python -m kaggle.features [text|lm|audio ...]   (default: all)
Paths come from DATA_DIR / FEAT_DIR env vars so the same code runs on Kaggle.
"""
import os
import re
import sys

import numpy as np
import pandas as pd

DATA_DIR = os.environ.get("DATA_DIR", "data")
FEAT_DIR = os.environ.get("FEAT_DIR", "features")

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
WORD = re.compile(r"[a-zA-Z']+")
FILLERS = {"um", "umm", "uh", "uhh", "ah", "ahh", "er", "erm", "hmm", "mm", "uhm"}
WEAK_END = {"a", "an", "the", "and", "but", "or", "so", "to", "of", "in", "on", "at", "for", "with",
            "my", "your", "his", "her", "our", "their", "is", "are", "was", "were", "i", "we", "you",
            "he", "she", "it", "they", "if", "that", "because", "than", "as", "by", "from"}
WEAK_START = {"and", "but", "or", "so", "because", "that", "which", "who", "when", "if", "while", "then"}


def load(split):
    """Transcripts joined onto the split CSV (transcript '' if Whisper returned nothing)."""
    df = pd.read_csv(f"{DATA_DIR}/{split}.csv")
    tr = pd.read_csv(f"{DATA_DIR}/{split}_transcripts.csv")[["filename", "transcript"]]
    df = df.merge(tr, on="filename", how="left")
    df["transcript"] = df["transcript"].fillna("")
    return df


def sentences(text, max_words=30):
    """Split on punctuation; chunk run-ons so unpunctuated transcripts still give units."""
    out = []
    for s in SENT_SPLIT.split(text.strip()):
        w = s.split()
        out += [" ".join(w[i:i + max_words]) for i in range(0, len(w), max_words)]
    return out or [""]


def text_features(text):
    words = [w.lower() for w in WORD.findall(text)]
    n = len(words)
    sents = [s for s in SENT_SPLIT.split(text.strip()) if s]
    sent_words = [[w.lower() for w in WORD.findall(s)] for s in sents]
    sent_words = [s for s in sent_words if s]
    tri = list(zip(words, words[1:], words[2:]))
    return {
        "word_count": n,
        "avg_word_len": np.mean([len(w) for w in words]) if n else 0,
        "type_token_ratio": len(set(words)) / n if n else 0,
        "sent_count": len(sent_words),
        "avg_sent_len": np.mean([len(s) for s in sent_words]) if sent_words else 0,
        "max_sent_len": max((len(s) for s in sent_words), default=0),
        "punct_per_word": len(re.findall(r"[.,!?;:]", text)) / n if n else 0,
        # Whisper hallucination loops repeat trigrams heavily (seen on label-0 clips)
        "trigram_repeat": 1 - len(set(tri)) / len(tri) if tri else 0,
        "filler_rate": sum(w in FILLERS for w in words) / n if n else 0,
        # immediate word repeats ("I I went") ~ restarts / self-corrections
        "word_repeat_rate": sum(a == b for a, b in zip(words, words[1:])) / n if n else 0,
        "weak_end_frac": np.mean([s[-1] in WEAK_END for s in sent_words]) if sent_words else 0,
        "weak_start_frac": np.mean([s[0] in WEAK_START for s in sent_words]) if sent_words else 0,
        "long_word_frac": sum(len(w) >= 7 for w in words) / n if n else 0,
    }


def lm_features(texts):
    """CoLA acceptability per sentence (mean/min/frac unacceptable) + GPT-2 perplexity per text."""
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    cola_name = "textattack/roberta-base-CoLA"
    ctok = AutoTokenizer.from_pretrained(cola_name)
    cola = AutoModelForSequenceClassification.from_pretrained(cola_name).to(dev).eval()
    gtok = AutoTokenizer.from_pretrained("gpt2")
    gpt = AutoModelForCausalLM.from_pretrained("gpt2").to(dev).eval()

    rows = []
    for i, text in enumerate(texts):
        sents = sentences(text)
        with torch.no_grad():
            enc = ctok(sents, return_tensors="pt", padding=True, truncation=True, max_length=64).to(dev)
            p_ok = cola(**enc).logits.softmax(-1)[:, 1].float().cpu().numpy()  # label 1 = acceptable
            ids = gtok(text or ".", return_tensors="pt", truncation=True, max_length=512).input_ids.to(dev)
            nll = gpt(ids, labels=ids).loss.item() if ids.shape[1] > 1 else 0.0
        rows.append({"cola_mean": p_ok.mean(), "cola_min": p_ok.min(), "cola_bad_frac": (p_ok < 0.5).mean(),
                     "gpt2_nll": nll})
        if i % 100 == 0:
            print(f"  lm {i}/{len(texts)}", flush=True)
    return pd.DataFrame(rows)


def audio_features(paths):
    """Silero VAD on 16 kHz mono audio -> duration, speech ratio, pause stats."""
    import soundfile as sf
    import torch
    import torchaudio.functional as AF
    from silero_vad import get_speech_timestamps, load_silero_vad

    vad = load_silero_vad()
    rows = []
    for i, p in enumerate(paths):
        wav, sr = sf.read(p, dtype="float32", always_2d=True)
        wav = torch.from_numpy(wav.mean(1))
        if sr != 16000:
            wav = AF.resample(wav, sr, 16000)
        dur = len(wav) / 16000
        segs = get_speech_timestamps(wav, vad, sampling_rate=16000, return_seconds=True)
        speech = sum(s["end"] - s["start"] for s in segs)
        pauses = np.array([b["start"] - a["end"] for a, b in zip(segs, segs[1:])])
        rows.append({
            "duration": dur,
            "speech_sec": speech,
            "speech_ratio": speech / dur if dur else 0,
            "n_segments": len(segs),
            "n_pauses_long": int((pauses > 1.0).sum()) if len(pauses) else 0,
            "pause_mean": pauses.mean() if len(pauses) else 0,
            "pause_max": pauses.max() if len(pauses) else 0,
            "rms": float(wav.pow(2).mean().sqrt()),
        })
        if i % 100 == 0:
            print(f"  audio {i}/{len(paths)}", flush=True)
    return pd.DataFrame(rows)


def build(group, split):
    df = load(split)
    if group == "text":
        feats = pd.DataFrame([text_features(t) for t in df["transcript"]])
    elif group == "lm":
        feats = lm_features(df["transcript"].tolist())
    elif group == "audio":
        feats = audio_features([f"{DATA_DIR}/{split}/{f}" for f in df["filename"]])
    else:
        raise ValueError(f"unknown group {group!r}")
    feats.insert(0, "filename", df["filename"].values)
    os.makedirs(FEAT_DIR, exist_ok=True)
    feats.to_csv(f"{FEAT_DIR}/{group}_{split}.csv", index=False)
    print(f"{group}_{split}: {feats.shape}")


if __name__ == "__main__":
    for g in sys.argv[1:] or ["text", "lm", "audio"]:
        for s in ("train", "test"):
            build(g, s)
