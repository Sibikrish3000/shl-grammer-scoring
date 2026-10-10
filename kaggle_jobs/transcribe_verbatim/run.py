"""Kaggle GPU job: verbatim-prompted Whisper-large-v3-turbo transcripts of full clips and train crops.

Plain Whisper "corrects" grammar ("he don't" -> "he doesn't"), removing the errors we score.
A prompt full of fillers and mistakes nudges it to transcribe verbatim.

Outputs (/kaggle/working):
  verbatim_full.csv   filename, split, transcript         (769 train + 216 test)
  verbatim_crops.csv  filename, crop, transcript          (4 test-length crops per train clip)
"""
import glob
import os
import time

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline

MODEL = "openai/whisper-large-v3-turbo"
PROMPT = ("Umm, so I goes to the, uh, the market yesterday and, like, he don't have "
          "no apples. Hmm, I mean... I was, I was think that it are good.")
OUT = "/kaggle/working"
DEBUG = os.environ.get("DEBUG") == "1"


def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    assert hits, name
    return hits[0]


root = os.path.dirname(find("train.csv"))
crops = pd.read_csv(find("crops.csv"))
full = pd.concat([pd.read_csv(f"{root}/train.csv").assign(split="train"),
                  pd.read_csv(f"{root}/test.csv").assign(split="test")], ignore_index=True)[["filename", "split"]]
if DEBUG:
    full, crops = full.head(8), crops.head(8)

proc = WhisperProcessor.from_pretrained(MODEL)
model = WhisperForConditionalGeneration.from_pretrained(MODEL, dtype=torch.float16).to("cuda").eval()
asr = pipeline("automatic-speech-recognition", model=model, tokenizer=proc.tokenizer,
               feature_extractor=proc.feature_extractor, device="cuda:0", dtype=torch.float16,
               chunk_length_s=30, batch_size=16)
GEN = {"language": "en", "task": "transcribe", "prompt_ids": proc.get_prompt_ids(PROMPT, return_tensors="pt").to("cuda")}


def audio(split, filename, n=None):
    wav, sr = sf.read(f"{root}/{split}/{filename}", dtype="float32", always_2d=True)
    wav = wav.mean(1)
    return {"raw": wav[:n] if n else wav, "sampling_rate": sr}


def run(items):
    t0 = time.time()
    out = [o["text"] for o in asr(items, generate_kwargs=GEN)]
    # some transformers versions echo the prompt at the start of the output
    out = [t.replace(PROMPT, "").strip() for t in out]
    print(f"{len(out)} clips in {time.time() - t0:.0f}s", flush=True)
    return out


full["transcript"] = run([audio(r.split, r.filename) for r in full.itertuples()])
full.to_csv(f"{OUT}/verbatim_full.csv", index=False)
crops["transcript"] = run([audio("train", r.filename, int(r.n_samples)) for r in crops.itertuples()])
crops[["filename", "crop", "transcript"]].to_csv(f"{OUT}/verbatim_crops.csv", index=False)
print(full.head(3).to_string(), "\nempty:", (full.transcript.str.len() == 0).sum(), (crops.transcript.str.len() == 0).sum())
