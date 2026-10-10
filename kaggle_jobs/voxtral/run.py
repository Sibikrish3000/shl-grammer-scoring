"""Kaggle GPU job (2x T4): Voxtral-Mini-3B audio-LLM features for full clips and train crops.

One forward pass per clip with [audio, rubric prompt]:
  * embedding: mean + std over the real (non-padding) audio tokens of the LM hidden states,
    averaged over layers at 20-60% depth -> 2 x hidden features. Attention is causal and the audio
    comes first, so these depend on the audio only.
  * judge: softmax over the next-token logits of "1".."5" after the rubric -> expected score.
Sharded: the main process spawns one worker per GPU (rows k::2), then merges.

Outputs (/kaggle/working): voxtral_full.npy / voxtral_crops.npy (float16, rows in the order of
voxtral_full_index.csv / crops.csv) and voxtral_judge_{full,crops}.csv.
"""
import glob
import math
import os
import subprocess
import sys

MODEL = "mistralai/Voxtral-Mini-3B-2507"
OUT = "/kaggle/working"
PROMPT = ("This is a 45-60 second spoken answer from an English speaking test. Rate the speaker's GRAMMAR "
          "(sentence structure, syntax, grammatical accuracy and complexity) on this rubric. Fillers and "
          "self-corrections are normal in speech.\n"
          "1: struggles with sentence structure and syntax; limited control of simple structures.\n"
          "2: simple structures with consistent basic mistakes; may leave sentences incomplete.\n"
          "3: decent grasp of sentence structure or of grammar, with errors in the other.\n"
          "4: strong control of structure and grammar; occasional minor errors.\n"
          "5: high accuracy; handles complex structures well; self-corrects when needed.\n"
          "Answer with a single digit from 1 to 5.")
TOK_PER_CHUNK, TOK_PER_SEC = 375, 12.5  # Voxtral audio: 30 s chunks of 375 tokens


def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    assert hits, name
    return hits[0]


def rows():
    import pandas as pd
    root = os.path.dirname(find("train.csv"))
    full = pd.concat([pd.read_csv(f"{root}/train.csv").assign(split="train"),
                      pd.read_csv(f"{root}/test.csv").assign(split="test")], ignore_index=True)[["filename", "split"]]
    full["n"] = 0
    crops = pd.read_csv(find("crops.csv")).assign(split="train").rename(columns={"n_samples": "n"})
    return root, full, crops


def worker(k, n):
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import AutoProcessor, VoxtralForConditionalGeneration

    root, full, crops = rows()
    proc = AutoProcessor.from_pretrained(MODEL)
    model = VoxtralForConditionalGeneration.from_pretrained(MODEL, dtype=torch.float16).to("cuda").eval()
    L = model.config.text_config.num_hidden_layers
    layers = list(range(int(0.2 * L), int(0.6 * L) + 1))  # hidden_states[0] = embeddings
    digits = [proc.tokenizer.encode(str(d), add_special_tokens=False)[-1] for d in range(1, 6)]
    assert len(set(digits)) == 5, digits
    tmp = f"/tmp/vox_{k}.wav"

    @torch.inference_mode()
    def one(split, filename, cut):
        wav, sr = sf.read(f"{root}/{split}/{filename}", dtype="float32", always_2d=True)
        wav = wav.mean(1)
        wav = wav[:cut] if cut else wav
        sf.write(tmp, wav, sr)
        conv = [{"role": "user", "content": [{"type": "audio", "path": tmp}, {"type": "text", "text": PROMPT}]}]
        x = proc.apply_chat_template(conv).to("cuda", dtype=torch.float16)
        out = model(**x, output_hidden_states=True)
        p = torch.softmax(out.logits[0, -1, digits].float(), 0).cpu().numpy()
        h = torch.stack([out.hidden_states[l][0] for l in layers]).float().mean(0)
        pos = (x["input_ids"][0] == model.config.audio_token_id).nonzero().squeeze(-1).tolist()
        dur = len(wav) / sr
        keep = [q for i, q in enumerate(pos)  # drop tokens that only cover padding of the last chunk
                if i % TOK_PER_CHUNK < math.ceil(min(max(dur - 30 * (i // TOK_PER_CHUNK), 0), 30) * TOK_PER_SEC)]
        h = h[keep]
        return torch.cat([h.mean(0), h.std(0)]).cpu().numpy(), p

    for name, df in (("full", full), ("crops", crops)):
        part = df.iloc[k::n]
        res = [one(r.split, r.filename, int(r.n)) for r in part.itertuples()]
        emb, prob = np.stack([e for e, _ in res]), np.stack([p for _, p in res])
        assert np.isfinite(emb).all() and np.isfinite(prob).all(), "non-finite (fp16 overflow?)"
        np.save(f"{OUT}/_{name}_emb_{k}.npy", emb.astype(np.float16))
        np.save(f"{OUT}/_{name}_prob_{k}.npy", prob)
        np.save(f"{OUT}/_{name}_idx_{k}.npy", part.index.values)
        print(f"worker {k}: {name} {emb.shape} layers {layers[0]}-{layers[-1]} of {L}", flush=True)


def main():
    import numpy as np
    import pandas as pd
    import torch
    n = torch.cuda.device_count()
    procs = [subprocess.Popen([sys.executable, __file__, "worker", str(k), str(n)],
                              env={**os.environ, "CUDA_VISIBLE_DEVICES": str(k)}) for k in range(n)]
    assert all(p.wait() == 0 for p in procs), "a worker failed"
    _, full, crops = rows()
    full.to_csv(f"{OUT}/voxtral_full_index.csv", index=False)
    for name, df in (("full", full), ("crops", crops)):
        idx = np.concatenate([np.load(f"{OUT}/_{name}_idx_{k}.npy") for k in range(n)])
        order = np.argsort(idx)
        emb = np.concatenate([np.load(f"{OUT}/_{name}_emb_{k}.npy") for k in range(n)])[order]
        prob = np.concatenate([np.load(f"{OUT}/_{name}_prob_{k}.npy") for k in range(n)])[order]
        assert len(emb) == len(df)
        np.save(f"{OUT}/voxtral_{name}.npy", emb)
        pd.DataFrame(prob, columns=[f"vp{d}" for d in range(1, 6)]).assign(
            v_mean=prob @ np.arange(1, 6)).to_csv(f"{OUT}/voxtral_judge_{name}.csv", index=False)
        print(name, emb.shape, "judge mean", round(float((prob @ np.arange(1, 6)).mean()), 3), flush=True)
    for f in glob.glob(f"{OUT}/_*.npy"):
        os.remove(f)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        worker(int(sys.argv[2]), int(sys.argv[3]))
    else:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.56", "mistral-common[audio]"],
                       check=True)
        main()
