# %% [markdown]
# # SHL Grammar Scoring Engine: Speech → Transcript → Features + DeBERTa → LightGBM stack
#
# **Task.** Predict a continuous MOS grammar score (0–5) for 45–60 s spoken English clips.
# 769 labelled training clips, 216 test clips. Leaderboard metrics: **Pearson r** and **RMSE**.
#
# ## Approach in one picture
#
# ```
#  wav ──► Whisper large-v3 ──► transcript ──┬──► text stats (structure, repetition, incompleteness)
#   │      (cached dataset)                  ├──► LM grammar signal (RoBERTa-CoLA acceptability, GPT-2 NLL)
#   │                                        └──► fine-tuned DeBERTa-v3-base regressor (OOF prediction)
#   └────► Silero VAD ──────────────────────────► audio stats (speech time, pauses, loudness)
#                                                            │
#                                    LightGBM regressor ◄────┘  (5-fold × 3-seed stratified CV)
# ```
#
# **Why these feature groups?** The rubric scores sentence structure, grammatical accuracy,
# incomplete sentences and self-correction.
# * *LM features* measure grammaticality directly: a RoBERTa model fine-tuned on CoLA
#   (Corpus of Linguistic Acceptability) scores how acceptable each sentence is, and GPT-2
#   negative log-likelihood measures how "unexpected" the word sequence is.
# * *Text features* encode the rubric's structural cues: incomplete sentences (sentences ending in a
#   function word such as "the"/"because"), fragments starting with a conjunction, restarts, lexical
#   variety, sentence length.
# * *Audio (VAD) features* capture what the transcript can't: how much real speech the clip contains
#   and how it is paced. Whisper also *hallucinates* fluent-looking loops on poor audio, so audio
#   evidence is needed to catch those clips.
#
# **Step 1 (transcription) is cached.** Whisper large-v3 over 985 clips takes ~2 h on a T4, so it
# was run once in a separate notebook and published as the dataset
# `sibikrish3000/shl-grammar-whisper-transcripts-ds`, attached here. The same dataset caches the
# extracted features (they are deterministic), so the notebook runs in minutes; all feature code is
# kept below and runs when `RECOMPUTE = True` or the cache is missing.

# %% [markdown]
# ## 0. Setup

# %%
# silero-vad is the only package not preinstalled on Kaggle
# !pip install -q silero-vad

# %%
import functools
import glob
import os
import re
import warnings

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio.functional as AF
from scipy.stats import pearsonr
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
plt.rcParams["figure.dpi"] = 100
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


INPUT_ROOT = os.environ.get("INPUT_ROOT", "/kaggle/input")  # override to test locally


def find(name):
    """Locate an input file anywhere under INPUT_ROOT (mount paths differ between setups)."""
    hits = glob.glob(f"{INPUT_ROOT}/**/{name}", recursive=True)
    assert hits, f"{name} not found under {INPUT_ROOT}"
    return hits[0]


DATA_DIR = os.path.dirname(find("train.csv"))
TRANS_DIR = os.path.dirname(find("train_transcripts.csv"))
print("audio/csv:", DATA_DIR, "\ntranscripts:", TRANS_DIR, "\ndevice:", DEVICE)

# %% [markdown]
# ## 1. Data loading and exploration

# %%
def load(split):
    """Split CSV joined with its cached Whisper transcript (empty string if none)."""
    df = pd.read_csv(f"{DATA_DIR}/{split}.csv")
    tr = pd.read_csv(f"{TRANS_DIR}/{split}_transcripts.csv")[["filename", "transcript"]]
    df = df.merge(tr, on="filename", how="left")
    df["transcript"] = df["transcript"].fillna("")
    df["path"] = DATA_DIR + f"/{split}/" + df["filename"]
    return df


train, test = load("train"), load("test")
print(train.shape, test.shape)

# Features are deterministic given audio + transcript, so they were computed once with the code
# below and cached in the transcripts dataset. Set RECOMPUTE = True to rebuild them from scratch
# (LM + VAD take ~10-15 min); the cached CSVs are exactly what that code produces.
RECOMPUTE = False


def cached(group, df, split, compute):
    """Load <group>_<split>.csv from the dataset aligned to df's rows, or compute it."""
    path = f"{TRANS_DIR}/{group}_{split}.csv"
    if not RECOMPUTE and os.path.exists(path):
        feats = pd.read_csv(path).set_index("filename").loc[df["filename"]]
        print(f"{group}_{split}: loaded from cache {feats.shape}")
        return feats.set_index(df.index)
    return compute(df)
assert train["path"].map(os.path.exists).all() and test["path"].map(os.path.exists).all()
train.head(3)

# %%
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
train["label"].value_counts().sort_index().plot.bar(ax=ax[0], color="steelblue")
ax[0].set(title="Training label distribution", xlabel="grammar score", ylabel="clips")
wc = train["transcript"].str.split().str.len()
ax[1].scatter(train["label"], wc, alpha=0.4)
ax[1].axhline(250, color="r", ls="--", label="> 250 words: likely Whisper loop")
ax[1].set(title="Transcript length vs label", xlabel="grammar score", ylabel="words")
ax[1].legend()
plt.tight_layout()
plt.show()
print("label mean / std:", round(train.label.mean(), 3), round(train.label.std(), 3))

# %% [markdown]
# **Observations**
# * Labels are half-point steps; the bulk sits at 2–5, plus a distinct group of **37 clips scored 0**.
#   Scores 1.0/1.5 are almost absent (4 clips).
# * A 60 s answer is normally 100–170 words. A handful of transcripts exceed 250 words, all
#   Whisper hallucination loops (the same sentence repeated), and they cluster on label-0 clips.
#   A trigram-repetition feature flags them.
# * Predicting the training mean for every clip gives RMSE ≈ label std ≈ **1.24**: the floor to beat.

# %%
print(train.loc[wc.idxmax(), "transcript"][:500], "...")

# %% [markdown]
# ## 2. Feature engineering
#
# ### 2a. Text features from the transcript
# Whisper punctuation is a model guess rather than ground truth, so sentence-level features are
# noisy for unpunctuated transcripts. They are kept as weak signals alongside
# punctuation-independent ones (repetition, lexical variety, word length).

# %%
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
WORD = re.compile(r"[a-zA-Z']+")
FILLERS = {"um", "umm", "uh", "uhh", "ah", "ahh", "er", "erm", "hmm", "mm", "uhm"}
# Sentence ending on one of these is usually cut off ("...and I went to the.")
WEAK_END = {"a", "an", "the", "and", "but", "or", "so", "to", "of", "in", "on", "at", "for", "with",
            "my", "your", "his", "her", "our", "their", "is", "are", "was", "were", "i", "we", "you",
            "he", "she", "it", "they", "if", "that", "because", "than", "as", "by", "from"}
# Sentence starting with one of these is often a fragment ("Because it was nice.")
WEAK_START = {"and", "but", "or", "so", "because", "that", "which", "who", "when", "if", "while", "then"}


def text_features(text):
    words = [w.lower() for w in WORD.findall(text)]
    n = len(words)
    sents = [[w.lower() for w in WORD.findall(s)] for s in SENT_SPLIT.split(text.strip())]
    sents = [s for s in sents if s]
    tri = list(zip(words, words[1:], words[2:]))
    return {
        "word_count": n,
        "avg_word_len": np.mean([len(w) for w in words]) if n else 0,
        "type_token_ratio": len(set(words)) / n if n else 0,       # lexical variety
        "sent_count": len(sents),
        "avg_sent_len": np.mean([len(s) for s in sents]) if sents else 0,
        "max_sent_len": max((len(s) for s in sents), default=0),   # run-on sentences
        "punct_per_word": len(re.findall(r"[.,!?;:]", text)) / n if n else 0,
        "trigram_repeat": 1 - len(set(tri)) / len(tri) if tri else 0,  # hallucination-loop flag
        "filler_rate": sum(w in FILLERS for w in words) / n if n else 0,
        "word_repeat_rate": sum(a == b for a, b in zip(words, words[1:])) / n if n else 0,  # restarts
        "weak_end_frac": np.mean([s[-1] in WEAK_END for s in sents]) if sents else 0,     # incomplete
        "weak_start_frac": np.mean([s[0] in WEAK_START for s in sents]) if sents else 0,  # fragments
        "long_word_frac": sum(len(w) >= 7 for w in words) / n if n else 0,  # vocabulary complexity
    }


def text_block(df):
    return pd.DataFrame([text_features(t) for t in df["transcript"]], index=df.index)


tx_train, tx_test = cached("text", train, "train", text_block), cached("text", test, "test", text_block)
tx_train.describe().T.round(3)

# %% [markdown]
# ### 2b. Language-model grammar features
# * **RoBERTa-CoLA** (`textattack/roberta-base-CoLA`): P(sentence is grammatically acceptable),
#   aggregated per clip as mean, minimum and fraction below 0.5. Long unpunctuated runs are chunked
#   into 30-word pieces so every transcript gives scorable units.
# * **GPT-2 NLL**: mean token negative log-likelihood of the whole transcript; ungrammatical word
#   order is less predictable.

# %%
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer


def sentences(text, max_words=30):
    out = []
    for s in SENT_SPLIT.split(text.strip()):
        w = s.split()
        out += [" ".join(w[i:i + max_words]) for i in range(0, len(w), max_words)]
    return out or [""]


@functools.cache
def lm_models():
    """Loaded only when features are actually computed (not when read from the cache)."""
    ctok = AutoTokenizer.from_pretrained("textattack/roberta-base-CoLA")
    cola = AutoModelForSequenceClassification.from_pretrained("textattack/roberta-base-CoLA").to(DEVICE).eval()
    gtok = AutoTokenizer.from_pretrained("gpt2")
    gpt = AutoModelForCausalLM.from_pretrained("gpt2").to(DEVICE).eval()
    return ctok, cola, gtok, gpt


@torch.no_grad()
def lm_features(text):
    ctok, cola, gtok, gpt = lm_models()
    enc = ctok(sentences(text), return_tensors="pt", padding=True, truncation=True, max_length=64).to(DEVICE)
    p_ok = cola(**enc).logits.softmax(-1)[:, 1].float().cpu().numpy()  # label 1 = acceptable
    ids = gtok(text or ".", return_tensors="pt", truncation=True, max_length=512).input_ids.to(DEVICE)
    nll = gpt(ids, labels=ids).loss.item() if ids.shape[1] > 1 else 0.0
    return {"cola_mean": p_ok.mean(), "cola_min": p_ok.min(), "cola_bad_frac": (p_ok < 0.5).mean(),
            "gpt2_nll": nll}


def lm_block(df):
    return pd.DataFrame([lm_features(t) for t in df["transcript"]], index=df.index)


lm_train, lm_test = cached("lm", train, "train", lm_block), cached("lm", test, "test", lm_block)
lm_train.describe().T.round(3)

# %% [markdown]
# ### 2c. Audio features with Voice Activity Detection (Silero VAD)
# **Preprocessing:** every wav is down-mixed to mono and resampled to 16 kHz (Silero's input rate).
# VAD returns speech segments; from them we derive total speech time, speech ratio, segment count,
# long pauses (> 1 s) and pause statistics, plus overall RMS loudness.

# %%
from silero_vad import get_speech_timestamps, load_silero_vad

vad = load_silero_vad()


def read_16k(path):
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(wav.mean(1))  # mono
    return AF.resample(wav, sr, 16000) if sr != 16000 else wav


def audio_features(path):
    wav = read_16k(path)
    dur = len(wav) / 16000
    segs = get_speech_timestamps(wav, vad, sampling_rate=16000, return_seconds=True)
    speech = sum(s["end"] - s["start"] for s in segs)
    pauses = np.array([b["start"] - a["end"] for a, b in zip(segs, segs[1:])])
    return {
        "duration": dur,
        "speech_sec": speech,
        "speech_ratio": speech / dur if dur else 0,
        "n_segments": len(segs),
        "n_pauses_long": int((pauses > 1.0).sum()) if len(pauses) else 0,
        "pause_mean": pauses.mean() if len(pauses) else 0,
        "pause_max": pauses.max() if len(pauses) else 0,
        "rms": float(wav.pow(2).mean().sqrt()),
    }


def audio_block(df):
    return pd.DataFrame([audio_features(p) for p in df["path"]], index=df.index)


au_train, au_test = cached("audio", train, "train", audio_block), cached("audio", test, "test", audio_block)
au_train.describe().T.round(3)

# %%
# Visualise VAD on one low- and one high-scoring clip
fig, axes = plt.subplots(2, 1, figsize=(13, 5), sharex=True)
for ax, idx in zip(axes, [train.label.idxmin(), train.label.idxmax()]):
    wav = read_16k(train.path[idx])
    t = np.arange(len(wav)) / 16000
    ax.plot(t, wav.numpy(), lw=0.3, color="gray")
    for s in get_speech_timestamps(wav, vad, sampling_rate=16000, return_seconds=True):
        ax.axvspan(s["start"], s["end"], color="green", alpha=0.2)
    ax.set_title(f"{train.filename[idx]}  label={train.label[idx]}  (green = detected speech)")
axes[-1].set_xlabel("seconds")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 2d. Stacking feature: fine-tuned DeBERTa-v3 prediction
# A `microsoft/deberta-v3-base` regressor was fine-tuned on the transcripts in a separate GPU
# notebook (`sibikrish3000/shl-grammar-deberta-v3-regression`: 5 folds × 5 epochs, lr 3e-5, MSE,
# dropout off, head bias initialised to the label mean). For **training rows** we use its
# **out-of-fold** prediction: each value comes from the fold model that never saw that clip, so
# LightGBM can learn how much to trust it without leakage. For **test rows** we use the average of
# the 5 fold models. DeBERTa reads the whole transcript in context, which captures grammatical
# errors that the hand-made features and sentence-level CoLA scores miss.


# %%
def need_deberta(df):
    raise FileNotFoundError("deberta_<split>.csv missing: run the DeBERTa notebook and add its outputs to the dataset")


db_train, db_test = cached("deberta", train, "train", need_deberta), cached("deberta", test, "test", need_deberta)
print(f"DeBERTa alone, OOF:  RMSE {np.sqrt(np.mean((train.label - db_train.deberta_oof) ** 2)):.4f}  "
      f"r {pearsonr(train.label, db_train.deberta_oof)[0]:.4f}")

# %% [markdown]
# ### 2e. Which features relate to the score?

# %%
X_all = pd.concat([tx_train, lm_train, au_train, db_train], axis=1)
Xt_all = pd.concat([tx_test, lm_test, au_test, db_test], axis=1)
GROUPS = {"text": list(tx_train.columns), "lm": list(lm_train.columns), "audio": list(au_train.columns),
          "deberta": list(db_train.columns)}

corr = X_all.apply(lambda c: pearsonr(c, train.label)[0]).sort_values()
corr.plot.barh(figsize=(7, 9), color=np.where(corr > 0, "seagreen", "indianred"),
               title="Pearson r of each feature with the grammar score")
plt.show()

grp = pd.cut(train.label, [-0.1, 0.5, 2.5, 3.5, 5], labels=["0", "1–2.5", "3–3.5", "4–5"])
pd.concat([X_all, grp.rename("score_band")], axis=1).groupby("score_band", observed=True)[
    ["cola_mean", "gpt2_nll", "speech_ratio", "rms", "trigram_repeat", "type_token_ratio"]].median().round(3)

# %% [markdown]
# * Higher **CoLA acceptability** and **speech time** go with higher scores; GPT-2 NLL and
#   trigram repetition go with lower scores.
# * The **label-0 clips are ~2× louder** (median RMS 0.126 vs 0.06) with heavy repetition in their
#   transcripts, which points to a different recording condition (clipped/noisy audio) on which
#   Whisper loops. Loudness is therefore a legitimate signal for that group.

# %% [markdown]
# ## 3. Model: LightGBM with repeated stratified K-fold CV
# * Small dataset (769 rows), so a shallow, regularised gradient-boosted tree model, with
#   predictions clipped to the valid range [0, 5].
# * **Stratified** 5-fold on half-point label bins (rare scores < 2 merged) so every fold sees the
#   label-0 group; repeated over 3 seeds and averaged to reduce split variance.
# * Out-of-fold (OOF) predictions give an honest estimate; test predictions are the average of all
#   15 fold models.

# %%
PARAMS = dict(objective="regression", learning_rate=0.01, n_estimators=800, num_leaves=15,
              min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
              reg_lambda=1.0, verbose=-1)
y = train["label"].values
bins = np.round(y * 2).astype(int)
bins = np.where(bins < 4, 0, bins)  # merge 0.0/1.0/1.5 so each stratum has enough members


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def cv_lgb(X, Xt, seeds=3, folds=5):
    oof, pred = np.zeros(len(X)), np.zeros(len(Xt))
    imp = pd.Series(0.0, index=X.columns)
    for seed in range(seeds):
        for tr, va in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, bins):
            m = lgb.LGBMRegressor(**PARAMS, random_state=seed).fit(X.iloc[tr], y[tr])
            oof[va] += m.predict(X.iloc[va]) / seeds
            pred += m.predict(Xt) / (seeds * folds)
            imp += pd.Series(m.booster_.feature_importance("gain"), index=X.columns)
    return np.clip(oof, 0, 5), np.clip(pred, 0, 5), imp / imp.sum()


# %% [markdown]
# ### 3a. Ablation: what does each feature group contribute?

# %%
base = GROUPS["text"] + GROUPS["lm"] + GROUPS["audio"]
rows = [{"features": "DeBERTa OOF alone (no LightGBM)", "n": 1, "CV RMSE": rmse(y, db_train.deberta_oof.values),
         "CV Pearson": pearsonr(y, db_train.deberta_oof)[0]}]
for name, cols in {"text": GROUPS["text"], "text + lm": GROUPS["text"] + GROUPS["lm"],
                   "text + audio": GROUPS["text"] + GROUPS["audio"], "text + lm + audio": base,
                   "text + lm + audio + deberta (final)": list(X_all.columns)}.items():
    oof, _, _ = cv_lgb(X_all[cols], Xt_all[cols])
    rows.append({"features": name, "n": len(cols), "CV RMSE": rmse(y, oof), "CV Pearson": pearsonr(y, oof)[0]})
ablation = pd.DataFrame(rows).set_index("features").round(4)
ablation

# %% [markdown]
# ## 4. Final model: all features

# %%
oof, test_pred, imp = cv_lgb(X_all, Xt_all)

# Training-data RMSE (required): one model refit on all 769 training clips, scored on them.
full = lgb.LGBMRegressor(**PARAMS, random_state=SEED).fit(X_all, y)
train_pred = np.clip(full.predict(X_all), 0, 5)

results = pd.DataFrame({
    "RMSE": [rmse(y, train_pred), rmse(y, oof), rmse(y, np.full_like(y, y.mean()))],
    "Pearson r": [pearsonr(y, train_pred)[0], pearsonr(y, oof)[0], np.nan],
}, index=["Training data (refit, in-sample)", "5-fold CV out-of-fold (honest)", "Baseline: predict mean"]).round(4)
results

# %%
print(f"TRAINING RMSE: {rmse(y, train_pred):.4f}")
print(f"CV (OOF) RMSE: {rmse(y, oof):.4f}   CV Pearson: {pearsonr(y, oof)[0]:.4f}")

# %% [markdown]
# ## 5. Evaluation visualisations

# %%
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
ax[0].scatter(y + np.random.uniform(-0.08, 0.08, len(y)), oof, alpha=0.4)
ax[0].plot([0, 5], [0, 5], "r--")
ax[0].set(xlabel="true score (jittered)", ylabel="OOF prediction",
          title=f"OOF predictions  r={pearsonr(y, oof)[0]:.3f}")
res = pd.DataFrame({"label": y, "abs_err": np.abs(oof - y)})
res.groupby("label")["abs_err"].mean().plot.bar(ax=ax[1], color="darkorange",
                                                 title="Mean absolute OOF error by true score")
imp.sort_values().tail(15).plot.barh(ax=ax[2], title="Feature importance (share of gain)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Submission

# %%
sub = pd.DataFrame({"filename": test["filename"], "label": test_pred})
assert len(sub) == 216 and sub.label.between(0, 5).all()
sub.to_csv("submission.csv", index=False)
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.hist(oof, bins=30, alpha=0.6, label="train OOF", density=True)
ax.hist(test_pred, bins=30, alpha=0.6, label="test", density=True)
ax.set(title="Prediction distribution: OOF vs test", xlabel="predicted score")
ax.legend()
plt.show()
sub.head()

# %% [markdown]
# ## 7. Report
#
# **Preprocessing.** Audio is down-mixed to mono and resampled to 16 kHz. Transcripts come from
# Whisper large-v3 (chunked long-form decoding, English, fp16 on GPU), computed once and cached as a
# dataset. Missing transcripts become empty strings; predictions are clipped to [0, 5].
#
# **Pipeline.** Four feature groups (26 features in total): transcript structure stats,
# LM-based grammaticality (RoBERTa-CoLA acceptability, GPT-2 NLL), Silero-VAD speech/pause stats,
# and the out-of-fold prediction of a fine-tuned DeBERTa-v3-base regressor. All of them feed a
# regularised LightGBM regressor trained with 5-fold × 3-seed stratified CV (a two-level stack).
#
# **Results.** See the tables above. The text-structure features alone are weak; the LM grammar
# features add a clear gain; the audio features add the largest gain among the hand-made groups,
# mostly by identifying the distinct label-0 group. The fine-tuned DeBERTa is the strongest single
# signal (it reads the full transcript in context) but misses what only the audio reveals, so
# stacking it with the other features beats both (CV RMSE ≈ 0.80 → 0.74, Pearson ≈ 0.76 → 0.80; a
# tuned linear blend of the two reached only ≈ 0.75). The training-data RMSE is reported as
# required, but it is in-sample and optimistic; the CV out-of-fold RMSE / Pearson are the honest
# estimate of test performance.
#
# **Limitations and next steps.**
# * Whisper normalises disfluencies and some grammar slips, so the transcript under-reports errors.
# * Hallucinated transcripts on poor audio corrupt text features for those clips; VAD-gated
#   transcription (e.g. faster-whisper with `vad_filter=True`) would reduce this.
# * DeBERTa overfits after ~3 epochs (train MSE → 0.07); more seeds, LoRA or a larger model
#   (deberta-v3-large) would likely add further gains.
