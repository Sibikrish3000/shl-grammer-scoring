# %% [markdown]
# # SHL Grammar Scoring: Approach B, DeBERTa-v3 regression on Whisper transcripts
#
# Fine-tunes `microsoft/deberta-v3-*` to predict the 0–5 grammar score directly from the transcript.
# Uses the **same stratified 5-fold split** as the LightGBM notebook, so the out-of-fold (OOF)
# predictions saved here can be stacked as a feature there without leakage.
#
# Outputs (`/kaggle/working`): `deberta_oof.csv` (filename, fold, oof) and `deberta_test.csv`
# (filename, pred = mean of the fold models), plus per-epoch validation curves.
#
# Set `DEBUG=1` in the environment for a tiny smoke run (subset, 2 folds, 1 epoch).

# %%
import gc
import glob
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

DEBUG = os.environ.get("DEBUG") == "1"
CFG = dict(
    model=os.environ.get("MODEL", "microsoft/deberta-v3-small" if DEBUG else "microsoft/deberta-v3-base"),
    max_len=384,    # transcripts avg ~105 words (~140 tokens); hallucination loops are truncated
    batch=8,
    lr=float(os.environ.get("LR", "3e-5")),
    weight_decay=0.01,
    epochs=1 if DEBUG else int(os.environ.get("EPOCHS", "5")),
    warmup=0.1,
    folds=2 if DEBUG else 5,
    seed=0,         # fold seed 0 = same split as the first seed of the LightGBM CV
)
INPUT_ROOT = os.environ.get("INPUT_ROOT", "/kaggle/input")
OUT = os.environ.get("OUT_DIR", "/kaggle/working")
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
AMP = DEVICE == "cuda"
print(CFG, DEVICE)


def find(name):
    hits = glob.glob(f"{INPUT_ROOT}/**/{name}", recursive=True)
    assert hits, f"{name} not found under {INPUT_ROOT}"
    return hits[0]


# %% [markdown]
# ## Data and folds

# %%
train = pd.read_csv(find("train_transcripts.csv"))
test = pd.read_csv(find("test_transcripts.csv"))
for d in (train, test):
    d["transcript"] = d["transcript"].fillna("").astype(str)
y = train["label"].values.astype(np.float32)
bins = np.round(y * 2).astype(int)
bins = np.where(bins < 4, 0, bins)  # identical strata to the LightGBM notebook
train["fold"] = -1
for f, (_, va) in enumerate(StratifiedKFold(CFG["folds"], shuffle=True, random_state=CFG["seed"]).split(train, bins)):
    train.loc[va, "fold"] = f
if DEBUG:
    train = train.groupby("fold").head(32).reset_index(drop=True)
    test = test.head(16)
    y = train["label"].values.astype(np.float32)
print(train.shape, test.shape, train.fold.value_counts().sort_index().tolist())

# %% [markdown]
# ## Model and training loop
# * `AutoModelForSequenceClassification` with `num_labels=1` = a linear regression head on the
#   pooled representation, trained with MSE.
# * Dropout is disabled: for regression, dropout noise on the hidden states makes the output scale
#   unstable and is a known source of worse RMSE on small datasets.
# * Linear warm-up then decay; fp16 autocast on GPU. The model after the final epoch is used
#   (no best-epoch picking on the validation fold, which would make the CV optimistic).

# %%
tok = AutoTokenizer.from_pretrained(CFG["model"])


def batches(texts, labels=None, shuffle=False):
    idx = list(range(len(texts)))

    def collate(ix):
        enc = tok([texts[i] for i in ix], truncation=True, max_length=CFG["max_len"], padding=True,
                  return_tensors="pt")
        if labels is not None:
            enc["labels"] = torch.tensor([labels[i] for i in ix], dtype=torch.float32)
        return enc

    return DataLoader(idx, batch_size=CFG["batch"] if shuffle else CFG["batch"] * 2, shuffle=shuffle,
                      collate_fn=collate)


def new_model():
    cfg = AutoConfig.from_pretrained(CFG["model"], num_labels=1, problem_type="regression",
                                     hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
                                     pooler_dropout=0.0)
    # fp32 master weights (transformers 5 otherwise keeps the checkpoint's fp16); autocast does mixed precision
    return AutoModelForSequenceClassification.from_pretrained(CFG["model"], config=cfg,
                                                              dtype=torch.float32).to(DEVICE)


@torch.no_grad()
def predict(model, texts):
    model.eval()
    out = []
    for b in batches(texts):
        b = {k: v.to(DEVICE) for k, v in b.items()}
        with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
            out.append(model(**b).logits.squeeze(-1).float().cpu().numpy())
    return np.clip(np.concatenate(out), 0, 5)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


# %%
torch.manual_seed(CFG["seed"])
oof = np.zeros(len(train), dtype=np.float32)
test_pred = np.zeros(len(test), dtype=np.float32)
history = []
texts_te = test["transcript"].tolist()
for f in range(CFG["folds"]):
    t0 = time.time()
    tr, va = train[train.fold != f], train[train.fold == f]
    model = new_model()
    model.classifier.bias.data.fill_(float(tr["label"].mean()))  # start at the mean, not 0
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    loader = batches(tr["transcript"].tolist(), tr["label"].tolist(), shuffle=True)
    steps = len(loader) * CFG["epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(CFG["warmup"] * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)
    for ep in range(CFG["epochs"]):
        model.train()
        losses = []
        for b in loader:
            b = {k: v.to(DEVICE) for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
                loss = model(**b).loss
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
        p = predict(model, va["transcript"].tolist())
        history.append({"fold": f, "epoch": ep + 1, "train_mse": float(np.mean(losses)),
                        "val_rmse": rmse(va["label"].values, p), "val_r": float(pearsonr(va["label"].values, p)[0])})
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in history[-1].items()}, flush=True)
    oof[va.index] = p
    test_pred += predict(model, texts_te) / CFG["folds"]
    print(f"fold {f} done in {time.time() - t0:.0f}s", flush=True)
    del model, opt
    gc.collect()
    torch.cuda.empty_cache() if AMP else None

# %% [markdown]
# ## Results

# %%
hist = pd.DataFrame(history)
cv = {"cv_rmse": rmse(y, oof), "cv_pearson": float(pearsonr(y, oof)[0])}
print(f"DeBERTa CV (OOF) RMSE: {cv['cv_rmse']:.4f}   CV Pearson: {cv['cv_pearson']:.4f}")
print(hist.groupby("epoch")[["train_mse", "val_rmse", "val_r"]].mean().round(4))

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
hist.groupby("epoch")[["val_rmse"]].mean().plot(ax=ax[0], marker="o", title="Mean validation RMSE per epoch")
ax[1].scatter(y + np.random.uniform(-0.08, 0.08, len(y)), oof, alpha=0.4)
ax[1].plot([0, 5], [0, 5], "r--")
ax[1].set(xlabel="true score (jittered)", ylabel="OOF prediction", title=f"DeBERTa OOF  r={cv['cv_pearson']:.3f}")
plt.tight_layout()
plt.show()

# %%
os.makedirs(OUT, exist_ok=True)
pd.DataFrame({"filename": train.filename, "fold": train.fold, "deberta_oof": oof}).to_csv(f"{OUT}/deberta_oof.csv", index=False)
pd.DataFrame({"filename": test.filename, "deberta_oof": test_pred}).to_csv(f"{OUT}/deberta_test.csv", index=False)
hist.to_csv(f"{OUT}/deberta_history.csv", index=False)
print("saved to", OUT)
