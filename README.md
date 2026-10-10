# SHL Hiring Assessment 2026: spoken grammar scoring

Predict a 0-5 grammar score from 45-60 s spoken-English clips (769 train / 216 test, metric RMSE + Pearson).
Public leaderboard: **0.3211** (rank ~6 / 392). Every submission and its score: [SUBMISSIONS.md](SUBMISSIONS.md).

## Approach
Frozen encoders (WavLM-Large, Whisper-large-v3 encoder, Voxtral-Mini-3B audio LLM, DeBERTa-v3-large over plain and
verbatim-prompted Whisper transcripts) -> Ridge/SVR per representation under **speaker-grouped CV** -> NNLS blend per
recording batch -> clip [2,5] -> speaker pooling -> zero gate for the 37 loud-noise clips. Final = mean of a full-clip
regime (v10) and a regime trained on test-length crops (v12).

## Layout
| path | what |
|---|---|
| `final_nb/` | **final deliverable notebook** (`final.ipynb` from `notebook/final.py` via jupytext); report, training RMSE, plots |
| `src/kaggle/` | local pipeline: `features.py` (hand/LM/VAD), `embed.py` (encoders), `crops.py`, `stack.py` (full-clip regime), `stack_crops.py` (crop regime), `post.py`, `avg.py` |
| `kaggle_jobs/` | Kaggle GPU scripts: verbatim transcription, Voxtral features (v1, v2), crop embeddings |
| `notebook/`, `deberta/`, `transcribe/` | earlier stages (LightGBM notebook v3/v5, fine-tuned DeBERTa, first transcription) |

## Kaggle datasets (all private, deterministic, computed once)
`shl-grammar-whisper-transcripts-ds` (transcripts, hand features, crops) · `shl-features` (embeddings, Voxtral, verbatim
transcripts, cached base predictions) · `shl-code` (this package).

## Reproduce
```bash
uv sync
uv run python -m kaggle.features && uv run python -m kaggle.embed      # features / embeddings (local, ~1 h on Apple M5)
OMP_NUM_THREADS=2 uv run python -m kaggle.stack_crops --out submissions/v12.csv
# or run the whole final pipeline from the cached datasets in ~3 min on CPU:
RECOMPUTE=1 INPUT_ROOT=<dir with the 3 datasets> OUT_DIR=out uv run python notebook/final.py
```
Note: `pyproject.toml` defines a `kaggle` console script that shadows the Kaggle CLI inside the venv; use `~/.local/bin/kaggle`.
