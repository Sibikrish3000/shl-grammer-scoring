# SHL Hiring Assessment 2026: spoken grammar scoring

Predict a 0-5 grammar score (MOS Likert) from 45-60 s spoken-English clips: 769 train / 216 test, evaluated by RMSE and Pearson.

| | |
|---|---|
| **Public leaderboard (RMSE)** | **0.3211** (rank ~6 of 392 when submitted; 0.6013 for the first baseline) |
| **Training RMSE** (in-sample, required) | **0.2366** |
| **Honest CV RMSE** (speaker-grouped, out-of-fold, all 769 train clips) | **0.4947**, Pearson 0.917 |
| Baseline (predict the mean) | 1.24 |

**Deliverable notebook: [`final_nb/final.ipynb`](final_nb/final.ipynb)**, executed, with outputs: exploration, speaker groups, zero gate,
models, blend, training RMSE, evaluation plots and a written report. Every submission and its public score is logged in
[SUBMISSIONS.md](SUBMISSIONS.md) (one git commit per submission).

## Approach

```
wav ─┬─► WavLM-Large (every layer, mean over time) ─────────────────┐
     ├─► Whisper-large-v3 encoder (every layer) ────────────────────┤  Ridge / SVR per representation
     ├─► Voxtral-Mini-3B audio LLM: hidden states + rubric digit ───┤  under speaker-grouped CV
     └─► Whisper transcripts (plain and verbatim-prompted)          │
              └─► DeBERTa-v3-large embeddings ────────────────────┘          │
                                       NNLS blend fitted per recording batch → clip [2,5] → speaker pooling → zero gate
```

Four things mattered more than any single model:

1. **Speaker-grouped validation.** Train speakers recorded ~3 answers each with near-identical grades; test speakers are unseen.
   Voice-similarity groups (WavLM layers 3 and 6) make CV honest; random K-fold leaks voice identity.
2. **Zero gate.** The 37 clips graded 0 are a loud noise batch; an energy-ratio rule flags all 37 with no false positive.
3. **Frozen encoders + simple heads** beat fine-tuning on 769 clips (fine-tuned DeBERTa: public 0.464; this stack: 0.32).
4. **Test-length crops.** 47% of test clips are ~45 s against 14% of train, and shorter audio lowers every prediction, so one of the two
   regimes also trains on 4 test-length crops per clip and uses verbatim-prompted transcripts (a prompt full of fillers and mistakes
   stops Whisper from "correcting" the grammar being graded). The final prediction is the mean of both regimes.

What did **not** help (all tested): a constant shift of predictions, Voxtral upper-layer / last-token features and judge digits,
hand text statistics of verbatim transcripts, Ridge alpha tuning (single models better, blend unchanged).

**Caveat.** The public set is 216 clips, so its score has a standard error of about 0.02. Differences of 0.003-0.005 between versions
are inside the noise; the grouped-CV numbers (and the private leaderboard) are the reliable measures.

## Layout

| path | what |
|---|---|
| `final_nb/` | final notebook (`final.ipynb`, generated from `notebook/final.py` with jupytext, then executed) |
| `src/kaggle/` | pipeline: `features.py` (hand / LM / VAD), `embed.py` (encoders), `crops.py`, `stack.py` (full-clip regime), `stack_crops.py` (crop regime), `post.py`, `avg.py`, `sweep.py`, `vbfeat.py`, `train_lgb.py` |
| `kaggle_jobs/` | Kaggle GPU scripts: verbatim transcription, Voxtral features (v1, v2), crop embeddings |
| `notebook/`, `deberta/`, `transcribe/` | earlier stages: LightGBM notebook (public 0.601), fine-tuned DeBERTa, first Whisper transcription |

## Reproducing

The competition data and everything derived from it (transcripts, embeddings, cached predictions) are **not** in this repo:
the competition rules forbid sharing the data, and the derived datasets are kept private on Kaggle. To rebuild:

```bash
uv sync
# data/ must contain the competition files: train.csv, test.csv, train/, test/
uv run python -m kaggle.features                     # hand / LM-grammar / VAD features
uv run python -m kaggle.embed                        # WavLM, Whisper encoder, DeBERTa-large (tens of minutes each on an Apple M5)
uv run python -m kaggle.crops                        # 4 test-length crops per train clip
# GPU steps on Kaggle (scripts in kaggle_jobs/); download their outputs to features/verbatim, features/voxtral, features/emb:
#   transcribe_verbatim (verbatim transcripts), voxtral (Voxtral features), embed_crops (WavLM / Whisper crop embeddings)
uv run python -m kaggle.embed text_verbatim          # DeBERTa-large over the verbatim transcripts
OMP_NUM_THREADS=2 uv run python -m kaggle.stack --models voxtral_ridge,voxtral_svr,audio_vox_ridge,hand_judge_svr,whisper_svr,text_ridge --out submissions/v10.csv
OMP_NUM_THREADS=2 uv run python -m kaggle.stack_crops --out submissions/v12.csv
uv run python -m kaggle.avg submissions/v13.csv submissions/v10.csv submissions/v12.csv   # the submitted file: mean of the two regimes
```

`notebook/final.py` runs the whole modelling pipeline (13 base models, nested CV, blend, training RMSE, plots) from the cached
features; with the features in place it takes about 3 minutes on CPU (`RECOMPUTE=1`). Note: use `~/.local/bin/kaggle` for the Kaggle CLI
when a `kaggle` Python package shadows it.

## Credits

Several ideas were taken from public solutions of this competition and re-implemented here: speaker-grouped validation, the zero gate and
per-batch NNLS stacking ([plutus123/Grammer_Scoring_Engine](https://github.com/plutus123/Grammer_Scoring_Engine)); multimodal
embeddings with non-negative stacking and speaker pooling ([shiv-ml/SHL-Hiring-Assessment](https://github.com/shiv-ml/SHL-Hiring-Assessment));
the verbatim Whisper prompt, test-length crops and Voxtral-Mini-3B audio-LLM features
([Techy-A/shl-grammar-scoring](https://github.com/Techy-A/shl-grammar-scoring); the structure of `kaggle_jobs/voxtral/run.py` follows their
extractor). Models used: WavLM-Large, Whisper-large-v3 / turbo, Voxtral-Mini-3B-2507, DeBERTa-v3-large, Silero VAD.
