# Submission log

Public leaderboard = RMSE-based, lower is better (#1 at 0.3064 on 2026-10-10).
gCV = speaker-grouped CV RMSE on graded (non-zero) train clips.

| Ver | Date | Kaggle ref | What changed | CV | Public |
|---|---|---|---|---|---|
| v3 | 2026-10-10 | 57036977 | LightGBM on text + CoLA/GPT-2 + Silero VAD features (notebook) | random-fold 0.801 | 0.6013 |
| v5 | 2026-10-10 | 57038122 | + fine-tuned DeBERTa-v3-base OOF stacked (notebook) | random-fold 0.737 | 0.4640 |
| v6 | 2026-10-10 | 57039135 | zero gate + speaker-grouped CV; NNLS per 45 s batch; hand LGB/SVR + WavLM-L L19-21 Ridge/SVR; clip [2,5]; speaker pool 0.5 | gCV 0.553 | 0.3753 |
| v7 | 2026-10-10 | 57039497 | v6 + Whisper-large-v3 encoder L30-32 Ridge/SVR | gCV 0.542 | 0.3598 |
| v8 | 2026-10-10 | 57039732 | v7 + frozen DeBERTa-v3-large transcript embeddings L21-23 Ridge/SVR | gCV 0.519 | 0.3345 |
