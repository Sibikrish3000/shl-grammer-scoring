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
| probe | 2026-10-10 | 57040069 | v8 - 0.12 constant shift (reference repo found +0.118 test bias) -> worse, no shift for us | - | 0.3416 |
| v9 | 2026-10-10 | 57040138 | 3-model NNLS stack: all-embedding concat Ridge + Whisper-enc SVR + DeBERTa-L text Ridge (same gCV as v8, simpler) | gCV 0.519 | 0.3357 |
| v10 | 2026-10-10 | 57041052 | v9 + Voxtral-Mini-3B hidden-state Ridge/SVR, audio-concat+Voxtral Ridge, judge-digit SVR (6 models) | gCV 0.513 | **0.3255** |
| v11 | 2026-10-10 | 57043310 | audio-only models (WavLM/Whisper/Voxtral Ridge/SVR) trained on full + 4 test-length crops, scored on crops; no text models | crop-val 0.525 | 0.3287 |
| v12 | 2026-10-10 | 57044529 | v11 + verbatim-prompt Whisper-turbo transcripts (crop transcripts) -> DeBERTa-L Ridge/SVR; 7 models all on full+4 crops | crop-val 0.516 | **0.3230** |
| v13 | 2026-10-10 | 57044537 | plain mean of v10 (full-clip models) and v12 (crop-trained models), corr 0.987 | - | **0.3211** |
| (no submit) | 2026-10-10 | - | Ridge alpha sweep on crop stack: alpha 3 best per model (voxtral 0.5555->0.5515) but stack CV unchanged 0.5162; verbatim hand features (vb_*) alone 0.845, zero NNLS weight -> dropped | crop-val 0.516 | - |
| (no submit) | 2026-10-10 | - | Voxtral v2 (errors prompt; upper layers 60-90% pooled; last-token states): single models 0.565 / 0.573, judge digit corr -0.41 but stack crop-val 0.5164 (pooled/last) and 0.5184 (+judge) vs 0.5163 -> no gain, corr(v12,v14)=0.9975, not submitted | crop-val 0.516 | - |
| (final nb) | 2026-10-10 | - | final notebook (final_nb/, Kaggle: shl-grammar-scoring-speaker-aware-stack) rebuilds v13 bit-identically; training RMSE 0.2366, grouped-CV RMSE 0.4947 / Pearson 0.917 (all train clips) | 0.4947 | 0.3211 (= v13) |
