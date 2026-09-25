# Marathi Indic-Speak Adapter Evaluation

Evaluated rows: **15** from the official Rasa Marathi test split.

| Measure | What it tells you | Result |
|---|---|---:|
| Intelligibility (WER / CER) | How reliably ASR can recover the reference text | **66.94% / 20.77%** |
| Response time | Local time until the complete WAV is returned | **mean 72.931s; p50 65.219s; p95 177.865s** |

## Reproducibility

- Dataset: `ai4bharat/Rasa` / `Marathi` / `test`
- Dataset revision: `632f55c7ac590219d41cd7adffce5b440e4604f5`
- Base model: `bodhan-ai/indic-speak` at `76e0814f189321efe550011850cd288dae366aad`
- Adapter: `PrakashPask/marathi-indic-speak-qlora` at `e4f292cbc78a7262b3d89b751ab49c90da053cef`
- ASR: `openai/whisper-large-v3-turbo` at `41f01f3fe87f28c78e2fbf8b568835947dd65ed9`
- Selection seed: `3407`
- Exact train-text overlap excluded: `True`
- Generation GPU: `NVIDIA L4`
- Successful generations: `15`
- Failed generations: `0`

## Interpretation note

WER and CER are corpus error rates calculated by `jiwer` after NFC, case, punctuation, and whitespace normalization. Recorded synthesis failures receive an empty ASR hypothesis and remain in the metric; missing files and ASR infrastructure errors stop report creation.

Response time is full-waveform latency from the local non-streaming Indic-Speak call. It must not be compared directly with the model card's approximately 200 ms delay-before-audio value because its serving setup and measurement protocol are not published.

The model card does not publish the sample set, ASR model, text normalization, hardware, or full latency protocol behind 26.49 / 17.86, so this run is a reproducible project result, not a reproduction claim.
