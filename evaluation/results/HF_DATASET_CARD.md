---
language:
- mr
task_categories:
- text-to-speech
- automatic-speech-recognition
tags:
- marathi
- tts-evaluation
- wer
- cer
pretty_name: Marathi Indic-Speak QLoRA Evaluation
---

# Marathi Indic-Speak QLoRA Evaluation

This dataset stores reproducible evaluation outputs for
`PrakashPask/marathi-indic-speak-qlora` on the official Rasa Marathi test split.

Each uploaded run contains the selected text and conditioning metadata, the
exact Rasa reference WAV, adapter-generated WAV, Whisper transcription,
per-sample WER/CER, and measured full-waveform response time.


Latest packaged run: `rasa_marathi_n015_seed3407` (15 rows)

- Corpus WER: 66.9388%
- Corpus CER: 20.7738%
- Base model: `bodhan-ai/indic-speak`
- Adapter: `PrakashPask/marathi-indic-speak-qlora`
- ASR: `openai/whisper-large-v3-turbo`
- Source dataset: `ai4bharat/Rasa`, Marathi test split

Reference recordings come from Rasa (CC-BY-4.0). Generated artifacts are
derived from Indic-Speak and the published PEFT adapter; consult their model
cards and licenses before redistribution or commercial use.

Repository: https://huggingface.co/datasets/PrakashPask/TTS-Evaluvation
