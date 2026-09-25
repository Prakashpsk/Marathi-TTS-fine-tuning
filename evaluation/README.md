# Marathi adapter evaluation

This folder is independent of the project's existing training and evaluation
code. It evaluates only these three measurements on rows selected directly from
the official `ai4bharat/Rasa` Marathi `test` split:

| Measure | Meaning | Output |
|---|---|---|
| WER | Word errors in an ASR transcript of generated audio | Corpus percentage |
| CER | Character errors in an ASR transcript of generated audio | Corpus percentage |
| Response time | Local wall time until Indic-Speak returns the complete WAV | Mean, p50, and p95 |

## Completed Lightning AI evaluation

A 15-row evaluation completed successfully on an NVIDIA L4. The values below come directly from `results/summary.json`. The selection was deterministic (`seed=3407`), used the official Rasa Marathi `test` split, and excluded normalized transcripts that also occur in the training split.

| Measurement | Result |
|---|---:|
| Evaluated rows | 15 |
| Successful generations | 15 |
| Failed generations | 0 |
| Corpus WER | 66.9388% |
| Corpus CER | 20.7738% |
| Mean full-waveform response time | 72.930726 s |
| Median (P50) response time | 65.218673 s |
| P95 response time | 177.864861 s |
| Minimum response time | 21.259229 s |
| Maximum response time | 191.277885 s |
| Generation GPU | NVIDIA L4 |
| ASR | `openai/whisper-large-v3-turbo` |

The exact resolved revisions were:

- Base model: `76e0814f189321efe550011850cd288dae366aad`
- Adapter: `e4f292cbc78a7262b3d89b751ab49c90da053cef`
- ASR: `41f01f3fe87f28c78e2fbf8b568835947dd65ed9`
- Dataset: `632f55c7ac590219d41cd7adffce5b440e4604f5`

This is an initial 15-sample adapter evaluation, not a statistically strong benchmark. It also does not establish improvement over the base model because the base model was not evaluated on the identical rows with the same ASR and decoding configuration.

The complete local artifacts are in `evaluation/results/`, including `summary.json`, `REPORT.md`, the selected sample metadata, reference and generated audio, ASR transcripts, and the exported evaluation dataset.

## Why these libraries are used

There are reliable prebuilt components, so the script does not implement its
own speech recognizer or edit-distance metric:

- `datasets` reads the official Rasa `test` split.
- Indic-Speak's pinned official `inference.py` generates 24 kHz audio.
- `peft.PeftModel.from_pretrained` applies
  `PrakashPask/marathi-indic-speak-qlora` to the causal language model inside
  that official TTS pipeline.
- Transformers Whisper (`openai/whisper-small` by default) transcribes Marathi.
- `jiwer` calculates corpus WER and CER.
- Python's monotonic `time.perf_counter` measures local response time, with a
  CUDA synchronization before and after each call.

The short example that loads `AutoModelForCausalLM` plus `PeftModel` is enough
to obtain model logits, but not a WAV. Indic-Speak also needs its prompt
formatting, code unpacking, SNAC quantizer, and Vocos waveform decoder.
Therefore, this evaluator loads the complete official `TTS` pipeline first and
attaches the PEFT adapter to the pipeline's internal causal language model. This
is the same adapter operation, placed where audio generation can work end to
end.

## Installation

Use Python 3.10 or newer in a clean virtual environment if possible. Install a
CUDA-enabled PyTorch build appropriate for the machine, then install the
remaining packages:

    python -m pip install -r evaluation/requirements.txt

On Windows, replace `python` with `py -3` if the `python` command is not
configured.

Log in to Hugging Face and accept any access conditions shown on the Rasa and
Indic-Speak repository pages:

    hf auth login

The token can instead be supplied through the standard `HF_TOKEN` environment
variable.
## Lightning AI standalone usage

When this folder is uploaded directly as
`/teamspace/studios/this_studio/evaluvation`, run:

    cd /teamspace/studios/this_studio/evaluvation
    python -m pip install -r requirements.txt
    hf auth login

Run a 15-sample evaluation:

    python evaluate.py \
      --num-samples 15 \
      --asr-model openai/whisper-large-v3-turbo \
      --asr-max-new-tokens 440 \
      --fail-fast

The official test rows whose normalized text occurs in training are excluded by
default. Do not add `--allow-train-text-overlap` to a formal evaluation.

If generation has already completed and only ASR scoring must be repeated:

    python evaluate.py \
      --stage score \
      --num-samples 15 \
      --asr-model openai/whisper-large-v3-turbo \
      --asr-max-new-tokens 440 \
      --overwrite \
      --fail-fast

`openai/whisper-small` reached its decoder-token limit on one generated sample
in the observed run. Whisper Large V3 Turbo completed all 15 rows. WER values
above 100% are valid when ASR word insertions exceed the number of reference
words.

## Run any sample count

From the project root:

    python evaluation/evaluate.py --num-samples 15
    python evaluation/evaluate.py --num-samples 20
    python evaluation/evaluate.py --num-samples 30
    python evaluation/evaluate.py --num-samples 100

Any positive count up to the number of valid test rows is accepted. Each count
gets a separate directory, for example:

    evaluation/results/rasa_marathi_n020_seed3407/
    evaluation/results/rasa_marathi_n030_seed3407/
    evaluation/results/rasa_marathi_n100_seed3407/

This prevents a 20-row run from overwriting a 100-row run. `--seed` changes the
deterministic sample. The selection is performed before synthesis, is balanced
between the two Marathi Rasa speaker labels, and never uses generated quality
to choose or remove rows. By default, rows whose normalized transcript occurs
in the Rasa `train` split are excluded, so the result measures unseen text
rather than the known official train/test transcript overlap. Use
`--allow-train-text-overlap` only when a literal sample of the published test
distribution is required; that choice is recorded in the report.

For a custom result location:

    python evaluation/evaluate.py --num-samples 30 --output-dir D:\eval_runs\n30

## Run one stage at a time

The default `all` stage selects, generates, and scores. Long jobs can instead be
resumed stage by stage:

    python evaluation/evaluate.py --num-samples 30 --stage select
    python evaluation/evaluate.py --num-samples 30 --stage generate
    python evaluation/evaluate.py --num-samples 30 --stage score

Completed WAVs and ASR transcripts are reused only when their configuration and
SHA-256 hashes match. Add `--overwrite` when a run should be rebuilt. A recorded
synthesis failure stays in the WER/CER denominator as an empty hypothesis. A
missing/corrupt WAV or ASR infrastructure failure stops report creation instead
of publishing a misleading metric; fix the error and rerun the score stage.

Useful alternatives:

    python evaluation/evaluate.py --num-samples 30 --asr-model openai/whisper-large-v3-turbo
    python evaluation/evaluate.py --num-samples 30 --asr-device cpu --stage score
    python evaluation/evaluate.py --help

Use the same ASR checkpoint for every model being compared. Changing the ASR
checkpoint can change WER and CER even when the TTS audio is identical.

## Files produced

- `selected_samples.json` and `.csv`: exact official test rows used.
- `selection_summary.json`: dataset revision, seed, rejection counts, and sample
  distribution.
- `audio/`: adapter-generated WAV files.
- `generation_config.json`: resolved base/adapter revisions, decoding
  configuration, GPU, software versions, and a configuration signature.
- `generation_manifest.json`: generation status, audio duration, and measured
  response time plus WAV SHA-256 for every row.
- `asr_config.json`: resolved Whisper revision and scoring configuration.
- `asr_transcripts.json` and `.csv`: reference, ASR hypothesis, and per-row
  WER/CER.
- `summary.json`: machine-readable corpus metrics.
- `REPORT.md`: the final two-row project report table and reproducibility
  details.

WER and CER in `summary.json` are true corpus rates (total edit errors divided
by total reference units), not a simple average of per-row percentages. Text is
normalized with Unicode NFC; punctuation and symbols are removed while
Devanagari combining marks are preserved.

## Response-time limitation

Indic-Speak's local `TTS` call is non-streaming: it returns only after the whole
waveform is ready. Consequently this script reports full-waveform response time,
not time to first audio. The unmeasured warm-up call reduces one-time CUDA setup
bias. A longer sentence normally takes more time than a short sentence.

The model card publishes an approximately 200 ms delay-before-audio number, but
does not specify the serving implementation, hardware, or detailed protocol.
It is therefore not directly comparable to this local measurement. Likewise,
the card does not publish enough detail to reproduce its 26.49 WER / 17.86 CER
exactly; this script creates a transparent project-specific result instead.

## Hardware guidance

- NVIDIA GPU: 16 GB VRAM is a practical minimum for the official BF16 TTS
  loader; 24 GB is recommended for a smoother 100-row run.
- System RAM: 32 GB recommended.
- Free disk: allow roughly 25-40 GB for Hugging Face caches, model files, and
  generated WAVs.
- ASR runs after TTS is unloaded, so both large models are not intentionally
  kept in VRAM together. Whisper can run on CPU with `--asr-device cpu`, but it
  will be much slower.

GPU type and sentence lengths affect response time. The evaluator records the
GPU name and relevant software versions in `generation_config.json` and
`summary.json` for paper reporting.

## Publish a completed run to Hugging Face

After `REPORT.md` and `summary.json` have been produced, package and upload the
run to the evaluation dataset repository:

    python evaluation/publish_results.py \
      --run-dir evaluation/results/rasa_marathi_n015_seed3407 \
      --repo-id PrakashPask/TTS-Evaluvation

The publisher uses the active `hf auth login` token. It downloads only the
exact selected Rasa reference recordings, writes `evaluation_dataset.jsonl`
and `.csv`, and uploads the run under `runs/<run-name>/`. Each row contains the
text and conditioning metadata, reference audio, predicted audio, Whisper
transcription, WER, CER, and total full-waveform latency.


Prepare and inspect the dataset locally without uploading:

    python evaluation/publish_results.py \
      --run-dir evaluation/results/rasa_marathi_n001_seed3407 \
      --prepare-only
## Download code or results from Lightning AI

From `/teamspace/studios/this_studio`, the uploaded standalone folder is named
`evaluvation` in the recorded run.

Create a code-only archive:

    cd /teamspace/studios/this_studio
    zip evaluation_code.zip \
      evaluvation/evaluate.py \
      evaluvation/publish_results.py \
      evaluvation/requirements.txt \
      evaluvation/README.md

Create an archive containing only the completed 15-sample result:

    zip -r evaluation_results_15.zip \
      evaluvation/results/rasa_marathi_n015_seed3407

Verify the archives before downloading them through the Lightning file browser:

    ls -lh evaluation_code.zip evaluation_results_15.zip
