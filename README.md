---
language:
- mr
library_name: peft
base_model: bodhan-ai/indic-speak
datasets:
- ai4bharat/Rasa
pipeline_tag: text-to-speech
tags:
- marathi
- text-to-speech
- speech-generation
- peft
- lora
- qlora
- snac
---
# Marathi Indic-Speak Adaptation with QLoRA

This repository contains a completed QLoRA adaptation of [`bodhan-ai/indic-speak`](https://huggingface.co/bodhan-ai/indic-speak) for Marathi text-to-speech using the Marathi configuration of [`ai4bharat/Rasa`](https://huggingface.co/datasets/ai4bharat/Rasa).

The experiment trains LoRA adapters on audio-token prediction while keeping the 4-bit base model, SNAC codec, and Vocos decoder frozen. The official Rasa `train` split is used for optimization and the official `test` split is held out for evaluation.
## Project links

- **Training code and notebook:** [GitHub repository](https://github.com/Prakashpsk/Marathi-TTS-fine-tuning)
- **Fine-tuned QLoRA adapter:** [Hugging Face model](https://huggingface.co/PrakashPask/marathi-indic-speak-qlora)
- **Base model:** [Bodhan AI Indic-Speak](https://huggingface.co/bodhan-ai/indic-speak)
- **Training dataset:** [AI4Bharat Rasa](https://huggingface.co/datasets/ai4bharat/Rasa)

## Results

| Measurement | Result |
|---|---:|
| Training examples | 26,960 |
| Test examples | 2,995 |
| Epochs | 1 |
| Optimizer steps | 3,370 |
| Train loss | 3.8043 |
| Test loss | 3.7393 |
| Runtime | 9,594.42 seconds (~2 h 40 min) |
| Throughput | 2.81 samples/second |
| Peak allocated GPU memory | 12.43 GiB |
| GPU | NVIDIA H200 |
| Compute dtype | `torch.bfloat16` |

Loss measures teacher-forced audio-token prediction. It does not directly measure pronunciation, naturalness, speaker similarity, or intelligibility.

## Method

The notebook:

1. Pins the exact Indic-Speak model and Rasa dataset revisions.
2. Loads the official Marathi train and test splits.
3. Decodes audio to mono and resamples it to 24 kHz when necessary.
4. Encodes audio with the frozen `hubertsiuzdak/snac_24khz` codec.
5. Packs the three SNAC levels into Indic-Speak's seven-token-per-frame layout.
6. Builds prompts with Indic-Speak's official `build_prompt`, including text, style, and gender-derived speaker conditioning.
7. Masks prompt tokens so loss is calculated only over audio and termination tokens.
8. Fine-tunes LoRA adapters on the training split and evaluates once on the test split.
9. Reloads the adapter into the official Indic-Speak TTS implementation and generates WAV comparisons through Vocos.

### QLoRA configuration

| Setting | Value |
|---|---|
| Base model | `bodhan-ai/indic-speak` |
| Quantization | 4-bit NF4 with double quantization |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Trainable parameters | 24,313,856 (0.7312%) |
| Learning rate | `1e-4` |
| Micro-batch / gradient accumulation | 1 / 8 |
| Optimizer | Paged AdamW 8-bit |
| Scheduler | Cosine with 170 warmup steps |
| Gradient checkpointing | Enabled |
| Seed | 3407 |

The immutable revisions and complete settings are recorded in [`run_config.json`](run_config.json).

## Dataset and conditioning

All 26,960 training rows and 2,995 test rows were successfully encoded. No gender, style, row-count, duration, or arbitrary token-length filter was applied. Both genders and all available styles were retained.

Rasa has no speaker ID, so gender is only a proxy for conditioning:

```text
MALE / M   -> MarathiRasaMale
FEMALE / F -> MarathiRasaFemale
```

These labels represent gender groups, not verified individual speakers.

The split-overlap audit found:

- 0 overlapping filenames
- 0 overlapping decoded-audio hashes
- 825 exact transcript overlaps

Repeated transcripts do not mean the recordings are duplicated, but they should be considered when interpreting test loss.

## Audio fertility

Audio fertility is the number of base SNAC frames per whitespace-delimited transcript word. Each frame is serialized as seven audio tokens.

| Split | Clips | Words | SNAC frames | Mean frames/word | Median | P05 | P95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 26,960 | 305,293 | 2,116,724 | 7.6020 | 7.0 | 4.8333 | 12.5000 |
| Test | 2,995 | 33,840 | 233,611 | 7.5799 | 7.0 | 4.8967 | 12.3333 |

Whitespace segmentation is an approximation for Marathi. Fertility is a preprocessing diagnostic, not a speech-quality metric. Per-clip values are in [`manifest.json`](manifest.json), and summaries are in [`audio_fertility.json`](audio_fertility.json).

## Repository contents

| Path | Description |
|---|---|
| `IndicSpeak_Marathi_Rasa_QLoRA_.ipynb` | End-to-end training and evaluation notebook |
| `adapter/` | Final PEFT LoRA adapter and tokenizer |
| `checkpoints/` | Resumable optimizer, scheduler, and model checkpoints |
| `train_tokens/`, `test_tokens/` | Preprocessed Hugging Face datasets |
| `manifest.json` | Per-example metadata, hashes, lengths, and fertility |
| `run_config.json` | Pinned revisions and run settings |
| `metrics.json` | Final training and test metrics |
| `training_log.json` | Step-level Trainer history |
| `loss.png` | Training-loss curve |
| `preprocessing_stats.json` | Split statistics and overlap audit |
| `requirements.lock.txt` | Captured Python environment |
| `baseline_*.wav`, `finetuned_*.wav` | Generated comparison pairs |
| `audio_comparison.json` | Comparison prompts, settings, filenames, and durations |
| `reference_*.wav` | Source examples exported during preprocessing |

## Reproducing the run

Use a fresh Linux/Colab CUDA environment. Accept the access conditions for Indic-Speak and Rasa, then provide a Hugging Face read token through `HF_TOKEN` or the notebook's hidden prompt. Run `IndicSpeak_Marathi_Rasa_QLoRA_.ipynb` from top to bottom.

The run used an NVIDIA H200. Although measured peak allocation was 12.43 GiB, total requirements vary with sequence length, model loading, temporary allocations, and library versions. A CUDA GPU with at least 24 GiB is a practical starting point.

## Loading the adapter

The following example loads the pinned base language model and attaches the PEFT adapter. Accept the Indic-Speak access conditions on Hugging Face before running it.

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL_ID = "bodhan-ai/indic-speak"
BASE_MODEL_REVISION = "76e0814f189321efe550011850cd288dae366aad"
ADAPTER_ID = "PrakashPask/marathi-indic-speak-qlora"

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    revision=BASE_MODEL_REVISION,
    torch_dtype="auto",
    device_map="auto",
)

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_ID,
    is_trainable=False,
)
model.eval()
```

This loads the adapted language model but does not by itself generate a WAV file. For speech synthesis, construct Bodhan's official Indic-Speak TTS object and attach the adapter to its language-model attribute:

```python
from peft import PeftModel

tts.lm = PeftModel.from_pretrained(
    tts.lm,
    "PrakashPask/marathi-indic-speak-qlora",
    is_trainable=False,
)
tts.lm.eval()
```

Use the official prompt construction, audio-token conversion, and Vocos decoding path. The notebook contains the complete revision-pinned reload procedure.
## Generated comparisons

Four baseline/fine-tuned pairs use the same prompt, speaker label, random seed, temperature (`0.6`), and top-p (`0.9`). Three prompts come from the held-out test manifest and one is a manually added neutral Marathi prompt.

The fine-tuned single-word `PROPER NOUN` example reached `max_new_tokens=2520` without emitting the end-of-speech token. Its 30.72-second output is likely truncated/runaway generation and must be treated as a failure.

Subjective listening is still required to assess:

- Marathi pronunciation and intelligibility
- Missing, repeated, or hallucinated words
- Naturalness and prosody
- Gender/style conditioning and voice consistency
- Baseline-versus-adapter preference

No perceptual-quality improvement is claimed from token loss alone.

## Limitations

- Training ran for only one epoch.
- Test loss is not a perceptual speech-quality measurement.
- Possible overlap with the base model's pretraining data is unknown.
- Gender conditioning is not a stable speaker identity.
- The official splits contain 825 exact-text overlaps.
- At least one short prompt caused runaway generation.
- Fertility outliers were reported but not removed.
- Marathi text encoding/rendering should be verified when moving artifacts between environments.
- No formal listening study or ASR-based intelligibility evaluation has been completed.

## Licenses and attribution

This project uses Indic-Speak from Bodhan AI / AI4Bharat, the Rasa dataset, SNAC, and Indic-Speak's official Vocos inference path. Follow the Indic Open Model License and Rasa's CC-BY-4.0 terms when distributing models, outputs, or derivatives. The supplied Orpheus notebook credits Unsloth and Etherll and carries LGPL-3.0 terms.

## Reference papers

The implementation is based on the following methods:

1. **SNAC — Multi-Scale Neural Audio Codec**  
   Hubert Siuzdak, Florian Grötschla, and Luca A. Lanzendörfer. *SNAC: Multi-Scale Neural Audio Codec*. arXiv:2410.14411, 2024.  
   Paper: https://arxiv.org/abs/2410.14411  
   In this project, the frozen `hubertsiuzdak/snac_24khz` model encodes 24 kHz speech into hierarchical discrete audio codes. Its three codec levels are packed into the seven-codebook token layout expected by Indic-Speak. SNAC is used as the audio tokenizer and is not fine-tuned.

2. **LoRA — Low-Rank Adaptation of Large Language Models**  
   Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685, 2021.  
   Paper: https://arxiv.org/abs/2106.09685  
   This project freezes the pretrained model weights and trains low-rank adapter matrices in the attention and MLP projection layers. With rank 16, only 24,313,856 parameters (0.7312% of the model) are trainable.

3. **QLoRA — Efficient Finetuning of Quantized LLMs**  
   Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. *QLoRA: Efficient Finetuning of Quantized LLMs*. arXiv:2305.14314, 2023.  
   Paper: https://arxiv.org/abs/2305.14314  
   The base Indic-Speak language model is loaded with 4-bit NF4 quantization and double quantization, while gradients update the LoRA adapters using `bfloat16` computation. The run also uses a paged 8-bit AdamW optimizer, following the memory-efficient QLoRA approach.

### BibTeX

```bibtex
@article{siuzdak2024snac,
  title   = {SNAC: Multi-Scale Neural Audio Codec},
  author  = {Siuzdak, Hubert and Gr{\"o}tschla, Florian and Lanzend{\"o}rfer, Luca A.},
  journal = {arXiv preprint arXiv:2410.14411},
  year    = {2024}
}

@article{hu2021lora,
  title   = {LoRA: Low-Rank Adaptation of Large Language Models},
  author  = {Hu, Edward J. and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  journal = {arXiv preprint arXiv:2106.09685},
  year    = {2021}
}

@article{dettmers2023qlora,
  title   = {QLoRA: Efficient Finetuning of Quantized LLMs},
  author  = {Dettmers, Tim and Pagnoni, Artidoro and Holtzman, Ari and Zettlemoyer, Luke},
  journal = {arXiv preprint arXiv:2305.14314},
  year    = {2023}
}
```
