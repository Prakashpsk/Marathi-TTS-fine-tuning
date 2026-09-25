"""Evaluate the Marathi Indic-Speak PEFT adapter on the official Rasa test split.

Only three metrics are reported:

* corpus word error rate (WER)
* corpus character error rate (CER)
* local full-waveform response time

The official Indic-Speak inference code returns a complete waveform, so the
timing in this script is not streaming time-to-first-audio.  Selection is done
before synthesis and failed generations remain in the WER/CER denominator.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import platform
import statistics
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = HERE / "results"

DEFAULT_DATASET = "ai4bharat/Rasa"
DEFAULT_DATASET_CONFIG = "Marathi"
DEFAULT_DATASET_REVISION = "632f55c7ac590219d41cd7adffce5b440e4604f5"

DEFAULT_BASE_MODEL = "bodhan-ai/indic-speak"
DEFAULT_MODEL_REVISION = "76e0814f189321efe550011850cd288dae366aad"
DEFAULT_ADAPTER = "PrakashPask/marathi-indic-speak-qlora"
DEFAULT_ASR_MODEL = "openai/whisper-small"

SAMPLE_RATE = 24_000
ASR_SAMPLE_RATE = 16_000


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    """Write JSON atomically so an interrupted run keeps its last checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def archive_published_report(run_dir: Path, reason: str) -> None:
    published = [run_dir / "summary.json", run_dir / "REPORT.md"]
    existing = [path for path in published if path.exists()]
    if not existing:
        return
    archive_dir = run_dir / "previous_reports"
    archive_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.time_ns()
    for path in existing:
        path.replace(archive_dir / f"{stamp}_{path.name}")
    write_json(
        run_dir / "evaluation_status.json",
        {"status": "invalidated", "reason": reason},
    )


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_environment() -> dict[str, Any]:
    packages = {}
    for name in (
        "accelerate",
        "datasets",
        "huggingface-hub",
        "jiwer",
        "peft",
        "snac",
        "torch",
        "transformers",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    information: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }
    try:
        import torch

        information.update(
            {
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "cudnn_version": (
                    torch.backends.cudnn.version() if torch.cuda.is_available() else None
                ),
                "gpu_names": [
                    torch.cuda.get_device_name(index)
                    for index in range(torch.cuda.device_count())
                ],
            }
        )
    except ImportError:
        information.update(
            {
                "cuda_available": False,
                "cuda_version": None,
                "cudnn_version": None,
                "gpu_names": [],
            }
        )
    return information


def normalize_text(text: str) -> str:
    """Normalize Marathi text without discarding Devanagari vowel marks."""
    text = unicodedata.normalize("NFC", str(text)).lower()
    kept: list[str] = []
    for character in text:
        category = unicodedata.category(character)
        if character.isspace() or category[0] in {"L", "M", "N"}:
            kept.append(character)
        else:
            kept.append(" ")
    return " ".join("".join(kept).split())


def frequency(rows: Iterable[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get(field, ""))] += 1
    return dict(sorted(counts.items()))


def speaker_from_gender(gender: Any) -> str | None:
    value = str(gender or "").strip().upper()
    if value in {"M", "MALE"}:
        return "MarathiRasaMale"
    if value in {"F", "FEMALE"}:
        return "MarathiRasaFemale"
    return None


def stable_key(seed: int, row: dict[str, Any]) -> str:
    identity = "|".join(
        [
            str(seed),
            str(row.get("filename", "")),
            str(row.get("text", "")),
            str(row.get("gender", "")),
            str(row.get("style", "")),
        ]
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def balanced_sample(
    candidates: Sequence[dict[str, Any]], count: int, seed: int
) -> list[dict[str, Any]]:
    """Select deterministically, balancing the two adapter speaker labels."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        groups[row["speaker_label"]].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: stable_key(seed, row))

    labels = ["MarathiRasaFemale", "MarathiRasaMale"]
    if not all(groups[label] for label in labels):
        raise RuntimeError("The Rasa Marathi test split did not contain both genders.")

    # Alternate which gender receives the extra row for odd sample counts.
    female_target = count // 2 + (count % 2 if seed % 2 == 0 else 0)
    male_target = count - female_target
    targets = {
        "MarathiRasaFemale": female_target,
        "MarathiRasaMale": male_target,
    }
    if any(targets[label] > len(groups[label]) for label in labels):
        raise ValueError(
            "Requested sample count cannot be balanced with the available rows: "
            + str({label: len(groups[label]) for label in labels})
        )

    selected = [
        row
        for label in labels
        for row in groups[label][: targets[label]]
    ]
    selected.sort(key=lambda row: stable_key(seed, row))
    return selected


def resolve_run_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()
    name = f"rasa_marathi_n{args.num_samples:03d}_seed{args.seed}"
    return (DEFAULT_RESULTS_ROOT / name).resolve()


def select_samples(args: argparse.Namespace, run_dir: Path) -> Path:
    selected_path = run_dir / "selected_samples.json"
    selection_config = {
        "schema_version": 2,
        "dataset_id": args.dataset_id,
        "dataset_config": args.dataset_config,
        "dataset_revision": args.dataset_revision,
        "split": "test",
        "num_samples": args.num_samples,
        "seed": args.seed,
        "allow_train_text_overlap": args.allow_train_text_overlap,
        "deduplicate_normalized_test_text": True,
        "balance_adapter_speaker_labels": True,
    }
    if selected_path.exists() and not args.overwrite:
        selected = read_json(selected_path)
        summary_path = run_dir / "selection_summary.json"
        saved_summary = read_json(summary_path) if summary_path.exists() else {}
        if len(selected) != args.num_samples:
            raise ValueError(
                f"{selected_path} has {len(selected)} rows, but --num-samples is "
                f"{args.num_samples}. Choose another --output-dir or use --overwrite."
            )
        if saved_summary.get("selection_config") != selection_config:
            raise ValueError(
                "The existing selection was created with different dataset, seed, "
                "count, or overlap settings. Choose another --output-dir or use "
                "--overwrite."
            )
        if saved_summary.get("selected_samples_sha256") != canonical_hash(selected):
            raise ValueError(
                f"{selected_path} does not match its recorded hash. Use a clean "
                "--output-dir or regenerate it with --overwrite."
            )
        print(f"Reusing {len(selected)} previously selected rows: {selected_path}")
        return selected_path

    try:
        from datasets import Audio, load_dataset
    except ImportError as error:
        raise RuntimeError(
            "Missing datasets. Install evaluation/requirements.txt first."
        ) from error

    print(
        f"Loading {args.dataset_id}/{args.dataset_config} official test split "
        f"at revision {args.dataset_revision} ...",
        flush=True,
    )
    train_texts: set[str] = set()
    if not args.allow_train_text_overlap:
        print(
            "Scanning train transcripts to exclude exact text overlap ...",
            flush=True,
        )
        train_dataset = load_dataset(
            args.dataset_id,
            args.dataset_config,
            split="train",
            streaming=True,
            revision=args.dataset_revision,
        )
        train_columns = train_dataset.column_names or []
        if "audio" in train_columns:
            train_dataset = train_dataset.cast_column("audio", Audio(decode=False))
        for source in train_dataset:
            normalized = normalize_text(source.get("text") or "")
            if normalized:
                train_texts.add(normalized)

    dataset = load_dataset(
        args.dataset_id,
        args.dataset_config,
        split="test",
        streaming=True,
        revision=args.dataset_revision,
    )
    # The source audio is not needed for TTS evaluation.  Disabling decoding
    # avoids downloading/decoding every WAV while scanning test metadata.
    test_columns = dataset.column_names or []
    if "audio" in test_columns:
        dataset = dataset.cast_column("audio", Audio(decode=False))

    candidates: list[dict[str, Any]] = []
    rejected: dict[str, int] = defaultdict(int)
    seen_text: set[str] = set()
    for source_index, source in enumerate(dataset):
        text = unicodedata.normalize("NFC", str(source.get("text") or "")).strip()
        normalized = normalize_text(text)
        speaker = speaker_from_gender(source.get("gender"))
        style = str(source.get("style") or "").strip().upper()
        if not normalized:
            rejected["empty_text"] += 1
            continue
        if normalized in train_texts:
            rejected["exact_train_text_overlap"] += 1
            continue
        if normalized in seen_text:
            rejected["duplicate_normalized_text"] += 1
            continue
        if speaker is None:
            rejected["unsupported_gender"] += 1
            continue
        if not style:
            rejected["missing_style"] += 1
            continue
        seen_text.add(normalized)
        candidates.append(
            {
                "source_index": source_index,
                "filename": str(source.get("filename") or f"row_{source_index:06d}"),
                "text": text,
                "language": str(source.get("language") or "Marathi"),
                "gender": str(source.get("gender") or ""),
                "style": style,
                "source_duration_seconds": (
                    float(source["duration"])
                    if source.get("duration") is not None
                    else None
                ),
                "speaker_label": speaker,
            }
        )

    if args.num_samples > len(candidates):
        raise ValueError(
            f"Requested {args.num_samples} rows, but only {len(candidates)} valid unique "
            "rows were found in the official test split."
        )
    selected = balanced_sample(candidates, args.num_samples, args.seed)
    for eval_id, row in enumerate(selected):
        row["eval_id"] = eval_id

    run_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        archive_published_report(
            run_dir, "The selected evaluation rows were regenerated."
        )
    write_json(selected_path, selected)
    write_csv(run_dir / "selected_samples.csv", selected)
    audit = {
        "selection_config": selection_config,
        "selected_samples_sha256": canonical_hash(selected),
        "dataset_id": args.dataset_id,
        "dataset_config": args.dataset_config,
        "split": "test",
        "dataset_revision": args.dataset_revision,
        "seed": args.seed,
        "valid_unique_candidates": len(candidates),
        "selected_count": len(selected),
        "selection_method": (
            "output-blind deterministic SHA-256 ordering, balanced by adapter speaker label"
        ),
        "exact_train_text_overlap_excluded": not args.allow_train_text_overlap,
        "unique_train_transcripts_scanned": len(train_texts),
        "selected_by_gender": frequency(selected, "gender"),
        "selected_by_style": frequency(selected, "style"),
        "selected_by_speaker_label": frequency(selected, "speaker_label"),
        "rejected": dict(sorted(rejected.items())),
    }
    write_json(run_dir / "selection_summary.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    return selected_path


def require_selected(args: argparse.Namespace, run_dir: Path) -> Path:
    path = run_dir / "selected_samples.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run with --stage select (or --stage all) first."
        )
    selected = read_json(path)
    summary_path = run_dir / "selection_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing {summary_path}. Run --stage select with this evaluator."
        )
    selection_summary = read_json(summary_path)
    saved = selection_summary.get("selection_config", {})
    requested = {
        "dataset_id": args.dataset_id,
        "dataset_config": args.dataset_config,
        "dataset_revision": args.dataset_revision,
        "num_samples": args.num_samples,
        "seed": args.seed,
        "allow_train_text_overlap": args.allow_train_text_overlap,
    }
    mismatched = {
        key: {"saved": saved.get(key), "requested": value}
        for key, value in requested.items()
        if saved.get(key) != value
    }
    if mismatched:
        raise ValueError(
            "Current command does not match the saved selection: "
            + json.dumps(mismatched, ensure_ascii=False)
        )
    if len(selected) != args.num_samples:
        raise ValueError(
            f"Selected manifest has {len(selected)} rows; command requested "
            f"{args.num_samples}."
        )
    if selection_summary.get("selected_samples_sha256") != canonical_hash(selected):
        raise ValueError(f"{path} does not match its recorded SHA-256")
    return path


def cuda_synchronize() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def save_wav(path: Path, waveform: Any, sample_rate: int = SAMPLE_RATE) -> None:
    import numpy as np
    from scipy.io import wavfile

    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(waveform, dtype=np.float32).reshape(-1)
    audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    wavfile.write(path, sample_rate, pcm)


def load_official_inference(model_dir: Path):
    inference_path = model_dir / "inference.py"
    if not inference_path.exists():
        raise FileNotFoundError(
            f"The pinned Indic-Speak snapshot has no inference.py: {inference_path}"
        )
    module_name = "_official_indic_speak_inference"
    sys.path.insert(0, str(model_dir))
    specification = importlib.util.spec_from_file_location(module_name, inference_path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Could not import {inference_path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


def find_language_model(tts: Any) -> tuple[str, Any]:
    from transformers import PreTrainedModel

    candidates = [
        (name, value)
        for name, value in vars(tts).items()
        if isinstance(value, PreTrainedModel)
    ]
    if len(candidates) != 1:
        names = [name for name, _ in candidates]
        raise RuntimeError(
            "Could not identify exactly one language model inside the official TTS "
            f"pipeline. Candidates: {names}"
        )
    return candidates[0]


def resolve_generation_snapshots(args: argparse.Namespace) -> tuple[Path, Path, dict[str, Any]]:
    from huggingface_hub import snapshot_download

    print(f"Downloading/loading base model {args.base_model} ...", flush=True)
    base_dir = Path(
        snapshot_download(repo_id=args.base_model, revision=args.model_revision)
    ).resolve()
    print(f"Downloading/loading adapter {args.adapter_model} ...", flush=True)
    adapter_dir = Path(
        snapshot_download(
            repo_id=args.adapter_model,
            revision=args.adapter_revision,
        )
    ).resolve()
    snapshots = {
        "base_model": args.base_model,
        "base_requested_revision": args.model_revision,
        "base_resolved_revision": base_dir.name,
        "adapter_model": args.adapter_model,
        "adapter_requested_revision": args.adapter_revision,
        "adapter_resolved_revision": adapter_dir.name,
    }
    return base_dir, adapter_dir, snapshots


def load_tts_with_adapter(
    args: argparse.Namespace, base_dir: Path, adapter_dir: Path
):
    from peft import PeftModel

    inference_module = load_official_inference(base_dir)
    tts = inference_module.TTS(str(base_dir))

    language_model_name, base_language_model = find_language_model(tts)
    print(
        f"Attaching PEFT adapter {args.adapter_model} to "
        f"TTS.{language_model_name} ...",
        flush=True,
    )
    adapted_language_model = PeftModel.from_pretrained(
        base_language_model,
        str(adapter_dir),
        is_trainable=False,
    )
    adapted_language_model.eval()
    adapted_language_model.config.use_cache = True
    setattr(tts, language_model_name, adapted_language_model)
    return tts


def synthesize(tts: Any, sample: dict[str, Any], args: argparse.Namespace):
    return tts(
        sample["text"],
        speaker=sample["speaker_label"],
        style=sample["style"],
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed + int(sample["eval_id"]),
        stock=False,
    )


def generate_samples(args: argparse.Namespace, run_dir: Path) -> Path:
    import numpy as np

    selected = read_json(require_selected(args, run_dir))
    manifest_path = run_dir / "generation_manifest.json"
    existing = read_json(manifest_path) if manifest_path.exists() else []
    existing_ids = [int(row["eval_id"]) for row in existing]
    if len(existing_ids) != len(set(existing_ids)):
        raise ValueError(f"{manifest_path} contains duplicate eval_id rows")
    existing_by_id = {int(row["eval_id"]): row for row in existing}
    output_rows: list[dict[str, Any]] = []

    base_dir, adapter_dir, snapshots = resolve_generation_snapshots(args)
    generation_config = {
        "schema_version": 2,
        "selected_samples_sha256": canonical_hash(selected),
        "snapshots": snapshots,
        "decoding": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "stock": False,
            "sample_rate": SAMPLE_RATE,
        },
        "seed": args.seed,
        "warmup_excluded": not args.skip_warmup,
        "environment": runtime_environment(),
    }
    generation_signature = canonical_hash(generation_config)
    generation_config["generation_signature"] = generation_signature
    generation_config_path = run_dir / "generation_config.json"
    previous_generation_config = (
        read_json(generation_config_path) if generation_config_path.exists() else {}
    )
    if (
        args.overwrite
        or previous_generation_config.get("generation_signature")
        != generation_signature
    ):
        archive_published_report(
            run_dir, "The synthesis configuration or generated audio changed."
        )
    write_json(generation_config_path, generation_config)

    audio_dir = run_dir / "audio"
    reusable_ids: set[int] = set()
    if not args.overwrite:
        for sample in selected:
            eval_id = int(sample["eval_id"])
            old = existing_by_id.get(eval_id)
            wav_path = audio_dir / f"{eval_id:04d}.wav"
            if (
                old is not None
                and old.get("status") == "ok"
                and old.get("generation_signature") == generation_signature
                and old.get("text") == sample["text"]
                and old.get("speaker_label") == sample["speaker_label"]
                and old.get("style") == sample["style"]
                and old.get("wav_sha256")
                and wav_path.exists()
                and file_sha256(wav_path) == old["wav_sha256"]
            ):
                reusable_ids.add(eval_id)

    pending = [
        sample for sample in selected if int(sample["eval_id"]) not in reusable_ids
    ]
    if not pending:
        output_rows = [existing_by_id[int(sample["eval_id"])] for sample in selected]
        write_json(manifest_path, output_rows)
        print(f"Reusing all {len(selected)} generated WAV files.", flush=True)
        return manifest_path

    tts = load_tts_with_adapter(args, base_dir, adapter_dir)
    if not args.skip_warmup:
        print("Running one unmeasured GPU warm-up synthesis ...", flush=True)
        _ = synthesize(tts, selected[0], args)
        cuda_synchronize()

    for position, sample in enumerate(selected, start=1):
        eval_id = int(sample["eval_id"])
        wav_path = audio_dir / f"{eval_id:04d}.wav"
        old = existing_by_id.get(eval_id)
        if eval_id in reusable_ids:
            output_rows.append(old)
            print(
                f"[{position}/{len(selected)}] reuse {wav_path.name}", flush=True
            )
            continue

        row = {
            **sample,
            "wav_path": str(wav_path.relative_to(run_dir)).replace("\\", "/"),
            "wav_sha256": None,
            "generation_signature": generation_signature,
            "status": "failed",
            "error": None,
            "generated_duration_seconds": None,
            "response_time_seconds": None,
            "response_time_definition": (
                "wall time until the non-streaming local TTS call returned the full waveform"
            ),
        }
        try:
            cuda_synchronize()
            started = time.perf_counter()
            waveform = synthesize(tts, sample, args)
            cuda_synchronize()
            elapsed = time.perf_counter() - started
            waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
            if waveform.size == 0:
                raise RuntimeError("TTS returned an empty waveform")
            if not np.isfinite(waveform).all():
                raise RuntimeError("TTS returned NaN or infinite waveform values")
            save_wav(wav_path, waveform)
            row.update(
                {
                    "status": "ok",
                    "wav_sha256": file_sha256(wav_path),
                    "generated_duration_seconds": round(
                        waveform.size / SAMPLE_RATE, 6
                    ),
                    "response_time_seconds": round(elapsed, 6),
                }
            )
            print(
                f"[{position}/{len(selected)}] {wav_path.name}: "
                f"{row['generated_duration_seconds']:.2f}s audio, "
                f"{elapsed:.2f}s response time",
                flush=True,
            )
        except Exception as error:  # Keep failures in the evaluation denominator.
            row["error"] = f"{type(error).__name__}: {error}"
            print(
                f"[{position}/{len(selected)}] FAILED: {row['error']}",
                file=sys.stderr,
                flush=True,
            )
            if args.fail_fast:
                output_rows.append(row)
                write_json(manifest_path, output_rows)
                raise
        output_rows.append(row)
        write_json(manifest_path, output_rows)

    write_json(manifest_path, output_rows)
    del tts
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return manifest_path


def load_audio(path: Path, target_rate: int = ASR_SAMPLE_RATE):
    import numpy as np
    from scipy.io import wavfile
    from scipy.signal import resample_poly

    sample_rate, audio = wavfile.read(path)
    if audio.ndim > 1:
        audio = audio.astype(np.float32).mean(axis=1)
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        scale = float(max(abs(info.min), info.max))
        audio = audio.astype(np.float32) / scale
    else:
        audio = audio.astype(np.float32)
    if int(sample_rate) != target_rate:
        divisor = math.gcd(int(sample_rate), target_rate)
        audio = resample_poly(
            audio,
            target_rate // divisor,
            int(sample_rate) // divisor,
        ).astype(np.float32)
    return np.nan_to_num(audio), target_rate


def resolve_asr_snapshot(
    model_id: str, revision: str
) -> tuple[Path, str]:
    from huggingface_hub import snapshot_download

    model_dir = Path(
        snapshot_download(repo_id=model_id, revision=revision)
    ).resolve()
    return model_dir, model_dir.name


class WhisperASR:
    """Small direct Transformers wrapper; no ffmpeg executable is required."""

    def __init__(self, model_dir: Path, device: str, max_new_tokens: int):
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch = torch
        self.device = torch.device(device)
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(str(model_dir))
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            str(model_dir),
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()

    def transcribe(self, wav_path: Path) -> str:
        audio, sample_rate = load_audio(wav_path)
        if len(audio) == 0:
            return ""
        chunk_samples = 25 * sample_rate
        pieces: list[str] = []
        with self.torch.inference_mode():
            for start in range(0, len(audio), chunk_samples):
                chunk = audio[start : start + chunk_samples]
                inputs = self.processor(
                    chunk,
                    sampling_rate=sample_rate,
                    return_tensors="pt",
                    return_attention_mask=True,
                )
                features = inputs.input_features.to(
                    self.device, dtype=self.model.dtype
                )
                generate_options: dict[str, Any] = {
                    "language": "marathi",
                    "task": "transcribe",
                    "max_new_tokens": self.max_new_tokens,
                }
                if hasattr(inputs, "attention_mask"):
                    generate_options["attention_mask"] = inputs.attention_mask.to(
                        self.device
                    )
                token_ids = self.model.generate(features, **generate_options)
                if token_ids.shape[-1] >= self.max_new_tokens:
                    raise RuntimeError(
                        "Whisper reached --asr-max-new-tokens; the transcript "
                        "may be truncated. Increase the limit and rescore."
                    )
                text = self.processor.batch_decode(
                    token_ids, skip_special_tokens=True
                )[0].strip()
                if text:
                    pieces.append(text)
        return " ".join(pieces)


def percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return (
        ordered[lower] * (upper - position)
        + ordered[upper] * (position - lower)
    )


def round_or_none(value: float | None, places: int = 4) -> float | None:
    return None if value is None else round(value, places)


def build_report(summary: dict[str, Any]) -> str:
    metrics = summary["metrics"]
    latency = metrics["response_time_seconds"]
    gpu_names = summary["generation_environment"].get("gpu_names") or []
    gpu_label = ", ".join(gpu_names) if gpu_names else "CPU / not recorded"
    lines = [
        "# Marathi Indic-Speak Adapter Evaluation",
        "",
        f"Evaluated rows: **{summary['evaluated_rows']}** from the official Rasa Marathi test split.",
        "",
        "| Measure | What it tells you | Result |",
        "|---|---|---:|",
        (
            "| Intelligibility (WER / CER) | How reliably ASR can recover the "
            f"reference text | **{metrics['wer_percent']:.2f}% / "
            f"{metrics['cer_percent']:.2f}%** |"
        ),
        (
            "| Response time | Local time until the complete WAV is returned "
            f"| **mean {latency['mean']:.3f}s; p50 {latency['p50']:.3f}s; "
            f"p95 {latency['p95']:.3f}s** |"
            if latency["mean"] is not None
            else "| Response time | Local time until the complete WAV is returned | **No successful generations** |"
        ),
        "",
        "## Reproducibility",
        "",
        f"- Dataset: `{summary['dataset']['id']}` / `{summary['dataset']['config']}` / `test`",
        f"- Dataset revision: `{summary['dataset']['revision']}`",
        f"- Base model: `{summary['models']['base']}` at `{summary['models']['base_revision']}`",
        f"- Adapter: `{summary['models']['adapter']}` at `{summary['models']['adapter_revision']}`",
        f"- ASR: `{summary['models']['asr']}` at `{summary['models']['asr_revision']}`",
        f"- Selection seed: `{summary['seed']}`",
        f"- Exact train-text overlap excluded: `{summary['dataset']['train_text_overlap_excluded']}`",
        f"- Generation GPU: `{gpu_label}`",
        f"- Successful generations: `{summary['successful_generations']}`",
        f"- Failed generations: `{summary['failed_generations']}`",
        "",
        "## Interpretation note",
        "",
        (
            "WER and CER are corpus error rates calculated by `jiwer` after NFC, "
            "case, punctuation, and whitespace normalization. Recorded synthesis "
            "failures receive an empty ASR hypothesis and remain in the metric; "
            "missing files and ASR infrastructure errors stop report creation."
        ),
        "",
        (
            "Response time is full-waveform latency from the local non-streaming "
            "Indic-Speak call. It must not be compared directly with the model card's "
            "approximately 200 ms delay-before-audio value because its serving setup "
            "and measurement protocol are not published."
        ),
        "",
        (
            "The model card does not publish the sample set, ASR model, text "
            "normalization, hardware, or full latency protocol behind 26.49 / 17.86, "
            "so this run is a reproducible project result, not a reproduction claim."
        ),
        "",
    ]
    return "\n".join(lines)


def score_samples(args: argparse.Namespace, run_dir: Path) -> Path:
    try:
        import jiwer
    except ImportError as error:
        raise RuntimeError(
            "Missing jiwer. Install evaluation/requirements.txt first."
        ) from error

    selected = read_json(require_selected(args, run_dir))
    generation_path = run_dir / "generation_manifest.json"
    if not generation_path.exists():
        raise FileNotFoundError(
            f"Missing {generation_path}. Run --stage generate (or --stage all) first."
        )
    generated = read_json(generation_path)
    selected_ids = [int(row["eval_id"]) for row in selected]
    generated_ids = [int(row["eval_id"]) for row in generated]
    if len(generated_ids) != len(set(generated_ids)):
        raise ValueError(f"{generation_path} contains duplicate eval_id rows")
    if set(generated_ids) != set(selected_ids):
        missing = sorted(set(selected_ids) - set(generated_ids))
        extra = sorted(set(generated_ids) - set(selected_ids))
        raise ValueError(
            "Generation is incomplete or belongs to another selection. "
            f"Missing IDs: {missing}; extra IDs: {extra}. Run --stage generate."
        )
    generated_by_id = {int(row["eval_id"]): row for row in generated}
    generated = [generated_by_id[eval_id] for eval_id in selected_ids]

    generation_config_path = run_dir / "generation_config.json"
    if not generation_config_path.exists():
        raise FileNotFoundError(
            f"Missing {generation_config_path}. Regenerate with this evaluator."
        )
    generation_config = read_json(generation_config_path)
    generation_signature = generation_config.get("generation_signature")
    if not generation_signature:
        raise ValueError("generation_config.json has no generation signature")
    unsigned_generation_config = dict(generation_config)
    unsigned_generation_config.pop("generation_signature", None)
    if canonical_hash(unsigned_generation_config) != generation_signature:
        raise ValueError("generation_config.json failed its integrity check")
    if generation_config.get("selected_samples_sha256") != canonical_hash(selected):
        raise ValueError(
            "Generated audio belongs to a different selected-sample manifest. "
            "Run --stage generate."
        )
    for sample, row in zip(selected, generated):
        if row.get("generation_signature") != generation_signature:
            raise ValueError(
                "Generation rows contain mixed or stale configurations. "
                "Run --stage generate --overwrite."
            )
        for field in ("text", "speaker_label", "style"):
            if row.get(field) != sample.get(field):
                raise ValueError(
                    f"Generation row {row['eval_id']} has stale {field!r} metadata. "
                    "Run --stage generate."
                )
        if row.get("status") == "ok":
            wav_path = run_dir / row["wav_path"]
            if not wav_path.exists():
                raise FileNotFoundError(f"Generated WAV is missing: {wav_path}")
            if not row.get("wav_sha256"):
                raise ValueError(f"Generation row {row['eval_id']} has no WAV hash")
            if file_sha256(wav_path) != row["wav_sha256"]:
                raise ValueError(
                    f"Generated WAV checksum mismatch: {wav_path}. Regenerate it."
                )

    transcript_path = run_dir / "asr_transcripts.json"
    existing: list[dict[str, Any]] = []
    if transcript_path.exists() and not args.overwrite:
        existing = read_json(transcript_path)
    existing_ids = [int(row["eval_id"]) for row in existing]
    if len(existing_ids) != len(set(existing_ids)):
        raise ValueError(f"{transcript_path} contains duplicate eval_id rows")
    existing_by_id = {int(row["eval_id"]): row for row in existing}

    print(f"Resolving ASR model {args.asr_model} ...", flush=True)
    asr_dir, asr_resolved_revision = resolve_asr_snapshot(
        args.asr_model, args.asr_revision
    )
    asr_config = {
        "schema_version": 2,
        "model": args.asr_model,
        "requested_revision": args.asr_revision,
        "resolved_revision": asr_resolved_revision,
        "max_new_tokens": args.asr_max_new_tokens,
        "language": "marathi",
        "task": "transcribe",
        "chunk_seconds": 25,
        "normalization_version": 1,
    }
    asr_signature = canonical_hash(asr_config)
    asr_config["asr_signature"] = asr_signature
    write_json(run_dir / "asr_config.json", asr_config)

    asr: WhisperASR | None = None
    infrastructure_errors: list[str] = []
    scored_rows: list[dict[str, Any]] = []
    for position, sample in enumerate(selected, start=1):
        eval_id = int(sample["eval_id"])
        generation = generated_by_id[eval_id]
        reference = normalize_text(sample["text"])
        reusable = existing_by_id.get(eval_id)
        can_reuse = (
            reusable is not None
            and not args.overwrite
            and reusable.get("asr_signature") == asr_signature
            and reusable.get("generation_signature") == generation_signature
            and reusable.get("generation_status") == generation.get("status")
            and reusable.get("wav_sha256") == generation.get("wav_sha256")
            and reusable.get("reference_normalized") == reference
            and reusable.get("asr_status") in {"ok", "generation_failed"}
        )
        if can_reuse:
            scored_rows.append(reusable)
            print(
                f"[{position}/{len(selected)}] reuse ASR transcript {eval_id:04d}",
                flush=True,
            )
            continue

        raw_hypothesis = ""
        hypothesis = ""
        asr_error = None
        evaluation_note = None
        if generation.get("status") != "ok":
            asr_status = "generation_failed"
            evaluation_note = (
                "Synthesis failed; an empty hypothesis is included in corpus WER/CER. "
                + str(generation.get("error"))
            )
        else:
            wav_path = run_dir / generation["wav_path"]
            try:
                if asr is None:
                    print(f"Loading ASR model {args.asr_model} ...", flush=True)
                    asr = WhisperASR(
                        asr_dir,
                        args.asr_device,
                        args.asr_max_new_tokens,
                    )
                raw_hypothesis = asr.transcribe(wav_path)
                hypothesis = normalize_text(raw_hypothesis)
                asr_status = "ok"
            except Exception as error:
                asr_status = "failed"
                asr_error = f"{type(error).__name__}: {error}"
                infrastructure_errors.append(f"eval_id={eval_id}: {asr_error}")
                if args.fail_fast:
                    raise

        row = {
            "eval_id": eval_id,
            "filename": sample["filename"],
            "reference": sample["text"],
            "reference_normalized": reference,
            "hypothesis": raw_hypothesis,
            "hypothesis_normalized": hypothesis,
            "asr_model": args.asr_model,
            "asr_requested_revision": args.asr_revision,
            "asr_resolved_revision": asr_resolved_revision,
            "asr_signature": asr_signature,
            "asr_status": asr_status,
            "asr_error": asr_error,
            "evaluation_note": evaluation_note,
            "generation_status": generation.get("status"),
            "generation_signature": generation_signature,
            "wav_sha256": generation.get("wav_sha256"),
            "wer_percent": round(100.0 * jiwer.wer(reference, hypothesis), 4),
            "cer_percent": round(100.0 * jiwer.cer(reference, hypothesis), 4),
        }
        scored_rows.append(row)
        write_json(transcript_path, scored_rows)
        print(
            f"[{position}/{len(selected)}] ASR {eval_id:04d}: "
            f"{asr_status}; WER {row['wer_percent']:.2f}% "
            f"CER {row['cer_percent']:.2f}%",
            flush=True,
        )

    write_json(transcript_path, scored_rows)
    write_csv(run_dir / "asr_transcripts.csv", scored_rows)
    if infrastructure_errors:
        raise RuntimeError(
            "ASR evaluation failed for one or more rows, so no summary was "
            "published. Fix the errors and rerun --stage score. "
            + " | ".join(infrastructure_errors[:5])
        )

    references = [row["reference_normalized"] for row in scored_rows]
    hypotheses = [row["hypothesis_normalized"] for row in scored_rows]
    corpus_wer = 100.0 * jiwer.wer(references, hypotheses)
    corpus_cer = 100.0 * jiwer.cer(references, hypotheses)
    successful = [row for row in generated if row.get("status") == "ok"]
    response_times = [
        float(row["response_time_seconds"])
        for row in successful
        if row.get("response_time_seconds") is not None
    ]
    latency = {
        "mean": round_or_none(statistics.fmean(response_times) if response_times else None, 6),
        "p50": round_or_none(statistics.median(response_times) if response_times else None, 6),
        "p95": round_or_none(percentile(response_times, 0.95), 6),
        "minimum": round_or_none(min(response_times) if response_times else None, 6),
        "maximum": round_or_none(max(response_times) if response_times else None, 6),
        "unit": "seconds",
        "definition": (
            "wall time until the local non-streaming TTS call returns the complete waveform"
        ),
        "warmup_excluded": generation_config["warmup_excluded"],
    }
    selection_summary = read_json(run_dir / "selection_summary.json")
    selection_config = selection_summary["selection_config"]
    snapshots = generation_config["snapshots"]
    summary = {
        "evaluated_rows": len(scored_rows),
        "successful_generations": len(successful),
        "failed_generations": len(selected) - len(successful),
        "seed": selection_config["seed"],
        "dataset": {
            "id": selection_config["dataset_id"],
            "config": selection_config["dataset_config"],
            "split": "test",
            "revision": selection_config["dataset_revision"],
            "train_text_overlap_excluded": (
                not selection_config["allow_train_text_overlap"]
            ),
        },
        "models": {
            "base": snapshots["base_model"],
            "base_revision": snapshots["base_resolved_revision"],
            "adapter": snapshots["adapter_model"],
            "adapter_revision": snapshots["adapter_resolved_revision"],
            "asr": args.asr_model,
            "asr_revision": asr_resolved_revision,
        },
        "metrics": {
            "wer_percent": round(corpus_wer, 4),
            "cer_percent": round(corpus_cer, 4),
            "response_time_seconds": latency,
        },
        "normalization": (
            "Unicode NFC; lowercase; keep Unicode letters, combining marks, and numbers; "
            "replace punctuation/symbols with spaces; collapse whitespace"
        ),
        "metric_library": "jiwer",
        "generation_environment": generation_config["environment"],
        "scoring_environment": runtime_environment(),
        "generation_signature": generation_signature,
        "asr_signature": asr_signature,
    }
    summary_path = run_dir / "summary.json"
    write_json(summary_path, summary)
    report_path = run_dir / "REPORT.md"
    report_path.write_text(build_report(summary), encoding="utf-8")
    print("\n" + build_report(summary), flush=True)
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Marathi Indic-Speak adapter WER, CER, and full-waveform "
            "response time on N official Rasa test rows."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stage", choices=("all", "select", "generate", "score"), default="all"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of official test rows; for example 20, 30, or 100.",
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")

    parser.add_argument("--dataset-id", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--dataset-revision", default=DEFAULT_DATASET_REVISION)
    parser.add_argument(
        "--allow-train-text-overlap",
        action="store_true",
        help=(
            "Allow test rows whose normalized transcript also occurs in train. "
            "By default they are excluded to measure unseen text."
        ),
    )

    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--adapter-model", default=DEFAULT_ADAPTER)
    parser.add_argument("--adapter-revision", default="main")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=2520)
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Do not run the default unmeasured warm-up synthesis.",
    )

    parser.add_argument(
        "--asr-model",
        default=DEFAULT_ASR_MODEL,
        help="A Transformers Whisper speech-to-text checkpoint.",
    )
    parser.add_argument("--asr-revision", default="main")
    parser.add_argument(
        "--asr-device",
        default="auto",
        help="auto, cpu, cuda, or a device such as cuda:1.",
    )
    parser.add_argument("--asr-max-new-tokens", type=int, default=400)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.num_samples < 1:
        raise ValueError("--num-samples must be at least 1")
    if not 0.0 < args.top_p <= 1.0:
        raise ValueError("--top-p must be in (0, 1]")
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be at least 1")
    if not 1 <= args.asr_max_new_tokens <= 440:
        raise ValueError("--asr-max-new-tokens must be between 1 and 440")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    run_dir = resolve_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}", flush=True)

    if args.stage in {"all", "select"}:
        select_samples(args, run_dir)
    if args.stage in {"all", "generate"}:
        generate_samples(args, run_dir)
    if args.stage in {"all", "score"}:
        score_samples(args, run_dir)


if __name__ == "__main__":
    main()
