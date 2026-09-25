"""Package and upload a completed evaluation run as a Hugging Face dataset.

The exported table joins the selected Rasa metadata, exact reference audio,
adapter-generated audio, Whisper transcription, row-level WER/CER, and the
measured full-waveform latency.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any, Sequence


SAMPLE_RATE = 24_000
DEFAULT_REPO_ID = "PrakashPask/TTS-Evaluvation"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty evaluation dataset")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_reference_audio(payload: Any) -> tuple[Any, int]:
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    if not isinstance(payload, dict):
        raise TypeError(f"Expected an audio mapping, received {type(payload).__name__}")
    raw = payload.get("bytes")
    source = io.BytesIO(raw) if raw is not None else payload.get("path")
    if source is None:
        raise ValueError("The Rasa audio row contains neither bytes nor a path")
    waveform, sample_rate = sf.read(source, dtype="float32", always_2d=True)
    waveform = waveform.mean(axis=1)
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError("The decoded Rasa reference waveform is empty or non-finite")
    if int(sample_rate) != SAMPLE_RATE:
        divisor = math.gcd(int(sample_rate), SAMPLE_RATE)
        waveform = resample_poly(
            waveform,
            SAMPLE_RATE // divisor,
            int(sample_rate) // divisor,
        ).astype(np.float32)
        sample_rate = SAMPLE_RATE
    return waveform, int(sample_rate)


def save_reference_wav(path: Path, waveform: Any, sample_rate: int) -> None:
    import numpy as np
    import soundfile as sf

    audio = np.asarray(waveform, dtype=np.float32).reshape(-1)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sample_rate, subtype="PCM_16")


def validate_completed_run(run_dir: Path) -> dict[str, Path]:
    required = {
        "selected": run_dir / "selected_samples.json",
        "selection": run_dir / "selection_summary.json",
        "generation": run_dir / "generation_manifest.json",
        "generation_config": run_dir / "generation_config.json",
        "transcripts": run_dir / "asr_transcripts.json",
        "asr_config": run_dir / "asr_config.json",
        "summary": run_dir / "summary.json",
        "report": run_dir / "REPORT.md",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "The evaluation run is incomplete. Missing: " + ", ".join(missing)
        )
    return required


def export_reference_audio(
    run_dir: Path,
    selected: Sequence[dict[str, Any]],
    selection_summary: dict[str, Any],
    overwrite: bool,
) -> dict[int, dict[str, Any]]:
    config = selection_summary["selection_config"]
    targets = {int(row["source_index"]): row for row in selected}
    if len(targets) != len(selected):
        raise ValueError("Selected rows contain duplicate source_index values")

    reference_dir = run_dir / "reference_audio"
    exported: dict[int, dict[str, Any]] = {}
    pending_indices: set[int] = set()
    for source_index, row in targets.items():
        eval_id = int(row["eval_id"])
        path = reference_dir / f"{eval_id:04d}.wav"
        if path.exists() and not overwrite:
            exported[eval_id] = {
                "path": path,
                "sha256": file_sha256(path),
            }
        else:
            pending_indices.add(source_index)

    if pending_indices:
        try:
            from datasets import Audio, load_dataset
        except ImportError as error:
            raise RuntimeError(
                "Missing datasets. Install evaluation/requirements.txt first."
            ) from error
        print(
            f"Downloading {len(pending_indices)} selected Rasa reference WAVs ...",
            flush=True,
        )
        dataset = load_dataset(
            config["dataset_id"],
            config["dataset_config"],
            split="test",
            streaming=True,
            revision=config["dataset_revision"],
        )
        if "audio" not in (dataset.column_names or []):
            raise ValueError("The selected Rasa split has no audio column")
        dataset = dataset.cast_column("audio", Audio(decode=False))
        highest = max(pending_indices)
        for source_index, source in enumerate(dataset):
            if source_index in pending_indices:
                row = targets[source_index]
                eval_id = int(row["eval_id"])
                waveform, sample_rate = decode_reference_audio(source["audio"])
                path = reference_dir / f"{eval_id:04d}.wav"
                save_reference_wav(path, waveform, sample_rate)
                exported[eval_id] = {
                    "path": path,
                    "sha256": file_sha256(path),
                    "duration_seconds": round(len(waveform) / sample_rate, 6),
                }
                pending_indices.remove(source_index)
                print(f"Reference {eval_id:04d}: {path.name}", flush=True)
            if source_index >= highest or not pending_indices:
                break
        if pending_indices:
            raise RuntimeError(
                "Could not recover selected Rasa source rows: "
                + str(sorted(pending_indices))
            )

    import wave

    for item in exported.values():
        if "duration_seconds" not in item:
            with wave.open(str(item["path"]), "rb") as handle:
                item["duration_seconds"] = round(
                    handle.getnframes() / handle.getframerate(), 6
                )
    return exported


def build_dataset_rows(
    run_dir: Path, paths: dict[str, Path], overwrite: bool
) -> list[dict[str, Any]]:
    selected = read_json(paths["selected"])
    selection_summary = read_json(paths["selection"])
    generated = read_json(paths["generation"])
    transcripts = read_json(paths["transcripts"])

    generated_by_id = {int(row["eval_id"]): row for row in generated}
    transcripts_by_id = {int(row["eval_id"]): row for row in transcripts}
    selected_ids = {int(row["eval_id"]) for row in selected}
    if set(generated_by_id) != selected_ids or set(transcripts_by_id) != selected_ids:
        raise ValueError(
            "Selected, generated, and transcribed row IDs do not match. "
            "Finish evaluation before publishing."
        )

    references = export_reference_audio(
        run_dir, selected, selection_summary, overwrite=overwrite
    )
    rows: list[dict[str, Any]] = []
    for sample in selected:
        eval_id = int(sample["eval_id"])
        generation = generated_by_id[eval_id]
        transcript = transcripts_by_id[eval_id]
        predicted_relative = generation.get("wav_path")
        if generation.get("status") == "ok":
            if not predicted_relative:
                raise ValueError(f"Successful generation {eval_id} has no WAV path")
            predicted_path = run_dir / predicted_relative
            if not predicted_path.exists():
                raise FileNotFoundError(f"Missing generated audio: {predicted_path}")
            if file_sha256(predicted_path) != generation.get("wav_sha256"):
                raise ValueError(f"Generated WAV checksum mismatch: {predicted_path}")

        reference = references[eval_id]
        rows.append(
            {
                "eval_id": eval_id,
                "source_filename": sample["filename"],
                "source_index": sample["source_index"],
                "text": sample["text"],
                "normalized_text": transcript["reference_normalized"],
                "language": sample["language"],
                "gender": sample["gender"],
                "style": sample["style"],
                "speaker_label": sample["speaker_label"],
                "reference_audio": str(reference["path"].relative_to(run_dir)).replace("\\", "/"),
                "predicted_audio": predicted_relative,
                "asr_transcription": transcript["hypothesis"],
                "normalized_asr_transcription": transcript["hypothesis_normalized"],
                "wer_percent": transcript["wer_percent"],
                "cer_percent": transcript["cer_percent"],

                "total_latency_seconds": generation.get("response_time_seconds"),
                "total_latency_definition": generation.get("response_time_definition"),
                "reference_duration_seconds": reference["duration_seconds"],
                "predicted_duration_seconds": generation.get("generated_duration_seconds"),
                "generation_status": generation.get("status"),
                "generation_error": generation.get("error"),
                "asr_status": transcript.get("asr_status"),
                "asr_error": transcript.get("asr_error"),
                "reference_audio_sha256": reference["sha256"],
                "predicted_audio_sha256": generation.get("wav_sha256"),
            }
        )

    write_jsonl(run_dir / "evaluation_dataset.jsonl", rows)
    write_csv(run_dir / "evaluation_dataset.csv", rows)
    return rows


def write_dataset_card(
    run_dir: Path, repo_id: str, rows: Sequence[dict[str, Any]]
) -> Path:
    summary = read_json(run_dir / "summary.json")
    card = f"""---
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


Latest packaged run: `{run_dir.name}` ({len(rows)} rows)

- Corpus WER: {summary['metrics']['wer_percent']:.4f}%
- Corpus CER: {summary['metrics']['cer_percent']:.4f}%
- Base model: `{summary['models']['base']}`
- Adapter: `{summary['models']['adapter']}`
- ASR: `{summary['models']['asr']}`
- Source dataset: `ai4bharat/Rasa`, Marathi test split

Reference recordings come from Rasa (CC-BY-4.0). Generated artifacts are
derived from Indic-Speak and the published PEFT adapter; consult their model
cards and licenses before redistribution or commercial use.

Repository: https://huggingface.co/datasets/{repo_id}
"""
    path = run_dir / "HF_DATASET_CARD.md"
    path.write_text(card, encoding="utf-8")
    return path


def upload_run(
    run_dir: Path,
    repo_id: str,
    path_in_repo: str,
    private: bool,
    card_path: Path,
) -> str:
    try:
        from huggingface_hub import HfApi, login
    except ImportError as error:
        raise RuntimeError(
            "Missing huggingface_hub. Install evaluation/requirements.txt first."
        ) from error

    api = HfApi()
    try:
        identity = api.whoami()
    except Exception:
        print("No active Hugging Face login; opening token prompt ...", flush=True)
        login()
        identity = api.whoami()
    print(f"Uploading as Hugging Face user {identity.get('name', 'unknown')} ...")

    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=str(run_dir),
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo=path_in_repo,
        ignore_patterns=["previous_reports/**", "*.tmp"],
        commit_message=f"Add evaluation run {run_dir.name}",
    )
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Update dataset card for {run_dir.name}",
    )
    return f"https://huggingface.co/datasets/{repo_id}/tree/main/{path_in_repo}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package and upload one completed TTS evaluation run."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--path-in-repo",
        help="Destination folder. Defaults to runs/<local run directory name>.",
    )
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--overwrite-reference-audio", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Build the reference audio and tables without uploading.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    paths = validate_completed_run(run_dir)
    rows = build_dataset_rows(
        run_dir,
        paths,
        overwrite=args.overwrite_reference_audio,
    )
    card_path = write_dataset_card(run_dir, args.repo_id, rows)
    print(f"Prepared {len(rows)} rows in {run_dir}", flush=True)
    if args.prepare_only:
        print("Preparation complete; upload skipped (--prepare-only).")
        return
    path_in_repo = args.path_in_repo or f"runs/{run_dir.name}"
    url = upload_run(
        run_dir,
        args.repo_id,
        path_in_repo.strip("/"),
        args.private,
        card_path,
    )
    print(f"Uploaded evaluation dataset: {url}", flush=True)


if __name__ == "__main__":
    main()