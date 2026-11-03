"""Common Japanese test data, on-disk WAVs and measurements for local TTS."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

from .engines import MODEL_REPOS, ROOT, download, load_engine

DEFAULT_CASES = ROOT / "cases/japanese.json"
RESULTS = ROOT / "results"


def load_cases(path: Path) -> list[dict[str, str]]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("Cases must be a non-empty JSON array")
    ids: set[str] = set()
    for item in cases:
        if not isinstance(item, dict) or not all(
            isinstance(item.get(k), str) and item[k].strip()
            for k in ("id", "category", "text")
        ):
            raise ValueError("Each case requires non-empty id/category/text strings")
        identifier = item["id"]
        if not all(ch.isascii() and (ch.isalnum() or ch in "-_") for ch in identifier):
            raise ValueError(f"Unsafe case id: {identifier}")
        if identifier in ids:
            raise ValueError(f"Duplicate case id: {identifier}")
        ids.add(identifier)
    return cases


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as audio:
        return audio.getnframes() / audio.getframerate()


def write_measurement(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def selected_cases(cases_path: Path, case_id: str | None) -> list[dict[str, str]]:
    cases = load_cases(cases_path)
    if case_id is None:
        return cases
    selected = [case for case in cases if case["id"] == case_id]
    if not selected:
        raise ValueError(f"Unknown case ID: {case_id}")
    return selected


def generate(args: argparse.Namespace) -> int:
    cases = selected_cases(Path(args.cases), args.case)
    # Fail before preparing output if weights are missing; loading time excluded from RTF.
    start_load = time.perf_counter()
    synthesizer, configuration = load_engine(
        args.model, speaker=args.qwen_speaker, num_step=args.omnivoice_steps
    )
    load_seconds = round(time.perf_counter() - start_load, 3)
    folder = RESULTS / "audio" / args.model
    folder.mkdir(parents=True, exist_ok=True)
    failures = 0
    for case in cases:
        output = folder / f"{case['id']}.wav"
        # Remove a stale success so a failed rerun cannot look successful.
        output.unlink(missing_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "case_id": case["id"],
            "category": case["category"],
            "text": case["text"],
            "configuration": configuration,
            "system": platform.platform(),
            "python": platform.python_version(),
            "load_seconds": load_seconds,
            "status": "ok",
        }
        start = time.perf_counter()
        try:
            synthesizer(case["text"], output)
            elapsed = time.perf_counter() - start
            seconds = wav_duration(output)
            if seconds <= 0:
                raise ValueError("Generated audio is empty")
            record.update({
                "seconds": round(elapsed, 4),
                "audio_seconds": round(seconds, 4),
                "rtf": round(elapsed / seconds, 4),
                "bytes": output.stat().st_size,
                "audio_path": str(output.relative_to(ROOT)),
            })
            print(f"OK {args.model}/{case['id']} ({elapsed:.2f}s, RTF={record['rtf']})")
        except Exception as exc:
            failures += 1
            output.unlink(missing_ok=True)
            record.update(status="error", error=f"{type(exc).__name__}: {exc}")
            print(f"ERROR {args.model}/{case['id']}: {exc}", file=sys.stderr)
        write_measurement(RESULTS / "measurements.jsonl", record)
    return 1 if failures else 0


def compare() -> int:
    path = RESULTS / "measurements.jsonl"
    if not path.exists():
        print("No measurements yet. Run mise run generate:qwen3 or generate:omnivoice.")
        return 1
    latest: dict[tuple[str, str], dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            latest[(row["model"], row["case_id"])] = row
    print("model | success/total | avg RTF | errors")
    print("--- | --- | --- | ---")
    for model in sorted({key[0] for key in latest}):
        rows = [v for (name, _), v in latest.items() if name == model]
        ok = [r for r in rows if r["status"] == "ok"]
        rtfs = [r["rtf"] for r in ok if r.get("rtf") is not None]
        avg = f"{sum(rtfs) / len(rtfs):.3f}" if rtfs else "n/a"
        print(f"{model} | {len(ok)}/{len(rows)} | {avg} | {len(rows) - len(ok)}")
    return 0


def play(model: str, case: str) -> None:
    path = RESULTS / "audio" / model / f"{case}.wav"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Generate samples first.")
    if sys.platform != "darwin":
        raise RuntimeError("Audio playback currently uses macOS 'afplay'. Open the WAV manually on other OSes.")
    subprocess.run(["afplay", str(path)], check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline Japanese TTS comparison")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="List cases and supported models")
    validate = sub.add_parser("validate", help="Check common cases")
    validate.add_argument("--cases", default=str(DEFAULT_CASES))
    download_cmd = sub.add_parser("download", help="Explicitly download model weights")
    download_cmd.add_argument("--model", required=True, choices=sorted(MODEL_REPOS))
    synth = sub.add_parser("generate", help="Generate WAVs for common cases")
    synth.add_argument("--model", required=True, choices=sorted(MODEL_REPOS))
    synth.add_argument("--cases", default=str(DEFAULT_CASES))
    synth.add_argument("--case", help="One case ID instead of all")
    synth.add_argument("--qwen-speaker", default="Ono_Anna")
    synth.add_argument("--omnivoice-steps", type=int, default=16)
    playback = sub.add_parser("play", help="Play a generated WAV on macOS")
    playback.add_argument("--model", required=True, choices=sorted(MODEL_REPOS))
    playback.add_argument("--case", default="basic-01")
    sub.add_parser("compare", help="Summarize latest results")
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            for key, repo in MODEL_REPOS.items():
                print(f"{key:10} {repo}")
            for case in load_cases(DEFAULT_CASES):
                print(f"{case['id']:14} {case['category']:16} {case['text']}")
            return 0
        if args.command == "validate":
            print(f"Validated {len(load_cases(Path(args.cases)))} cases")
            return 0
        if args.command == "download":
            print(f"Weights saved to: {download(args.model)}")
            return 0
        if args.command == "generate":
            if args.omnivoice_steps <= 0:
                raise ValueError("--omnivoice-steps must be positive")
            return generate(args)
        if args.command == "play":
            play(args.model, args.case)
            return 0
        if args.command == "compare":
            return compare()
    except (ValueError, FileNotFoundError, OSError, RuntimeError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 2
