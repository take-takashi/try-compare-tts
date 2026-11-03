"""Local TTS experiments: baseline vs. deterministic quality-improvement profiles."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import wave
from uuid import uuid4
from datetime import datetime, timezone
from pathlib import Path

from .engines import MODEL_REPOS, ROOT, download, load_engine
from .profiles import DEFAULT_RULES, PROFILES, load_rules, prepare
from .quality import audio_metrics, quality_failures, trim_dummy_speech

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


def validate(cases_path: Path, rules_path: Path) -> int:
    cases = load_cases(cases_path)
    rules = load_rules(rules_path)
    ids = {case["id"] for case in cases}
    unknown = set(rules["cases"]) - ids
    if unknown:
        raise ValueError(f"Unknown reading override IDs: {', '.join(sorted(unknown))}")
    for case in cases:
        for profile in PROFILES:
            prepare(case, profile, rules)
    print(f"Validated {len(cases)} cases and {len(rules['cases'])} case-specific reading rules")
    return 0


def validate_run_id(run_id: str) -> str:
    if not run_id or len(run_id) > 100 or not all(
        ch.isascii() and (ch.isalnum() or ch in "-_") for ch in run_id
    ):
        raise ValueError("Run ID must contain only ASCII letters, digits, '-' or '_'")
    return run_id


def audio_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate(args: argparse.Namespace) -> int:
    cases = selected_cases(Path(args.cases), args.case)
    rules = load_rules(Path(args.rules))
    prepared = [(case, prepare(case, args.profile, rules)) for case in cases]
    run_id = validate_run_id(args.run_id or uuid4().hex)
    generation_id = uuid4().hex
    print(f"Run ID: {run_id}", flush=True)
    start_load = time.perf_counter()
    load_error = None
    configuration = {}
    try:
        synthesizer, configuration = load_engine(
            args.model, speaker=args.qwen_speaker, num_step=args.omnivoice_steps
        )
    except Exception as exc:
        load_error = exc
    load_seconds = round(time.perf_counter() - start_load, 3)
    folder = RESULTS / "audio" / run_id / generation_id / args.model / args.profile
    folder.mkdir(parents=True, exist_ok=True)
    failures = 0
    for case, input_text in prepared:
        output = folder / f"{case['id']}.wav"
        padded = folder / f"{case['id']}.padded.wav"
        record = {
            "run_id": run_id,
            "generation_id": generation_id,
            "record_id": uuid4().hex,
            "cases_sha256": audio_hash(Path(args.cases)),
            "rules_sha256": audio_hash(Path(args.rules)),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "profile": args.profile,
            "case_id": case["id"],
            "category": case["category"],
            "text": case["text"],
            "synthesis_text": input_text.synthesized,
            "reading_changed": input_text.reading_changed,
            "trimmed_dummy": input_text.trimmed,
            "quality_assurance": "basic_checks_only",
            "dummy_boundary_verified": False if input_text.trimmed else None,
            "configuration": configuration,
            "system": platform.platform(),
            "python": platform.python_version(),
            "load_seconds": load_seconds,
            "status": "error",
        }
        start_all = time.perf_counter()
        attempt = 0
        try:
            if load_error is not None:
                raise RuntimeError(f"Model load failed: {type(load_error).__name__}: {load_error}")
            for attempt in range(1, args.max_retries + 2):
                start = time.perf_counter()
                output.unlink(missing_ok=True)
                try:
                    synthesizer(input_text.synthesized, output)
                    generation_seconds = time.perf_counter() - start
                    if input_text.trimmed:
                        # Preserve the actual padded result (including unsuccessful ones).
                        output.replace(padded)
                        output.write_bytes(padded.read_bytes())
                        trim_info = trim_dummy_speech(output)
                    else:
                        trim_info = {}
                    metrics = audio_metrics(output)
                    qa_errors = quality_failures(metrics)
                    if qa_errors:
                        raise ValueError("Quality gate: " + ", ".join(qa_errors))
                    break
                except Exception:
                    if attempt > args.max_retries:
                        raise
            elapsed = time.perf_counter() - start_all
            seconds = metrics["audio_seconds"]
            record.update({
                "status": "ok",
                "attempts": attempt,
                "seconds": round(generation_seconds, 4),
                "total_seconds": round(elapsed, 4),
                "audio_seconds": seconds,
                "rtf": round(generation_seconds / seconds, 4),
                "total_rtf": round(elapsed / seconds, 4),
                "bytes": output.stat().st_size,
                "audio_path": str(output.relative_to(ROOT) if output.is_relative_to(ROOT) else output),
                "audio_sha256": audio_hash(output),
                "quality": metrics,
                **trim_info,
            })
            if padded.exists():
                record["padded_audio_path"] = str(padded.relative_to(ROOT) if padded.is_relative_to(ROOT) else padded)
            print(f"OK {args.model}/{args.profile}/{case['id']} (attempts={attempt}, total RTF={record['total_rtf']})")
        except Exception as exc:
            failures += 1
            output.unlink(missing_ok=True)
            record.update({"status": "error", "attempts": attempt,
                           "error": f"{type(exc).__name__}: {exc}",
                           "total_seconds": round(time.perf_counter() - start_all, 4)})
            if padded.exists():
                record["padded_audio_path"] = str(padded.relative_to(ROOT) if padded.is_relative_to(ROOT) else padded)
            print(f"ERROR {args.model}/{args.profile}/{case['id']}: {exc}", file=sys.stderr)
        write_measurement(RESULTS / "measurements.jsonl", record)
    return 1 if failures else 0


def load_measurements(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def latest_run(records: list[dict], run_id: str | None) -> tuple[str, dict]:
    if run_id is not None:
        validate_run_id(run_id)
    else:
        run_id = next((row["run_id"] for row in reversed(records) if row.get("run_id")), None)
    if not run_id:
        raise ValueError("No run-scoped measurements found. Generate a new run first; legacy results are not mixed in.")
    latest = {}
    for row in records:
        if row.get("run_id") == run_id:
            latest[(row["model"], row["profile"], row["case_id"])] = row
    if not latest:
        raise ValueError(f"No measurements for run ID: {run_id}")
    return run_id, latest


def compare(run_id: str | None = None) -> int:
    run_id, latest = latest_run(load_measurements(RESULTS / "measurements.jsonl"), run_id)
    print(f"Run ID: {run_id}")
    # Reusing a run ID with changed settings or text must not silently produce a comparison.
    for field in ("cases_sha256", "rules_sha256", "system", "python"):
        if len({r.get(field) for r in latest.values()}) > 1:
            raise ValueError(f"Mixed {field} in run; use a new run ID")
    for model in {k[0] for k in latest}:
        configs = {json.dumps(r.get("configuration"), sort_keys=True) for (m, _, _), r in latest.items()
                   if m == model and r.get("status") == "ok"}
        if len(configs) > 1:
            raise ValueError(f"Mixed configurations for {model}; use a new run ID")
    for case_id in {k[2] for k in latest}:
        if len({r.get("text") for (_, _, c), r in latest.items() if c == case_id}) > 1:
            raise ValueError(f"Mixed source texts for {case_id}; use a new run ID")
    groups = {(m, p) for m, p, _ in latest}
    case_sets = {frozenset(c for m, p, c in latest if (m, p) == group) for group in groups}
    if len(case_sets) > 1:
        print("WARNING: model/profile groups contain different cases; averages are not directly comparable.")
    print("model | profile | success/total | avg RTF | avg total RTF | attempts | failures")
    print("--- | --- | --- | --- | --- | --- | ---")
    for model, profile in sorted({(k[0], k[1]) for k in latest}):
        rows = [r for (m, p, _), r in latest.items() if (m, p) == (model, profile)]
        ok = [r for r in rows if r["status"] == "ok"]
        def average(key: str) -> str:
            values = [r[key] for r in ok if isinstance(r.get(key), (int, float))]
            return f"{sum(values) / len(values):.3f}" if values else "n/a"
        attempts = sum(r.get("attempts", 1) for r in rows)
        print(f"{model} | {profile} | {len(ok)}/{len(rows)} | {average('rtf')} | {average('total_rtf')} | {attempts} | {len(rows) - len(ok)}")
    asr_path = RESULTS / "asr.jsonl"
    if asr_path.exists():
        asr_latest = {}
        for row in load_measurements(asr_path):
            key = (row.get("model"), row.get("profile"), row.get("case_id"))
            measurement = latest.get(key)
            if (measurement and measurement.get("status") == "ok"
                    and row.get("record_id") == measurement.get("record_id")
                    and row.get("audio_sha256") == measurement.get("audio_sha256")):
                path = ROOT / measurement["audio_path"]
                if path.is_file() and audio_hash(path) == measurement["audio_sha256"]:
                    asr_latest[key] = row
        print("ASR transcript proxy: model | profile | mean CER | samples")
        for model, profile in sorted({(m, p) for m, p, _ in asr_latest}):
            vals = [r["cer"] for (m, p, _), r in asr_latest.items()
                    if (m, p) == (model, profile) and r.get("status") == "ok"]
            if vals:
                print(f"{model} | {profile} | {sum(vals) / len(vals):.3f} | {len(vals)}")
    print("NOTE: Success means basic WAV checks only. Dummy boundaries and pronunciation are not verified; ASR CER is a rough proxy.")
    return 0


def play(model: str, case: str, profile: str, run_id: str | None = None) -> None:
    _, latest = latest_run(load_measurements(RESULTS / "measurements.jsonl"), run_id)
    record = latest.get((model, profile, case))
    if not record or record.get("status") != "ok":
        raise FileNotFoundError("No successful audio for this model/profile/case in the selected run.")
    path = ROOT / record["audio_path"]
    if not path.is_file() or audio_hash(path) != record.get("audio_sha256"):
        raise ValueError(f"Audio missing or modified: {path}")
    if sys.platform != "darwin":
        raise RuntimeError("Audio playback uses macOS 'afplay'. Open WAVs manually on other OSes.")
    subprocess.run(["afplay", str(path)], check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline Japanese TTS profile comparison")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="List cases and supported models")
    validate_cmd = sub.add_parser("validate", help="Check common cases and reading rules")
    validate_cmd.add_argument("--cases", default=str(DEFAULT_CASES))
    validate_cmd.add_argument("--rules", default=str(DEFAULT_RULES))
    download_cmd = sub.add_parser("download", help="Explicitly download model weights")
    download_cmd.add_argument("--model", required=True, choices=sorted(MODEL_REPOS))
    synth = sub.add_parser("generate", help="Generate WAVs for a profile")
    synth.add_argument("--model", required=True, choices=sorted(MODEL_REPOS))
    synth.add_argument("--profile", choices=PROFILES, default="raw")
    synth.add_argument("--cases", default=str(DEFAULT_CASES))
    synth.add_argument("--rules", default=str(DEFAULT_RULES))
    synth.add_argument("--case", help="One case ID instead of all")
    synth.add_argument("--run-id", default=os.environ.get("TTSBENCH_RUN_ID"), help="Experiment ID; generated if omitted")
    synth.add_argument("--max-retries", type=int, default=0, help="Extra attempts after failed WAV quality checks")
    synth.add_argument("--qwen-speaker", default="Ono_Anna")
    synth.add_argument("--omnivoice-steps", type=int, default=16)
    playback = sub.add_parser("play", help="Play a generated WAV on macOS")
    playback.add_argument("--model", required=True, choices=sorted(MODEL_REPOS))
    playback.add_argument("--case", default="basic-01")
    playback.add_argument("--profile", choices=PROFILES, default="raw")
    playback.add_argument("--run-id", help="Experiment ID (default: latest)")
    comparison = sub.add_parser("compare", help="Compare profiles and summarize speed and basic checks")
    comparison.add_argument("--run-id", help="Experiment ID (default: latest)")
    sub.add_parser("download:asr", help="Download offline ASR model explicitly")
    asr_cmd = sub.add_parser("evaluate:asr", help="ASR CER proxy for generated audio")
    asr_cmd.add_argument("--model", choices=sorted(MODEL_REPOS))
    asr_cmd.add_argument("--profile", choices=PROFILES)
    asr_cmd.add_argument("--case")
    asr_cmd.add_argument("--run-id", help="Experiment ID (default: latest)")
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            for key, repo in MODEL_REPOS.items():
                print(f"{key:10} {repo}")
            for case in load_cases(DEFAULT_CASES):
                print(f"{case['id']:14} {case['category']:16} {case['text']}")
            print("Profiles: " + ", ".join(PROFILES))
            return 0
        if args.command == "validate":
            return validate(Path(args.cases), Path(args.rules))
        if args.command == "download":
            print(f"Weights saved to: {download(args.model)}")
            return 0
        if args.command == "generate":
            if args.omnivoice_steps <= 0 or args.max_retries < 0:
                raise ValueError("--omnivoice-steps must be positive and --max-retries nonnegative")
            return generate(args)
        if args.command == "play":
            play(args.model, args.case, args.profile, args.run_id)
            return 0
        if args.command == "compare":
            return compare(args.run_id)
        if args.command == "download:asr":
            from .asr import download_asr
            print(f"ASR weights saved to: {download_asr()}")
            return 0
        if args.command == "evaluate:asr":
            from .asr import evaluate
            return evaluate(model=args.model, profile=args.profile, case_id=args.case,
                            run_id=args.run_id, results=RESULTS)
    except (ValueError, FileNotFoundError, OSError, RuntimeError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 2
