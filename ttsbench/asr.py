"""Optional offline ASR spot-check (a transcript proxy, not a pronunciation judge)."""
from __future__ import annotations

import json
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from .engines import ROOT

ASR_REPO = "Systran/faster-whisper-small"
ASR_DIR = ROOT / "models/asr-faster-whisper-small"


def download_asr() -> Path:
    from huggingface_hub import snapshot_download

    ASR_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=ASR_REPO, local_dir=str(ASR_DIR))
    return ASR_DIR


def comparable(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    # Punctuation and spaces are ignored; kanji/hiragana are NOT conflated.
    return "".join(c for c in text if not unicodedata.category(c).startswith(("P", "Z", "C", "S")))


def character_error_rate(reference: str, hypothesis: str) -> float:
    ref, hyp = comparable(reference), comparable(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, start=1):
        row = [i]
        for j, b in enumerate(hyp, start=1):
            row.append(min(row[-1] + 1, prev[j] + 1, prev[j - 1] + (a != b)))
        prev = row
    return round(prev[-1] / len(ref), 4)


def evaluate(*, model: str | None, profile: str | None, case_id: str | None,
             run_id: str | None, results: Path) -> int:
    from .cli import audio_hash, latest_run, load_measurements

    run_id, latest = latest_run(load_measurements(results / "measurements.jsonl"), run_id)
    candidates = [row for (m, p, c), row in latest.items()
                  if row.get("status") == "ok" and (not model or m == model)
                  and (not profile or p == profile) and (not case_id or c == case_id)]
    if not candidates:
        print("No successful measurements found for requested run/model/profile/case.")
        return 1
    if not ASR_DIR.exists() or not (ASR_DIR / "model.bin").is_file():
        raise FileNotFoundError("ASR weights absent. Run 'mise run download:asr' first.")
    from faster_whisper import WhisperModel

    engine = WhisperModel(str(ASR_DIR), device="cpu", compute_type="int8")
    failures = 0
    print(f"Run ID: {run_id}")
    for measurement in candidates:
        name, label, case = measurement["model"], measurement["profile"], measurement["case_id"]
        path = ROOT / measurement["audio_path"]
        result = {"timestamp": datetime.now(timezone.utc).isoformat(),
                  "run_id": run_id, "record_id": measurement["record_id"],
                  "audio_sha256": measurement["audio_sha256"],
                  "model": name, "profile": label, "case_id": case,
                  "asr_model": ASR_REPO, "expected": measurement["text"],
                  "audio_path": measurement["audio_path"]}
        try:
            if not path.is_file() or audio_hash(path) != measurement["audio_sha256"]:
                raise ValueError("Audio missing or modified since generation")
            segments, _ = engine.transcribe(str(path), language="ja", beam_size=5,
                                           condition_on_previous_text=False, vad_filter=False)
            transcript = "".join(seg.text for seg in segments).strip()
            result.update({"transcript": transcript, "cer": character_error_rate(measurement["text"], transcript), "status": "ok"})
            print(f"ASR {name}/{label}/{case} CER={result['cer']} {transcript}")
        except Exception as exc:
            failures += 1
            result.update(status="error", error=str(exc))
            print(f"ASR ERROR {name}/{label}/{case}: {exc}")
        out = results / "asr.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
    return 1 if failures else 0
