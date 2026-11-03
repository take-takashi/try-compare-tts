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


def evaluate(cases: list[dict], *, model: str | None, profile: str | None, case_id: str | None, results: Path) -> int:
    if not ASR_DIR.exists() or not (ASR_DIR / "model.bin").is_file():
        raise FileNotFoundError("ASR weights absent. Run 'mise run download:asr' first.")
    from faster_whisper import WhisperModel

    engine = WhisperModel(str(ASR_DIR), device="cpu", compute_type="int8")
    evaluated = 0
    failures = 0
    candidates = []
    models = [model] if model else ["qwen3", "omnivoice"]
    profiles = [profile] if profile else ["raw", "normalized", "dummy", "improved"]
    for name in models:
        for label in profiles:
            for case in cases:
                if case_id and case["id"] != case_id:
                    continue
                path = results / "audio" / name / label / f"{case['id']}.wav"
                if path.exists():
                    candidates.append((name, label, case, path))
    for name, label, case, path in candidates:
        result = {"timestamp": datetime.now(timezone.utc).isoformat(),
                  "model": name, "profile": label, "case_id": case["id"],
                  "asr_model": ASR_REPO, "expected": case["text"],
                  "audio_path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)}
        try:
            segments, _ = engine.transcribe(str(path), language="ja", beam_size=5,
                                           condition_on_previous_text=False, vad_filter=False)
            transcript = "".join(seg.text for seg in segments).strip()
            result.update({"transcript": transcript, "cer": character_error_rate(case["text"], transcript), "status": "ok"})
            print(f"ASR {name}/{label}/{case['id']} CER={result['cer']} {transcript}")
        except Exception as exc:
            failures += 1
            result.update(status="error", error=str(exc))
            print(f"ASR ERROR {name}/{label}/{case['id']}: {exc}")
        out = results / "asr.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
        evaluated += 1
    if not evaluated:
        print("No generated WAVs found for requested model/profile/case.")
        return 1
    return 1 if failures else 0
