"""Conservative WAV quality checks and experimental dummy speech cropping.

Cropping requires clear silence intervals; it NEVER guesses where the text begins.
These checks cannot establish Japanese reading correctness or speaker identity.
"""
from __future__ import annotations

import math
import sys
import wave
from array import array
from pathlib import Path


def samples_from_wav(path: Path) -> tuple[int, array]:
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getcomptype() != "NONE":
            raise ValueError("Quality checks require mono 16-bit PCM WAV")
        rate = wav.getframerate()
        samples = array("h", wav.readframes(wav.getnframes()))
    if sys.byteorder != "little":
        samples.byteswap()
    if not rate or not samples:
        raise ValueError("Empty WAV")
    return rate, samples


def audio_metrics(path: Path) -> dict[str, float]:
    rate, samples = samples_from_wav(path)
    length = len(samples)
    peak = max(abs(v) for v in samples) / 32768.0
    clipping = sum(1 for v in samples if abs(v) >= 32760) / length
    rms = math.sqrt(sum(v * v for v in samples) / length) / 32768.0
    return {
        "rms": round(rms, 5),
        "peak": round(peak, 5),
        "clipping_ratio": round(clipping, 5),
        "audio_seconds": round(length / rate, 4),
    }


def quality_failures(metrics: dict[str, float]) -> list[str]:
    failures = []
    if metrics["rms"] < 0.001:
        failures.append("too_quiet")
    if metrics["clipping_ratio"] > 0.01:
        failures.append("clipping")
    if metrics["audio_seconds"] < 0.15:
        failures.append("too_short")
    return failures


def trim_dummy_speech(path: Path, *, min_gap: float = 0.24) -> dict[str, float]:
    """Trim prefix/suffix around *two distinct* pronounced silence gaps.

    A wrong cut would corrupt a benchmark; require an early and a late gap,
    enough speech on each side, and enough speech remaining between them.
    Keep the source WAV as `.padded.wav` for listening to unsuccessful trials.
    """
    rate, samples = samples_from_wav(path)
    window = max(1, rate // 100)  # 10 ms
    magnitudes = [max(abs(s) for s in samples[i:i + window]) for i in range(0, len(samples), window)]
    maximum = max(magnitudes)
    if maximum < 150:
        raise ValueError("Dummy crop: no audible signal")
    threshold = max(180, round(maximum * 0.035))
    active = [v > threshold for v in magnitudes]
    voiced = [i for i, value in enumerate(active) if value]
    if not voiced:
        raise ValueError("Dummy crop: no active windows")
    first, last = voiced[0], voiced[-1]
    gaps: list[tuple[int, int]] = []
    start = None
    for i in range(first, last + 1):
        if not active[i] and start is None:
            start = i
        if active[i] and start is not None:
            if (i - start) * window / rate >= min_gap:
                gaps.append((start, i))
            start = None
    # Only accept the first/last marker pauses; other pauses inside the body
    # must not be mistaken for padding unless they lie near its boundaries.
    early = [g for g in gaps if g[0] < first + (last - first) * 0.35]
    late = [g for g in gaps if g[1] > first + (last - first) * 0.65]
    if not early or not late:
        raise ValueError("Dummy crop: separator pauses not reliably detected")
    left_gap, right_gap = early[0], late[-1]
    if left_gap == right_gap or left_gap[1] >= right_gap[0]:
        raise ValueError("Dummy crop: two independent separator pauses required")
    begin = left_gap[1] * window
    end = min(len(samples), right_gap[0] * window)
    if end - begin < rate * 0.30:
        raise ValueError("Dummy crop: retained body too short")
    # Preserve 30ms around the boundaries to avoid clipped consonants.
    begin = max(0, begin - round(rate * 0.03))
    end = min(len(samples), end + round(rate * 0.03))
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(samples[begin:end].tobytes())
    return {"trim_start_seconds": round(begin / rate, 3), "trim_end_seconds": round((len(samples) - end) / rate, 3)}
