"""Deterministic Japanese TTS input profiles; never modify the shared cases."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .engines import ROOT

DEFAULT_RULES = ROOT / "cases/reading_overrides.json"
PROFILES = ("raw", "normalized", "dummy", "improved")
PREFIX = "うん。……"
SUFFIX = "……はい。"


@dataclass(frozen=True)
class PreparedText:
    original: str
    synthesized: str
    profile: str
    trimmed: bool
    reading_changed: bool


def load_rules(path: Path = DEFAULT_RULES) -> dict:
    rules = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rules, dict) or not isinstance(rules.get("global"), dict) or not isinstance(rules.get("cases"), dict):
        raise ValueError("Reading rules require 'global' and 'cases' dictionaries")
    for source, replacement in rules["global"].items():
        if not isinstance(source, str) or not source or not isinstance(replacement, str) or not replacement:
            raise ValueError("Reading global rules must map non-empty strings")
    for case_id, override in rules["cases"].items():
        if not isinstance(case_id, str) or not isinstance(override, dict) or any(not isinstance(override.get(k), str) or not override[k] for k in ("source", "replacement")):
            raise ValueError("Reading case overrides require source/replacement strings")
    return rules


def prepare(case: dict[str, str], profile: str, rules: dict) -> PreparedText:
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile: {profile}")
    original = case["text"]
    normalized = original
    if profile in ("normalized", "improved"):
        mapping = rules["global"]
        if mapping:
            matcher = re.compile("|".join(re.escape(s) for s in sorted(mapping, key=lambda s: (-len(s), s))))
            normalized = matcher.sub(lambda match: mapping[match.group(0)], normalized)
        override = rules["cases"].get(case["id"])
        if override:
            if override["source"] != original:
                raise ValueError(f"Stale reading override for {case['id']} (source differs from case text)")
            normalized = override["replacement"]
    trimmed = profile in ("dummy", "improved")
    synthesized = PREFIX + normalized + SUFFIX if trimmed else normalized
    return PreparedText(original, synthesized, profile, trimmed, original != normalized)
