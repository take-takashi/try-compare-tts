import json
import math
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from ttsbench.asr import character_error_rate
from ttsbench.cli import load_measurements, main
from ttsbench.profiles import DEFAULT_RULES, load_rules, prepare
from ttsbench.quality import audio_metrics, quality_failures, trim_dummy_speech


def pcm_wav(path: Path, segments: list[tuple[float, bool]], rate: int = 16000):
    from array import array
    values = array("h")
    for seconds, audible in segments:
        n = round(seconds * rate)
        for j in range(n):
            values.append(round(3500 * math.sin(2 * math.pi * 260 * j / rate)) if audible else 0)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(values.tobytes())


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.rules = load_rules(DEFAULT_RULES)

    def test_raw_is_byte_for_byte_unchanged(self):
        c = {"id": "english-01", "text": "GitHub ActionsとPythonを使う"}
        p = prepare(c, "raw", self.rules)
        self.assertEqual(p.synthesized, c["text"])
        self.assertFalse(p.reading_changed)
        self.assertFalse(p.trimmed)

    def test_normalized_replaces_global_without_cascading(self):
        c = {"id": "unknown", "text": "GitHub ActionsとPythonを使う"}
        p = prepare(c, "normalized", self.rules)
        self.assertEqual(p.synthesized, "ギットハブ アクションズとパイソンを使う")
        self.assertTrue(p.reading_changed)

    def test_homograph_override_is_context_specific(self):
        c = {"id": "kanji-02", "text": "人気のない公園で、人気の歌手について話しました。"}
        p = prepare(c, "normalized", self.rules)
        self.assertIn("ひとけ", p.synthesized)
        self.assertIn("にんき", p.synthesized)

    def test_stale_override_fails(self):
        with self.assertRaisesRegex(ValueError, "Stale"):
            prepare({"id": "kanji-02", "text": "原稿変更"}, "normalized", self.rules)

    def test_improved_has_both_modifications(self):
        c = {"id": "unknown", "text": "Python"}
        p = prepare(c, "improved", self.rules)
        self.assertTrue(p.trimmed)
        self.assertTrue(p.reading_changed)
        self.assertIn("パイソン", p.synthesized)
        self.assertTrue(p.synthesized.startswith("うん"))

    def test_dummy_requires_two_separator_gaps(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "generated.wav"
            pcm_wav(f, [(0.25, True), (0.36, False), (1.10, True), (0.37, False), (0.24, True)])
            before = audio_metrics(f)["audio_seconds"]
            info = trim_dummy_speech(f)
            self.assertGreater(info["trim_start_seconds"], 0.15)
            self.assertLess(audio_metrics(f)["audio_seconds"], before)
            self.assertGreater(audio_metrics(f)["audio_seconds"], 1)

    def test_trim_fails_closed_when_pauses_absent(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "generated.wav"
            pcm_wav(f, [(1.5, True)])
            with self.assertRaisesRegex(ValueError, "separator"):
                trim_dummy_speech(f)
            self.assertEqual(audio_metrics(f)["audio_seconds"], 1.5)

    def test_quiet_audio_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "silent.wav"
            pcm_wav(f, [(1, False)])
            self.assertIn("too_quiet", quality_failures(audio_metrics(f)))

    def test_asr_metric_is_rough_not_a_reading_truth(self):
        self.assertEqual(character_error_rate("今日は。", "今日は"), 0)
        self.assertGreater(character_error_rate("御徒町", "おかちまち"), 0)

    def test_profiles_generate_in_separate_directories(self):
        with tempfile.TemporaryDirectory() as d:
            temp = Path(d)
            calls = []
            def fake_synth(text: str, path: Path):
                calls.append(text)
                pcm_wav(path, [(0.25, True), (0.36, False), (1.10, True), (0.37, False), (0.24, True)] if text.startswith("うん") else [(1, True)])
            with patch("ttsbench.cli.RESULTS", temp), patch("ttsbench.cli.load_engine", return_value=(fake_synth, {"test": True})):
                self.assertEqual(main(["generate", "--model", "qwen3", "--profile", "raw", "--case", "english-01", "--run-id", "experiment"]), 0)
                self.assertEqual(main(["generate", "--model", "qwen3", "--profile", "improved", "--case", "english-01", "--run-id", "experiment"]), 0)
            self.assertEqual(len(calls), 2)
            self.assertNotEqual(calls[0], calls[1])
            records = load_measurements(temp / "measurements.jsonl")
            self.assertEqual([r["profile"] for r in records], ["raw", "improved"])
            self.assertTrue(all(r["status"] == "ok" for r in records))
            self.assertTrue(all(Path(r["audio_path"]).exists() for r in records))
            self.assertTrue(Path(records[1]["padded_audio_path"]).exists())
            self.assertNotEqual(records[0]["generation_id"], records[1]["generation_id"])

    def test_quality_retry_repeats_and_records_attempts(self):
        with tempfile.TemporaryDirectory() as d:
            temp = Path(d)
            calls = []
            def fake_synth(text: str, path: Path):
                calls.append(text)
                pcm_wav(path, [(1, False)] if len(calls) == 1 else [(1, True)])
            with patch("ttsbench.cli.RESULTS", temp), patch("ttsbench.cli.load_engine", return_value=(fake_synth, {})):
                self.assertEqual(main(["generate", "--model", "qwen3", "--max-retries", "1", "--case", "basic-01"]), 0)
            self.assertEqual(len(calls), 2)
            record = load_measurements(temp / "measurements.jsonl")[0]
            self.assertEqual(record["attempts"], 2)
            self.assertEqual(record["status"], "ok")


if __name__ == "__main__":
    unittest.main()
