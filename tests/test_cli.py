import json
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from ttsbench.cli import RESULTS, load_cases, main, selected_cases, wav_duration, write_measurement
from ttsbench.engines import MODEL_REPOS, model_dir


class CasesTest(unittest.TestCase):
    def test_common_cases_are_valid(self):
        cases = load_cases(Path("cases/japanese.json"))
        self.assertGreaterEqual(len(cases), 8)
        self.assertEqual(len({item["id"] for item in cases}), len(cases))

    def test_duplicate_id_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            file = Path(name) / "cases.json"
            file.write_text(json.dumps([{"id": "same", "category": "basic", "text": "こんにちは"}] * 2), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_cases(file)

    def test_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            file = Path(name) / "cases.json"
            file.write_text(json.dumps([{"id": "../escape", "category": "basic", "text": "こんにちは"}]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                load_cases(file)

    def test_unknown_case_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown case"):
            selected_cases(Path("cases/japanese.json"), "does-not-exist")

    def test_audio_duration(self):
        with tempfile.TemporaryDirectory() as name:
            file = Path(name) / "test.wav"
            with wave.open(str(file), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(bytes(16000 * 2))
            self.assertEqual(wav_duration(file), 1.0)

    def test_jsonl(self):
        with tempfile.TemporaryDirectory() as name:
            file = Path(name) / "measurements.jsonl"
            write_measurement(file, {"text": "御徒町"})
            self.assertEqual(json.loads(file.read_text(encoding="utf-8")), {"text": "御徒町"})

    def test_models_are_under_repository(self):
        for key in MODEL_REPOS:
            self.assertEqual(model_dir(key).parent.name, "models")

    def test_qwen3_uses_1_7b_customvoice(self):
        self.assertEqual(MODEL_REPOS["qwen3"], "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
        self.assertEqual(model_dir("qwen3").name, "qwen3-1.7b-customvoice")

    def test_validation_cli(self):
        self.assertEqual(main(["validate"]), 0)

    def test_empty_model_download_is_not_started_by_generate(self):
        with patch("ttsbench.cli.load_engine", side_effect=FileNotFoundError("weights absent")) as loader:
            self.assertEqual(main(["generate", "--model", "omnivoice", "--case", "basic-01"]), 1)
            loader.assert_called_once()


if __name__ == "__main__":
    unittest.main()
