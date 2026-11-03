import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from ttsbench.cli import load_measurements, main
from test_profiles import pcm_wav


class RunTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.results = Path(self.directory.name) / "results"
        self.results_patch = patch("ttsbench.cli.RESULTS", self.results)
        self.results_patch.start()
        self.addCleanup(self.results_patch.stop)

    @staticmethod
    def synth(text, path):
        pcm_wav(path, [(1, True)])

    def generate(self, run, case="basic-01", config=None):
        with patch("ttsbench.cli.load_engine", return_value=(self.synth, config or {"device": "cpu"})):
            return main(["generate", "--model", "qwen3", "--case", case, "--run-id", run])

    def output(self, args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            code = main(args)
        return code, out.getvalue()

    def test_runs_are_isolated_and_latest_is_default(self):
        self.assertEqual(self.generate("first"), 0)
        self.assertEqual(self.generate("second", "kanji-02"), 0)
        code, output = self.output(["compare"])
        self.assertEqual(code, 0)
        self.assertIn("Run ID: second", output)
        self.assertIn("1/1", output)
        self.assertEqual(self.output(["compare", "--run-id", "first"])[0], 0)
        self.assertEqual(self.output(["compare", "--run-id", "missing"])[0], 1)
        records = load_measurements(self.results / "measurements.jsonl")
        self.assertNotEqual(records[0]["audio_path"], records[1]["audio_path"])

    def test_failed_reload_replaces_success_in_summary_but_keeps_old_audio(self):
        self.assertEqual(self.generate("same"), 0)
        first = load_measurements(self.results / "measurements.jsonl")[0]
        with patch("ttsbench.cli.load_engine", side_effect=RuntimeError("load failed")):
            self.assertEqual(main(["generate", "--model", "qwen3", "--case", "basic-01", "--run-id", "same"]), 1)
        self.assertTrue(Path(first["audio_path"]).exists())
        code, output = self.output(["compare", "--run-id", "same"])
        self.assertEqual(code, 0)
        self.assertIn("0/1", output)
        self.assertIn("| 0 | 1", output)

    def test_mixed_configuration_is_rejected(self):
        self.generate("same", "basic-01", {"speaker": "one"})
        self.generate("same", "kanji-02", {"speaker": "two"})
        code, output = self.output(["compare", "--run-id", "same"])
        self.assertEqual(code, 1)
        self.assertIn("Mixed configurations", output)

    def test_unsafe_run_id_rejected(self):
        self.assertEqual(self.generate("../escape"), 1)

    def test_asr_is_bound_to_record_and_audio_content(self):
        self.generate("same")
        asr_dir = Path(self.directory.name) / "asr"
        asr_dir.mkdir()
        (asr_dir / "model.bin").touch()

        class FakeWhisper:
            def __init__(self, *args, **kwargs):
                pass

            def transcribe(self, *args, **kwargs):
                return [types.SimpleNamespace(text="こんにちは")], None

        with patch("ttsbench.asr.ASR_DIR", asr_dir), patch.dict(sys.modules, {"faster_whisper": types.SimpleNamespace(WhisperModel=FakeWhisper)}):
            self.assertEqual(main(["evaluate:asr", "--run-id", "same"]), 0)
            self.assertIn("qwen3 | raw |", self.output(["compare", "--run-id", "same"])[1].split("ASR transcript proxy:")[1])
            self.generate("same")
            self.assertNotIn("qwen3 | raw |", self.output(["compare", "--run-id", "same"])[1].split("ASR transcript proxy:")[1])
            self.assertEqual(main(["evaluate:asr", "--run-id", "same"]), 0)
            record = load_measurements(self.results / "measurements.jsonl")[-1]
            Path(record["audio_path"]).write_bytes(b"modified")
            self.assertNotIn("qwen3 | raw |", self.output(["compare", "--run-id", "same"])[1].split("ASR transcript proxy:")[1])
            self.assertEqual(main(["evaluate:asr", "--run-id", "same"]), 1)


if __name__ == "__main__":
    unittest.main()
