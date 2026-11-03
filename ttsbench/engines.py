"""Isolated, local-only Qwen3-TTS and OmniVoice adapters.

Heavy optional imports happen only during download or synthesis.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
MODEL_REPOS = {
    "qwen3": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "omnivoice": "k2-fsa/OmniVoice",
}
MODEL_DIRS = {
    "qwen3": ROOT / "models/qwen3-0.6b-customvoice",
    "omnivoice": ROOT / "models/omnivoice",
}


def model_dir(model: str) -> Path:
    return MODEL_DIRS[model]


def download(model: str) -> Path:
    from huggingface_hub import snapshot_download

    path = model_dir(model)
    path.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=MODEL_REPOS[model], local_dir=str(path))
    return path


def select_device(torch) -> tuple[str, object]:
    if torch.cuda.is_available():
        return "cuda:0", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def load_engine(model_name: str, *, speaker: str = "Ono_Anna", num_step: int = 16) -> tuple[Callable[[str, Path], None], dict]:
    """Return a synthesizer and its config; load weights exactly once per invocation."""
    path = model_dir(model_name)
    if not (path / "config.json").exists():
        raise FileNotFoundError(
            f"Model not downloaded: {path}. Run 'mise run download:{model_name}' first."
        )

    import soundfile as sf
    import torch

    device, dtype = select_device(torch)
    if model_name == "qwen3":
        from qwen_tts import Qwen3TTSModel

        engine = Qwen3TTSModel.from_pretrained(
            str(path), device_map=device, dtype=dtype, attn_implementation="sdpa"
        )

        def synthesize(text: str, destination: Path) -> None:
            wavs, rate = engine.generate_custom_voice(
                text=text, language="Japanese", speaker=speaker
            )
            sf.write(destination, wavs[0], rate)

        options = {"speaker": speaker, "language": "Japanese"}
    elif model_name == "omnivoice":
        from omnivoice import OmniVoice

        engine = OmniVoice.from_pretrained(str(path), device_map=device, dtype=dtype)

        def synthesize(text: str, destination: Path) -> None:
            audio = engine.generate(text=text, language="ja", num_step=num_step)
            sf.write(destination, audio[0], engine.sampling_rate)

        options = {"voice": "auto", "language": "ja", "num_step": num_step}
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return synthesize, {
        "checkpoint": MODEL_REPOS[model_name],
        "local_dir": str(path.relative_to(ROOT)),
        "device": device,
        "dtype": str(dtype),
        **options,
    }
