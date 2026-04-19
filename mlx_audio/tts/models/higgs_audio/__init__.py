"""Higgs Audio v2 — MLX port."""
from .config import HiggsAudioConfig, HiggsTextConfig
from .higgs_audio import HiggsAudioModel
from .model import Model, ModelConfig
from .serve import HiggsAudioGenerationResult, HiggsAudioServer, build_prompt

__all__ = [
    # Framework-conforming entry points — used by mlx_audio.tts.utils.load
    # and `python -m mlx_audio.tts.generate`.
    "Model",
    "ModelConfig",
    # Lower-level building blocks.
    "HiggsAudioConfig",
    "HiggsTextConfig",
    "HiggsAudioModel",
    "build_prompt",
    # Additional Python API — composes model + codec + tokenizer with a
    # kwarg-rich generate(target_text=..., reference_audio_path=..., ...)
    # signature tailored to the Higgs serve flow.
    "HiggsAudioServer",
    "HiggsAudioGenerationResult",
]
