"""Native MLX implementation of Stable Audio 3."""

from .stable_audio_3 import Model, ModelConfig

DETECTION_HINTS = {
    "path_patterns": {"stable-audio-3", "stable_audio_3"},
}

__all__ = ["Model", "ModelConfig"]
