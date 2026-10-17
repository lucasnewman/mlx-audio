from .config import ModelConfig
from .phonon2 import Model

DETECTION_HINTS = {"model_type_aliases": {"parakeet_tdt_five_value"}}

__all__ = ["Model", "ModelConfig"]


def prepare_config(config, model_path):
    # Generic conversion would cast packed codes and may requantize row palettes.
    raise ValueError(
        "Phonon-2 conversion and requantization are unsupported. "
        "Load mlx-community/Phonon-2 directly with mlx_audio.stt.load."
    )
