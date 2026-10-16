from .config import ModelConfig
from .phonon2 import Model

DETECTION_HINTS = {"model_type_aliases": {"parakeet_tdt_five_value"}}

__all__ = ["Model", "ModelConfig"]


def prepare_config(config, model_path):
    # The generic converter casts every tensor, including packed uint32 codes,
    # and may requantize tables. Use the lossless model-specific converter.
    raise ValueError(
        "Use python -m mlx_audio.stt.models.phonon2.convert to convert Phonon-2 "
        "without casting or requantizing its packed codes and row palettes."
    )
