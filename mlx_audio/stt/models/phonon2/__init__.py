from .phonon2 import Model

DETECTION_HINTS = {"model_type_aliases": {"parakeet_tdt_five_value"}}

__all__ = ["Model"]


def prepare_config(config, model_path):
    # The generic converter casts every tensor, including packed uint32 codes,
    # and may requantize tables. Use the lossless loader conversion instead.
    raise ValueError(
        'Use mlx_audio.stt.load("FermionResearch/Phonon-2") to convert Phonon-2 '
        "without casting or requantizing its packed codes and row palettes."
    )
