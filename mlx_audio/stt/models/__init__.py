import importlib

__all__ = [
    "cohere_asr",
    "fireredasr2",
    "glmasr",
    "granite_speech",
    "granite_speech5_ctc",
    "granite_speech_nar",
    "lasr_ctc",
    "mimo_v2_asr",
    "moss_music",
    "moss_transcribe_diarize",
    "parakeet",
    "phonon",
    "qwen3_asr",
    "sensevoice",
    "voxtral",
    "voxtral_realtime",
    "wav2vec",
    "whisper",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return importlib.import_module(f"{__name__}.{name}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
