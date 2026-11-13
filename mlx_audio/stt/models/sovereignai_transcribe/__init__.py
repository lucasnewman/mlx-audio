from .config import ModelConfig
from .sovereignai_transcribe import Model, copy_supporting_files, write_model_card

DETECTION_HINTS = {
    "model_type_aliases": ["sovereignai_transcribe"],
    "architectures": ["SovereignAITranscribeForConditionalGeneration"],
    "path_patterns": [
        "sovereignai_transcribe",
        "sovereignai-transcribe",
        "sov-transcribe",
    ],
}

__all__ = ["Model", "ModelConfig", "copy_supporting_files", "write_model_card"]
