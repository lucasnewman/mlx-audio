from dataclasses import dataclass
from typing import Any, Dict, Optional

from mlx_audio.stt.models.parakeet.parakeet import ParakeetTDTArgs
from mlx_audio.utils import from_dict


@dataclass
class ModelConfig(ParakeetTDTArgs):
    """Parakeet TDT configuration with Phonon-2 packed module metadata."""

    phonon2_modules: list[dict]
    model_type: str = "phonon2"
    model_path: Optional[str] = None

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "ModelConfig":
        return from_dict(cls, params)
