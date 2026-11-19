from dataclasses import dataclass, field

from ..omnivoice.config import OmniVoiceConfig


@dataclass
class ModelConfig(OmniVoiceConfig):
    model_type: str = "reverb"
    echo: dict = field(default_factory=dict)

    @property
    def embedding_rank(self):
        return self.echo.get("architecture", {}).get("embedding_rank", 256)

    @property
    def prefix_blocking(self):
        return self.echo.get("prefix_blocking", {}).get("enabled", True)
