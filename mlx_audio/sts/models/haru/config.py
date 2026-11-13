"""Configuration for the Haru full-duplex speech model."""

from dataclasses import dataclass, field
from math import prod
from typing import Any

from mlx_audio.base import BaseModelArgs


@dataclass
class CodecConfig(BaseModelArgs):
    latent_dim: int = 64
    base_channels: int = 32
    ratios: tuple[int, ...] = (8, 5, 5, 4, 2, 2)
    encoder_depths: tuple[int, ...] = (3, 3, 3, 3, 3, 3, 8)
    decoder_depths: tuple[int, ...] = (8, 3, 3, 3, 3, 3, 3)
    norm_eps: float = 1e-5
    latent_std: float = 0.5

    def __post_init__(self):
        for name in ("ratios", "encoder_depths", "decoder_depths"):
            setattr(self, name, tuple(getattr(self, name)))
        if not self.ratios or min(self.ratios) < 2:
            raise ValueError("codec ratios must be at least 2")
        if any(
            len(d) != len(self.ratios) + 1
            for d in (self.encoder_depths, self.decoder_depths)
        ):
            raise ValueError("codec depths must have one entry per stage")

    @property
    def frame_samples(self):
        return prod(self.ratios)


@dataclass
class HeadConfig(BaseModelArgs):
    hidden_size: int = 2560
    latent_dim: int = 64
    num_layers: int = 4
    ffn_ratio: float = 3.0
    norm_eps: float = 1e-5
    interval_embed: float = 7.0
    num_steps: int = 1
    guidance_scale: float = 1.5
    guidance_range: tuple[float, float] = (1.0, 2.0)


@dataclass
class ModelConfig(BaseModelArgs):
    model_type: str = "haru"
    model_path: str | None = None
    sample_rate: int = 24000
    frame_samples: int = 3200
    latent_dim: int = 64
    pad_token_id: int = 248076
    eop_token_id: int = 248071
    dtype: str = "bfloat16"
    lm: dict[str, Any] = field(default_factory=dict)
    codec: CodecConfig | dict = field(default_factory=CodecConfig)
    head: HeadConfig | dict = field(default_factory=HeadConfig)

    def __post_init__(self):
        if self.model_type != "haru":
            raise ValueError(f"Unsupported model_type: {self.model_type}")
        if isinstance(self.codec, dict):
            self.codec = CodecConfig.from_dict(self.codec)
        if isinstance(self.head, dict):
            self.head = HeadConfig.from_dict(self.head)
        if self.codec.frame_samples != self.frame_samples:
            raise ValueError("codec ratios do not match frame_samples")
        if (
            self.codec.latent_dim != self.latent_dim
            or self.head.latent_dim != self.latent_dim
        ):
            raise ValueError("codec, head, and model latent dimensions must match")
        if self.head.hidden_size != self.hidden_size:
            raise ValueError("head and language model hidden sizes must match")
        if not self.lm.get("tie_word_embeddings", True):
            raise ValueError("Haru requires tied text embeddings")
        interval = self.lm.get("full_attention_interval", 4)
        count = self.lm.get("num_hidden_layers", 32)
        expected = [
            "full_attention" if (i + 1) % interval == 0 else "linear_attention"
            for i in range(count)
        ]
        if self.lm.get("layer_types", expected) != expected or count < interval:
            raise ValueError("Unsupported Haru attention layer schedule")

    @property
    def frame_rate(self):
        return self.sample_rate / self.frame_samples

    @property
    def hidden_size(self):
        return self.lm.get("hidden_size", 2560)

    @property
    def eos_token_id(self):
        return self.lm.get("eos_token_id")
