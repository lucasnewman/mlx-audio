"""Configuration for the one-step FastDialogueSidon separator."""

import json
import math
from dataclasses import dataclass, field, fields
from pathlib import Path

from mlx_audio.sts.models.dialogue_sidon.config import EncoderConfig


@dataclass
class HeadConfig:
    hidden_size: int = 768
    num_layers: int = 8
    num_heads: int = 12
    ffn_ratio: float = 4.0
    frequency_embedding_size: int = 256


@dataclass
class CTMConfig:
    alpha_t: float = 0.006352818571031094
    sigma_t: float = 0.9999797940254211
    c_skip: float = 0.010001031681895256
    c_out: float = 0.9998864531517029
    guidance_scale: float = 1.0


@dataclass
class GeneratorConfig:
    hidden_dim: int = 512
    intermediate_dim: int = 1536
    num_layers: int = 8
    kernel_size: int = 7


@dataclass
class VocoderConfig:
    num_iteration: int = 2
    latent_dim: int = 32
    sample_rate: int = 24000
    latent_rate: int = 50
    n_fft_stft: int = 1024
    latent_upsample: int = 2
    generator_args: GeneratorConfig = field(default_factory=GeneratorConfig)

    def __post_init__(self):
        if isinstance(self.generator_args, dict):
            self.generator_args = GeneratorConfig(**self.generator_args)
        if self.latent_upsample not in (1, 2):
            raise ValueError("latent_upsample must be 1 or 2")
        if self.latent_rate <= 0 or self.sample_rate % (
            self.latent_rate * self.latent_upsample
        ):
            raise ValueError("sample_rate must be divisible by the vocoder frame rate")
        if self.num_iteration < 1 or self.n_fft_stft < 2 or self.n_fft_stft % 2:
            raise ValueError("Expected positive iterations and an even FFT size")
        if self.generator_args.kernel_size % 2 != 1:
            raise ValueError("Vocoder kernel_size must be odd")


@dataclass
class ModelConfig:
    model_type: str = "fast_dialogue_sidon"
    sample_rate: int = 24000
    latent_dim: int = 32
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    diffusion: HeadConfig = field(default_factory=HeadConfig)
    ctm: CTMConfig = field(default_factory=CTMConfig)
    vocoder: VocoderConfig = field(default_factory=VocoderConfig)
    weight_format: str = "mlx"
    model_path: str = ""
    source_repo: str = "higurashi-ai/FastDialogueSidon"
    source_revision: str = ""

    def __post_init__(self):
        for name, kind in (
            ("encoder", EncoderConfig),
            ("diffusion", HeadConfig),
            ("ctm", CTMConfig),
            ("vocoder", VocoderConfig),
        ):
            if isinstance(getattr(self, name), dict):
                setattr(self, name, kind(**getattr(self, name)))
        if self.encoder.hidden_size % self.encoder.num_attention_heads:
            raise ValueError("Encoder hidden size must be divisible by its head count")
        if (
            self.diffusion.hidden_size % self.diffusion.num_heads
            or (self.diffusion.hidden_size // self.diffusion.num_heads) % 2
        ):
            raise ValueError("DiT head dimension must be an even integer")
        if self.latent_dim != self.vocoder.latent_dim:
            raise ValueError("Encoder and vocoder latent dimensions must match")
        if (
            self.sample_rate != self.vocoder.sample_rate
            or self.vocoder.latent_rate != 50
        ):
            raise ValueError("Expected matching sample rates and 50 Hz latents")
        if not all(math.isfinite(v) for v in vars(self.ctm).values()):
            raise ValueError("CTM constants must be finite")
        if self.ctm.guidance_scale != 1.0:
            raise ValueError("Only the published unguided CTM checkpoint is supported")
        if self.weight_format not in ("mlx", "source"):
            raise ValueError("weight_format must be 'mlx' or 'source'")

    @classmethod
    def from_dict(cls, config):
        config = dict(config)
        if (
            "encoder" not in config
            and config.get("model_type") == "fast-dialogue-sidon"
        ):
            root = Path(config.get("model_path", ""))
            if not (root / "encoder/config.json").is_file():
                raise ValueError("The source bundle requires encoder/config.json")
            encoder = json.loads((root / "encoder/config.json").read_text())
            if encoder.get("position_embeddings_type") != "relative_key" or encoder.get(
                "add_adapter", False
            ):
                raise ValueError("Expected a relative-key encoder without adapters")
            config["encoder"] = {
                f.name: encoder[f.name]
                for f in fields(EncoderConfig)
                if f.name in encoder
            }
            head = json.loads((root / "diffusion_head/config.json").read_text())
            if head.get("guidance", False) or not head.get("use_positional", True):
                raise ValueError("Expected an unguided DiT with rotary positions")
            if head["latent_size"] != 2 * config["latent_dim"] or head["cond_size"] != (
                2 * config["latent_dim"] + config["encoder"]["hidden_size"]
            ):
                raise ValueError("Inconsistent DiT latent or conditioning dimensions")
            config["diffusion"] = {
                "hidden_size": head["hidden_size"],
                "num_layers": head["head_layers"],
                "num_heads": head["num_heads"],
                "ffn_ratio": head["head_ffn_ratio"],
            }
            config["vocoder"] = json.loads((root / "vocoder/config.json").read_text())
            config["sample_rate"] = config["vocoder"]["sample_rate"]
            config["weight_format"] = "source"
        config["model_type"] = "fast_dialogue_sidon"
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in config.items() if k in names})
