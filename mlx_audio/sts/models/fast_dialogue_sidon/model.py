"""One-step two-speaker separation and restoration on Apple Silicon."""

import re
from pathlib import Path

import mlx.core as mx
from mlx import nn

from mlx_audio.sts.models.dialogue_sidon.encoder import Encoder
from mlx_audio.sts.models.dialogue_sidon.frontend import (
    extract_features,
    normalize_chunk,
)
from mlx_audio.sts.models.dialogue_sidon.separation import SeparationModel

from .config import ModelConfig
from .ctm import CTMHead
from .vocoder import WaveFitNeXtDecoder


def sanitize_source_weights(encoder, head, vocoder, stats):
    """Map the published PyTorch safetensors to MLX names and convolution layouts."""
    weights = {}
    for key, value in encoder.items():
        if key == "model.masked_spec_embed":
            continue  # Training-only SpecAugment state.
        if key.startswith("model."):
            key = "encoder." + key.removeprefix("model.").removeprefix("encoder.")
        key = key.replace("speaker_1.", "linear1.").replace("speaker_2.", "linear2.")
        if value.ndim == 3:
            value = value.transpose(0, 2, 1)
        weights[key] = value
    for key, value in head.items():
        key = re.sub(r"\.(mlp|adaLN_modulation)\.(\d+)\.", r".\1.layers.\2.", key)
        weights["diffusion_head." + key] = value
    for key, value in vocoder.items():
        if value.ndim == 3:
            value = (
                value.transpose(1, 2, 0)
                if key == "upsampler.weight"
                else value.transpose(0, 2, 1)
            )
        key = key.replace(".norm.norm.", ".norm.")
        key = key.replace(".input_norm.norm.", ".input_norm.")
        key = key.replace(".final_norm.norm.", ".final_norm.")
        weights["decoder." + key] = value
    weights.update({key: value.reshape(-1) for key, value in stats.items()})
    return weights


class Model(SeparationModel):
    default_num_steps = 1

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config.encoder)
        self.linear1 = nn.Linear(config.encoder.hidden_size, config.latent_dim)
        self.linear2 = nn.Linear(config.encoder.hidden_size, config.latent_dim)
        self.diffusion_head = CTMHead(
            config.diffusion,
            2 * config.latent_dim,
            config.encoder.hidden_size + 2 * config.latent_dim,
        )
        self.decoder = WaveFitNeXtDecoder(config.vocoder)
        self.latent_norm_mean = mx.zeros((2 * config.latent_dim,))
        self.latent_norm_std = mx.ones((2 * config.latent_dim,))

    @property
    def dtype(self):
        return self.linear1.weight.dtype

    def _validate_num_steps(self, num_steps):
        if num_steps != 1 or not isinstance(num_steps, int):
            raise ValueError(
                "FastDialogueSidon uses exactly one step; num_steps must be 1"
            )

    def encode(self, input_features, attention_mask=None):
        features = self.encoder(input_features.astype(self.dtype), attention_mask)
        return features, self.linear1(features), self.linear2(features)

    def sample_latents(self, conditioning, num_steps=1, initial_noise=None, key=None):
        self._validate_num_steps(num_steps)
        shape = (*conditioning.shape[:2], 2 * self.config.latent_dim)
        if initial_noise is not None and initial_noise.shape != shape:
            raise ValueError(
                f"Expected initial_noise shape {shape}, got {initial_noise.shape}"
            )
        # Noise and the CTM update stay in float32, including with half weights.
        noise = (
            mx.random.normal(shape, key=key)
            if initial_noise is None
            else initial_noise.astype(mx.float32)
        )
        t = mx.full((shape[0],), 999.0)
        s = mx.zeros((shape[0],))
        velocity = self.diffusion_head(
            noise.astype(self.dtype), t, s, conditioning.astype(self.dtype)
        ).astype(mx.float32)
        c = self.config.ctm
        x0 = c.alpha_t * noise - c.sigma_t * velocity
        return c.c_skip * noise + c.c_out * x0

    def predict_latents(self, waveform, num_steps=1, initial_noise=None, key=None):
        """Mono 16 kHz audio -> normalized [1, frames, 2 * latent_dim] latents."""
        # Match the reference's exact-length path: at least 3.2 seconds, rounded
        # to 20 ms. Padding precedes utterance-level feature normalization.
        length = max(51200, ((waveform.shape[0] + 319) // 320) * 320)
        waveform = mx.pad(waveform, ((0, length - waveform.shape[0]),))
        features, _ = extract_features(normalize_chunk(waveform))
        features, first, second = self.encode(features)
        # Unlike DialogueSidon, these speaker heads already predict normalized
        # latents. Applying the decoder statistics here changes conditioning.
        conditioning = mx.concatenate((first, second, features), axis=-1)
        mx.eval(conditioning)
        return self.sample_latents(conditioning, num_steps, initial_noise, key)

    def decode_latents(self, latents):
        raw = latents.astype(mx.float32) * self.latent_norm_std + self.latent_norm_mean
        first, second = mx.split(raw, 2, axis=-1)
        dtype = self.decoder.generator[0].spec_proj.weight.dtype
        # Decode sequentially to bound activation memory on long windows.
        first = self.decoder(first.astype(dtype))
        mx.eval(first)
        second = self.decoder(second.astype(dtype))
        result = mx.stack((first, second), axis=1).astype(mx.float32)
        mx.eval(result)
        return result

    def sanitize(self, weights):
        if self.config.weight_format == "source":
            root = Path(self.config.model_path)
            weights = sanitize_source_weights(
                mx.load(str(root / "encoder/model.safetensors")),
                mx.load(str(root / "diffusion_head/model.safetensors")),
                mx.load(str(root / "vocoder/model.safetensors")),
                weights,
            )
        expected = (2 * self.config.latent_dim,)
        for name in ("latent_norm_mean", "latent_norm_std"):
            if name not in weights or weights[name].shape != expected:
                raise ValueError(f"Expected {name} with shape {expected}")
            weights[name] = weights[name].astype(mx.float32)
            if not mx.all(mx.isfinite(weights[name])).item():
                raise ValueError(f"{name} must be finite")
        if not mx.all(weights["latent_norm_std"] > 0).item():
            raise ValueError("latent_norm_std must be positive")
        return weights
