"""Offline, two-speaker DialogueSidon separation on Apple Silicon."""

import mlx.core as mx
from mlx import nn

from mlx_audio.codec.models.descript.dac import Decoder

from .config import ModelConfig
from .diffusion import DiffusionHead, DPMSolver
from .encoder import Encoder
from .frontend import extract_features, normalize_chunk
from .separation import SeparationModel, SeparationResult, align_speakers  # noqa: F401


class Model(SeparationModel):
    output_suffix = "_separated"

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config.encoder)
        self.linear1 = nn.Linear(config.encoder.hidden_size, config.latent_dim)
        self.linear2 = nn.Linear(config.encoder.hidden_size, config.latent_dim)
        self.diffusion_head = DiffusionHead(
            config.diffusion,
            config.latent_dim * 2,
            config.encoder.hidden_size + config.latent_dim * 2,
        )
        self.decoder = Decoder(
            input_channel=config.latent_dim,
            channels=config.decoder_channels,
            rates=config.decoder_rates,
            d_out=1,
        )
        # Config-owned constants, not learned model parameters.
        self._latent_mean = mx.array(
            config.latent_norm_mean or [0.0] * (config.latent_dim * 2)
        )
        self._latent_std = mx.array(
            config.latent_norm_std or [1.0] * (config.latent_dim * 2)
        )

    def _validate_num_steps(self, num_steps):
        DPMSolver(self.config.diffusion, num_steps)

    @property
    def dtype(self):
        return self.linear1.weight.dtype

    def normalize_latents(self, latents):
        if not self.config.latent_norm_initialized:
            return latents
        return (
            (latents.astype(mx.float32) - self._latent_mean) / self._latent_std
        ).astype(latents.dtype)

    def denormalize_latents(self, latents):
        if not self.config.latent_norm_initialized:
            return latents
        return (
            latents.astype(mx.float32) * self._latent_std + self._latent_mean
        ).astype(latents.dtype)

    def encode(self, input_features, attention_mask=None):
        features = self.encoder(input_features.astype(self.dtype), attention_mask)
        return features, self.linear1(features), self.linear2(features)

    def sample_latents(self, conditioning, num_steps=30, initial_noise=None, key=None):
        shape = (*conditioning.shape[:2], self.config.latent_dim * 2)
        if initial_noise is not None and initial_noise.shape != shape:
            raise ValueError(
                f"Expected initial_noise shape {shape}, got {initial_noise.shape}"
            )
        latents = (
            mx.random.normal(shape, dtype=self.dtype, key=key)
            if initial_noise is None
            else initial_noise.astype(self.dtype)
        )
        scheduler = DPMSolver(self.config.diffusion, num_steps)
        for t in scheduler.timesteps:
            timesteps = mx.full((shape[0],), int(t), dtype=mx.float32)
            prediction = self.diffusion_head(latents, timesteps, conditioning)
            latents = scheduler.step(prediction, latents)
            mx.eval(latents, scheduler.previous)
        return latents

    def predict_latents(self, waveform, num_steps=30, initial_noise=None, key=None):
        """One mono 16 kHz chunk -> normalized [1, frames, 2 * latent_dim]."""
        features, mask = extract_features(normalize_chunk(waveform))
        features, first, second = self.encode(features, mask)
        predicted = mx.concatenate((first, second), axis=-1)
        conditioning = mx.concatenate(
            (self.normalize_latents(predicted), features), axis=-1
        )
        mx.eval(conditioning)
        return self.sample_latents(conditioning, num_steps, initial_noise, key)

    def decode_latents(self, latents):
        """Normalized [batch, frames, 2 * latent_dim] -> [batch, 2, samples]."""
        latents = self.denormalize_latents(latents).astype(self.dtype)
        first, second = mx.split(latents, 2, axis=-1)
        # Sequential decoding limits decoder activation memory for long chunks.
        first = self.decoder(first)[:, :, 0]
        mx.eval(first)
        second = self.decoder(second)[:, :, 0]
        output = mx.stack((first, second), axis=1)
        mx.eval(output)
        return output
