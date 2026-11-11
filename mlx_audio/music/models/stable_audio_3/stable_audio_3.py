"""Native MLX Stable Audio 3 small pipeline."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import sentencepiece as spm

from mlx_audio.tts.models.base import BaseModelArgs, GenerationResult

from . import t5gemma
from .dit import COND_DIM, FOURIER_DIM, IO_CHANNELS, DiT, expo_fourier_features
from .same import SAMEDecoder

TEXT_ENCODER_DIR = "t5gemma-b-b-ul2"
TEXT_MAX_LENGTH = 256
TEXT_VOCAB_SIZE = 256000
SECONDS_MAX = 384.0
SAMPLES_PER_LATENT = 4096
LOGSNR_START = -6.2
LOGSNR_END = 2.0

_KEY_PREFIXES = (
    ("model.model.", "dit."),
    ("conditioner.conditioners.prompt.padding_embedding", "padding_embedding"),
    ("conditioner.conditioners.seconds_total.embedder.embedding.1.", "seconds_proj."),
    ("pretransform.model.bottleneck.running_std", "decoder.running_std"),
    ("pretransform.model.decoder.layers.1.", "decoder.project_in."),
    ("pretransform.model.decoder.layers.3.", "decoder."),
)
_UNUSED_KEYS = (
    "pretransform.model.encoder.",
    "pretransform.model.bottleneck.bias",
    "pretransform.model.bottleneck.scaling_factor",
)


@dataclass
class ModelConfig(BaseModelArgs):
    sample_rate: int = 44100
    sample_size: int = 5292032


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(TEXT_VOCAB_SIZE, COND_DIM)
        self.encoder = t5gemma.Encoder(
            COND_DIM,
            2048,
            12,
            12,
            64,
            num_hidden_layers=12,
            query_pre_attn_scalar=64.0,
        )

    def __call__(self, input_ids: mx.array, attention_mask: mx.array) -> mx.array:
        return self.encoder(self.embed_tokens(input_ids), attention_mask)


def sampling_schedule(steps: int) -> mx.array:
    t = mx.linspace(1.0, 0.0, steps + 1, dtype=mx.float32)
    shifted = mx.sigmoid(-(LOGSNR_END - t * (LOGSNR_END - LOGSNR_START)))
    shifted = mx.where(t <= 0, mx.zeros_like(shifted), shifted)
    return mx.where(t >= 1, mx.ones_like(shifted), shifted)


def guided_velocity(
    x: mx.array,
    sigma: mx.array,
    cond_v: mx.array,
    uncond_v: mx.array,
    cfg_scale: float,
) -> mx.array:
    cond = x - sigma * cond_v
    diff = sigma * (uncond_v - cond_v)
    norm = mx.sqrt((cond * cond).sum(axis=(-2, -1), keepdims=True))
    unit = cond / mx.maximum(norm, 1e-8)
    orthogonal = diff - (diff * unit).sum(axis=(-2, -1), keepdims=True) * unit
    return (x - (cond + (cfg_scale - 1.0) * orthogonal)) / sigma


def sample_pingpong(model_fn, x: mx.array, sigmas: mx.array, key: mx.array) -> mx.array:
    last = sigmas.shape[0] - 2
    for step in range(last + 1):
        sigma, next_sigma = sigmas[step], sigmas[step + 1]
        denoised = x - sigma * model_fn(x, sigma[None])
        if step == last:
            x = denoised
        else:
            key, noise_key = mx.random.split(key)
            noise = mx.random.normal(x.shape, dtype=x.dtype, key=noise_key)
            x = (1.0 - next_sigma) * denoised + next_sigma * noise
        mx.eval(x)
    return x


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.dit = DiT()
        self.decoder = SAMEDecoder()
        self.padding_embedding = mx.zeros((COND_DIM,))
        self.seconds_proj = nn.Linear(FOURIER_DIM, COND_DIM)

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    @property
    def max_duration(self) -> int:
        return self.config.sample_size // self.config.sample_rate

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        sanitized = {}
        for key, value in weights.items():
            if key.endswith("inv_freq") or key.startswith(_UNUSED_KEYS):
                continue
            for old, new in _KEY_PREFIXES:
                if key.startswith(old):
                    key = new + key[len(old) :]
                    break
            sanitized[key] = value
        return sanitized

    @staticmethod
    def post_load_hook(model: "Model", model_path: Path) -> "Model":
        directory = Path(model_path) / TEXT_ENCODER_DIR
        weights = []
        for key, value in mx.load(str(directory / "model.safetensors")).items():
            if key.startswith("model.encoder.embed_tokens"):
                weights.append((key[len("model.encoder.") :], value))
            elif key.startswith("model.encoder."):
                weights.append((key[len("model.") :], value))
        encoder = TextEncoder()
        encoder.load_weights(weights, strict=True)
        mx.eval(encoder.parameters())
        encoder.eval()
        model.text_encoder = encoder
        model.tokenizer = spm.SentencePieceProcessor(
            model_file=str(directory / "tokenizer.model")
        )
        return model

    def _encode_text(self, text: str) -> mx.array:
        tokens = self.tokenizer.encode(text)[:TEXT_MAX_LENGTH]
        ids = np.zeros((1, TEXT_MAX_LENGTH), dtype=np.int32)
        ids[0, : len(tokens)] = tokens
        mask = mx.arange(TEXT_MAX_LENGTH)[None] < len(tokens)
        hidden = self.text_encoder(mx.array(ids), mask).astype(mx.float32)
        keep = mask[..., None].astype(mx.float32)
        return hidden * keep + self.padding_embedding[None, None] * (1 - keep)

    def _seconds_embedding(self, seconds: float) -> mx.array:
        # The seconds conditioner was only trained on whole seconds.
        whole_seconds = math.ceil(seconds)
        return self.seconds_proj(
            expo_fourier_features(mx.array([whole_seconds / SECONDS_MAX]))
        )

    def _sample(
        self,
        text: str,
        seconds: float,
        steps: int,
        seed: int,
        cfg_scale: float,
        negative_prompt: Optional[str],
    ) -> mx.array:
        length = max(1, math.ceil(seconds * self.sample_rate / SAMPLES_PER_LATENT))
        seconds_embed = self._seconds_embedding(seconds)
        cross_attn = mx.concatenate(
            [self._encode_text(text), seconds_embed[:, None]], axis=1
        )
        if cfg_scale != 1.0:
            uncond = (
                mx.concatenate(
                    [self._encode_text(negative_prompt), seconds_embed[:, None]],
                    axis=1,
                )
                if negative_prompt and negative_prompt.strip()
                else mx.zeros_like(cross_attn)
            )
            cross_attn = mx.concatenate([cross_attn, uncond], axis=0)
            global_cond = mx.concatenate([seconds_embed, seconds_embed], axis=0)
        else:
            global_cond = seconds_embed

        def model_fn(x: mx.array, t: mx.array) -> mx.array:
            if cfg_scale == 1.0:
                return self.dit(x, t, cross_attn, global_cond)
            doubled = mx.concatenate([x, x], axis=0)
            velocity = self.dit(
                doubled, mx.concatenate([t, t], axis=0), cross_attn, global_cond
            )
            cond_v, uncond_v = mx.split(velocity, 2, axis=0)
            return guided_velocity(x, t[:, None, None], cond_v, uncond_v, cfg_scale)

        key, noise_key = mx.random.split(mx.random.key(seed))
        noise = mx.random.normal((1, length, IO_CHANNELS), key=noise_key)
        return sample_pingpong(model_fn, noise, sampling_schedule(steps), key)

    def generate(
        self,
        text: str,
        *,
        duration: float = 30.0,
        steps: int = 8,
        seed: int = 0,
        cfg_scale: float = 1.0,
        negative_prompt: Optional[str] = None,
        lyrics: Optional[str] = None,
    ) -> Iterator[GenerationResult]:
        if lyrics is not None:
            raise ValueError("Stable Audio 3 does not take lyrics")
        if not text.strip():
            raise ValueError("A text prompt is required")
        if duration <= 0 or duration > self.max_duration:
            raise ValueError(
                f"duration must be between 0 and {self.max_duration:g} seconds"
            )
        if steps < 1:
            raise ValueError("steps must be at least 1")

        started = time.perf_counter()
        latents = self._sample(text, duration, steps, seed, cfg_scale, negative_prompt)
        audio = self.decoder.decode(latents)[0, : round(duration * self.sample_rate)]
        waveform = mx.clip(audio, -1.0, 1.0)
        mx.eval(waveform)
        if not np.isfinite(np.asarray(waveform)).all():
            raise RuntimeError("Stable Audio 3 produced non-finite audio")

        elapsed = time.perf_counter() - started
        samples = int(waveform.shape[0])
        audio_seconds = samples / self.sample_rate
        milliseconds = int((audio_seconds % 1) * 1000)
        yield GenerationResult(
            audio=waveform,
            samples=samples,
            sample_rate=self.sample_rate,
            segment_idx=0,
            token_count=int(latents.shape[1]),
            audio_duration=(
                f"{int(audio_seconds // 3600):02d}:"
                f"{int((audio_seconds % 3600) // 60):02d}:"
                f"{int(audio_seconds % 60):02d}.{milliseconds:03d}"
            ),
            real_time_factor=elapsed / max(audio_seconds, 1e-9),
            prompt={"text": text, "seed": seed, "steps": steps},
            audio_samples={
                "samples": samples,
                "samples-per-sec": samples / max(elapsed, 1e-9),
                "requested_seconds": duration,
            },
            processing_time_seconds=elapsed,
            peak_memory_usage=mx.get_peak_memory() / 1e9,
        )


__all__ = ["Model", "ModelConfig"]
