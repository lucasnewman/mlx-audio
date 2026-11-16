"""Diffusion transformer of Stable Audio 3."""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

IO_CHANNELS = 256
HEAD_DIM = 64
ROPE_DIMS = 32
COND_DIM = 768
LOCAL_ADD_COND_DIM = 257
NUM_MEMORY_TOKENS = 64
FOURIER_DIM = 256


@dataclass(frozen=True)
class DiTConfig:
    embed_dim: int = 1024
    depth: int = 20
    ff_mult: float = 4.0
    differential: bool = False


def expo_fourier_features(t: mx.array) -> mx.array:
    half = FOURIER_DIM // 2
    ramp = mx.arange(half, dtype=mx.float32) / (half - 1)
    log_min, log_max = math.log(0.5), math.log(10000.0)
    freqs = mx.exp(ramp * (log_max - log_min) + log_min)
    args = t[:, None].astype(mx.float32) * freqs * 2 * math.pi
    return mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)


def to_heads(x: mx.array) -> mx.array:
    batch, length, _ = x.shape
    return x.reshape(batch, length, -1, HEAD_DIM).transpose(0, 2, 1, 3)


def from_heads(x: mx.array) -> mx.array:
    batch, _, length, _ = x.shape
    return x.transpose(0, 2, 1, 3).reshape(batch, length, -1)


def rope(x: mx.array) -> mx.array:
    return mx.fast.rope(
        x, ROPE_DIMS, traditional=False, base=10000.0, scale=1.0, offset=0
    )


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.gamma = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.gamma, self.eps)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, differential: bool):
        super().__init__()
        self.parts = 5 if differential else 3
        self.to_qkv = nn.Linear(dim, self.parts * dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        self.q_norm = RMSNorm(HEAD_DIM, eps=1e-6)
        self.k_norm = RMSNorm(HEAD_DIM, eps=1e-6)

    def _attend(self, q: mx.array, k: mx.array, v: mx.array) -> mx.array:
        q = rope(self.q_norm(q))
        k = rope(self.k_norm(k))
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=HEAD_DIM**-0.5)

    def __call__(self, x: mx.array) -> mx.array:
        parts = [
            to_heads(part) for part in mx.split(self.to_qkv(x), self.parts, axis=-1)
        ]
        out = self._attend(parts[0], parts[1], parts[2])
        if self.parts == 5:
            out = out - self._attend(parts[3], parts[4], parts[2])
        return self.to_out(from_heads(out))


class CrossAttention(nn.Module):
    def __init__(self, dim: int, differential: bool):
        super().__init__()
        self.differential = differential
        self.to_q = nn.Linear(dim, (2 if differential else 1) * dim, bias=False)
        self.to_kv = nn.Linear(dim, (3 if differential else 2) * dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        self.q_norm = RMSNorm(HEAD_DIM, eps=1e-6)
        self.k_norm = RMSNorm(HEAD_DIM, eps=1e-6)

    def _attend(self, q: mx.array, k: mx.array, v: mx.array) -> mx.array:
        return mx.fast.scaled_dot_product_attention(
            self.q_norm(q), self.k_norm(k), v, scale=HEAD_DIM**-0.5
        )

    def __call__(self, x: mx.array, context: mx.array) -> mx.array:
        if self.differential:
            q, q_diff = (to_heads(part) for part in mx.split(self.to_q(x), 2, axis=-1))
            k, k_diff, v = (
                to_heads(part) for part in mx.split(self.to_kv(context), 3, axis=-1)
            )
            out = self._attend(q, k, v) - self._attend(q_diff, k_diff, v)
        else:
            k, v = (
                to_heads(part) for part in mx.split(self.to_kv(context), 2, axis=-1)
            )
            out = self._attend(to_heads(self.to_q(x)), k, v)
        return self.to_out(from_heads(out))


class GatedProjection(nn.Module):
    def __init__(self, dim: int, inner: int, sinusoidal: bool = False):
        super().__init__()
        self.proj = nn.Linear(dim, 2 * inner)
        self.sinusoidal = sinusoidal

    def __call__(self, x: mx.array) -> mx.array:
        value, gate = mx.split(self.proj(x), 2, axis=-1)
        return value * (mx.sin(gate * math.pi) if self.sinusoidal else nn.silu(gate))


class FeedForward(nn.Module):
    def __init__(self, dim: int, inner: int, sinusoidal: bool = False):
        super().__init__()
        self.ff = [GatedProjection(dim, inner, sinusoidal), None, nn.Linear(inner, dim)]

    def __call__(self, x: mx.array) -> mx.array:
        return self.ff[2](self.ff[0](x))


def _mlp(layers: list[nn.Linear], x: mx.array) -> mx.array:
    return layers[2](nn.silu(layers[0](x)))


class TransformerBlock(nn.Module):
    def __init__(self, config: DiTConfig):
        super().__init__()
        dim = config.embed_dim
        self.pre_norm = RMSNorm(dim, eps=1e-5)
        self.self_attn = SelfAttention(dim, config.differential)
        self.cross_attend_norm = RMSNorm(dim, eps=1e-5)
        self.cross_attn = CrossAttention(dim, config.differential)
        self.ff_norm = RMSNorm(dim, eps=1e-5)
        self.ff = FeedForward(dim, int(dim * config.ff_mult))
        self.to_scale_shift_gate = mx.zeros((6 * dim,))
        self.to_local_embed = [
            nn.Linear(LOCAL_ADD_COND_DIM, dim),
            None,
            nn.Linear(dim, dim),
        ]

    def __call__(
        self, x: mx.array, context: mx.array, adaln: mx.array, local: mx.array
    ) -> mx.array:
        scale_self, shift_self, gate_self, scale_ff, shift_ff, gate_ff = mx.split(
            (self.to_scale_shift_gate + adaln)[:, None, :], 6, axis=-1
        )
        h = self.pre_norm(x) * (1 + scale_self) + shift_self
        x = x + self.self_attn(h) * mx.sigmoid(1 - gate_self)
        x = x + self.cross_attn(self.cross_attend_norm(x), context)
        pad = mx.zeros((x.shape[0], NUM_MEMORY_TOKENS, x.shape[-1]), dtype=x.dtype)
        x = x + mx.concatenate([pad, _mlp(self.to_local_embed, local)], axis=1)
        h = self.ff_norm(x) * (1 + scale_ff) + shift_ff
        return x + self.ff(h) * mx.sigmoid(1 - gate_ff)


class ContinuousTransformer(nn.Module):
    def __init__(self, config: DiTConfig):
        super().__init__()
        dim = config.embed_dim
        self.project_in = nn.Linear(IO_CHANNELS, dim, bias=False)
        self.project_out = nn.Linear(dim, IO_CHANNELS, bias=False)
        self.memory_tokens = mx.zeros((NUM_MEMORY_TOKENS, dim))
        self.global_cond_embedder = [
            nn.Linear(dim, dim),
            None,
            nn.Linear(dim, 6 * dim),
        ]
        self.layers = [TransformerBlock(config) for _ in range(config.depth)]

    def __call__(
        self, x: mx.array, context: mx.array, global_embed: mx.array
    ) -> mx.array:
        batch, length, _ = x.shape
        memory = mx.broadcast_to(
            self.memory_tokens[None], (batch, *self.memory_tokens.shape)
        )
        x = mx.concatenate([memory, self.project_in(x)], axis=1)
        adaln = _mlp(self.global_cond_embedder, global_embed)
        local = mx.zeros((batch, length, LOCAL_ADD_COND_DIM), dtype=x.dtype)
        for layer in self.layers:
            x = layer(x, context, adaln, local)
        return self.project_out(x[:, NUM_MEMORY_TOKENS:, :])


class DiT(nn.Module):
    def __init__(self, config: DiTConfig = DiTConfig()):
        super().__init__()
        dim = config.embed_dim
        self.preprocess_conv = nn.Conv1d(IO_CHANNELS, IO_CHANNELS, 1, bias=False)
        self.postprocess_conv = nn.Conv1d(IO_CHANNELS, IO_CHANNELS, 1, bias=False)
        self.to_cond_embed = [
            nn.Linear(COND_DIM, dim, bias=False),
            None,
            nn.Linear(dim, dim, bias=False),
        ]
        self.to_global_embed = [
            nn.Linear(COND_DIM, dim, bias=False),
            None,
            nn.Linear(dim, dim, bias=False),
        ]
        self.to_timestep_embed = [
            nn.Linear(FOURIER_DIM, dim),
            None,
            nn.Linear(dim, dim),
        ]
        self.transformer = ContinuousTransformer(config)

    def __call__(
        self, x: mx.array, t: mx.array, cross_attn: mx.array, global_cond: mx.array
    ) -> mx.array:
        context = _mlp(self.to_cond_embed, cross_attn)
        timestep = _mlp(self.to_timestep_embed, expo_fourier_features(t))
        global_embed = _mlp(self.to_global_embed, global_cond) + timestep
        h = self.transformer(self.preprocess_conv(x) + x, context, global_embed)
        return self.postprocess_conv(h) + h
