"""Diffusion transformer of Stable Audio 3 small."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

IO_CHANNELS = 256
EMBED_DIM = 1024
DEPTH = 20
NUM_HEADS = 16
HEAD_DIM = 64
ROPE_DIMS = 32
FF_INNER = 4096
COND_DIM = 768
LOCAL_ADD_COND_DIM = 257
NUM_MEMORY_TOKENS = 64
FOURIER_DIM = 256


def expo_fourier_features(t: mx.array) -> mx.array:
    half = FOURIER_DIM // 2
    ramp = mx.arange(half, dtype=mx.float32) / (half - 1)
    log_min, log_max = math.log(0.5), math.log(10000.0)
    freqs = mx.exp(ramp * (log_max - log_min) + log_min)
    args = t[:, None].astype(mx.float32) * freqs * 2 * math.pi
    return mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)


def _to_heads(x: mx.array) -> mx.array:
    batch, length, _ = x.shape
    return x.reshape(batch, length, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)


def _from_heads(x: mx.array) -> mx.array:
    batch, _, length, _ = x.shape
    return x.transpose(0, 2, 1, 3).reshape(batch, length, EMBED_DIM)


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
    def __init__(self):
        super().__init__()
        self.to_qkv = nn.Linear(EMBED_DIM, 3 * EMBED_DIM, bias=False)
        self.to_out = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False)
        self.q_norm = RMSNorm(HEAD_DIM, eps=1e-6)
        self.k_norm = RMSNorm(HEAD_DIM, eps=1e-6)

    def __call__(self, x: mx.array) -> mx.array:
        q, k, v = (_to_heads(part) for part in mx.split(self.to_qkv(x), 3, axis=-1))
        q = rope(self.q_norm(q))
        k = rope(self.k_norm(k))
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=HEAD_DIM**-0.5)
        return self.to_out(_from_heads(out))


class CrossAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False)
        self.to_kv = nn.Linear(EMBED_DIM, 2 * EMBED_DIM, bias=False)
        self.to_out = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False)
        self.q_norm = RMSNorm(HEAD_DIM, eps=1e-6)
        self.k_norm = RMSNorm(HEAD_DIM, eps=1e-6)

    def __call__(self, x: mx.array, context: mx.array) -> mx.array:
        k, v = (_to_heads(part) for part in mx.split(self.to_kv(context), 2, axis=-1))
        q = self.q_norm(_to_heads(self.to_q(x)))
        out = mx.fast.scaled_dot_product_attention(
            q, self.k_norm(k), v, scale=HEAD_DIM**-0.5
        )
        return self.to_out(_from_heads(out))


class GatedProjection(nn.Module):
    def __init__(self, dim: int, inner: int):
        super().__init__()
        self.proj = nn.Linear(dim, 2 * inner)

    def __call__(self, x: mx.array) -> mx.array:
        value, gate = mx.split(self.proj(x), 2, axis=-1)
        return value * nn.silu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim: int, inner: int):
        super().__init__()
        self.ff = [GatedProjection(dim, inner), None, nn.Linear(inner, dim)]

    def __call__(self, x: mx.array) -> mx.array:
        return self.ff[2](self.ff[0](x))


def _mlp(layers: list[nn.Linear], x: mx.array) -> mx.array:
    return layers[2](nn.silu(layers[0](x)))


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_norm = RMSNorm(EMBED_DIM, eps=1e-5)
        self.self_attn = SelfAttention()
        self.cross_attend_norm = RMSNorm(EMBED_DIM, eps=1e-5)
        self.cross_attn = CrossAttention()
        self.ff_norm = RMSNorm(EMBED_DIM, eps=1e-5)
        self.ff = FeedForward(EMBED_DIM, FF_INNER)
        self.to_scale_shift_gate = mx.zeros((6 * EMBED_DIM,))
        self.to_local_embed = [
            nn.Linear(LOCAL_ADD_COND_DIM, EMBED_DIM),
            None,
            nn.Linear(EMBED_DIM, EMBED_DIM),
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
        pad = mx.zeros((x.shape[0], NUM_MEMORY_TOKENS, EMBED_DIM), dtype=x.dtype)
        x = x + mx.concatenate([pad, _mlp(self.to_local_embed, local)], axis=1)
        h = self.ff_norm(x) * (1 + scale_ff) + shift_ff
        return x + self.ff(h) * mx.sigmoid(1 - gate_ff)


class ContinuousTransformer(nn.Module):
    def __init__(self, depth: int):
        super().__init__()
        self.project_in = nn.Linear(IO_CHANNELS, EMBED_DIM, bias=False)
        self.project_out = nn.Linear(EMBED_DIM, IO_CHANNELS, bias=False)
        self.memory_tokens = mx.zeros((NUM_MEMORY_TOKENS, EMBED_DIM))
        self.global_cond_embedder = [
            nn.Linear(EMBED_DIM, EMBED_DIM),
            None,
            nn.Linear(EMBED_DIM, 6 * EMBED_DIM),
        ]
        self.layers = [TransformerBlock() for _ in range(depth)]

    def __call__(
        self, x: mx.array, context: mx.array, global_embed: mx.array
    ) -> mx.array:
        batch, length, _ = x.shape
        memory = mx.broadcast_to(
            self.memory_tokens[None], (batch, NUM_MEMORY_TOKENS, EMBED_DIM)
        )
        x = mx.concatenate([memory, self.project_in(x)], axis=1)
        adaln = _mlp(self.global_cond_embedder, global_embed)
        local = mx.zeros((batch, length, LOCAL_ADD_COND_DIM), dtype=x.dtype)
        for layer in self.layers:
            x = layer(x, context, adaln, local)
        return self.project_out(x[:, NUM_MEMORY_TOKENS:, :])


class DiT(nn.Module):
    def __init__(self, depth: int = DEPTH):
        super().__init__()
        self.preprocess_conv = nn.Conv1d(IO_CHANNELS, IO_CHANNELS, 1, bias=False)
        self.postprocess_conv = nn.Conv1d(IO_CHANNELS, IO_CHANNELS, 1, bias=False)
        self.to_cond_embed = [
            nn.Linear(COND_DIM, EMBED_DIM, bias=False),
            None,
            nn.Linear(EMBED_DIM, EMBED_DIM, bias=False),
        ]
        self.to_global_embed = [
            nn.Linear(COND_DIM, EMBED_DIM, bias=False),
            None,
            nn.Linear(EMBED_DIM, EMBED_DIM, bias=False),
        ]
        self.to_timestep_embed = [
            nn.Linear(FOURIER_DIM, EMBED_DIM),
            None,
            nn.Linear(EMBED_DIM, EMBED_DIM),
        ]
        self.transformer = ContinuousTransformer(depth)

    def __call__(
        self, x: mx.array, t: mx.array, cross_attn: mx.array, global_cond: mx.array
    ) -> mx.array:
        context = _mlp(self.to_cond_embed, cross_attn)
        timestep = _mlp(self.to_timestep_embed, expo_fourier_features(t))
        global_embed = _mlp(self.to_global_embed, global_cond) + timestep
        h = self.transformer(self.preprocess_conv(x) + x, context, global_embed)
        return self.postprocess_conv(h) + h
