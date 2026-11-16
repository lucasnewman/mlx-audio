"""SAME decoders of Stable Audio 3: SAME-S for small, SAME-L for medium."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .dit import HEAD_DIM, FeedForward, from_heads, rope, to_heads

LATENT_DIM = 256
PATCH_CHANNELS = 512
PATCH_SIZE = 256
STRIDE = 16
SUB_CHUNK = STRIDE + 1
EFFECTIVE_CHUNK = 34
SHIFT = EFFECTIVE_CHUNK // 2
CHUNK = 8
OVERLAP = 2
SLIDING_WINDOW_CHUNK = 128
SLIDING_WINDOW_OVERLAP = 8
WINDOW = 3 * SUB_CHUNK
FF_MULT = 3


@dataclass(frozen=True)
class SAMEConfig:
    dim: int = 768
    depth: int = 6
    sinusoidal_blocks: int = 0
    conv_mapping: bool = True
    sliding_window: bool = False


class DynamicTanh(nn.Module):
    def __init__(self, dims: int):
        super().__init__()
        self.alpha = mx.ones((1,))
        self.gamma = mx.ones((dims,))
        self.beta = mx.zeros((dims,))

    def __call__(self, x: mx.array) -> mx.array:
        return self.gamma * mx.tanh(self.alpha * x) + self.beta


def _window_mask(dtype: mx.Dtype) -> mx.array:
    q = mx.arange(SUB_CHUNK)[:, None]
    kv = mx.arange(WINDOW)[None, :]
    valid = (kv >= q) & (kv <= q + 2 * SUB_CHUNK)
    return mx.where(valid, mx.array(0.0, dtype=dtype), mx.array(-1e9, dtype=dtype))


class DifferentialAttention(nn.Module):
    def __init__(self, dim: int, sliding_window: bool = False):
        super().__init__()
        self.sliding_window = sliding_window
        self.to_qkv = nn.Linear(dim, 5 * dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        self.q_norm = DynamicTanh(HEAD_DIM)
        self.k_norm = DynamicTanh(HEAD_DIM)

    def _attend(self, q: mx.array, k: mx.array, v: mx.array) -> mx.array:
        q = rope(self.q_norm(q))
        k = rope(self.k_norm(k))
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=HEAD_DIM**-0.5)

    def _attend_windowed(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        q_diff: mx.array,
        k_diff: mx.array,
    ) -> mx.array:
        q, k = rope(self.q_norm(q)), rope(self.k_norm(k))
        q_diff, k_diff = rope(self.q_norm(q_diff)), rope(self.k_norm(k_diff))
        batch, heads, length, _ = q.shape
        groups = length // SUB_CHUNK
        padding = [(0, 0), (0, 0), (SUB_CHUNK, SUB_CHUNK), (0, 0)]
        padded_length = length + 2 * SUB_CHUNK
        shape = [batch, heads, groups, WINDOW, HEAD_DIM]
        strides = [
            heads * padded_length * HEAD_DIM,
            padded_length * HEAD_DIM,
            SUB_CHUNK * HEAD_DIM,
            HEAD_DIM,
            1,
        ]
        windows = [
            mx.as_strided(mx.pad(part, padding), shape=shape, strides=strides)
            .transpose(0, 2, 1, 3, 4)
            .reshape(batch * groups, heads, WINDOW, HEAD_DIM)
            for part in (k, k_diff, v)
        ]
        positions = mx.arange(groups)[:, None] * SUB_CHUNK + mx.arange(WINDOW)[None]
        boundary = mx.where(
            (positions >= SUB_CHUNK) & (positions < length + SUB_CHUNK),
            mx.array(0.0, dtype=q.dtype),
            mx.array(-1e9, dtype=q.dtype),
        )[:, None, :]
        mask = mx.broadcast_to(
            (_window_mask(q.dtype) + boundary)[None], (batch, groups, SUB_CHUNK, WINDOW)
        ).reshape(batch * groups, 1, SUB_CHUNK, WINDOW)
        grouped = [
            part.reshape(batch, heads, groups, SUB_CHUNK, HEAD_DIM)
            .transpose(0, 2, 1, 3, 4)
            .reshape(batch * groups, heads, SUB_CHUNK, HEAD_DIM)
            for part in (q, q_diff)
        ]
        out = mx.fast.scaled_dot_product_attention(
            mx.concatenate(grouped, axis=1),
            mx.concatenate([windows[0], windows[1]], axis=1),
            mx.concatenate([windows[2], windows[2]], axis=1),
            scale=HEAD_DIM**-0.5,
            mask=mask,
        )
        main, diff = mx.split(out, 2, axis=1)
        return (
            (main - diff)
            .reshape(batch, groups, heads, SUB_CHUNK, HEAD_DIM)
            .transpose(0, 2, 1, 3, 4)
            .reshape(batch, heads, length, HEAD_DIM)
        )

    def __call__(self, x: mx.array) -> mx.array:
        q, k, v, q_diff, k_diff = (
            to_heads(part) for part in mx.split(self.to_qkv(x), 5, axis=-1)
        )
        if self.sliding_window and x.shape[1] > SUB_CHUNK:
            out = self._attend_windowed(q, k, v, q_diff, k_diff)
        else:
            out = self._attend(q, k, v) - self._attend(q_diff, k_diff, v)
        return self.to_out(from_heads(out))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, sinusoidal: bool, sliding_window: bool):
        super().__init__()
        self.pre_norm = DynamicTanh(dim)
        self.self_attn = DifferentialAttention(dim, sliding_window)
        self.ff_norm = DynamicTanh(dim)
        self.ff = FeedForward(dim, FF_MULT * dim, sinusoidal)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.self_attn(self.pre_norm(x))
        return x + self.ff(self.ff_norm(x))


class SAMEDecoder(nn.Module):
    def __init__(self, config: SAMEConfig = SAMEConfig()):
        super().__init__()
        self.config = config
        dim = config.dim
        self.running_std = mx.ones((1,))
        self.project_in = nn.Linear(LATENT_DIM, dim)
        self.new_tokens = mx.zeros((1, 1, dim))
        self.transformers = [
            TransformerBlock(
                dim,
                config.depth - index < config.sinusoidal_blocks,
                config.sliding_window,
            )
            for index in range(config.depth)
        ]
        kernel = 3 if config.conv_mapping else 1
        self.mapping = nn.Conv1d(dim, PATCH_CHANNELS, kernel, padding=kernel // 2)

    def _run_chunks(
        self, x: mx.array, blocks: list[TransformerBlock], chunks: int
    ) -> mx.array:
        batch, _, dim = x.shape
        x = x.reshape(batch * chunks, EFFECTIVE_CHUNK, dim)
        for block in blocks:
            x = block(x)
        return x.reshape(batch, chunks * EFFECTIVE_CHUNK, dim)

    def _run_shifted_chunks(self, x: mx.array) -> mx.array:
        split = len(self.transformers) // 2
        chunks = x.shape[1] // EFFECTIVE_CHUNK
        x = self._run_chunks(x, self.transformers[:split], chunks)
        x = mx.concatenate([x[:, :SHIFT, :], x, x[:, -SHIFT:, :]], axis=1)
        x = self._run_chunks(x, self.transformers[split:], chunks + 1)
        return x[:, SHIFT:-SHIFT, :]

    def __call__(self, latents: mx.array) -> mx.array:
        batch, length, _ = latents.shape
        dim = self.config.dim
        x = self.project_in(latents * self.running_std)
        tokens = mx.broadcast_to(self.new_tokens[None], (batch, length, STRIDE, dim))
        x = mx.concatenate([x[:, :, None, :], tokens], axis=2)
        x = x.reshape(batch, length * SUB_CHUNK, dim)

        if self.config.sliding_window:
            for block in self.transformers:
                x = block(x)
        else:
            x = self._run_shifted_chunks(x)

        x = x.reshape(batch * length, SUB_CHUNK, dim)[:, 1:, :]
        return self.mapping(x.reshape(batch, length * STRIDE, dim))

    def _decode_windows(self, latents: mx.array, chunk: int, overlap: int) -> mx.array:
        total = latents.shape[1]
        kernel = chunk + 2 * overlap
        first = chunk + overlap
        pieces = [self(latents[:, :kernel])[:, : first * STRIDE]]
        position = first
        while position + chunk + overlap <= total:
            window = latents[:, position - overlap : position + chunk + overlap]
            pieces.append(
                self(window)[:, overlap * STRIDE : (overlap + chunk) * STRIDE]
            )
            mx.eval(pieces[-1])
            position += chunk
        if position < total:
            last = self(latents[:, total - kernel :])
            pieces.append(last[:, -(total - position) * STRIDE :])
        return mx.concatenate(pieces, axis=1)

    def decode(self, latents: mx.array) -> mx.array:
        total = latents.shape[1]
        if self.config.sliding_window:
            chunk, overlap = SLIDING_WINDOW_CHUNK, SLIDING_WINDOW_OVERLAP
        else:
            chunk, overlap = CHUNK, OVERLAP
        if total > chunk + 2 * overlap:
            patches = self._decode_windows(latents, chunk, overlap)
        elif self.config.sliding_window or total % 2 == 0:
            patches = self(latents)
        elif total > 6:
            patches = self._decode_windows(latents, 2, 2)
        else:
            even = mx.concatenate([latents, latents[:, -1:]], axis=1)
            patches = self(even)[:, : total * STRIDE]
        batch, length, _ = patches.shape
        audio = patches.reshape(batch, length, 2, PATCH_SIZE).transpose(0, 1, 3, 2)
        return audio.reshape(batch, length * PATCH_SIZE, 2)
