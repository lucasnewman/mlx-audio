"""SAME-S decoder of Stable Audio 3 small."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .dit import HEAD_DIM, FeedForward, rope

LATENT_DIM = 256
DIM = 768
DEPTH = 6
NUM_HEADS = 12
FF_INNER = 2304
PATCH_CHANNELS = 512
PATCH_SIZE = 256
STRIDE = 16
SUB_CHUNK = STRIDE + 1
EFFECTIVE_CHUNK = 34
SHIFT = EFFECTIVE_CHUNK // 2
CHUNK = 8
OVERLAP = 2


class DynamicTanh(nn.Module):
    def __init__(self, dims: int):
        super().__init__()
        self.alpha = mx.ones((1,))
        self.gamma = mx.ones((dims,))
        self.beta = mx.zeros((dims,))

    def __call__(self, x: mx.array) -> mx.array:
        return self.gamma * mx.tanh(self.alpha * x) + self.beta


class DifferentialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_qkv = nn.Linear(DIM, 5 * DIM, bias=False)
        self.to_out = nn.Linear(DIM, DIM, bias=False)
        self.q_norm = DynamicTanh(HEAD_DIM)
        self.k_norm = DynamicTanh(HEAD_DIM)

    def _attend(self, q: mx.array, k: mx.array, v: mx.array) -> mx.array:
        q = rope(self.q_norm(q))
        k = rope(self.k_norm(k))
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=HEAD_DIM**-0.5)

    def __call__(self, x: mx.array) -> mx.array:
        batch, length, _ = x.shape
        q, k, v, q_diff, k_diff = (
            part.reshape(batch, length, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
            for part in mx.split(self.to_qkv(x), 5, axis=-1)
        )
        out = self._attend(q, k, v) - self._attend(q_diff, k_diff, v)
        return self.to_out(out.transpose(0, 2, 1, 3).reshape(batch, length, DIM))


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_norm = DynamicTanh(DIM)
        self.self_attn = DifferentialAttention()
        self.ff_norm = DynamicTanh(DIM)
        self.ff = FeedForward(DIM, FF_INNER)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.self_attn(self.pre_norm(x))
        return x + self.ff(self.ff_norm(x))


class SAMEDecoder(nn.Module):
    def __init__(self, depth: int = DEPTH):
        super().__init__()
        self.running_std = mx.ones((1,))
        self.project_in = nn.Linear(LATENT_DIM, DIM)
        self.new_tokens = mx.zeros((1, 1, DIM))
        self.transformers = [TransformerBlock() for _ in range(depth)]
        self.mapping = nn.Conv1d(DIM, PATCH_CHANNELS, 3, padding=1)

    def _run_chunks(
        self, x: mx.array, blocks: list[TransformerBlock], chunks: int
    ) -> mx.array:
        batch, _, dim = x.shape
        x = x.reshape(batch * chunks, EFFECTIVE_CHUNK, dim)
        for block in blocks:
            x = block(x)
        return x.reshape(batch, chunks * EFFECTIVE_CHUNK, dim)

    def __call__(self, latents: mx.array) -> mx.array:
        batch, length, _ = latents.shape
        x = self.project_in(latents * self.running_std)
        tokens = mx.broadcast_to(self.new_tokens[None], (batch, length, STRIDE, DIM))
        x = mx.concatenate([x[:, :, None, :], tokens], axis=2)
        x = x.reshape(batch, length * SUB_CHUNK, DIM)

        split = len(self.transformers) // 2
        chunks = x.shape[1] // EFFECTIVE_CHUNK
        x = self._run_chunks(x, self.transformers[:split], chunks)
        x = mx.concatenate([x[:, :SHIFT, :], x, x[:, -SHIFT:, :]], axis=1)
        x = self._run_chunks(x, self.transformers[split:], chunks + 1)
        x = x[:, SHIFT:-SHIFT, :]

        x = x.reshape(batch * length, SUB_CHUNK, DIM)[:, 1:, :]
        return self.mapping(x.reshape(batch, length * STRIDE, DIM))

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
        if total > CHUNK + 2 * OVERLAP:
            patches = self._decode_windows(latents, CHUNK, OVERLAP)
        elif total % 2 == 0:
            patches = self(latents)
        elif total > 6:
            patches = self._decode_windows(latents, 2, 2)
        else:
            even = mx.concatenate([latents, latents[:, -1:]], axis=1)
            patches = self(even)[:, : total * STRIDE]
        batch, length, _ = patches.shape
        audio = patches.reshape(batch, length, 2, PATCH_SIZE).transpose(0, 1, 3, 2)
        return audio.reshape(batch, length * PATCH_SIZE, 2)
