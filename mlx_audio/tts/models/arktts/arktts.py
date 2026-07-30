"""MLX port of Audio8-TTS-Preview-0.6b (model_type: arktts).

Mirrors the reference modeling_arktts.py structure: a DualAR transformer — a
slow AR stack predicting one semantic token per audio frame, and a fast AR
stack predicting the frame's residual codec codebooks — plus the bundled
44.1 kHz codec (codec.py). Sampling reproduces the reference exactly:
semantic-range logit filtering, the legacy top-k/top-p order (filter before
temperature), exponential-race sampling, and RAS repetition rescue.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import BaseModelArgs, GenerationResult
from .codec import ArkttsCodec


@dataclass
class ModelConfig(BaseModelArgs):
    model_type: str = "arktts"
    vocab_size: int = 155776
    dim: int = 896
    n_layer: int = 24
    n_head: int = 14
    n_local_heads: int = 2
    head_dim: int = 64
    intermediate_size: int = 4864
    max_seq_len: int = 2048
    rope_base: float = 1_000_000
    norm_eps: float = 1e-6
    attention_qkv_bias: bool = True
    attention_qk_norm: bool = False
    attention_o_bias: bool = False
    tie_word_embeddings: bool = True
    codebook_size: int = 4096
    num_codebooks: int = 10
    semantic_begin_id: int = 151678
    semantic_end_id: int = 155773
    n_fast_layer: int = 4
    fast_dim: int = 896
    fast_n_head: int = 14
    fast_n_local_heads: int = 2
    fast_head_dim: int = 64
    fast_intermediate_size: int = 4864
    fast_attention_qkv_bias: bool = False
    fast_attention_qk_norm: bool = False
    fast_attention_o_bias: bool = False
    norm_fastlayer_input: bool = True
    codec_filename: str = "codec.pth"
    codec_sample_rate: int = 44100
    codec_frame_size: int = 2048
    codec_post_n_layer: int = 8
    codec_post_n_head: int = 16
    codec_post_n_local_heads: int = 8
    codec_post_intermediate_size: int = 1216
    ras_window_size: int = 10
    ras_temperature: float = 1.0
    ras_top_p: float = 0.9
    eos_token_id: int = 151645
    pad_token_id: int = 151643
    sample_rate: int = 44100


def _precompute_rope(length: int, head_dim: int, base: float) -> mx.array:
    frequencies = 1.0 / (
        base
        ** (mx.arange(0, head_dim, 2).astype(mx.float32)[: head_dim // 2] / head_dim)
    )
    phases = mx.arange(length).astype(mx.float32)[:, None] * frequencies[None, :]
    # reference stores the (real, imag) table in bf16; keep the same rounding
    return mx.stack((mx.cos(phases), mx.sin(phases)), axis=-1).astype(mx.bfloat16)


def _apply_rope(x: mx.array, rope: mx.array) -> mx.array:
    # x: (B, T, H, D); rope: (T, D/2, 2) or (B, T, D/2, 2)
    shaped = x.astype(mx.float32).reshape(*x.shape[:-1], -1, 2)
    rope = rope.astype(mx.float32)
    if rope.ndim == 3:
        rope = rope[None, :, None]
    elif rope.ndim == 4:
        rope = rope[:, :, None]
    else:
        raise ValueError(f"Unexpected RoPE shape: {tuple(rope.shape)}")
    output = mx.stack(
        (
            shaped[..., 0] * rope[..., 0] - shaped[..., 1] * rope[..., 1],
            shaped[..., 1] * rope[..., 0] + shaped[..., 0] * rope[..., 1],
        ),
        axis=-1,
    )
    return output.flatten(3).astype(x.dtype)


class ArkttsRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = float(eps)
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        xf = x.astype(mx.float32)
        normalized = xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + self.eps)
        return normalized.astype(x.dtype) * self.weight


class ArkttsKVCache:
    def __init__(
        self, batch_size: int, max_length: int, heads: int, head_dim: int, dtype
    ):
        shape = (batch_size, heads, max_length, head_dim)
        self.keys = mx.zeros(shape, dtype=dtype)
        self.values = mx.zeros(shape, dtype=dtype)

    def update(self, start: int, keys: mx.array, values: mx.array):
        length = keys.shape[2]
        self.keys[:, :, start : start + length] = keys
        self.values[:, :, start : start + length] = values
        return self.keys, self.values


class ArkttsAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_head: int,
        n_local_heads: int,
        head_dim: int,
        qkv_bias: bool,
        output_bias: bool,
        qk_norm: bool,
        norm_eps: float,
    ):
        super().__init__()
        total = (n_head + 2 * n_local_heads) * head_dim
        self.wqkv = nn.Linear(dim, total, bias=qkv_bias)
        self.wo = nn.Linear(n_head * head_dim, dim, bias=output_bias)
        self.n_head = int(n_head)
        self.n_local_heads = int(n_local_heads)
        self.head_dim = int(head_dim)
        self.qk_norm = bool(qk_norm)
        if self.qk_norm:
            self.q_norm = ArkttsRMSNorm(head_dim, norm_eps)
            self.k_norm = ArkttsRMSNorm(head_dim, norm_eps)
        self.kv_cache: Optional[ArkttsKVCache] = None

    def __call__(
        self,
        x: mx.array,
        rope: mx.array,
        attention_mask: Optional[mx.array],
        cache_start: Optional[int] = None,
    ) -> mx.array:
        batch, length, _ = x.shape
        query_size = self.n_head * self.head_dim
        kv_size = self.n_local_heads * self.head_dim
        qkv = self.wqkv(x)
        query, key, value = mx.split(qkv, [query_size, query_size + kv_size], axis=-1)
        query = query.reshape(batch, length, self.n_head, self.head_dim)
        key = key.reshape(batch, length, self.n_local_heads, self.head_dim)
        value = value.reshape(batch, length, self.n_local_heads, self.head_dim)
        if self.qk_norm:
            query = self.q_norm(query)
            key = self.k_norm(key)
        query = _apply_rope(query, rope).transpose(0, 2, 1, 3)
        key = _apply_rope(key, rope).transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)
        if self.kv_cache is not None:
            if cache_start is None:
                raise ValueError("cache_start is required when KV cache is enabled")
            key, value = self.kv_cache.update(cache_start, key, value)
        output = mx.fast.scaled_dot_product_attention(
            query,
            key,
            value,
            scale=1.0 / math.sqrt(self.head_dim),
            mask=attention_mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, query_size)
        return self.wo(output)


class ArkttsFeedForward(nn.Module):
    def __init__(self, dim: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(dim, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, dim, bias=False)
        self.w3 = nn.Linear(dim, intermediate_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class ArkttsTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_size: int,
        n_head: int,
        n_local_heads: int,
        head_dim: int,
        qkv_bias: bool,
        output_bias: bool,
        qk_norm: bool,
        norm_eps: float,
    ):
        super().__init__()
        self.attention = ArkttsAttention(
            dim,
            n_head,
            n_local_heads,
            head_dim,
            qkv_bias,
            output_bias,
            qk_norm,
            norm_eps,
        )
        self.feed_forward = ArkttsFeedForward(dim, intermediate_size)
        self.ffn_norm = ArkttsRMSNorm(dim, norm_eps)
        self.attention_norm = ArkttsRMSNorm(dim, norm_eps)

    def __call__(self, x, rope, attention_mask, cache_start=None):
        hidden = x + self.attention(
            self.attention_norm(x), rope, attention_mask, cache_start
        )
        return hidden + self.feed_forward(self.ffn_norm(hidden))


class ArkttsModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.codebook_embeddings = nn.Embedding(
            config.codebook_size * config.num_codebooks, config.dim
        )
        self.layers = [
            ArkttsTransformerBlock(
                config.dim,
                config.intermediate_size,
                config.n_head,
                config.n_local_heads,
                config.head_dim,
                config.attention_qkv_bias,
                config.attention_o_bias,
                config.attention_qk_norm,
                config.norm_eps,
            )
            for _ in range(config.n_layer)
        ]
        self.norm = ArkttsRMSNorm(config.dim, config.norm_eps)
        self.fast_project_in = (
            nn.Linear(config.dim, config.fast_dim)
            if config.fast_dim != config.dim
            else nn.Identity()
        )
        self.fast_embeddings = nn.Embedding(config.codebook_size, config.fast_dim)
        self.fast_layers = [
            ArkttsTransformerBlock(
                config.fast_dim,
                config.fast_intermediate_size,
                config.fast_n_head,
                config.fast_n_local_heads,
                config.fast_head_dim,
                config.fast_attention_qkv_bias,
                config.fast_attention_o_bias,
                config.fast_attention_qk_norm,
                config.norm_eps,
            )
            for _ in range(config.n_fast_layer)
        ]
        self.fast_norm = ArkttsRMSNorm(config.fast_dim, config.norm_eps)
        self.fast_output = nn.Linear(config.fast_dim, config.codebook_size, bias=False)
        self._freqs_cis = _precompute_rope(
            config.max_seq_len, config.head_dim, config.rope_base
        )
        self._fast_freqs_cis = _precompute_rope(
            config.num_codebooks, config.fast_head_dim, config.rope_base
        )

    # -- embedding -----------------------------------------------------------
    def _embed(self, input_ids: mx.array) -> mx.array:
        config = self.config
        codebook_embeds = []
        for index in range(config.num_codebooks):
            codebook_embeds.append(
                self.codebook_embeddings(
                    input_ids[:, index + 1] + index * config.codebook_size
                )
            )
        codebook_sum = mx.stack(codebook_embeds, axis=1).sum(axis=1)
        semantic = mx.logical_and(
            input_ids[:, 0] >= config.semantic_begin_id,
            input_ids[:, 0] <= config.semantic_end_id,
        )
        codebook_sum = mx.where(semantic[..., None], codebook_sum, 0.0)
        return self.embeddings(input_ids[:, 0]) + codebook_sum

    @staticmethod
    def _causal_mask(
        attention_mask: mx.array, query_positions: mx.array, key_length: int
    ) -> mx.array:
        if attention_mask.shape[1] < key_length:
            attention_mask = mx.pad(
                attention_mask, ((0, 0), (0, key_length - attention_mask.shape[1]))
            )
        key_positions = mx.arange(key_length)
        causal = key_positions[None, :] <= query_positions[:, None]
        return mx.logical_and(
            causal[None, None],
            attention_mask[:, None, None, :key_length].astype(mx.bool_),
        )

    # -- prefill (no-cache) forward, for parity ------------------------------
    def __call__(self, input_ids: mx.array, attention_mask: Optional[mx.array] = None):
        config = self.config
        if input_ids.ndim != 3 or input_ids.shape[1] != config.num_codebooks + 1:
            raise ValueError(
                f"input_ids must have shape [B, {config.num_codebooks + 1}, T]"
            )
        batch, _, length = input_ids.shape
        if attention_mask is None:
            attention_mask = mx.ones((batch, length), dtype=mx.int64)
        position_ids = mx.maximum(
            mx.cumsum(attention_mask.astype(mx.int64), axis=-1) - 1, 0
        )
        rope = self._freqs_cis[position_ids]
        mask = self._causal_mask(attention_mask, mx.arange(length), length)
        hidden = self._embed(input_ids)
        for layer in self.layers:
            hidden = layer(hidden, rope, mask)
        normalized = self.norm(hidden)
        logits = normalized @ self.embeddings.weight.T
        return logits, hidden

    # -- cached decode steps -------------------------------------------------
    def _setup_generation_caches(self, batch_size: int, dtype):
        config = self.config
        for layer in self.layers:
            layer.attention.kv_cache = ArkttsKVCache(
                batch_size,
                config.max_seq_len,
                config.n_local_heads,
                config.head_dim,
                dtype,
            )
        for layer in self.fast_layers:
            layer.attention.kv_cache = ArkttsKVCache(
                batch_size,
                config.num_codebooks,
                config.fast_n_local_heads,
                config.fast_head_dim,
                dtype,
            )

    def _clear_generation_caches(self):
        for layer in list(self.layers) + list(self.fast_layers):
            layer.attention.kv_cache = None

    def _slow_step(
        self,
        input_ids: mx.array,
        cache_start: int,
        cache_positions: mx.array,
        position_ids: mx.array,
        attention_mask: mx.array,
    ):
        config = self.config
        hidden = self._embed(input_ids)
        rope = self._freqs_cis[position_ids]
        mask = self._causal_mask(attention_mask, cache_positions, config.max_seq_len)
        for layer in self.layers:
            hidden = layer(hidden, rope, mask, cache_start)
        hidden = hidden[:, -1:]
        normalized = self.norm(hidden)
        logits = (normalized @ self.embeddings.weight.T)[:, -1]
        fast_hidden = normalized if config.norm_fastlayer_input else hidden
        return logits, fast_hidden

    def _fast_step(self, hidden: mx.array, position: int) -> mx.array:
        config = self.config
        rope = self._fast_freqs_cis[mx.array([position])]
        key_mask = mx.ones((hidden.shape[0], config.num_codebooks), dtype=mx.bool_)
        mask = self._causal_mask(key_mask, mx.array([position]), config.num_codebooks)
        for layer in self.fast_layers:
            hidden = layer(hidden, rope, mask, position)
        return self.fast_output(self.fast_norm(hidden))[:, -1]

    # -- sampling (mirrors the reference exactly) ----------------------------
    @staticmethod
    def _legacy_top_k_top_p(scores: mx.array, top_k: int, top_p: float) -> mx.array:
        sorted_indices = mx.argsort(-scores, axis=-1)
        sorted_scores = mx.take_along_axis(scores, sorted_indices, axis=-1)
        cumulative = mx.cumsum(mx.softmax(sorted_scores, axis=-1), axis=-1)
        positions = mx.arange(scores.shape[-1])
        remove_sorted = mx.logical_or(cumulative > top_p, positions[None, :] >= top_k)
        remove_sorted[:, 0] = False
        remove = mx.zeros_like(remove_sorted)
        remove = mx.put_along_axis(remove, sorted_indices, remove_sorted, axis=-1)
        return mx.where(remove, mx.array(-mx.inf, dtype=scores.dtype), scores)

    def _semantic_filter(self, scores: mx.array) -> mx.array:
        config = self.config
        filtered = mx.full(scores.shape, -mx.inf, dtype=scores.dtype)
        filtered[:, config.semantic_begin_id : config.semantic_end_id + 1] = scores[
            :, config.semantic_begin_id : config.semantic_end_id + 1
        ]
        filtered[:, config.eos_token_id] = scores[:, config.eos_token_id]
        return filtered

    @staticmethod
    def _sample(scores: mx.array, rng_key) -> mx.array:
        probabilities = mx.softmax(scores, axis=-1)
        random = mx.random.uniform(shape=probabilities.shape, key=rng_key)
        noise = -mx.log(random)
        return mx.argmax(probabilities / noise, axis=-1).astype(mx.int64)

    def _processed_scores(self, scores, top_k, top_p, temperature, semantic: bool):
        if semantic:
            scores = self._semantic_filter(scores)
        scores = self._legacy_top_k_top_p(scores, top_k, top_p)
        return scores / max(temperature, 1e-5)

    def _sample_semantic(
        self, logits, top_k, top_p, temperature, previous, do_sample, rng_keys
    ):
        config = self.config
        regular_scores = self._processed_scores(logits, top_k, top_p, temperature, True)
        if not do_sample:
            return mx.argmax(regular_scores, axis=-1).astype(mx.int64)
        normal = self._sample(regular_scores, rng_keys[0])
        high_scores = self._processed_scores(
            logits, top_k, config.ras_top_p, config.ras_temperature, True
        )
        high = self._sample(high_scores, rng_keys[1])
        if previous is None:
            return normal
        repeated = mx.any(previous == normal[:, None], axis=1)
        semantic = mx.logical_and(
            normal >= config.semantic_begin_id, normal <= config.semantic_end_id
        )
        return mx.where(mx.logical_and(repeated, semantic), high, normal)

    def _generate_codebooks(
        self, slow_hidden, semantic, top_k, top_p, temperature, do_sample, rng_key
    ):
        config = self.config
        hidden = self.fast_project_in(slow_hidden)
        self._fast_step(hidden, 0)
        current = mx.clip(
            semantic - config.semantic_begin_id, 0, config.codebook_size - 1
        )
        codebooks = [current]
        hidden = self.fast_embeddings(current)[:, None]
        for position in range(1, config.num_codebooks):
            scores = self._fast_step(hidden, position)
            scores = self._processed_scores(scores, top_k, top_p, temperature, False)
            if do_sample:
                rng_key, sub = mx.random.split(rng_key)
                current = self._sample(scores, sub)
            else:
                current = mx.argmax(scores, axis=-1).astype(mx.int64)
            codebooks.append(current)
            hidden = self.fast_embeddings(current)[:, None]
        return mx.stack(codebooks, axis=1)

    # -- prompt building -----------------------------------------------------
    def _prepare_prompt(
        self,
        prefix_input_ids: np.ndarray,
        suffix_input_ids: np.ndarray,
        reference_codes: Optional[np.ndarray] = None,
        reference_code_lengths: Optional[np.ndarray] = None,
    ):
        """Single-row (B=1 per row) numpy prompt assembly, mirroring the reference."""
        config = self.config
        rows = []
        batch_size = len(prefix_input_ids)
        for batch_index in range(batch_size):
            prefix = np.asarray(prefix_input_ids[batch_index], dtype=np.int64)
            suffix = np.asarray(suffix_input_ids[batch_index], dtype=np.int64)
            if reference_codes is None:
                semantic_row = np.concatenate((prefix, suffix))
                values = np.zeros(
                    (config.num_codebooks + 1, semantic_row.size), dtype=np.int64
                )
                values[0] = semantic_row
            else:
                length = int(reference_code_lengths[batch_index])
                codes = np.asarray(
                    reference_codes[batch_index][:, :length], dtype=np.int64
                )
                semantic_codes = codes[0] + config.semantic_begin_id
                semantic_row = np.concatenate((prefix, semantic_codes, suffix))
                values = np.zeros(
                    (config.num_codebooks + 1, semantic_row.size), dtype=np.int64
                )
                values[0] = semantic_row
                values[1:, prefix.size : prefix.size + length] = codes
            rows.append(values)
        max_length = max(row.shape[1] for row in rows)
        prompt = np.zeros(
            (batch_size, config.num_codebooks + 1, max_length), dtype=np.int64
        )
        prompt[:, 0] = config.pad_token_id
        prompt_mask = np.zeros((batch_size, max_length), dtype=np.int64)
        for batch_index, row in enumerate(rows):
            start = max_length - row.shape[1]
            prompt[batch_index, :, start:] = row
            prompt_mask[batch_index, start:] = 1
        return mx.array(prompt), mx.array(prompt_mask)

    # -- generation ----------------------------------------------------------
    def generate_codes(
        self,
        prefix_input_ids,
        suffix_input_ids,
        reference_codes=None,
        reference_code_lengths=None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        seed: Optional[int] = None,
    ) -> mx.array:
        """Returns generated codec codes (B, num_codebooks, T). Batch size 1."""
        config = self.config
        prompt, prompt_mask = self._prepare_prompt(
            prefix_input_ids, suffix_input_ids, reference_codes, reference_code_lengths
        )
        batch_size, _, prompt_width = prompt.shape
        if prompt_width >= config.max_seq_len:
            raise ValueError(
                f"Prompt length {prompt_width} must be smaller than {config.max_seq_len}"
            )
        max_new_tokens = min(max_new_tokens, config.max_seq_len - prompt_width)
        dtype = self.embeddings.weight.dtype
        self._setup_generation_caches(batch_size, dtype)
        rng_key = mx.random.key(
            seed if seed is not None else int(time.time_ns() % (2**63))
        )

        position_ids = mx.maximum(mx.cumsum(prompt_mask, axis=-1) - 1, 0)
        logits, slow_hidden = self._slow_step(
            prompt, 0, mx.arange(prompt_width), position_ids, prompt_mask
        )
        prompt_lengths = prompt_mask.sum(axis=-1)
        previous = None
        finished = mx.zeros((batch_size,), dtype=mx.bool_)
        code_lengths = mx.zeros((batch_size,), dtype=mx.int64)
        generated_frames = []
        prompt_mask_np = prompt_mask

        for step in range(max_new_tokens):
            active_before = mx.logical_not(finished)
            keys = mx.random.split(rng_key, 4)
            rng_key = keys[3]
            semantic = self._sample_semantic(
                logits,
                top_k,
                top_p,
                temperature,
                previous,
                do_sample,
                (keys[0], keys[1]),
            )
            codebooks = self._generate_codebooks(
                slow_hidden, semantic, top_k, top_p, temperature, do_sample, keys[2]
            )
            emitted = mx.logical_and(active_before, semantic != config.eos_token_id)
            frame = mx.where(emitted[:, None], codebooks, -1)
            generated_frames.append(frame)
            code_lengths = code_lengths + emitted.astype(mx.int64)

            if previous is None:
                previous = mx.zeros(
                    (batch_size, config.ras_window_size), dtype=mx.int64
                )
            else:
                previous = mx.concatenate((previous[:, 1:], semantic[:, None]), axis=1)
            finished = mx.logical_or(finished, semantic == config.eos_token_id)
            mx.eval(finished, frame, code_lengths)
            if bool(mx.all(finished)):
                break

            next_column = mx.concatenate((semantic[:, None], codebooks), axis=1)[
                ..., None
            ]
            new_valid = active_before.astype(mx.int64)[:, None]
            prompt_mask_np = mx.concatenate((prompt_mask_np, new_valid), axis=1)
            physical_position = prompt_width + step
            token_position = (prompt_lengths + step)[:, None]
            logits, slow_hidden = self._slow_step(
                next_column,
                physical_position,
                mx.array([physical_position]),
                token_position,
                prompt_mask_np,
            )

        self._clear_generation_caches()
        if generated_frames:
            codes = mx.stack(generated_frames, axis=2)
            max_valid = int(mx.max(code_lengths).item()) if code_lengths.size else 0
            codes = codes[:, :, :max_valid]
        else:
            codes = mx.zeros((batch_size, config.num_codebooks, 0), dtype=mx.int64)
        return codes


class Model(nn.Module):
    """mlx-audio entry point wrapping ArkttsModel + ArkttsCodec + the prompt builder."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = ArkttsModel(config)
        self.codec = ArkttsCodec(config)
        self._tokenizer = None
        self.model_path: Optional[Path] = None

    @property
    def sample_rate(self) -> int:
        return self.config.codec_sample_rate

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
        model.model_path = Path(model_path)
        return model

    # tokenizer / prompt -----------------------------------------------------
    def _load_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            # the reference processor pins fix_mistral_regex=False; match it
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path), use_fast=True, fix_mistral_regex=False
            )
        return self._tokenizer

    @staticmethod
    def _clean_text(text: str) -> str:
        return " ".join(str(text).strip().split())

    def _format_reference_text(self, text: str) -> str:
        import re

        cleaned = self._clean_text(text)
        if re.search(r"<\|speaker:\d+\|>", cleaned):
            return cleaned
        return f"<|speaker:0|>{cleaned}"

    def _prompt_segments(
        self, text: str, reference_text: Optional[str], has_reference: bool
    ):
        tokenizer = self._load_tokenizer()

        def encode_parts(parts):
            out = []
            for part in parts:
                out.extend(tokenizer.encode(part, add_special_tokens=False))
            return np.asarray(out, dtype=np.int64)

        target = self._clean_text(text)
        if not target:
            raise ValueError("text must not be empty")
        if not has_reference:
            full = encode_parts(
                [
                    "<|im_start|>system\n",
                    "convert the provided text to speech",
                    "<|im_end|>\n",
                    "<|im_start|>user\n",
                    target,
                    "<|im_end|>\n",
                    "<|im_start|>assistant\n<|voice|>",
                ]
            )
            return full, np.asarray([], dtype=np.int64)
        if not reference_text:
            raise ValueError(
                "reference_text is required when a reference voice is provided"
            )
        prefix = encode_parts(
            [
                "<|im_start|>system\n",
                "convert the provided text to speech reference to the following:\n\nText:\n",
                self._format_reference_text(reference_text),
                "\n\nSpeech:\n",
            ]
        )
        suffix = encode_parts(
            [
                "<|im_end|>\n",
                "<|im_start|>user\n",
                target,
                "<|im_end|>\n",
                "<|im_start|>assistant\n<|voice|>",
            ]
        )
        return prefix, suffix

    # audio ------------------------------------------------------------------
    def _load_reference_audio(self, ref_audio) -> mx.array:
        import soundfile as sf

        from mlx_audio.utils import resample_audio

        if isinstance(ref_audio, (str, Path)):
            array, source_rate = sf.read(
                str(ref_audio), dtype="float32", always_2d=True
            )
            array = array.mean(axis=1)
        else:
            array = np.asarray(ref_audio, dtype=np.float32)
            source_rate = self.config.codec_sample_rate
        if int(source_rate) != self.config.codec_sample_rate:
            array = np.asarray(
                resample_audio(array, int(source_rate), self.config.codec_sample_rate),
                dtype=np.float32,
            )
        return mx.array(array)

    def encode_reference(self, ref_audio) -> tuple[mx.array, mx.array]:
        audio = self._load_reference_audio(ref_audio)
        codes, code_lengths = self.codec.encode(
            audio[None], mx.array([audio.shape[0]], dtype=mx.int64)
        )
        return codes, code_lengths

    # generation -------------------------------------------------------------
    def generate(
        self,
        text: str,
        ref_audio=None,
        ref_text: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        max_tokens: int = 512,
        do_sample: bool = True,
        seed: Optional[int] = None,
        verbose: bool = False,
        **kwargs,
    ):
        start = time.time()
        has_reference = ref_audio is not None
        prefix, suffix = self._prompt_segments(text, ref_text, has_reference)
        reference_codes = reference_code_lengths = None
        if has_reference:
            codes, lengths = self.encode_reference(ref_audio)
            mx.eval(codes, lengths)
            reference_codes = [np.asarray(codes[0])]
            reference_code_lengths = np.asarray(lengths)

        codes = self.model.generate_codes(
            [prefix],
            [suffix],
            reference_codes,
            reference_code_lengths,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            seed=seed,
        )
        mx.eval(codes)
        token_count = codes.shape[-1]
        if token_count == 0:
            raise RuntimeError("Model generated no audio frames")
        waveform = self.codec.decode(codes)[0]
        mx.eval(waveform)
        elapsed = time.time() - start
        samples = waveform.shape[0]
        duration = samples / self.sample_rate
        yield GenerationResult(
            audio=waveform,
            samples=samples,
            sample_rate=self.sample_rate,
            segment_idx=0,
            token_count=token_count,
            audio_duration=time.strftime("%H:%M:%S", time.gmtime(duration)),
            real_time_factor=elapsed / duration if duration else 0.0,
            prompt={"text": text, "ref_text": ref_text},
            audio_samples={"samples-per-sec": samples / elapsed if elapsed else 0.0},
            processing_time_seconds=elapsed,
            peak_memory_usage=mx.get_peak_memory() / 1e9,
        )

    # weights ----------------------------------------------------------------
    def sanitize(self, weights: dict) -> dict:
        """Remap reference checkpoints to this module tree.

        Handles: LM keys (prefix under model.), codec keys (prefix under codec.),
        weight-norm folding (both parametrizations and legacy g/v), conv layout
        transposes, Snake alpha reshape, and dropped rope buffers.

        Idempotent: weights already carrying the model./codec. prefixes (i.e. a
        pre-converted mlx repo) pass through untouched.
        """
        if all(key.startswith(("model.", "codec.")) for key in weights):
            return weights
        out = {}
        pending: dict[str, dict] = {}

        def fold_weight_norm(g: mx.array, v: mx.array) -> mx.array:
            # PyTorch weight_norm dim=0: per-output-channel norm over (I, K)
            norm = mx.sqrt(
                mx.sum(mx.square(v.reshape(v.shape[0], -1)), axis=1, keepdims=True)
            ).reshape(v.shape[0], *([1] * (v.ndim - 1)))
            return g * v / norm

        for key, value in weights.items():
            if key.endswith(("freqs_cis", "causal_mask")):
                continue
            if key.startswith("generator."):
                key = key[len("generator.") :]
            if not key.startswith(("model.", "codec.")):
                # raw reference LM checkpoint keys arrive unprefixed;
                # raw codec.pth keys arrive as encoder./decoder./quantizer.
                if key.split(".")[0] in ("encoder", "decoder", "quantizer"):
                    key = "codec." + key
                else:
                    key = "model." + key
            # collect weight-norm pairs for folding
            if ".parametrizations.weight.original0" in key:
                base = key.replace(".parametrizations.weight.original0", ".weight")
                pending.setdefault(base, {})["g"] = value
                continue
            if ".parametrizations.weight.original1" in key:
                base = key.replace(".parametrizations.weight.original1", ".weight")
                pending.setdefault(base, {})["v"] = value
                continue
            if key.endswith(".weight_g"):
                base = key[: -len(".weight_g")] + ".weight"
                pending.setdefault(base, {})["g"] = value
                continue
            if key.endswith(".weight_v"):
                base = key[: -len(".weight_v")] + ".weight"
                pending.setdefault(base, {})["v"] = value
                continue
            out[key] = value

        for base, pair in pending.items():
            out[base] = fold_weight_norm(pair["g"], pair["v"])

        remapped = {}
        for key, value in out.items():
            if key.startswith("codec."):
                if key.endswith("alpha"):
                    # Snake (1, C, 1) -> (1, 1, C)
                    value = value.transpose(0, 2, 1)
                elif key.endswith(".conv.weight") and value.ndim == 3:
                    if isinstance_transpose(key):
                        # ConvTranspose1d: torch (I, O, K) -> mlx (O, K, I)
                        value = value.transpose(1, 2, 0)
                    else:
                        # Conv1d: torch (O, I, K) -> mlx (O, K, I)
                        value = value.transpose(0, 2, 1)
                elif (
                    ".in_proj.weight" in key or ".out_proj.weight" in key
                ) and value.ndim == 3:
                    # VQ 1x1 convs: torch (O, I, 1) -> mlx (O, 1, I)
                    value = value.transpose(0, 2, 1)
            remapped[key] = value
        return remapped


def isinstance_transpose(key: str) -> bool:
    """Conv keys that belong to ArkttsCausalConvTranspose1d modules:
    decoder blocks 1..4 hold theirs at block.1.conv; the quantizer's upsample
    stages hold theirs at upsample.<i>.0.conv (the .1 ConvNeXt dwconv is a
    regular grouped conv)."""
    import re

    return (
        re.search(r"decoder\.model\.[1-4]\.block\.1\.conv\.weight$", key) is not None
        or re.search(r"quantizer\.upsample\.\d+\.0\.conv\.weight$", key) is not None
    )
