import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
from mlx_lm.models import cache as cache_utils
from mlx_lm.models.base import BaseModelArgs, create_attention_mask
from mlx_lm.models.rope_utils import initialize_rope
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.utils import stream_generate
from snac import SNAC
from tqdm import tqdm
from transformers import AutoTokenizer

from ..base import GenerationResult


@dataclass
class ModelConfig(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    head_dim: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    attention_bias: bool = False
    mlp_bias: bool = False
    rope_theta: float = 500000.0
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = True
    tokeniser_name: str = "mlx-community/orpheus-3b-0.1-ft-bf16"

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


# TODO: Convert to mlx
snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
snac_model = snac_model.to("cpu")


# TODO: Convert to mlx
def redistribute_codes(code_list):
    layer_1 = []
    layer_2 = []
    layer_3 = []
    for i in range((len(code_list) + 1) // 7):
        layer_1.append(code_list[7 * i])
        layer_2.append(code_list[7 * i + 1] - 4096)
        layer_3.append(code_list[7 * i + 2] - (2 * 4096))
        layer_3.append(code_list[7 * i + 3] - (3 * 4096))
        layer_2.append(code_list[7 * i + 4] - (4 * 4096))
        layer_3.append(code_list[7 * i + 5] - (5 * 4096))
        layer_3.append(code_list[7 * i + 6] - (6 * 4096))
    codes = [
        torch.tensor(layer_1).unsqueeze(0),
        torch.tensor(layer_2).unsqueeze(0),
        torch.tensor(layer_3).unsqueeze(0),
    ]
    audio_hat = snac_model.decode(codes)
    return audio_hat


class Attention(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        self.head_dim = head_dim = args.head_dim or args.hidden_size // n_heads

        self.scale = head_dim**-0.5
        if hasattr(args, "attention_bias"):
            attention_bias = args.attention_bias
        else:
            attention_bias = False

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=attention_bias)

        self.rope = initialize_rope(
            self.head_dim,
            args.rope_theta,
            args.rope_traditional,
            args.rope_scaling,
            args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()

        dim = args.hidden_size
        hidden_dim = args.intermediate_size
        if hasattr(args, "mlp_bias"):
            mlp_bias = args.mlp_bias
        else:
            mlp_bias = False

        self.gate_proj = nn.Linear(dim, hidden_dim, bias=mlp_bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=mlp_bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=mlp_bias)

    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.args = args

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out


class LlamaModel(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache=None,
    ):
        h = self.embed_tokens(inputs)

        if mask is None:
            mask = create_attention_mask(h, cache)

        if cache is None:
            cache = [None] * len(self.layers)

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, cache=c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelConfig, **kwargs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokeniser_name)
        self.model = LlamaModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache=None,
    ):
        out = self.model(inputs, mask, cache)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def sanitize(self, weights):
        # Remove unused precomputed rotary freqs
        weights = {
            k: v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k
        }
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights

    @property
    def layers(self):
        return self.model.layers

    def parse_output(self, input_ids):
        token_to_find = 128257
        token_to_remove = 128258

        # MLX doesn't have nonzero, so we need to create indices manually
        mask = input_ids == token_to_find
        indices = []
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j]:
                    indices.append((i, j))
        token_indices = [[], []]
        for i, j in indices:
            token_indices[0].append(i)
            token_indices[1].append(j)

        token_indices = mx.array(token_indices)

        if len(token_indices[1]) > 0:
            last_occurrence_idx = int(token_indices[1][-1])
            cropped_tensor = input_ids[:, last_occurrence_idx + 1 :]
        else:
            cropped_tensor = input_ids

        mask = cropped_tensor != token_to_remove

        processed_rows = []

        for row in cropped_tensor:
            # Create a mask and filter manually since boolean indexing isn't supported
            row_list = row.tolist()
            masked_row = mx.array([val for val in row_list if val != token_to_remove])
            processed_rows.append(masked_row)

        code_lists = []

        for row in processed_rows:
            row_length = row.shape[0]
            new_length = (row_length // 7) * 7
            trimmed_row = row[:new_length]
            trimmed_row = [t - 128266 for t in trimmed_row]
            code_lists.append(trimmed_row)

        return code_lists

    def generate(
        self,
        text,
        voice: str,
        temperature: float = 1.0,
        top_p: float = 0.95,
        split_pattern: str = "\n",
        max_tokens: int = 1200,
        verbose: bool = False,
        **kwargs,
    ):
        prompt = text.replace("\\n", "\n").replace("\\t", "\t")
        prompts = prompt.split(split_pattern)
        prompts = [f"{voice}: " + p for p in prompts]

        all_input_ids = []

        for prompt in prompts:
            input_ids = mx.array(self.tokenizer(prompt, return_tensors="pt").input_ids)
            all_input_ids.append(input_ids)

        start_token = mx.array([[128259]], dtype=mx.int64)  # Start of human
        end_tokens = mx.array(
            [[128009, 128260]], dtype=mx.int64
        )  # End of text, End of human

        all_modified_input_ids = []
        for input_ids in all_input_ids:
            modified_input_ids = mx.concatenate(
                [start_token, input_ids, end_tokens], axis=1
            )  # SOH SOT Text EOT EOH
            all_modified_input_ids.append(modified_input_ids)

        input_ids = mx.concatenate(all_modified_input_ids, axis=0)

        sampler = make_sampler(temperature, top_p, top_k=kwargs.get("top_k", 1))
        logits_processors = make_logits_processors(
            kwargs.get("logit_bias", None),
            kwargs.get("repetition_penalty", 1.1),
            kwargs.get("repetition_context_size", 20),
        )

        time_start = time.time()
        # TODO: Support batch processing as in the Colab: https://github.com/canopyai/Orpheus-TTS
        for i, response in enumerate(
            tqdm(
                stream_generate(
                    self,
                    tokenizer=self.tokenizer,
                    prompt=input_ids.squeeze(0),
                    max_tokens=max_tokens,
                    sampler=sampler,
                    logits_processors=logits_processors,
                ),
                total=max_tokens,
                disable=not verbose,
            )
        ):
            next_token = mx.array([response.token])
            input_ids = mx.concatenate([input_ids, next_token[None, :]], axis=1)
            if i % 50 == 0:
                mx.metal.clear_cache()

            if next_token == 128258:
                break

        code_lists = self.parse_output(input_ids)

        my_samples = []
        for code_list in code_lists:
            code_list = torch.from_dlpack(mx.array(code_list))
            samples = redistribute_codes(code_list)
            samples = mx.array(samples)
            my_samples.append(samples)

        time_end = time.time()

        if len(prompts) != len(my_samples):
            raise Exception("Number of prompts and samples do not match")
        else:
            for i in range(len(my_samples)):
                audio = my_samples[i][0]

                samples = audio.shape[0] if audio is not None else 0
                assert samples > 0, "No audio generated"

                # Calculate token count
                token_count = len(input_ids) if input_ids is not None else 0

                # Calculate audio duration in seconds
                sample_rate = 24000  # Assuming 24kHz sample rate, adjust if different
                audio_duration_seconds = samples / sample_rate * audio.shape[1]

                # Calculate real-time factor (RTF)
                rtf = audio_duration_seconds / (time_end - time_start)

                # Format duration as HH:MM:SS.mmm
                duration_mins = int(audio_duration_seconds // 60)
                duration_secs = int(audio_duration_seconds % 60)
                duration_ms = int((audio_duration_seconds % 1) * 1000)
                duration_hours = int(audio_duration_seconds // 3600)
                duration_str = f"{duration_hours:02d}:{duration_mins:02d}:{duration_secs:02d}.{duration_ms:03d}"

                yield GenerationResult(
                    audio=audio[0],
                    samples=samples,
                    segment_idx=i,
                    token_count=token_count,
                    audio_duration=duration_str,
                    real_time_factor=rtf,
                    prompt={
                        "tokens": token_count,
                        "tokens-per-sec": (
                            round(token_count / audio_duration_seconds, 2)
                            if audio_duration_seconds > 0
                            else 0
                        ),
                    },
                    audio_samples={
                        "samples": samples,
                        "samples-per-sec": (
                            round(samples / audio_duration_seconds, 2)
                            if audio_duration_seconds > 0
                            else 0
                        ),
                    },
                    processing_time_seconds=time_end - time_start,
                    peak_memory_usage=mx.metal.get_peak_memory() / 1e9,
                )
