# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import mlx.core as mx
from mlx_lm.models.cache import BatchKVCache, KVCache

from mlx_audio.tts.models.base import BatchGenerationResult


def _format_duration(seconds: float) -> str:
    """Format duration as HH:MM:SS.mmm."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


@dataclass
class Qwen3BatchInputs:
    input_embeds: mx.array
    trailing_text_hidden: mx.array
    tts_pad_embed: mx.array
    attention_mask: mx.array
    left_padding: List[int]
    prefill_lens: List[int]
    trailing_lens: List[int]


@dataclass
class _PendingRequest:
    sequence_idx: int
    text: str
    voice: Optional[str] = None
    instruct: Optional[str] = None


@dataclass
class _ActiveRequest:
    sequence_idx: int
    text: str
    voice: Optional[str]
    instruct: Optional[str]
    cache: List[Any]
    input_embeds: mx.array
    trailing_text_hidden: mx.array
    tts_pad_embed: mx.array
    generated_codes: List[mx.array] = field(default_factory=list)
    generated_token_ids: List[int] = field(default_factory=list)
    trailing_idx: int = 0


class Qwen3ContinuousBatchEngine:
    """Step-wise non-streaming Qwen3 TTS batch engine.

    Each request owns its KV cache. At each decode step, active request caches
    are merged into a temporary BatchKVCache, advanced together, and extracted
    back into per-request caches. This allows new requests to be admitted at
    step boundaries without rebuilding existing request state.
    """

    def __init__(
        self,
        model,
        *,
        temperature: float = 0.9,
        lang_code: str = "auto",
        max_tokens: int = 4096,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.05,
        max_batch_size: int = 8,
        verbose: bool = False,
    ):
        if model.speech_tokenizer is None:
            raise ValueError("Speech tokenizer not loaded")

        self.model = model
        self.temperature = temperature
        self.lang_code = lang_code
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.max_batch_size = max(1, max_batch_size)
        self.verbose = verbose

        config = model.config.talker_config
        self.config = config
        self.eos_token_id = config.codec_eos_token_id
        self.suppress_tokens = [
            i
            for i in range(config.vocab_size - 1024, config.vocab_size)
            if i != self.eos_token_id
        ]

        self._pending: List[_PendingRequest] = []
        self._active: List[_ActiveRequest] = []
        self._start_time = time.time()
        self._step_count = 0

    @property
    def active_count(self) -> int:
        return len(self._active)

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def available_slots(self) -> int:
        return max(0, self.max_batch_size - len(self._active))

    @property
    def idle(self) -> bool:
        return not self._pending and not self._active

    @property
    def active_sequence_ids(self) -> List[int]:
        return [request.sequence_idx for request in self._active]

    @property
    def pending_sequence_ids(self) -> List[int]:
        return [request.sequence_idx for request in self._pending]

    def submit(
        self,
        *,
        sequence_idx: int,
        text: str,
        voice: Optional[str] = None,
        instruct: Optional[str] = None,
    ) -> None:
        self._pending.append(
            _PendingRequest(
                sequence_idx=sequence_idx,
                text=text,
                voice=voice,
                instruct=instruct,
            )
        )

    def cancel(self, sequence_idx: int) -> bool:
        original_pending = len(self._pending)
        original_active = len(self._active)
        self._pending = [
            request
            for request in self._pending
            if request.sequence_idx != sequence_idx
        ]
        self._active = [
            request
            for request in self._active
            if request.sequence_idx != sequence_idx
        ]
        return (
            len(self._pending) != original_pending
            or len(self._active) != original_active
        )

    def step(self) -> List[BatchGenerationResult]:
        completed: List[BatchGenerationResult] = []

        if self._active:
            completed.extend(self._advance_active())

        if self.available_slots > 0 and self._pending:
            completed.extend(self._admit_pending())

        self._step_count += 1
        if self._step_count > 0 and self._step_count % 50 == 0:
            mx.clear_cache()

        return completed

    def _take_pending_batch(self) -> List[_PendingRequest]:
        batch_size = min(self.available_slots, len(self._pending))
        batch = self._pending[:batch_size]
        del self._pending[:batch_size]
        return batch

    def _admit_pending(self) -> List[BatchGenerationResult]:
        pending = self._take_pending_batch()
        if not pending:
            return []

        if self.verbose:
            print(
                "[qwen3-continuous] admitting "
                f"{len(pending)} request(s); active={len(self._active)}",
                flush=True,
            )

        if self.max_tokens <= 0:
            return [self._empty_result(request.sequence_idx) for request in pending]

        batch_inputs = self.model._prepare_batch_inputs_with_metadata(
            [request.text for request in pending],
            language=self.lang_code,
            speakers=[request.voice for request in pending],
            instructs=[request.instruct for request in pending],
        )
        mx.eval(
            batch_inputs.input_embeds,
            batch_inputs.trailing_text_hidden,
            batch_inputs.tts_pad_embed,
            batch_inputs.attention_mask,
        )

        cache = self._make_batch_cache(batch_inputs.left_padding)
        logits, hidden = self.model.talker(
            batch_inputs.input_embeds,
            cache=cache,
            attention_mask=batch_inputs.attention_mask,
        )

        next_token_batch, code_tokens, all_codes = self._sample_code_tokens(
            logits,
            hidden,
            generated_tokens_per_seq=[[] for _ in pending],
        )
        finished = next_token_batch[:, 0] == self.eos_token_id
        next_input_embeds = self._next_input_embeds_for_batch(
            batch_inputs.trailing_text_hidden,
            batch_inputs.tts_pad_embed,
            mx.zeros((len(pending), 1), dtype=mx.int32),
            code_tokens,
        )
        mx.eval(all_codes, next_input_embeds, finished)

        caches_by_request = self._extract_batch_cache_rows(cache, len(pending))
        finished_cpu = finished.tolist()
        token_ids_cpu = next_token_batch[:, 0].tolist()

        completed: List[BatchGenerationResult] = []
        for index, request in enumerate(pending):
            trailing_len = batch_inputs.trailing_lens[index]
            state = _ActiveRequest(
                sequence_idx=request.sequence_idx,
                text=request.text,
                voice=request.voice,
                instruct=request.instruct,
                cache=caches_by_request[index],
                input_embeds=next_input_embeds[index : index + 1],
                trailing_text_hidden=batch_inputs.trailing_text_hidden[
                    index : index + 1, :trailing_len, :
                ],
                tts_pad_embed=batch_inputs.tts_pad_embed,
                trailing_idx=1,
            )

            if not finished_cpu[index]:
                state.generated_token_ids.append(token_ids_cpu[index])
                state.generated_codes.append(all_codes[index : index + 1])

            if finished_cpu[index] or len(state.generated_codes) >= self.max_tokens:
                completed.append(self._decode_state(state))
            else:
                self._active.append(state)

        return completed

    def _advance_active(self) -> List[BatchGenerationResult]:
        states = list(self._active)
        batch_size = len(states)

        cache = self._merge_state_caches(states)
        input_embeds = mx.concatenate(
            [state.input_embeds for state in states], axis=0
        )
        attention_mask = self._attention_mask_for_states(states)

        logits, hidden = self.model.talker(
            input_embeds,
            cache=cache,
            attention_mask=attention_mask,
        )
        next_token_batch, code_tokens, all_codes = self._sample_code_tokens(
            logits,
            hidden,
            generated_tokens_per_seq=[
                state.generated_token_ids for state in states
            ],
        )
        finished = next_token_batch[:, 0] == self.eos_token_id
        next_input_embeds = self._next_input_embeds_for_states(states, code_tokens)
        mx.eval(all_codes, next_input_embeds, finished)

        caches_by_state = self._extract_batch_cache_rows(cache, batch_size)
        finished_cpu = finished.tolist()
        token_ids_cpu = next_token_batch[:, 0].tolist()

        completed: List[BatchGenerationResult] = []
        still_active: List[_ActiveRequest] = []
        for index, state in enumerate(states):
            state.cache = caches_by_state[index]
            state.input_embeds = next_input_embeds[index : index + 1]

            if not finished_cpu[index]:
                state.generated_token_ids.append(token_ids_cpu[index])
                state.generated_codes.append(all_codes[index : index + 1])
                state.trailing_idx += 1

            if finished_cpu[index] or len(state.generated_codes) >= self.max_tokens:
                completed.append(self._decode_state(state))
            else:
                still_active.append(state)

        self._active = still_active
        return completed

    def _sample_code_tokens(
        self,
        logits: mx.array,
        hidden: mx.array,
        *,
        generated_tokens_per_seq: List[List[int]],
    ) -> Tuple[mx.array, List[mx.array], mx.array]:
        next_token_batch = self.model._sample_token_batch(
            logits,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            generated_tokens_per_seq=generated_tokens_per_seq,
            suppress_tokens=self.suppress_tokens,
            eos_token_id=self.eos_token_id,
        )

        code_tokens = [next_token_batch]
        code_hidden = hidden[:, -1:, :]
        code_cache = self.model.talker.code_predictor.make_cache()

        for code_idx in range(self.config.num_code_groups - 1):
            if code_idx == 0:
                code_0_embed = self.model.talker.get_input_embeddings()(
                    next_token_batch
                )
                code_input = mx.concatenate([code_hidden, code_0_embed], axis=1)
            else:
                code_embed = self.model.talker.code_predictor.codec_embedding[
                    code_idx - 1
                ](code_tokens[-1])
                code_input = code_embed

            code_logits, code_cache, _ = self.model.talker.code_predictor(
                code_input,
                cache=code_cache,
                generation_step=code_idx,
            )
            next_code = self.model._sample_token_batch(
                code_logits,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
            )
            code_tokens.append(next_code)

        all_codes = mx.concatenate(code_tokens, axis=1)
        return next_token_batch, code_tokens, all_codes

    def _next_input_embeds_for_batch(
        self,
        trailing_text_hidden: mx.array,
        tts_pad_embed: mx.array,
        trailing_indices: mx.array,
        code_tokens: List[mx.array],
    ) -> mx.array:
        batch_size = trailing_text_hidden.shape[0]
        max_trailing_len = trailing_text_hidden.shape[1]
        batch_arange = mx.arange(batch_size)
        clamped_indices = mx.minimum(trailing_indices[:, 0], max_trailing_len - 1)
        text_embeds = trailing_text_hidden[batch_arange, clamped_indices, :][
            :, None, :
        ]

        exhausted = trailing_indices[:, 0] >= max_trailing_len
        pad_broadcast = mx.broadcast_to(tts_pad_embed, text_embeds.shape)
        text_embeds = mx.where(exhausted[:, None, None], pad_broadcast, text_embeds)
        return text_embeds + self._codec_embeds_for_tokens(code_tokens)

    def _next_input_embeds_for_states(
        self,
        states: List[_ActiveRequest],
        code_tokens: List[mx.array],
    ) -> mx.array:
        text_embeds = []
        for state in states:
            if state.trailing_idx < state.trailing_text_hidden.shape[1]:
                text_embeds.append(
                    state.trailing_text_hidden[
                        :, state.trailing_idx : state.trailing_idx + 1, :
                    ]
                )
            else:
                text_embeds.append(state.tts_pad_embed)

        return mx.concatenate(text_embeds, axis=0) + self._codec_embeds_for_tokens(
            code_tokens
        )

    def _codec_embeds_for_tokens(self, code_tokens: List[mx.array]) -> mx.array:
        codec_embed = self.model.talker.get_input_embeddings()(code_tokens[0])
        for index, code in enumerate(code_tokens[1:]):
            codec_embed = (
                codec_embed
                + self.model.talker.code_predictor.codec_embedding[index](code)
            )
        return codec_embed

    def _attention_mask_for_states(self, states: List[_ActiveRequest]) -> mx.array:
        cache_lengths = [
            (
                state.cache[0].offset
                if state.cache and state.cache[0].keys is not None
                else 0
            )
            for state in states
        ]
        max_cache_len = max(cache_lengths) if cache_lengths else 0
        mask_rows = []
        for cache_len in cache_lengths:
            left_pad = max_cache_len - cache_len
            mask_rows.append(
                mx.concatenate(
                    [
                        mx.zeros((1, left_pad)),
                        mx.ones((1, cache_len + 1)),
                    ],
                    axis=1,
                )
            )
        return mx.concatenate(mask_rows, axis=0)

    def _make_batch_cache(self, left_padding: List[int]) -> List[BatchKVCache]:
        return [
            BatchKVCache(left_padding)
            for _ in range(self.config.num_hidden_layers)
        ]

    def _merge_state_caches(
        self, states: List[_ActiveRequest]
    ) -> List[BatchKVCache]:
        return [
            KVCache.merge([state.cache[layer_idx] for state in states])
            for layer_idx in range(len(states[0].cache))
        ]

    def _extract_batch_cache_rows(
        self,
        cache: List[Any],
        batch_size: int,
    ) -> List[List[Any]]:
        rows: List[List[Any]] = [[] for _ in range(batch_size)]
        for layer_cache in cache:
            for index in range(batch_size):
                rows[index].append(layer_cache.extract(index))
        return rows

    def _decode_state(self, state: _ActiveRequest) -> BatchGenerationResult:
        if not state.generated_codes:
            return self._empty_result(state.sequence_idx)

        audio = self.model._decode_generated_codes(state.generated_codes)
        duration_seconds = audio.shape[0] / self.model.sample_rate
        return BatchGenerationResult(
            audio=audio,
            sequence_idx=state.sequence_idx,
            samples=audio.shape[0],
            sample_rate=self.model.sample_rate,
            token_count=len(state.generated_token_ids),
            audio_duration=_format_duration(duration_seconds),
            processing_time_seconds=time.time() - self._start_time,
            peak_memory_usage=mx.get_peak_memory() / 1e9,
        )

    def _empty_result(self, sequence_idx: int) -> BatchGenerationResult:
        audio = mx.zeros((0,), dtype=mx.float32)
        return BatchGenerationResult(
            audio=audio,
            sequence_idx=sequence_idx,
            samples=0,
            sample_rate=self.model.sample_rate,
            token_count=0,
            audio_duration=_format_duration(0.0),
            processing_time_seconds=time.time() - self._start_time,
            peak_memory_usage=mx.get_peak_memory() / 1e9,
        )
