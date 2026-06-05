from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterator, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.qwen3 import Qwen3Model

from ..base import GenerationResult
from .config import HiggsAudioV3Config
from .generation import (
    HiggsSamplerState,
    apply_delay_pattern,
    reverse_delay_pattern,
    step,
)
from .prompt import AUDIO_PLACEHOLDER_ID, HiggsAudioV3PromptBuilder, ReferenceCodes

ModelConfig = HiggsAudioV3Config


def _format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


class Model(nn.Module):
    def __init__(self, config: HiggsAudioV3Config):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.backbone = Qwen3Model(config.text_config.to_qwen3_args())
        self.multimodal_embedding = nn.Embedding(
            config.audio_num_codebooks * config.audio_codebook_size,
            config.text_config.hidden_size,
        )
        self._tokenizer = None
        self._prompt_builder: Optional[HiggsAudioV3PromptBuilder] = None
        self._codec = None

    @property
    def layers(self):
        return self.backbone.layers

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def codec(self):
        return self._codec

    def __call__(
        self,
        input_ids: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        return self.backbone(input_ids, cache=cache, input_embeddings=input_embeddings)

    def model_quant_predicate(self, name: str, module: nn.Module) -> bool:
        if name.startswith("multimodal_embedding"):
            return False
        return isinstance(module, (nn.Linear, nn.Embedding))

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
        from tokenizers import Tokenizer
        from transformers import PreTrainedTokenizerFast

        from ....codec.models.higgs_audio.higgs_audio import HiggsAudioTokenizer

        raw = Tokenizer.from_file(str(Path(model_path) / "tokenizer.json"))
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=raw,
            eos_token="<|endoftext|>",
            pad_token="<|endoftext|>",
        )
        model._tokenizer = tokenizer
        model._prompt_builder = HiggsAudioV3PromptBuilder(tokenizer)

        audio_tokenizer_dir = Path(model_path) / "audio_tokenizer"
        if audio_tokenizer_dir.exists():
            model._codec = HiggsAudioTokenizer.from_pretrained(str(model_path))
        else:
            model._codec = HiggsAudioTokenizer.from_higgs_tts_checkpoint(
                str(model_path)
            )
        return model

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        result: dict[str, mx.array] = {}
        for key, value in weights.items():
            if key.startswith("tied.embedding.text_embedding."):
                result[
                    "backbone.embed_tokens."
                    + key[len("tied.embedding.text_embedding.") :]
                ] = value
            elif key.startswith("body.layers."):
                result["backbone.layers." + key[len("body.layers.") :]] = value
            elif key.startswith("body.norm."):
                result["backbone.norm." + key[len("body.norm.") :]] = value
            elif key.startswith("tied.embedding.modality_embeddings.0.embedding."):
                result[
                    "multimodal_embedding."
                    + key[len("tied.embedding.modality_embeddings.0.embedding.") :]
                ] = value
            elif key.startswith("tied.embedding.modality_embeddings.0.model."):
                continue
            elif key.startswith("tied.head."):
                continue
            else:
                result[key] = value
        return result

    def _embed_audio_codes(self, codes: mx.array) -> mx.array:
        if codes.ndim == 1:
            codes = codes[None, :]
        if codes.ndim != 2 or codes.shape[1] != self.config.audio_num_codebooks:
            raise ValueError(
                f"audio codes must be [T, {self.config.audio_num_codebooks}], "
                f"got {codes.shape}"
            )
        offsets = (
            mx.arange(self.config.audio_num_codebooks, dtype=mx.int32)
            * self.config.audio_codebook_size
        )
        fused_ids = codes.astype(mx.int32) + offsets
        return mx.sum(self.multimodal_embedding(fused_ids), axis=-2)

    def _audio_logits(self, hidden: mx.array) -> mx.array:
        flat = self.multimodal_embedding.as_linear(hidden)
        return flat.reshape(
            *hidden.shape[:-1],
            self.config.audio_num_codebooks,
            self.config.audio_codebook_size,
        )

    def _text_embeddings(self, token_ids: list[int]) -> mx.array:
        if not token_ids:
            return mx.zeros((0, self.config.text_config.hidden_size))
        ids = mx.array([token_ids], dtype=mx.int32)
        return self.backbone.embed_tokens(ids)[0]

    def _build_prompt_embeddings(
        self,
        text: str,
        references: list[ReferenceCodes],
    ) -> tuple[mx.array, int]:
        if self._prompt_builder is None:
            raise RuntimeError(
                "Tokenizer missing. Load via mlx_audio.tts.utils.load()."
            )

        prompt = self._prompt_builder.build_prompt(text, references=references)
        pieces = []
        cursor = 0
        for start, delayed_codes in prompt.audio_segments:
            pieces.append(self._text_embeddings(prompt.token_ids[cursor:start]))
            pieces.append(self._embed_audio_codes(delayed_codes))
            cursor = start + int(delayed_codes.shape[0])

        tail_ids = prompt.token_ids[cursor:]
        if any(token_id == AUDIO_PLACEHOLDER_ID for token_id in tail_ids):
            raise ValueError("Internal prompt error: unresolved audio placeholder")
        pieces.append(self._text_embeddings(tail_ids))

        full = mx.concatenate([piece for piece in pieces if piece.shape[0] > 0], axis=0)
        return full[None], len(prompt.token_ids)

    def _normalize_audio(self, audio: Any) -> np.ndarray:
        if isinstance(audio, (str, Path)):
            from mlx_audio.utils import load_audio

            audio = load_audio(str(audio), sample_rate=self.sample_rate)
        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim == 2:
            if arr.shape[0] <= 2:
                arr = arr.mean(axis=0)
            else:
                arr = arr.mean(axis=-1)
        return arr.reshape(-1).astype(np.float32, copy=False)

    def _encode_reference_audio(self, audio: Any) -> mx.array:
        if self._codec is None:
            raise RuntimeError("Codec missing. Load via mlx_audio.tts.utils.load().")
        audio_np = self._normalize_audio(audio)
        if audio_np.shape[0] < self.sample_rate:
            audio_np = np.pad(audio_np, (0, self.sample_rate - audio_np.shape[0]))
        waveform = mx.array(audio_np).reshape(1, -1, 1)
        codes = self._codec.encode(waveform)[0].astype(mx.int32)
        return apply_delay_pattern(
            codes,
            boc_id=self.config.audio_boc_token_id,
            eoc_id=self.config.audio_eoc_token_id,
        )

    def _normalize_references(
        self,
        ref_audio=None,
        ref_text=None,
        references=None,
        ref_audios=None,
        ref_texts=None,
    ) -> list[ReferenceCodes]:
        audios = _as_list(ref_audios if ref_audios is not None else ref_audio)
        texts = _as_list(ref_texts if ref_texts is not None else ref_text)

        if references:
            for item in references:
                if not isinstance(item, dict):
                    continue
                audio = None
                for key in ("audio", "audio_path", "path", "ref_audio"):
                    if key in item and item[key] is not None:
                        audio = item[key]
                        break
                if audio is None:
                    continue
                audios.append(audio)
                texts.append(item.get("text") or item.get("ref_text"))

        if not audios:
            return []
        if texts and len(texts) != len(audios):
            raise ValueError("ref_audio and ref_text lists must have the same length")
        if not texts:
            texts = [None] * len(audios)

        refs = []
        for audio, text in zip(audios, texts):
            refs.append(
                ReferenceCodes(
                    codes=self._encode_reference_audio(audio),
                    text=str(text) if text is not None else None,
                )
            )
        return refs

    def _decode_audio(self, delayed_rows: list[mx.array]) -> mx.array:
        if self._codec is None:
            raise RuntimeError("Codec missing. Load via mlx_audio.tts.utils.load().")
        if not delayed_rows:
            return mx.zeros((0,), dtype=mx.float32)
        delayed = mx.stack(delayed_rows, axis=0).astype(mx.int32)
        if delayed.shape[0] < self.config.audio_num_codebooks:
            return mx.zeros((0,), dtype=mx.float32)
        raw_codes = reverse_delay_pattern(delayed)
        audio = self._codec.decode(raw_codes)
        return audio.astype(mx.float32).reshape(-1)

    def generate(
        self,
        text: str,
        voice: Optional[str] = None,
        ref_audio=None,
        ref_text=None,
        references=None,
        ref_audios=None,
        ref_texts=None,
        max_new_tokens: Optional[int] = None,
        max_new_frames: Optional[int] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
        fade_in_ms: float = 30.0,
        fade_out_ms: float = 15.0,
        stream: bool = False,
        **kwargs,
    ) -> Iterator[GenerationResult]:
        del voice, kwargs
        if seed is not None:
            mx.random.seed(int(seed))

        limit = max_new_tokens
        if limit is None:
            limit = max_new_frames
        if limit is None:
            limit = max_tokens
        if limit is None:
            limit = 2048

        start = time.perf_counter()
        references_list = self._normalize_references(
            ref_audio=ref_audio,
            ref_text=ref_text,
            references=references,
            ref_audios=ref_audios,
            ref_texts=ref_texts,
        )
        prompt_embeds, prompt_tokens = self._build_prompt_embeddings(
            text, references_list
        )
        mx.eval(prompt_embeds)

        cache = make_prompt_cache(self)
        dummy = mx.zeros((1, prompt_embeds.shape[1]), dtype=mx.int32)
        hidden = self.backbone(dummy, cache=cache, input_embeddings=prompt_embeds)
        last_hidden = hidden[:, -1, :]

        state = HiggsSamplerState(num_codebooks=self.config.audio_num_codebooks)
        delayed_rows: list[mx.array] = []
        for _ in range(int(limit)):
            logits = self._audio_logits(last_hidden)[0]
            codes = step(
                logits,
                state,
                temperature=float(temperature),
                top_p=top_p,
                top_k=top_k,
                boc_id=self.config.audio_boc_token_id,
                eoc_id=self.config.audio_eoc_token_id,
            )
            delayed_rows.append(codes)
            if state.generation_done:
                break
            next_embed = self._embed_audio_codes(codes)[None]
            decode_dummy = mx.zeros((1, 1), dtype=mx.int32)
            hidden = self.backbone(
                decode_dummy,
                cache=cache,
                input_embeddings=next_embed,
            )
            last_hidden = hidden[:, -1, :]

        audio = self._decode_audio(delayed_rows)
        mx.eval(audio)
        audio_np = np.array(audio).astype(np.float32, copy=False)

        n_in = int(fade_in_ms * self.sample_rate / 1000.0)
        n_out = int(fade_out_ms * self.sample_rate / 1000.0)
        if n_in > 0 and audio_np.size > n_in:
            audio_np[:n_in] *= np.linspace(0.0, 1.0, n_in, dtype=np.float32)
        if n_out > 0 and audio_np.size > n_out:
            audio_np[-n_out:] *= np.linspace(1.0, 0.0, n_out, dtype=np.float32)

        elapsed = time.perf_counter() - start
        audio = mx.array(audio_np)
        samples = int(audio.shape[0])
        duration_s = samples / self.sample_rate if self.sample_rate else 0.0
        token_count = len(delayed_rows)
        yield GenerationResult(
            audio=audio,
            samples=samples,
            sample_rate=self.sample_rate,
            segment_idx=0,
            token_count=token_count,
            audio_duration=_format_duration(duration_s),
            real_time_factor=round(elapsed / duration_s, 3) if duration_s else 0.0,
            prompt={
                "tokens": prompt_tokens,
                "completion_tokens": token_count,
                "tokens-per-sec": (
                    round(token_count / elapsed, 2) if elapsed > 0 else 0.0
                ),
            },
            audio_samples={
                "samples": samples,
                "samples-per-sec": round(samples / elapsed, 2) if elapsed > 0 else 0.0,
            },
            processing_time_seconds=elapsed,
            peak_memory_usage=mx.get_peak_memory() / 1e9,
            is_streaming_chunk=bool(stream),
            is_final_chunk=bool(stream),
        )
