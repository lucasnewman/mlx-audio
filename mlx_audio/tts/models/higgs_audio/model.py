"""Higgs Audio v2 — framework-conforming entry point.

Defines `Model` and `ModelConfig` in the shape expected by
`mlx_audio.tts.utils.load()` and the top-level
`python -m mlx_audio.tts.generate` CLI.

`Model` subclasses `HiggsAudioModel` so the safetensors checkpoint loads
directly against it (no key remapping). Tokenizer + codec are attached
via `post_load_hook`; `generate(text, voice=..., ref_audio=..., ref_text=...)`
matches the framework's standard signature and yields a
`mlx_audio.tts.models.base.GenerationResult`.

The richer, kwarg-style API (`HiggsAudioServer.from_pretrained(...).generate(target_text=...)`)
in `serve.py` is preserved as an additional Python entrypoint.
"""

from __future__ import annotations

import time
from typing import Iterator, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import GenerationResult
from .config import HiggsAudioConfig
from .higgs_audio import HiggsAudioModel
from .serve import build_prompt

# Framework loader reads `module.ModelConfig.from_dict(config_json)`; aliasing
# HiggsAudioConfig keeps a single source of truth.
ModelConfig = HiggsAudioConfig


_DEFAULT_CODEC_REPO = "mlx-community/higgs-audio-v2-tokenizer"


def _format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


class Model(HiggsAudioModel):
    """Framework-conforming wrapper around `HiggsAudioModel`.

    Inherits the full model structure so the checkpoint loads without any
    key remapping. Adds a tokenizer + codec (populated via `post_load_hook`)
    and a `generate()` method matching the standard TTS interface.
    """

    def __init__(self, config: HiggsAudioConfig):
        super().__init__(config)
        self._config = config
        self._tokenizer = None
        self._codec = None
        self._sample_rate = 24000

    # --- framework hooks ------------------------------------------------

    def model_quant_predicate(self, name: str, module: nn.Module) -> bool:
        """Quantize everything except the audio head + codebook embeddings.

        These two components are most sensitive to quantization noise: q4/q6
        pushed the audio head to collapse to stream-EOS or drift a semitone.
        Keeping them at bf16 preserves voice character; the Llama backbone
        and text head compress cleanly.
        """
        if not isinstance(module, (nn.Linear, nn.Embedding)):
            return False
        protected = ("audio_codebook_embeddings", "audio_decoder_proj.audio_lm_head")
        return not any(p in name for p in protected)

    @classmethod
    def post_load_hook(cls, model: "Model", model_path) -> "Model":
        """Attach the HF text tokenizer (from model_path) and the Higgs codec
        (from the default mlx-community repo, overridable post-hoc via
        `model.codec = ...`)."""
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer

        from ....codec.models.higgs_audio.higgs_audio import HiggsAudioTokenizer

        model._tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        codec_dir = snapshot_download(repo_id=_DEFAULT_CODEC_REPO)
        model._codec = HiggsAudioTokenizer.from_pretrained(codec_dir)
        return model

    # --- accessors ------------------------------------------------------

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def tokenizer(self):
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value) -> None:
        self._tokenizer = value

    @property
    def codec(self):
        return self._codec

    @codec.setter
    def codec(self, value) -> None:
        self._codec = value

    # --- generation -----------------------------------------------------

    def generate(  # type: ignore[override]
        self,
        text: str,
        voice: Optional[str] = None,
        ref_audio=None,
        ref_text: Optional[str] = None,
        max_new_frames: int = 1200,
        temperature: float = 0.7,
        top_p: Optional[float] = 0.95,
        top_k: Optional[int] = None,
        ras_win_len: Optional[int] = 7,
        ras_max_repeat: int = 2,
        sampling_warmup_frames: int = 0,
        fade_in_ms: float = 30.0,
        fade_out_ms: float = 15.0,
        verbose: bool = False,
        **kwargs,  # absorb framework kwargs we don't use (speed, lang_code, ...)
    ) -> Iterator[GenerationResult]:
        """Synthesize `text` and yield a single `GenerationResult`.

        Voice cloning: pass `ref_audio` (mono 24 kHz mx.array or numpy array,
        loaded by the top-level CLI from `--ref_audio`) together with
        `ref_text` (its transcript). Without `ref_audio`, runs smart-voice
        mode — works but less reliable; recommended to always supply a
        reference for production use.
        """
        if self._tokenizer is None or self._codec is None:
            raise RuntimeError(
                "Model not fully loaded — tokenizer/codec missing. Load via "
                "mlx_audio.tts.utils.load(...) so post_load_hook runs."
            )

        start = time.perf_counter()

        ref_audio_24k = None
        if ref_audio is not None:
            ref_audio_24k = np.asarray(ref_audio, dtype=np.float32).reshape(-1)

        full_embeds, audio_out_mask, _info = build_prompt(
            text,
            ref_text=ref_text,
            ref_audio_24k=ref_audio_24k,
            config=self._config,
            tokenizer=self._tokenizer,
            codec=self._codec,
            embed_tokens=self.embed_tokens,
            audio_codebook_embeddings=self.audio_codebook_embeddings,
        )

        # Call the inherited HiggsAudioModel.generate (delay-pattern state machine).
        aligned, gen_info = HiggsAudioModel.generate(
            self,
            inputs_embeds=full_embeds,
            audio_out_mask=audio_out_mask,
            max_new_frames=max_new_frames,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            ras_win_len=ras_win_len,
            ras_max_repeat=ras_max_repeat,
            sampling_warmup_frames=sampling_warmup_frames,
        )
        mx.eval(aligned)

        pcm = self._codec.decode(aligned.T)
        mx.eval(pcm)
        pcm_np = np.array(pcm).astype(np.float32).reshape(-1)

        sr = self._sample_rate
        n_in = int(fade_in_ms * sr / 1000.0)
        n_out = int(fade_out_ms * sr / 1000.0)
        if n_in > 0 and pcm_np.size > n_in:
            pcm_np[:n_in] *= np.linspace(0.0, 1.0, n_in, dtype=np.float32)
        if n_out > 0 and pcm_np.size > n_out:
            pcm_np[-n_out:] *= np.linspace(1.0, 0.0, n_out, dtype=np.float32)

        audio_mx = mx.array(pcm_np)
        samples = audio_mx.shape[0]
        elapsed = time.perf_counter() - start
        duration_s = samples / sr
        rtf = elapsed / duration_s if duration_s > 0 else 0.0
        tok = gen_info["num_frames_aligned"]

        yield GenerationResult(
            audio=audio_mx,
            samples=samples,
            sample_rate=sr,
            segment_idx=0,
            token_count=tok,
            audio_duration=_format_duration(duration_s),
            real_time_factor=round(rtf, 3),
            prompt={
                "tokens": tok,
                "tokens-per-sec": round(tok / elapsed, 2) if elapsed > 0 else 0.0,
            },
            audio_samples={
                "samples": samples,
                "samples-per-sec": round(samples / elapsed, 2) if elapsed > 0 else 0.0,
            },
            processing_time_seconds=elapsed,
            peak_memory_usage=mx.get_peak_memory() / 1e9,
        )
