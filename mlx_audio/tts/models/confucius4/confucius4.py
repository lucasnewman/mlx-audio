# Copyright (c) 2026 Hert4 (https://github.com/Hert4)
# MLX port of Confucius4-TTS (netease-youdao, Apache-2.0).
# Licensed under the Apache License, Version 2.0.
"""Confucius4-TTS: multilingual, cross-lingual, zero-shot voice-cloning TTS in MLX.

Pipeline: w2v-bert semantic features + CAMPPlus speaker emb -> T2S (GPT-2) ->
S2A flow-matching (DiT+WaveNet) -> BigVGAN vocoder. All MLX; frontend DSP numpy.
"""
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_audio.tts.models.base import BaseModelArgs, GenerationResult
from mlx_audio.tts.models.chatterbox.s3gen.xvector import CAMPPlus

from .prefix import T2SPrefixMLX
from .s2a import S2AEstimator
from .t2s import T2SMLX
from .vocoder import BigVGANMLX
from .w2vbert import W2VBertMLX

LANGUAGE_TOKEN = {  # subset; matches Confucius LANGUAGE_TOKEN_MAP
    "zh": "请用中文朗读接下来的文字",
    "en": "请用英文朗读接下来的文字",
    "vi": "请用越南语朗读接下来的文字",
    "ja": "请用日语朗读接下来的文字",
    "ko": "请用韩语朗读接下来的文字",
    "th": "请用泰语朗读接下来的文字",
}


@dataclass
class ModelConfig(BaseModelArgs):
    model_path: str = ""
    sample_rate: int = 22050
    model_type: str = "confucius4"
    quant_bits: int = 8  # bits used if T2S weights are quantized (int8/int4)
    quant_group_size: int = 64


def _ref_mel(audio16k):
    import librosa

    SR, NFFT, HOP, WIN = 22050, 1024, 256, 1024
    a = librosa.resample(audio16k, orig_sr=16000, target_sr=SR)
    mb = librosa.filters.mel(sr=SR, n_fft=NFFT, n_mels=80, fmin=0, fmax=None)
    hann = np.hanning(WIN + 1)[:-1].astype(np.float32)
    pad = (NFFT - HOP) // 2
    y = np.pad(a, (pad, pad), mode="reflect")
    nfr = 1 + (len(y) - NFFT) // HOP
    fr = np.stack([y[i * HOP : i * HOP + NFFT] * hann for i in range(nfr)], 0)
    spec = np.sqrt(np.abs(np.fft.rfft(fr, NFFT, axis=1)).T ** 2 + 1e-9)
    return np.log(np.clip(mb @ spec, 1e-5, None)).T[None].astype(np.float32)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.sample_rate = config.sample_rate
        d = Path(config.model_path)
        self.w2v = W2VBertMLX(
            str(d / "w2vbert_mlx.safetensors"),
            group_size=config.quant_group_size,
            bits=config.quant_bits,
        )
        self.prefix = T2SPrefixMLX(str(d / "t2s_model.safetensors"))
        self.t2s = T2SMLX(
            str(d / "t2s_model.safetensors"),
            group_size=config.quant_group_size,
            bits=config.quant_bits,
        )
        self.s2a = S2AEstimator(str(d / "s2a_mlx.safetensors"))
        self.voc = BigVGANMLX(str(d / "bigvgan_mlx.safetensors"))
        self.stats = np.load(str(d / "w2v_stats.npz"))
        from mlx.utils import tree_unflatten

        self.camp = CAMPPlus(feat_dim=80, embedding_size=192)
        self.camp.update(
            tree_unflatten(list(mx.load(str(d / "campplus.safetensors")).items()))
        )
        mx.eval(self.camp.parameters())
        self.camp.eval()
        # torch-free preprocessing: numpy fbank + `tokenizers` (Rust) tokenizer
        from tokenizers import Tokenizer

        from .features import fbank_160

        self._fbank = fbank_160
        ff = np.load(str(d / "fbank_filters.npz"))
        self._mel, self._win = ff["mel"], ff["window"]
        self._tok = Tokenizer.from_file(str(d / "checkpoints" / "tokenizer.json"))

    def sanitize(self, weights):
        return {}  # sub-weights are loaded from the model dir in __init__

    def load_weights(self, *args, **kwargs):
        # Components self-load from the model dir in __init__; the standard
        # tree loader is bypassed. No-op keeps mlx_audio.tts.utils.load() happy.
        return self

    def generate(
        self,
        text: str,
        ref_audio: str,
        lang: str = "vi",
        temperature: float = 0.8,
        top_k: int = 30,
        top_p: float = 0.8,
        repetition_penalty: float = 10.0,
        seed: int = 0,
        **kwargs,
    ):
        import soundfile as sf

        t0 = time.time()
        audio, sr = sf.read(ref_audio)
        audio = audio.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.mean(1)

        feats = self._fbank(audio, self._mel, self._win)
        h17 = np.array(self.w2v.hidden17(mx.array(feats)))
        cond_vec = mx.array((h17 - self.stats["mean"]) / self.stats["std"])
        style = mx.array(np.array(self.camp.inference(mx.array(audio))).reshape(1, 192))
        ref_mel = mx.array(_ref_mel(audio))

        lt = LANGUAGE_TOKEN.get(lang, LANGUAGE_TOKEN["en"])
        ids = self._tok.encode(f"You are a helpful assistant. {lt}:{text}").ids
        cond_emb = self.prefix.cond_emb(cond_vec)
        text_emb = self.prefix.text_emb(mx.array([ids]))
        codes, latent = self.t2s.generate(
            cond_emb,
            text_emb,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            rep_pen=repetition_penalty,
            seed=seed,
        )

        T_ref = ref_mel.shape[1]
        mu = self.s2a.build_mu(mx.array(codes[None]), mx.array(latent), T_ref)
        mx.random.seed(seed)
        z = mx.random.normal((1, 80, mu.shape[1]))
        mel = self.s2a.solve_euler(
            z,
            mx.transpose(ref_mel, (0, 2, 1)),
            mu,
            style,
            mx.linspace(0, 1, 26),
            cfg=0.7,
        )[:, :, T_ref:]
        wav = self.voc(mel)
        mx.eval(wav)
        wav = mx.array(np.array(wav).reshape(-1))

        samples = wav.shape[0]
        dt = time.time() - t0
        dur = samples / self.sample_rate
        yield GenerationResult(
            audio=wav,
            samples=samples,
            sample_rate=self.sample_rate,
            segment_idx=0,
            token_count=len(codes),
            audio_duration=f"{dur:.2f}s",
            real_time_factor=round(dt / dur, 2) if dur else 0.0,
            prompt={"tokens": len(codes)},
            audio_samples={"samples": samples},
            processing_time_seconds=round(dt, 2),
            peak_memory_usage=0.0,
            is_final_chunk=True,
        )
