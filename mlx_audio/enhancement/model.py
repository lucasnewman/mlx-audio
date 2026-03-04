"""DeepFilterNet MLX model runtime.

This module implements the full DeepFilterNet inference pipeline in MLX:
- STFT/ISTFT with Vorbis window
- ERB + DF feature extraction and normalization
- Neural network inference
- Spectral reconstruction and delay compensation
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional, Union

import mlx.core as mx
import numpy as np

from mlx_audio import audio_io
from mlx_audio.dsp import istft, stft

from .config import DeepFilterNet2Config, DeepFilterNet3Config, DeepFilterNetConfig
from .network import DfNet
from .weight_loader import load_weights as load_df_weights

DEFAULT_MODEL_DIRS = {
    1: [
        "/Users/kylehowells/Developer/Example-Projects/mlx-audio/models/DeepFilterNet1",
        "/Users/kylehowells/Developer/Example-Projects/DeepFilterNet/DeepFilterNet/models/extracted/DeepFilterNet",
    ],
    2: [
        "/Users/kylehowells/Developer/Example-Projects/mlx-audio/models/DeepFilterNet2",
        "/Users/kylehowells/Developer/Example-Projects/DeepFilterNet2/DeepFilterNet2/models/DeepFilterNet2",
    ],
    3: [
        "/Users/kylehowells/Developer/Example-Projects/mlx-audio/models/DeepFilterNet3",
        "/Users/kylehowells/Developer/Example-Projects/DeepFilterNet/DeepFilterNet/models/extracted/DeepFilterNet3",
    ],
}

DEFAULT_CONFIGS = {
    "DeepFilterNet": DeepFilterNetConfig,
    "DeepFilterNet2": DeepFilterNet2Config,
    "DeepFilterNet3": DeepFilterNet3Config,
}


def resolve_model_dir(version: int, model_dir: Optional[Union[str, Path]] = None) -> Path:
    if model_dir is not None:
        p = Path(model_dir).expanduser().resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"Model directory not found: {p}")

    for cand in DEFAULT_MODEL_DIRS.get(version, []):
        p = Path(cand)
        if p.exists():
            return p

    raise FileNotFoundError(
        f"No local model directory found for DeepFilterNet{version}. "
        f"Tried: {DEFAULT_MODEL_DIRS.get(version, [])}"
    )


class DeepFilterNetModel:
    """Pure-MLX DeepFilterNet inference runtime."""

    def __init__(self, model: DfNet, config: DeepFilterNetConfig, model_dir: Path):
        self.model = model
        self.config = config
        self.model_dir = model_dir

        # From libDF: wnorm = 1 / (fft_size^2 / (2 * hop_size))
        self.wnorm = 1.0 / (config.fft_size * config.fft_size / (2.0 * config.hop_size))

        self._vorbis = self._vorbis_window(config.fft_size)

        # Filterbanks loaded from model weights via weight_loader fallback
        self.erb_fb = self.model.erb_fb
        self.erb_inv_fb = self.model.mask.erb_inv_fb

    @classmethod
    def from_pretrained(
        cls,
        version: int = 3,
        model_dir: Optional[Union[str, Path]] = None,
    ) -> "DeepFilterNetModel":
        model_dir_resolved = resolve_model_dir(version=version, model_dir=model_dir)

        config_path = model_dir_resolved / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config.json in model directory: {model_dir_resolved}")

        with open(config_path, encoding="utf-8") as f:
            config_dict = json.load(f)

        model_version = config_dict.get("model_version", f"DeepFilterNet{version}")
        config_class = DEFAULT_CONFIGS.get(model_version, DeepFilterNetConfig)
        config = config_class.from_dict(config_dict)

        model = DfNet(config)

        weights_path = model_dir_resolved / "model.safetensors"
        if not weights_path.exists():
            npz_fallback = model_dir_resolved / "weights.npz"
            if npz_fallback.exists():
                weights_path = npz_fallback
            else:
                raise FileNotFoundError(f"Missing model weights in {model_dir_resolved}")

        weights = mx.load(str(weights_path))
        loaded = load_df_weights(model, weights)
        print(f"Loaded DeepFilterNet weights: {loaded} tensors from {weights_path}")

        return cls(model=model, config=config, model_dir=model_dir_resolved)

    def enhance_file(self, input_path: Union[str, Path], output_path: Union[str, Path]) -> Path:
        input_path = Path(input_path)
        output_path = Path(output_path)

        audio, sr = audio_io.read(str(input_path), always_2d=False, dtype="float32")
        if sr != self.config.sample_rate:
            raise ValueError(
                f"Expected {self.config.sample_rate} Hz audio, got {sr} Hz: {input_path}"
            )

        if audio.ndim > 1:
            audio = audio[:, 0]

        enhanced = self.enhance_array(audio)
        audio_io.write(str(output_path), enhanced, self.config.sample_rate)
        return output_path

    def enhance_array(self, audio: np.ndarray) -> np.ndarray:
        x = mx.array(audio.astype(np.float32))
        p = self.config

        orig_len = x.shape[0]

        # Match DeepFilterNet inference path: end-pad by fft size
        x = mx.pad(x, [(0, p.fft_size)])

        # STFT [T, F], no center padding
        spec = stft(
            x,
            n_fft=p.fft_size,
            hop_length=p.hop_size,
            win_length=p.fft_size,
            window=self._vorbis,
            center=False,
        )

        # Match libDF analysis normalization
        spec = spec * self.wnorm

        alpha = self._norm_alpha()

        # ERB features
        spec_mag_sq = (mx.real(spec) ** 2 + mx.imag(spec) ** 2)  # [T, F]
        erb = spec_mag_sq @ self.erb_fb  # [T, E]
        erb_db = 10.0 * mx.log10(mx.maximum(erb, 1e-10))
        erb_norm = self._band_mean_norm(erb_db, alpha, p.nb_erb)
        feat_erb = erb_norm[None, None, :, :]  # [B, 1, T, E]

        # DF complex features
        df_spec = spec[:, : p.nb_df]  # [T, D]
        df_re, df_im = self._band_unit_norm(df_spec, alpha, p.nb_df)
        feat_df = mx.stack([df_re, df_im], axis=-1)[None, None, :, :, :]  # [B, 1, T, D, 2]

        spec_in = mx.stack([mx.real(spec), mx.imag(spec)], axis=-1)[None, None, :, :, :]

        spec_e, _mask, _lsnr, _df_coefs = self.model(spec_in, feat_erb, feat_df)

        enh = spec_e[0, 0]  # [T, F, 2]
        enh_complex = enh[..., 0] + 1j * enh[..., 1]  # [T, F]

        # Undo libDF normalization for ISTFT implementation
        enh_complex = enh_complex / self.wnorm

        # istft expects [F, T]
        enh_complex = mx.transpose(enh_complex, (1, 0))
        audio_out = istft(
            enh_complex,
            hop_length=p.hop_size,
            win_length=p.fft_size,
            window=self._vorbis,
            center=False,
            length=orig_len + p.fft_size,
            normalized=False,
        )

        # `mlx_audio.dsp.istft` alignment differs from libDF's synthesis by one hop.
        # For DeepFilterNet configs (hop = fft/2), this yields zero net delay here.
        d = max(0, p.fft_size - 2 * p.hop_size)
        audio_out = audio_out[d : orig_len + d]

        y = np.array(audio_out, dtype=np.float32)
        return np.clip(y, -1.0, 1.0)

    def _norm_alpha(self) -> float:
        # Same as DeepFilterNet defaults (tau=1.0)
        return math.exp(-self.config.hop_size / self.config.sample_rate)

    def _band_mean_norm(self, x: mx.array, alpha: float, nb_bands: int) -> mx.array:
        # libDF init state: linearly spaced [-60, -90]
        state = mx.linspace(-60.0, -90.0, nb_bands)
        outs = []
        for i in range(x.shape[0]):
            state = x[i] * (1.0 - alpha) + state * alpha
            outs.append((x[i] - state) / 40.0)
        return mx.stack(outs, axis=0)

    def _band_unit_norm(self, x: mx.array, alpha: float, nb_freqs: int):
        # libDF init state: linearly spaced [0.001, 0.0001]
        state = mx.linspace(0.001, 0.0001, nb_freqs)
        mag = mx.abs(x)

        outs_r = []
        outs_i = []
        for i in range(x.shape[0]):
            state = mag[i] * (1.0 - alpha) + state * alpha
            denom = mx.sqrt(state + 1e-8)
            outs_r.append(mx.real(x[i]) / denom)
            outs_i.append(mx.imag(x[i]) / denom)

        return mx.stack(outs_r, axis=0), mx.stack(outs_i, axis=0)

    @staticmethod
    def _vorbis_window(size: int) -> mx.array:
        # libDF Vorbis window:
        # sin(0.5*pi*sin(0.5*pi*(n+0.5)/(N/2))^2)
        n = np.arange(size, dtype=np.float32)
        n_half = size // 2
        inner = np.sin(0.5 * np.pi * (n + 0.5) / n_half)
        window = np.sin(0.5 * np.pi * inner * inner)
        return mx.array(window.astype(np.float32))
