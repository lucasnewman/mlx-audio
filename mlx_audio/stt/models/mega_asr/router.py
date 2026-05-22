from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.dsp import hanning, mel_filters, stft


class LogMel80(nn.Module):
    def __init__(self):
        super().__init__()
        self.sample_rate = 16000
        self.n_fft = 400
        self.hop_length = 160
        self.win_length = 400
        self.window = hanning(self.win_length, periodic=True)
        self.filters = mel_filters(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=80,
            f_min=0.0,
            f_max=8000.0,
            norm="slaney",
            mel_scale="slaney",
        )

    def __call__(self, waveform: mx.array) -> mx.array:
        if waveform.ndim != 1:
            waveform = mx.squeeze(waveform)

        spec = stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            pad_mode="reflect",
        )
        power = mx.abs(spec) ** 2.0
        mel = power @ self.filters.T
        log_mel = mx.log10(mx.maximum(mel, 1e-10))
        return (log_mel + 4.0) / 4.0


class BatchNorm1d(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((num_features,))
        self.bias = mx.zeros((num_features,))
        self.running_mean = mx.zeros((num_features,))
        self.running_var = mx.ones((num_features,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return (x - self.running_mean) / mx.sqrt(
            self.running_var + self.eps
        ) * self.weight + self.bias


class ConvFrontend(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(80, 128, kernel_size=3, stride=2, padding=1)
        self.bn1 = BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn2 = BatchNorm1d(256)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.gelu(self.bn1(self.conv1(x)))
        x = nn.gelu(self.bn2(self.conv2(x)))
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 850):
        super().__init__()
        position = mx.arange(max_len, dtype=mx.float32)[:, None]
        div_term = mx.exp(
            mx.arange(0, d_model, 2, dtype=mx.float32) * (-math.log(10000.0) / d_model)
        )
        pe = mx.zeros((1, max_len, d_model), dtype=mx.float32)
        pe[:, :, 0::2] = mx.sin(position * div_term)
        pe[:, :, 1::2] = mx.cos(position * div_term)
        self.pe = pe

    def __call__(self, x: mx.array) -> mx.array:
        return x + self.pe[:, : x.shape[1]]
