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


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 4):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def __call__(self, x: mx.array) -> mx.array:
        batch_size, steps, _ = x.shape
        q = self.q_proj(x).reshape(batch_size, steps, self.nhead, self.head_dim)
        k = self.k_proj(x).reshape(batch_size, steps, self.nhead, self.head_dim)
        v = self.v_proj(x).reshape(batch_size, steps, self.nhead, self.head_dim)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 3, 1)
        v = v.transpose(0, 2, 1, 3)

        attn = mx.softmax((q * self.scale) @ k, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(batch_size, steps, self.d_model)
        return self.out_proj(out)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 4, dim_feedforward: int = 1024):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadSelfAttention(d_model=d_model, nhead=nhead)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.self_attn(self.norm1(x))
        x = x + self.linear2(nn.gelu(self.linear1(self.norm2(x))))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 4, dim_feedforward: int = 1024, num_layers: int = 1):
        super().__init__()
        self.layers = [
            TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
            )
            for _ in range(num_layers)
        ]
        self.norm = nn.LayerNorm(d_model)

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class AttentionPooling(nn.Module):
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.query = nn.Linear(d_model, 1)

    def __call__(self, x: mx.array) -> mx.array:
        weights = mx.softmax(mx.squeeze(self.query(x), axis=-1), axis=-1)
        return mx.sum(weights[..., None] * x, axis=1)


class ClassifierHead(nn.Module):
    def __init__(self, d_model: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))
