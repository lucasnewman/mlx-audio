from functools import lru_cache
from typing import Union

import mlx.core as mx
import numpy as np

from mlx_audio.utils import stft, mel_filters, hanning


def dynamic_range_compression(x: mx.array, C: float = 1.0, clip_val: float = 1e-5) -> mx.array:
    return mx.log(mx.maximum(x, clip_val) * C)


def spectral_normalize(magnitudes: mx.array) -> mx.array:
    return dynamic_range_compression(magnitudes)


@lru_cache(maxsize=32)
def _get_mel_basis(sampling_rate: int, n_fft: int, num_mels: int, fmin: float, fmax: float) -> mx.array:
    return mel_filters(sample_rate=sampling_rate, n_fft=n_fft, n_mels=num_mels, f_min=fmin, f_max=fmax, norm="slaney", mel_scale="slaney")


@lru_cache(maxsize=8)
def _get_hann_window(win_size: int) -> mx.array:
    return hanning(win_size)


def mel_spectrogram(
    y: Union[mx.array, np.ndarray],
    n_fft: int = 1920,
    num_mels: int = 80,
    sampling_rate: int = 24000,
    hop_size: int = 480,
    win_size: int = 1920,
    fmin: float = 0,
    fmax: float = 8000,
    center: bool = False,
) -> mx.array:
    """Compute mel spectrogram compatible with Matcha-TTS.

    Ported from https://github.com/shivammehta25/Matcha-TTS/blob/main/matcha/utils/audio.py
    with default values according to Cosyvoice's config.

    Args:
        y: Input waveform, shape (batch, time) or (time,)
        n_fft: FFT size
        num_mels: Number of mel bins
        sampling_rate: Sample rate
        hop_size: Hop size
        win_size: Window size
        fmin: Minimum frequency
        fmax: Maximum frequency
        center: Whether to center the frames

    Returns:
        Mel spectrogram, shape (batch, num_mels, time)
    """
    if isinstance(y, np.ndarray):
        y = mx.array(y, dtype=mx.float32)

    if y.ndim == 1:
        y = y[None, :]

    y_min = mx.min(y)
    y_max = mx.max(y)
    if y_min < -1.0:
        print(f"min value is {y_min}")
    if y_max > 1.0:
        print(f"max value is {y_max}")

    mel_basis = _get_mel_basis(sampling_rate, n_fft, num_mels, fmin, fmax)
    window = _get_hann_window(win_size)

    pad_amount = int((n_fft - hop_size) / 2)
    if pad_amount > 0:
        batch_size = y.shape[0]
        padded_y = []

        for i in range(batch_size):
            y_i = y[i]
            left_pad = y_i[1 : pad_amount + 1][::-1]
            right_pad = y_i[-(pad_amount + 1) : -1][::-1] if pad_amount > 1 else y_i[-2:-1]
            y_padded = mx.concatenate([left_pad, y_i, right_pad])
            padded_y.append(y_padded)

        y = mx.stack(padded_y)

    batch_size = y.shape[0]
    specs = []

    for i in range(batch_size):
        spec = stft(y[i], n_fft=n_fft, hop_length=hop_size, win_length=win_size, window=window, center=center, pad_mode="reflect")
        spec_mag = mx.abs(spec)
        mel_spec = mel_basis @ spec_mag
        specs.append(mel_spec)

    spec = mx.stack(specs)
    spec = spectral_normalize(spec)

    return spec
