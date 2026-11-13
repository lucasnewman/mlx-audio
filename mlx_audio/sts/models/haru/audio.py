"""Mono audio loading and voice-reference preparation."""

from pathlib import Path

import mlx.core as mx
import numpy as np


def load_audio(audio, target_rate=24000, sample_rate=None):
    from mlx_audio.utils import load_audio as load_file

    if isinstance(audio, (str, Path)):
        return np.asarray(
            load_file(str(audio), sample_rate=target_rate), dtype=np.float32
        )
    waveform = np.asarray(audio, dtype=np.float32)
    if waveform.ndim != 1:
        raise ValueError("Haru audio arrays must be mono [samples]")
    if sample_rate is not None and sample_rate != target_rate:
        from mlx_audio.resample import sinc_resample

        waveform = np.asarray(
            sinc_resample(mx.array(waveform), sample_rate, target_rate)
        )
    return waveform


def prepare_reference(audio, max_seconds=180.0, sample_rate=24000):
    """Trim silence with 20 ms RMS windows, then keep the end of the clip."""
    if max_seconds <= 0:
        raise ValueError("max_seconds must be positive")
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError("Voice references must be mono [samples]")
    window = sample_rate // 50
    count = len(audio) // window
    if count:
        rms = np.sqrt(
            np.mean(audio[: count * window].reshape(count, window) ** 2, axis=1) + 1e-12
        )
        voiced = np.flatnonzero(20 * np.log10(rms) > -45.0)
        if not len(voiced):
            return audio[:0]
        margin = int(0.2 * sample_rate)
        audio = audio[
            max(0, int(voiced[0]) * window - margin) : min(
                len(audio), (int(voiced[-1]) + 1) * window + margin
            )
        ]
    return audio[-int(max_seconds * sample_rate) :]
