from __future__ import annotations

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
