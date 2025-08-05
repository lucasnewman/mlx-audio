from functools import lru_cache

import mlx.core as mx
import numpy as np
from scipy import signal

from mlx_audio.utils import stft, mel_filters


@lru_cache()
def mel_basis(hp):
    assert hp.fmax <= hp.sample_rate // 2
    return mel_filters(
        sample_rate=hp.sample_rate, n_fft=hp.n_fft, n_mels=hp.num_mels, f_min=hp.fmin, f_max=hp.fmax, norm="slaney", mel_scale="htk"
    )


def preemphasis(wav, hp):
    assert hp.preemphasis != 0
    if isinstance(wav, mx.array):
        wav_np = np.array(wav)
    else:
        wav_np = wav

    # Apply preemphasis filter
    wav_np = signal.lfilter([1, -hp.preemphasis], [1], wav_np)
    wav_np = np.clip(wav_np, -1, 1)

    return mx.array(wav_np)


def melspectrogram(wav, hp, pad=True):
    if not isinstance(wav, mx.array):
        wav = mx.array(wav)

    # Run through pre-emphasis
    if hp.preemphasis > 0:
        wav = preemphasis(wav, hp)

    assert mx.abs(wav).max() - 1 < 1e-07

    # Do the stft
    spec_complex = _stft(wav, hp, pad=pad)

    # Get the magnitudes
    spec_magnitudes = mx.abs(spec_complex)

    if hp.mel_power != 1.0:
        spec_magnitudes = mx.power(spec_magnitudes, hp.mel_power)

    # Get the mel and convert magnitudes->db
    mel = mx.matmul(mel_basis(hp), spec_magnitudes)

    if hp.mel_type == "db":
        mel = _amp_to_db(mel, hp)

    # Normalise the mel from db to 0,1
    if hp.normalized_mels:
        mel = _normalize(mel, hp).astype(mx.float32)

    assert not pad or mel.shape[1] == 1 + len(wav) // hp.hop_size  # Sanity check

    return mel  # (M, T)


def _stft(y, hp, pad=True):
    return stft(
        y,
        n_fft=hp.n_fft,
        hop_length=hp.hop_size,
        win_length=hp.win_size,
        center=pad,
        pad_mode="reflect",
    )


def _amp_to_db(x, hp):
    return 20 * mx.log10(mx.maximum(mx.array(hp.stft_magnitude_min), x))


def _db_to_amp(x):
    return mx.power(10.0, x * 0.05)


def _normalize(s, hp, headroom_db=15):
    min_level_db = 20 * mx.log10(mx.array(hp.stft_magnitude_min))
    s = (s - min_level_db) / (-min_level_db + headroom_db)
    return s
