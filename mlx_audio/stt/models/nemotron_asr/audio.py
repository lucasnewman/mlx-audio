"""Log-mel spectrogram preprocessing matching NeMo's AudioToMelSpectrogramPreprocessor.

Differences from the Parakeet preprocessor:
  * 128 mel bins (vs 80).
  * ``normalize: NA`` => the features are NOT normalized (NeMo's normalize_batch
    falls through and returns the input unchanged for an unrecognized type).

Everything else matches NeMo at inference time: preemphasis 0.97, symmetric Hann
window, power spectrogram (mag_power=2.0), Slaney mel filters, log with a small
additive guard, and no dither (dither is training-only in NeMo).
"""

import mlx.core as mx

from mlx_audio.utils import STR_TO_WINDOW_FN, hanning, mel_filters, stft

from .config import PreprocessArgs


def log_mel_spectrogram(x: mx.array, args: PreprocessArgs) -> mx.array:
    """Compute (1, T, features) log-mel features for a mono waveform ``x`` (samples,)."""
    original_dtype = x.dtype

    if args.pad_to > 0 and x.shape[-1] < args.pad_to:
        pad_length = args.pad_to - x.shape[-1]
        x = mx.pad(x, ((0, pad_length),), constant_values=args.pad_value)

    window_fn = STR_TO_WINDOW_FN.get(args.window, None)
    window = window_fn(args.win_length) if window_fn else hanning(args.win_length)

    # Preemphasis high-pass filter: y[n] = x[n] - alpha * x[n-1] (NeMo default 0.97).
    if args.preemph and args.preemph > 0:
        x = mx.concat([x[:1], x[1:] - args.preemph * x[:-1]], axis=0)

    # Center-pad the window to n_fft (matches torch.stft / NeMo).
    if window.shape[0] < args.n_fft:
        left = (args.n_fft - window.shape[0]) // 2
        right = args.n_fft - window.shape[0] - left
        window = mx.concatenate(
            [
                mx.zeros((left,), dtype=window.dtype),
                window,
                mx.zeros((right,), dtype=window.dtype),
            ]
        )

    x = stft(x, args.n_fft, args.hop_length, args.n_fft, window, pad_mode="constant")
    # Power spectrum (mag_power = 2.0).
    x = mx.square(mx.abs(x)).astype(original_dtype)

    filters = mel_filters(
        args.sample_rate, args.n_fft, args.features, norm="slaney", mel_scale="slaney"
    )
    x = filters.astype(x.dtype) @ x.T

    log_guard = mx.array(args.log_zero_guard_value, dtype=x.dtype)
    x = mx.log(x + log_guard)

    # normalize: NeMo only normalizes for "per_feature" / "all_features". "NA" (and
    # anything else) is a no-op, which is what this model was trained with.
    if args.normalize == "per_feature":
        mean = mx.mean(x, axis=1, keepdims=True)
        n = max(x.shape[1] - 1, 1)
        variance = mx.sum((x - mean) ** 2, axis=1, keepdims=True) / n
        std = mx.sqrt(variance)
        x = (x - mean) / (std + 1e-5)
    elif args.normalize == "all_features":
        x = (x - mx.mean(x)) / (mx.std(x) + 1e-5)

    x = mx.expand_dims(x.T, axis=0)
    return x.astype(original_dtype)
