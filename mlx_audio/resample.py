"""Audio resampling: a high-quality polyphase FIR, whole-buffer and streaming,
and a torchaudio-compatible sinc for models that depend on torchaudio's output."""

import math
from collections.abc import Iterable
from typing import Optional

import mlx.core as mx
import numpy as np
from scipy import signal


def _polyphase_filter(orig_sample_rate: int, sample_rate: int):
    """Build the shared high-quality polyphase resampling configuration."""
    gcd = math.gcd(int(orig_sample_rate), int(sample_rate))
    up = sample_rate // gcd
    down = orig_sample_rate // gcd

    # kaiser_best-equivalent anti-aliasing FIR (resampy defaults): a long,
    # high-attenuation Kaiser sinc designed at the upsampled rate. Cutoff is at
    # ``rolloff / max(up, down)`` of the upsampled Nyquist.
    max_rate = max(up, down)
    num_zeros, rolloff, beta = 64, 0.9475937167399596, 14.769656459379492
    fir = signal.firwin(
        2 * num_zeros * max_rate + 1,
        rolloff / max_rate,
        window=("kaiser", beta),
    )
    return up, down, fir


def resample_audio_array(
    audio: np.ndarray,
    orig_sample_rate: int,
    sample_rate: int,
    axis: int = -1,
) -> np.ndarray:
    """Resample an in-memory array through the high-quality polyphase FIR."""
    if orig_sample_rate == sample_rate:
        return audio

    up, down, fir = _polyphase_filter(orig_sample_rate, sample_rate)
    return signal.resample_poly(
        audio,
        up,
        down,
        axis=axis,
        window=fir,
        padtype="edge",
    ).astype(np.float32, copy=False)


def resample_audio_chunks(
    chunks: Iterable[np.ndarray],
    orig_sample_rate: int,
    sample_rate: int,
    num_input_frames: int,
    chunk_duration_seconds: float = 1.0,
    max_output_frames: Optional[int] = None,
) -> np.ndarray:
    """Resample time-first audio chunks without materializing the full input.

    Core boundaries and overlap are aligned to the reduced polyphase ratio so
    each retained output region has the same phase and samples as a single
    whole-buffer ``resample_poly`` call.

    ``max_output_frames`` keeps only that many output frames, and stops reading
    chunks once they are written.
    """
    if chunk_duration_seconds <= 0:
        raise ValueError("chunk_duration_seconds must be positive")

    chunk_iter = iter(chunks)
    first = None
    for chunk in chunk_iter:
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.shape[0] > 0:
            first = chunk
            break

    if first is None:
        return np.empty((0,), dtype=np.float32)

    trailing_shape = first.shape[1:]
    num_input_frames = max(0, int(num_input_frames))
    if num_input_frames == 0:
        return np.empty((0, *trailing_shape), dtype=np.float32)

    if orig_sample_rate == sample_rate:
        if max_output_frames is not None:
            num_input_frames = min(num_input_frames, max_output_frames)
        parts = [first]
        frames = first.shape[0]
        for chunk in chunk_iter:
            if frames >= num_input_frames:
                break
            parts.append(np.asarray(chunk, dtype=np.float32))
            frames += parts[-1].shape[0]
        return np.concatenate(parts, axis=0)[:num_input_frames]

    up, down, fir = _polyphase_filter(orig_sample_rate, sample_rate)
    core_frames = max(
        down,
        int(chunk_duration_seconds * orig_sample_rate) // down * down,
    )

    # resample_poly prepends at most ``down`` filter taps for centering. Round
    # the required input support up to a multiple of ``down`` so an overlapped
    # segment begins at the same polyphase phase as the full signal.
    half_filter_length = (len(fir) - 1) // 2
    input_support = math.ceil((half_filter_length + down) / up)
    halo_frames = math.ceil(input_support / down) * down

    num_output_frames = math.ceil(num_input_frames * up / down)
    if max_output_frames is not None:
        num_output_frames = min(num_output_frames, max_output_frames)
    output = np.empty((num_output_frames, *trailing_shape), dtype=np.float32)

    buffer = first
    buffer_start = 0
    core_start = 0
    output_end = 0
    reached_eof = False

    while core_start < num_input_frames and output_end < num_output_frames:
        core_end = min(num_input_frames, core_start + core_frames)
        required_end = min(num_input_frames, core_end + halo_frames)

        while buffer_start + buffer.shape[0] < required_end and not reached_eof:
            try:
                part = np.asarray(next(chunk_iter), dtype=np.float32)
            except StopIteration:
                reached_eof = True
                break
            if part.shape[0] == 0:
                continue
            if part.shape[1:] != trailing_shape:
                raise ValueError("all audio chunks must have matching shapes")
            buffer = np.concatenate((buffer, part), axis=0)

        available_end = buffer_start + buffer.shape[0]
        if available_end < core_end:
            if not reached_eof:
                continue
            core_end = available_end
            required_end = available_end
            if core_end <= core_start:
                break

        read_start = max(0, core_start - halo_frames)
        read_end = min(required_end, available_end)
        segment = buffer[read_start - buffer_start : read_end - buffer_start]
        converted = signal.resample_poly(
            segment,
            up,
            down,
            axis=0,
            window=fir,
            padtype="edge",
        ).astype(np.float32, copy=False)

        segment_output_start = read_start * up // down
        output_start = core_start * up // down
        output_end = min(num_output_frames, math.ceil(core_end * up / down))
        local_start = output_start - segment_output_start
        output[output_start:output_end] = converted[
            local_start : local_start + output_end - output_start
        ]

        core_start = core_end
        next_read_start = max(0, core_start - halo_frames)
        drop_frames = next_read_start - buffer_start
        if drop_frames > 0:
            buffer = buffer[drop_frames:].copy()
            buffer_start = next_read_start

    return output[:output_end]


# torchaudio.functional.resample defaults, method "sinc_interp_hann".
_SINC_LOWPASS_FILTER_WIDTH = 6
_SINC_ROLLOFF = 0.99
_SINC_CHUNK_FRAMES = 1 << 16


def sinc_resample(
    waveform: mx.array, orig_sample_rate: int, sample_rate: int
) -> mx.array:
    """Resample a 1-D signal like ``torchaudio.functional.resample`` with defaults.

    Use it where the reference pipeline resamples with torchaudio and the model
    depends on its output. Its short Hann-windowed sinc passes tones up to 1 kHz
    above the new Nyquist at -9 to -17 dB. ``resample_audio_array`` suppresses
    them by more than 140 dB.

    Output sample ``k`` is row ``k % new`` of torchaudio's kernel applied to the
    padded input from ``floor(k * orig / new)`` on, with the reduced rates. Each
    row is zero outside ``2 * width + 1`` taps, so only those are gathered, one
    chunk at a time. Memory follows the signal, also for rates such as 44099 Hz.
    """
    if orig_sample_rate == sample_rate:
        return waveform
    gcd = math.gcd(int(orig_sample_rate), int(sample_rate))
    orig = int(orig_sample_rate) // gcd
    new = int(sample_rate) // gcd

    base_freq = min(orig, new) * _SINC_ROLLOFF
    width = math.ceil(_SINC_LOWPASS_FILTER_WIDTH * orig / base_freq)
    taps = 2 * width + 1

    # torchaudio's t from the first tap of each row on, in float64, because its
    # two terms nearly cancel.
    phases = np.arange(new)
    first = phases * orig // new
    t = (first[:, None] + np.arange(taps) - width) / orig - phases[:, None] / new
    t = np.clip(t * base_freq, -_SINC_LOWPASS_FILTER_WIDTH, _SINC_LOWPASS_FILTER_WIDTH)
    window = np.cos(t * np.pi / _SINC_LOWPASS_FILTER_WIDTH / 2) ** 2
    kernel = mx.array((np.sinc(t) * window * (base_freq / orig)).astype(np.float32))
    first = mx.array(first.astype(np.int32))

    padded = mx.pad(waveform.astype(mx.float32), (width, width + orig))
    offsets = mx.arange(taps)
    out_len = math.ceil(waveform.shape[0] * new / orig)
    chunks = []
    for start in range(0, out_len, _SINC_CHUNK_FRAMES):
        k = mx.arange(start, min(start + _SINC_CHUNK_FRAMES, out_len))
        phase = k % new
        rows = padded[(k // new * orig + first[phase])[:, None] + offsets]
        chunks.append((rows * kernel[phase]).sum(axis=1))
        # Keeps one chunk's gather in memory at a time.
        mx.eval(chunks[-1])
    return mx.concatenate(chunks) if chunks else padded[:0]
