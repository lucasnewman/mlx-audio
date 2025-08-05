import math
from typing import Tuple

import mlx.core as mx
import numpy as np

EPSILON = 1.1920928955078125e-07
MILLISECONDS_TO_SECONDS = 0.001

HAMMING = "hamming"
HANNING = "hanning"
POVEY = "povey"
RECTANGULAR = "rectangular"
BLACKMAN = "blackman"


def _get_epsilon(dtype=mx.float32):
    """Get epsilon value for the given dtype"""
    return mx.array(EPSILON, dtype=dtype)


def _next_power_of_2(x: int) -> int:
    """Returns the smallest power of 2 that is greater than x"""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def _get_strided(waveform: mx.array, window_size: int, window_shift: int, snip_edges: bool) -> mx.array:
    """Given a waveform (1D array of size num_samples), it returns a 2D array (m, window_size)
    representing how the window is shifted along the waveform. Each row is a frame.
    """
    assert waveform.ndim == 1
    num_samples = waveform.shape[0]

    if snip_edges:
        if num_samples < window_size:
            return mx.zeros((0, 0), dtype=waveform.dtype)
        else:
            m = 1 + (num_samples - window_size) // window_shift
    else:
        m = (num_samples + (window_shift // 2)) // window_shift
        pad = window_size // 2 - window_shift // 2
        if pad > 0:
            # Reflect padding
            pad_left = waveform[-pad:][::-1]
            pad_right = waveform[::-1]
            waveform = mx.concatenate([pad_left, waveform, pad_right], axis=0)
        else:
            # Trim the waveform at the front
            pad_right = waveform[::-1]
            waveform = mx.concatenate([waveform[-pad:], pad_right], axis=0)

    # Create strided view using indexing
    indices = mx.arange(window_size)[None, :] + mx.arange(m)[:, None] * window_shift
    return waveform[indices]


def _feature_window_function(
    window_type: str,
    window_size: int,
    blackman_coeff: float,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Returns a window function with the given type and size"""
    if window_type == HANNING:
        n = mx.arange(window_size, dtype=dtype)
        return 0.5 - 0.5 * mx.cos(2 * mx.pi * n / (window_size - 1))
    elif window_type == HAMMING:
        n = mx.arange(window_size, dtype=dtype)
        return 0.54 - 0.46 * mx.cos(2 * mx.pi * n / (window_size - 1))
    elif window_type == POVEY:
        n = mx.arange(window_size, dtype=dtype)
        hann = 0.5 - 0.5 * mx.cos(2 * mx.pi * n / (window_size - 1))
        return mx.power(hann, 0.85)
    elif window_type == RECTANGULAR:
        return mx.ones(window_size, dtype=dtype)
    elif window_type == BLACKMAN:
        a = 2 * math.pi / (window_size - 1)
        n = mx.arange(window_size, dtype=dtype)
        return blackman_coeff - 0.5 * mx.cos(a * n) + (0.5 - blackman_coeff) * mx.cos(2 * a * n)
    else:
        raise Exception("Invalid window type " + window_type)


def _get_log_energy(strided_input: mx.array, epsilon: mx.array, energy_floor: float) -> mx.array:
    """Returns the log energy of size (m) for a strided_input (m,*)"""
    log_energy = mx.log(mx.maximum(mx.sum(strided_input**2, axis=1), epsilon))
    if energy_floor == 0.0:
        return log_energy
    return mx.maximum(log_energy, mx.array(math.log(energy_floor), dtype=log_energy.dtype))


def _get_waveform_and_window_properties(
    waveform: mx.array,
    channel: int,
    sample_frequency: float,
    frame_shift: float,
    frame_length: float,
    round_to_power_of_two: bool,
    preemphasis_coefficient: float,
) -> Tuple[mx.array, int, int, int]:
    """Gets the waveform and window properties"""
    channel = max(channel, 0)
    assert channel < waveform.shape[0], f"Invalid channel {channel} for size {waveform.shape[0]}"
    waveform = waveform[channel, :]  # size (n)
    window_shift = int(sample_frequency * frame_shift * MILLISECONDS_TO_SECONDS)
    window_size = int(sample_frequency * frame_length * MILLISECONDS_TO_SECONDS)
    padded_window_size = _next_power_of_2(window_size) if round_to_power_of_two else window_size

    assert 2 <= window_size <= len(waveform), f"choose a window size {window_size} that is [2, {len(waveform)}]"
    assert 0 < window_shift, "`window_shift` must be greater than 0"
    assert padded_window_size % 2 == 0, (
        "the padded `window_size` must be divisible by two. use `round_to_power_of_two` or change `frame_length`"
    )
    assert 0.0 <= preemphasis_coefficient <= 1.0, "`preemphasis_coefficient` must be between [0,1]"
    assert sample_frequency > 0, "`sample_frequency` must be greater than zero"
    return waveform, window_shift, window_size, padded_window_size


def _get_window(
    waveform: mx.array,
    padded_window_size: int,
    window_size: int,
    window_shift: int,
    window_type: str,
    blackman_coeff: float,
    snip_edges: bool,
    raw_energy: bool,
    energy_floor: float,
    dither: float,
    remove_dc_offset: bool,
    preemphasis_coefficient: float,
) -> Tuple[mx.array, mx.array]:
    """Gets a window and its log energy"""
    dtype = waveform.dtype
    epsilon = _get_epsilon(dtype)

    # size (m, window_size)
    strided_input = _get_strided(waveform, window_size, window_shift, snip_edges)

    if dither != 0.0:
        key = mx.random.key(0)
        rand_gauss = mx.random.normal(key=key, shape=strided_input.shape, dtype=dtype)
        strided_input = strided_input + rand_gauss * dither

    if remove_dc_offset:
        row_means = mx.mean(strided_input, axis=1, keepdims=True)
        strided_input = strided_input - row_means

    if raw_energy:
        signal_log_energy = _get_log_energy(strided_input, epsilon, energy_floor)

    if preemphasis_coefficient != 0.0:
        offset_strided_input = mx.pad(strided_input, [(0, 0), (1, 0)], mode="edge")
        strided_input = strided_input - preemphasis_coefficient * offset_strided_input[:, :-1]

    window_function = _feature_window_function(window_type, window_size, blackman_coeff, dtype)
    strided_input = strided_input * window_function[None, :]

    if padded_window_size != window_size:
        padding_right = padded_window_size - window_size
        strided_input = mx.pad(strided_input, [(0, 0), (0, padding_right)], constant_values=0)

    if not raw_energy:
        signal_log_energy = _get_log_energy(strided_input, epsilon, energy_floor)

    return strided_input, signal_log_energy


def _subtract_column_mean(tensor: mx.array, subtract_mean: bool) -> mx.array:
    """Subtracts the column mean of the tensor if subtract_mean=True"""
    if subtract_mean:
        col_means = mx.mean(tensor, axis=0, keepdims=True)
        tensor = tensor - col_means
    return tensor


def mel_scale_scalar(freq: float) -> float:
    """Convert frequency to mel scale"""
    return 1127.0 * math.log(1.0 + freq / 700.0)


def mel_scale(freq: mx.array) -> mx.array:
    """Convert frequency array to mel scale"""
    return 1127.0 * mx.log(1.0 + freq / 700.0)


def inverse_mel_scale_scalar(mel_freq: float) -> float:
    """Convert mel frequency to linear frequency"""
    return 700.0 * (math.exp(mel_freq / 1127.0) - 1.0)


def inverse_mel_scale(mel_freq: mx.array) -> mx.array:
    """Convert mel frequency array to linear frequency"""
    return 700.0 * (mx.exp(mel_freq / 1127.0) - 1.0)


def vtln_warp_freq(
    vtln_low_cutoff: float,
    vtln_high_cutoff: float,
    low_freq: float,
    high_freq: float,
    vtln_warp_factor: float,
    freq: mx.array,
) -> mx.array:
    """VTLN warping function"""
    assert vtln_low_cutoff > low_freq, "be sure to set the vtln_low option higher than low_freq"
    assert vtln_high_cutoff < high_freq, "be sure to set the vtln_high option lower than high_freq"

    l = vtln_low_cutoff * max(1.0, vtln_warp_factor)
    h = vtln_high_cutoff * min(1.0, vtln_warp_factor)
    scale = 1.0 / vtln_warp_factor
    Fl = scale * l  # F(l)
    Fh = scale * h  # F(h)
    assert l > low_freq and h < high_freq

    scale_left = (Fl - low_freq) / (l - low_freq)
    scale_right = (high_freq - Fh) / (high_freq - h)

    res = mx.zeros_like(freq)

    outside_low_high_freq = (freq < low_freq) | (freq > high_freq)
    before_l = freq < l
    before_h = freq < h
    after_h = freq >= h

    res = mx.where(after_h, high_freq + scale_right * (freq - high_freq), res)
    res = mx.where(before_h & ~after_h, scale * freq, res)
    res = mx.where(before_l & ~before_h, low_freq + scale_left * (freq - low_freq), res)
    res = mx.where(outside_low_high_freq, freq, res)

    return res


def vtln_warp_mel_freq(
    vtln_low_cutoff: float,
    vtln_high_cutoff: float,
    low_freq: float,
    high_freq: float,
    vtln_warp_factor: float,
    mel_freq: mx.array,
) -> mx.array:
    """VTLN warping in mel frequency domain"""
    return mel_scale(vtln_warp_freq(vtln_low_cutoff, vtln_high_cutoff, low_freq, high_freq, vtln_warp_factor, inverse_mel_scale(mel_freq)))


def get_mel_banks(
    num_bins: int,
    window_length_padded: int,
    sample_freq: float,
    low_freq: float,
    high_freq: float,
    vtln_low: float,
    vtln_high: float,
    vtln_warp_factor: float,
) -> Tuple[mx.array, mx.array]:
    """Create mel filterbank matrix"""
    assert num_bins > 3, "Must have at least 3 mel bins"
    assert window_length_padded % 2 == 0
    num_fft_bins = window_length_padded // 2
    nyquist = 0.5 * sample_freq

    if high_freq <= 0.0:
        high_freq += nyquist

    assert (0.0 <= low_freq < nyquist) and (0.0 < high_freq <= nyquist) and (low_freq < high_freq), (
        f"Bad values: low-freq {low_freq} and high-freq {high_freq} vs. nyquist {nyquist}"
    )

    fft_bin_width = sample_freq / window_length_padded
    mel_low_freq = mel_scale_scalar(low_freq)
    mel_high_freq = mel_scale_scalar(high_freq)

    mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_bins + 1)

    if vtln_high < 0.0:
        vtln_high += nyquist

    assert vtln_warp_factor == 1.0 or ((low_freq < vtln_low < high_freq) and (0.0 < vtln_high < high_freq) and (vtln_low < vtln_high)), (
        f"Bad VTLN values: vtln-low {vtln_low} and vtln-high {vtln_high}"
    )

    bin_idx = mx.arange(num_bins)[:, None]
    left_mel = mel_low_freq + bin_idx * mel_freq_delta
    center_mel = mel_low_freq + (bin_idx + 1.0) * mel_freq_delta
    right_mel = mel_low_freq + (bin_idx + 2.0) * mel_freq_delta

    if vtln_warp_factor != 1.0:
        left_mel = vtln_warp_mel_freq(vtln_low, vtln_high, low_freq, high_freq, vtln_warp_factor, left_mel)
        center_mel = vtln_warp_mel_freq(vtln_low, vtln_high, low_freq, high_freq, vtln_warp_factor, center_mel)
        right_mel = vtln_warp_mel_freq(vtln_low, vtln_high, low_freq, high_freq, vtln_warp_factor, right_mel)

    center_freqs = inverse_mel_scale(center_mel).squeeze(1)

    mel = mel_scale(fft_bin_width * mx.arange(num_fft_bins))[None, :]

    up_slope = (mel - left_mel) / (center_mel - left_mel)
    down_slope = (right_mel - mel) / (right_mel - center_mel)

    if vtln_warp_factor == 1.0:
        bins = mx.maximum(mx.zeros(1), mx.minimum(up_slope, down_slope))
    else:
        bins = mx.zeros_like(up_slope)
        up_idx = (mel > left_mel) & (mel <= center_mel)
        down_idx = (mel > center_mel) & (mel < right_mel)
        bins = mx.where(up_idx, up_slope, bins)
        bins = mx.where(down_idx, down_slope, bins)

    return bins, center_freqs


def fbank(
    waveform: mx.array,
    blackman_coeff: float = 0.42,
    channel: int = -1,
    dither: float = 0.0,
    energy_floor: float = 1.0,
    frame_length: float = 25.0,
    frame_shift: float = 10.0,
    high_freq: float = 0.0,
    htk_compat: bool = False,
    low_freq: float = 20.0,
    min_duration: float = 0.0,
    num_mel_bins: int = 23,
    preemphasis_coefficient: float = 0.97,
    raw_energy: bool = True,
    remove_dc_offset: bool = True,
    round_to_power_of_two: bool = True,
    sample_frequency: float = 16000.0,
    snip_edges: bool = True,
    subtract_mean: bool = False,
    use_energy: bool = False,
    use_log_fbank: bool = True,
    use_power: bool = True,
    vtln_high: float = -500.0,
    vtln_low: float = 100.0,
    vtln_warp: float = 1.0,
    window_type: str = POVEY,
) -> mx.array:
    """Create a filterbank from a raw audio signal.

    This matches the input/output of Kaldi's compute-fbank-feats.

    Args:
        waveform: Array of audio of size (c, n) where c is in the range [0,2)
        blackman_coeff: Constant coefficient for generalized Blackman window
        channel: Channel to extract (-1 -> expect mono, 0 -> left, 1 -> right)
        dither: Dithering constant (0.0 means no dither)
        energy_floor: Floor on energy (absolute, not relative) in Spectrogram computation
        frame_length: Frame length in milliseconds
        frame_shift: Frame shift in milliseconds
        high_freq: High cutoff frequency for mel bins (if <= 0, offset from Nyquist)
        htk_compat: If true, put energy last
        low_freq: Low cutoff frequency for mel bins
        min_duration: Minimum duration of segments to process (in seconds)
        num_mel_bins: Number of triangular mel-frequency bins
        preemphasis_coefficient: Coefficient for use in signal preemphasis
        raw_energy: If True, compute energy before preemphasis and windowing
        remove_dc_offset: Subtract mean from waveform on each frame
        round_to_power_of_two: If True, round window size to power of two
        sample_frequency: Waveform data sample frequency
        snip_edges: If True, end effects will be handled by outputting only frames that completely fit
        subtract_mean: Subtract mean of each feature file [CMS]
        use_energy: Add an extra dimension with energy to the FBANK output
        use_log_fbank: If true, produce log-filterbank, else produce linear
        use_power: If true, use power, else use magnitude
        vtln_high: High inflection point in VTLN warping function
        vtln_low: Low inflection point in VTLN warping function
        vtln_warp: VTLN warp factor
        window_type: Type of window

    Returns:
        A filterbank array of shape (m, num_mel_bins + use_energy)
    """
    dtype = waveform.dtype

    waveform, window_shift, window_size, padded_window_size = _get_waveform_and_window_properties(
        waveform, channel, sample_frequency, frame_shift, frame_length, round_to_power_of_two, preemphasis_coefficient
    )

    if len(waveform) < min_duration * sample_frequency:
        return mx.zeros((0, num_mel_bins + int(use_energy)), dtype=dtype)

    strided_input, signal_log_energy = _get_window(
        waveform,
        padded_window_size,
        window_size,
        window_shift,
        window_type,
        blackman_coeff,
        snip_edges,
        raw_energy,
        energy_floor,
        dither,
        remove_dc_offset,
        preemphasis_coefficient,
    )

    fft_result = mx.fft.rfft(strided_input, axis=1)
    spectrum = mx.abs(fft_result)

    if use_power:
        spectrum = spectrum**2.0

    mel_energies, _ = get_mel_banks(num_mel_bins, padded_window_size, sample_frequency, low_freq, high_freq, vtln_low, vtln_high, vtln_warp)
    mel_energies = mel_energies.astype(dtype)
    mel_energies = mx.pad(mel_energies, [(0, 0), (0, 1)], constant_values=0)
    mel_energies = spectrum @ mel_energies.T

    if use_log_fbank:
        mel_energies = mx.log(mx.maximum(mel_energies, _get_epsilon(dtype)))

    if use_energy:
        signal_log_energy = signal_log_energy[:, None]
        if htk_compat:
            mel_energies = mx.concatenate([mel_energies, signal_log_energy], axis=1)
        else:
            mel_energies = mx.concatenate([signal_log_energy, mel_energies], axis=1)

    mel_energies = _subtract_column_mean(mel_energies, subtract_mean)
    return mel_energies


# if __name__ == "__main__":
#     import numpy as np

#     try:
#         import torch
#         import torchaudio.compliance.kaldi as kaldi

#         torch_available = True
#     except ImportError:
#         torch_available = False
#         print("PyTorch/torchaudio not available, skipping comparison test")

#     np.random.seed(42)

#     duration = 2.0
#     sample_rate = 16000
#     num_samples = int(duration * sample_rate)
#     wave = np.random.randn(1, num_samples).astype(np.float32)

#     mlx_wave = mx.array(wave)
#     mlx_fbank = fbank(
#         mlx_wave,
#         num_mel_bins=80,
#     )
#     print(f"MLX fbank shape: {mlx_fbank.shape}")
#     print(f"MLX fbank stats - min: {mx.min(mlx_fbank):.4f}, max: {mx.max(mlx_fbank):.4f}, mean: {mx.mean(mlx_fbank):.4f}")

#     if torch_available:
#         # PyTorch/Kaldi computation
#         torch_wave = torch.from_numpy(wave)
#         torch_fbank = kaldi.fbank(
#             torch_wave,
#             num_mel_bins=80,
#         )

#         print(f"\nPyTorch fbank shape: {torch_fbank.shape}")
#         print(f"PyTorch fbank stats - min: {torch_fbank.min():.4f}, max: {torch_fbank.max():.4f}, mean: {torch_fbank.mean():.4f}")

#         mlx_fbank_np = np.array(mlx_fbank)
#         torch_fbank_np = torch_fbank.numpy()
#         assert mlx_fbank_np.shape == torch_fbank_np.shape, f"Shape mismatch: MLX {mlx_fbank_np.shape} vs PyTorch {torch_fbank_np.shape}"

#         abs_diff = np.abs(mlx_fbank_np - torch_fbank_np)
#         rel_diff = abs_diff / (np.abs(torch_fbank_np) + 1e-8)

#         print("\nComparison:")
#         print(f"Max absolute difference: {np.max(abs_diff):.6f}")
#         print(f"Mean absolute difference: {np.mean(abs_diff):.6f}")
#         print(f"Max relative difference: {np.max(rel_diff):.6f}")
#         print(f"Mean relative difference: {np.mean(rel_diff):.6f}")

#         tolerance = 1e-4
#         if np.allclose(mlx_fbank_np, torch_fbank_np, rtol=tolerance, atol=tolerance):
#             print(f"\n✓ Results match within tolerance {tolerance}")
#         else:
#             print(f"\n✗ Results differ beyond tolerance {tolerance}")
