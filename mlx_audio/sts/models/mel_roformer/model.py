# Copyright (c) 2026 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

"""Mel-Band-RoFormer for vocal source separation.

Architecture (Kim Vocal 2, 228M params):
    [B, 2, samples] → STFT → CaC interleave → BandSplit → 6× DualAxis →
    MaskEstimate → complex multiply → iSTFT → [B, 2, samples]

Reference implementations:
    - lucidrains/BS-RoFormer (MIT)
    - ZFTurbo/Music-Source-Separation-Training (MIT)
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import MelRoFormerConfig


@dataclass
class SeparationResult:
    """Result of audio source separation."""

    vocals: mx.array
    sample_rate: int
    duration_seconds: float
    processing_time_seconds: float


# ---------- Mel Filterbank ----------


class MelFilterbank:
    """Computes mel-scale frequency band boundaries and gather indices.

    Uses Slaney mel scale (librosa-compatible). Produces binarized triangular
    filters, then computes stereo-interleaved gather indices per band.
    """

    def __init__(self, config: MelRoFormerConfig):
        self.config = config
        fb = self._build_mel_filterbank()
        self.freq_indices, self.band_dims, self.num_bands_per_freq = self._compute_band_info(fb)

    def _build_mel_filterbank(self) -> np.ndarray:
        """Build binarized mel filterbank matrix [num_bands, freq_bins]."""
        sr = self.config.sample_rate
        n_fft = self.config.n_fft
        n_mels = self.config.num_bands
        freq_bins = self.config.freq_bins

        # Slaney mel scale (matches librosa)
        fmin, fmax = 0.0, sr / 2.0
        min_mel = self._hz_to_mel(fmin)
        max_mel = self._hz_to_mel(fmax)
        mels = np.linspace(min_mel, max_mel, n_mels + 2)
        freqs = self._mel_to_hz(mels)

        # FFT bin frequencies
        fft_freqs = np.linspace(0, sr / 2, freq_bins)

        # Triangular filters
        fb = np.zeros((n_mels, freq_bins))
        for i in range(n_mels):
            lower, center, upper = freqs[i], freqs[i + 1], freqs[i + 2]
            for j in range(freq_bins):
                if lower <= fft_freqs[j] <= center and center > lower:
                    fb[i, j] = (fft_freqs[j] - lower) / (center - lower)
                elif center < fft_freqs[j] <= upper and upper > center:
                    fb[i, j] = (upper - fft_freqs[j]) / (upper - center)

        # Binarize: non-zero → 1.0
        fb = (fb > 0).astype(np.float32)

        return fb

    def _compute_band_info(self, fb: np.ndarray):
        """Compute per-band gather indices for stereo CaC interleaving.

        Returns:
            freq_indices: list of np.ndarray — per-band gather indices into CaC spectrum
            band_dims: list of int — per-band input dimensions (for BandSplit projections)
            num_bands_per_freq: np.ndarray [freq_bins*2] — overlap count for scatter averaging
        """
        freq_bins = self.config.freq_bins
        num_bands = self.config.num_bands

        freq_indices = []
        num_bands_per_freq = np.zeros(freq_bins * 2, dtype=np.float32)

        for i in range(num_bands):
            # Find which frequency bins belong to this band
            bin_indices = np.where(fb[i] > 0)[0]
            if len(bin_indices) == 0:
                bin_indices = np.array([i])  # fallback

            # Stereo CaC interleave: each freq bin → 2 entries (L, R)
            cac_indices = []
            for b in bin_indices:
                cac_indices.extend([b * 2, b * 2 + 1])

            freq_indices.append(np.array(cac_indices, dtype=np.int32))
            for idx in cac_indices:
                if idx < len(num_bands_per_freq):
                    num_bands_per_freq[idx] += 1

            # Band dim = num_cac_entries × 2 (real + imag)
            pass

        band_dims = [len(idx) * 2 for idx in freq_indices]  # ×2 for real/imag
        num_bands_per_freq = np.maximum(num_bands_per_freq, 1.0)

        return freq_indices, band_dims, num_bands_per_freq

    @staticmethod
    def _hz_to_mel(hz):
        """Slaney mel scale."""
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    @staticmethod
    def _mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


# ---------- RoPE ----------


class RotaryEmbedding:
    """Precomputed rotary position embedding frequencies."""

    def __init__(self, dim_head: int, base: float = 10000.0):
        half = dim_head // 2
        self.freqs = 1.0 / (base ** (np.arange(0, half, dtype=np.float32) / half))
        self.freqs_mx = mx.array(self.freqs)

    def get_cos_sin(self, seq_len: int):
        """Returns (cos, sin) each of shape [seq_len, dim_head]."""
        t = mx.arange(seq_len).astype(mx.float32)
        freqs = mx.outer(t, self.freqs_mx)  # [seq_len, half]
        cos_val = mx.cos(freqs)
        sin_val = mx.sin(freqs)
        # Duplicate for full dim: [seq_len, dim_head]
        cos_val = mx.concatenate([cos_val, cos_val], axis=-1)
        sin_val = mx.concatenate([sin_val, sin_val], axis=-1)
        return cos_val, sin_val


def _apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Apply rotary position embeddings.

    x: [B, heads, T, dim_head]
    cos, sin: [T, dim_head]
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    rotated = mx.concatenate([-x2, x1], axis=-1)
    return x * cos + rotated * sin


# ---------- RoFormer Attention ----------


class RoFormerAttention(nn.Module):
    """Multi-head attention with RoPE and per-head sigmoid gates."""

    def __init__(self, dim: int, heads: int, dim_head: int):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head

        self.norm = nn.RMSNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_gates = nn.Linear(dim, heads, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.rotary_embed = RotaryEmbedding(dim_head)

    def __call__(self, x: mx.array) -> mx.array:
        B, T, D = x.shape
        h = self.norm(x)

        q = self.to_q(h).reshape(B, T, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        k = self.to_k(h).reshape(B, T, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        v = self.to_v(h).reshape(B, T, self.heads, self.dim_head).transpose(0, 2, 1, 3)

        # RoPE
        cos, sin = self.rotary_embed.get_cos_sin(T)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.dim_head)
        attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)

        # Per-head sigmoid gates
        gates = mx.sigmoid(self.to_gates(h))  # [B, T, heads]
        gates = gates.transpose(0, 2, 1)[..., None]  # [B, heads, T, 1]
        attn = attn * gates

        # Project out
        attn = attn.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.to_out(attn)


# ---------- Feed-Forward Network ----------


class RoFormerFFN(nn.Module):
    """Feed-forward: RMSNorm → Linear → GELU → Linear.

    Weight key structure matches PyTorch Sequential indices:
        net.0 = RMSNorm, net.1 = Linear(expand), net.4 = Linear(compress)
    """

    def __init__(self, dim: int, ff_mult: int = 4):
        super().__init__()
        ff_dim = dim * ff_mult
        self.net = [
            nn.RMSNorm(dim),     # index 0
            nn.Linear(dim, ff_dim),  # index 1
            None,                    # index 2 (GELU activation, no params)
            None,                    # index 3 (placeholder)
            nn.Linear(ff_dim, dim),  # index 4
        ]

    def __call__(self, x: mx.array) -> mx.array:
        h = self.net[0](x)      # RMSNorm
        h = self.net[1](h)      # Linear expand
        h = nn.gelu(h)          # GELU
        h = self.net[4](h)      # Linear compress
        return h


# ---------- Transformer Block ----------


class Transformer(nn.Module):
    """Single-axis transformer with RoPE attention + FFN + output RMSNorm."""

    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, ff_mult: int):
        super().__init__()
        self.layers = [
            [RoFormerAttention(dim, heads, dim_head), RoFormerFFN(dim, ff_mult)]
            for _ in range(depth)
        ]
        self.norm = nn.RMSNorm(dim)

    def __call__(self, x: mx.array) -> mx.array:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


# ---------- BandSplit ----------


class BandSplit(nn.Module):
    """Split spectrogram into mel-scale bands and project each to model dimension."""

    def __init__(self, config: MelRoFormerConfig):
        super().__init__()
        self.filterbank = MelFilterbank(config)
        # Per-band projection: RMSNorm → Linear
        self.to_features = [
            [nn.RMSNorm(bd), nn.Linear(bd, config.dim)]
            for bd in self.filterbank.band_dims
        ]

    def split(self, x: mx.array) -> mx.array:
        """Split CaC spectrogram into mel bands and project.

        Args:
            x: [B, freq_bins*2, T, 2] (CaC interleaved, real/imag)

        Returns:
            [B, T, num_bands, dim]
        """
        B, F2, T, _ = x.shape
        bands = []

        for i, (indices, proj) in enumerate(zip(
            self.filterbank.freq_indices, self.to_features
        )):
            # Gather frequencies for this band: [B, num_band_freqs, T, 2]
            gathered = x[:, indices, :, :]
            # Flatten last dims: [B, T, num_band_freqs * 2]
            n_freqs = gathered.shape[1]
            band_input = gathered.transpose(0, 2, 1, 3).reshape(B, T, n_freqs * 2)
            # Project: RMSNorm → Linear → [B, T, dim]
            h = proj[0](band_input)  # RMSNorm
            h = proj[1](h)            # Linear
            bands.append(h)

        # Stack bands: [B, T, num_bands, dim]
        return mx.stack(bands, axis=2)

    def merge(self, band_masks: list, freq_bins_times_two: int) -> mx.array:
        """Scatter per-band masks back to full spectrum with overlap averaging.

        Args:
            band_masks: list of [B, T, band_dim] arrays (one per band)
            freq_bins_times_two: target frequency dimension

        Returns:
            [B, freq_bins*2, T, 2]
        """
        B = band_masks[0].shape[0]
        T = band_masks[0].shape[1]

        # Initialize output
        output = mx.zeros((B, freq_bins_times_two, T, 2))

        for i, (indices, mask) in enumerate(zip(self.filterbank.freq_indices, band_masks)):
            n_freqs = len(indices)
            # Reshape mask: [B, T, n_freqs*2] → [B, T, n_freqs, 2] → [B, n_freqs, T, 2]
            mask_reshaped = mask.reshape(B, T, n_freqs, 2).transpose(0, 2, 1, 3)
            # Scatter-add to output at the band's frequency indices
            for j, idx in enumerate(indices):
                output = output.at[:, idx, :, :].add(mask_reshaped[:, j, :, :])

        # Average by overlap count
        num_bands = mx.array(self.filterbank.num_bands_per_freq).reshape(1, -1, 1, 1)
        output = output / num_bands

        return output


# ---------- Mask Estimator ----------


class MaskEstimator(nn.Module):
    """Estimate complex masks per frequency band via MLP + GLU.

    Per-band MLP: Linear→Tanh→Linear→Tanh→Linear(→2× for GLU)
    Weight keys use indices 0, 2, 4 (Tanh at 1, 3 has no params).
    """

    def __init__(self, config: MelRoFormerConfig, band_dims: list):
        super().__init__()
        dim = config.dim
        mlp_hidden = config.mlp_hidden

        # Per-band: 3 linear layers + GLU
        self.to_freqs = []
        for bd in band_dims:
            layers = [
                [nn.Linear(dim, mlp_hidden)],       # index 0: Linear + Tanh
                [nn.Linear(mlp_hidden, mlp_hidden)], # index 2: Linear + Tanh
                [nn.Linear(mlp_hidden, bd * 2)],     # index 4: Linear (doubled for GLU)
            ]
            self.to_freqs.append(layers)

    def __call__(self, x: mx.array) -> list:
        """Estimate per-band masks.

        Args:
            x: [B, T, num_bands, dim]

        Returns:
            list of [B, T, band_dim] arrays (one per band)
        """
        masks = []
        num_bands = x.shape[2]

        for i in range(num_bands):
            band_input = x[:, :, i, :]  # [B, T, dim]
            layers = self.to_freqs[i]

            h = mx.tanh(layers[0][0](band_input))   # Linear → Tanh
            h = mx.tanh(layers[1][0](h))             # Linear → Tanh
            h = layers[2][0](h)                       # Linear (doubled)

            # GLU: split in half, apply sigmoid gate
            half = h.shape[-1] // 2
            h = h[..., :half] * mx.sigmoid(h[..., half:])

            masks.append(h)

        return masks


# ---------- STFT / iSTFT ----------


def stft(audio: mx.array, n_fft: int, hop_length: int, window: mx.array) -> tuple:
    """Short-time Fourier Transform.

    Args:
        audio: [B, channels, samples]
        n_fft: FFT size
        hop_length: hop between frames
        window: [n_fft] Hann window

    Returns:
        (real, imag) each [B, channels, freq_bins, frames]
    """
    B, C, N = audio.shape
    pad = n_fft // 2
    # Center-pad
    audio_padded = mx.pad(audio, [(0, 0), (0, 0), (pad, pad)])

    # Extract frames
    N_padded = audio_padded.shape[2]
    num_frames = (N_padded - n_fft) // hop_length + 1

    # Reshape for frame extraction: [B*C, N_padded]
    flat = audio_padded.reshape(B * C, -1)

    frames = []
    for t in range(num_frames):
        start = t * hop_length
        frame = flat[:, start:start + n_fft]
        frames.append(frame)

    # Stack: [B*C, num_frames, n_fft]
    frames = mx.stack(frames, axis=1)

    # Apply window
    frames = frames * window

    # FFT (real FFT on last axis)
    spectrum = mx.fft.rfft(frames)  # [B*C, num_frames, freq_bins] complex

    # Extract real and imaginary
    real = spectrum.real().reshape(B, C, num_frames, -1).transpose(0, 1, 3, 2)
    imag = spectrum.imag().reshape(B, C, num_frames, -1).transpose(0, 1, 3, 2)

    return real, imag


def istft(real: mx.array, imag: mx.array, n_fft: int, hop_length: int,
          window: mx.array, length: int) -> mx.array:
    """Inverse Short-time Fourier Transform via overlap-add.

    Args:
        real: [B, channels, freq_bins, frames]
        imag: [B, channels, freq_bins, frames]
        n_fft: FFT size
        hop_length: hop between frames
        window: [n_fft] Hann window
        length: target output length in samples

    Returns:
        [B, channels, samples]
    """
    B, C, F, T = real.shape
    pad = n_fft // 2

    # Reconstruct complex spectrum: [B*C, T, F]
    spectrum = (real + 1j * imag).transpose(0, 1, 3, 2).reshape(B * C, T, F)

    # Inverse FFT: [B*C, T, n_fft]
    frames = mx.fft.irfft(spectrum, n=n_fft)

    # Apply window
    frames = frames * window

    # Overlap-add
    output_len = (T - 1) * hop_length + n_fft
    output = mx.zeros((B * C, output_len))
    window_sum = mx.zeros((1, output_len))

    w2 = window * window
    for t_idx in range(T):
        start = t_idx * hop_length
        output = output.at[:, start:start + n_fft].add(frames[:, t_idx, :])
        window_sum = window_sum.at[:, start:start + n_fft].add(w2)

    # Normalize by window sum
    window_sum = mx.maximum(window_sum, 1e-8)
    output = output / window_sum

    # Remove center padding and trim
    output = output[:, pad:pad + length]
    output = output.reshape(B, C, -1)

    return output


# ---------- Main Model ----------


class MelRoFormer(nn.Module):
    """Mel-Band-RoFormer for vocal source separation.

    Pipeline: STFT → CaC → BandSplit → DualAxisTransformer → MaskEstimate → iSTFT
    """

    def __init__(self, config: MelRoFormerConfig = MelRoFormerConfig()):
        super().__init__()
        self.config = config

        self.band_split = BandSplit(config)

        # 6× dual-axis: [time_transformer, freq_transformer] per depth
        self.layers = [
            [
                Transformer(config.dim, 1, config.heads, config.dim_head, config.ff_mult),
                Transformer(config.dim, 1, config.heads, config.dim_head, config.ff_mult),
            ]
            for _ in range(config.depth)
        ]

        self.mask_estimators = [
            MaskEstimator(config, self.band_split.filterbank.band_dims)
        ]

    def __call__(self, audio: mx.array) -> mx.array:
        """Separate vocals from audio.

        Args:
            audio: [B, 2, samples] stereo audio at 44.1kHz

        Returns:
            [B, 2, samples] separated vocal audio
        """
        original_length = audio.shape[2]
        config = self.config

        # Hann window
        window = mx.array(np.hanning(config.n_fft + 1)[:-1].astype(np.float32))

        # Step 1: STFT → real/imag [B, 2, freq_bins, T]
        stft_real, stft_imag = stft(audio, config.n_fft, config.hop_length, window)
        B = stft_real.shape[0]
        freq_bins = stft_real.shape[2]
        T = stft_real.shape[3]

        # Step 2: CaC interleave — [B, 2, freq_bins, T] → [B, freq_bins*2, T]
        real_interleaved = stft_real.transpose(0, 2, 1, 3).reshape(B, freq_bins * 2, T)
        imag_interleaved = stft_imag.transpose(0, 2, 1, 3).reshape(B, freq_bins * 2, T)

        # Stack real/imag: [B, freq_bins*2, T, 2]
        stft_repr = mx.stack([real_interleaved, imag_interleaved], axis=-1)

        # Step 3: BandSplit → [B, T, num_bands, dim]
        x = self.band_split.split(stft_repr)
        Nb = x.shape[2]
        D = x.shape[3]

        # Step 4: 6× dual-axis transformer
        for time_tf, freq_tf in self.layers:
            # Time attention: [B, T, Nb, D] → [B*Nb, T, D]
            time_in = x.transpose(0, 2, 1, 3).reshape(B * Nb, T, D)
            time_out = time_tf(time_in)
            x = time_out.reshape(B, Nb, T, D).transpose(0, 2, 1, 3)

            # Freq attention: [B, T, Nb, D] → [B*T, Nb, D]
            freq_in = x.reshape(B * T, Nb, D)
            freq_out = freq_tf(freq_in)
            x = freq_out.reshape(B, T, Nb, D)

        # Step 5: Mask estimation → list of [B, T, band_dim]
        masks = self.mask_estimators[0](x)

        # Step 6: Merge masks → [B, freq_bins*2, T, 2]
        full_mask = self.band_split.merge(masks, freq_bins * 2)

        # Step 7: Complex multiply (input × mask)
        input_real = stft_repr[..., 0]  # [B, freq_bins*2, T]
        input_imag = stft_repr[..., 1]
        mask_real = full_mask[..., 0]
        mask_imag = full_mask[..., 1]

        out_real = input_real * mask_real - input_imag * mask_imag
        out_imag = input_real * mask_imag + input_imag * mask_real

        # Step 8: De-interleave → [B, 2, freq_bins, T]
        real_deint = out_real.reshape(B, freq_bins, 2, T).transpose(0, 2, 1, 3)
        imag_deint = out_imag.reshape(B, freq_bins, 2, T).transpose(0, 2, 1, 3)

        # Step 9: iSTFT → [B, 2, samples]
        separated = istft(real_deint, imag_deint,
                         config.n_fft, config.hop_length, window, original_length)

        return separated

    def sanitize(self, weights: dict) -> dict:
        """Sanitize PyTorch weights for MLX.

        Handles key remapping from PyTorch BS-RoFormer format to MLX module paths.
        The main transformation is splitting packed to_qkv into separate to_q/to_k/to_v.
        """
        new_weights = {}

        for key, value in weights.items():
            # Split packed QKV into separate projections
            if "to_qkv.weight" in key:
                prefix = key.replace("to_qkv.weight", "")
                # value shape: [3*inner_dim, dim] → split into 3
                third = value.shape[0] // 3
                new_weights[f"{prefix}to_q.weight"] = value[:third]
                new_weights[f"{prefix}to_k.weight"] = value[third:2*third]
                new_weights[f"{prefix}to_v.weight"] = value[2*third:]
            else:
                new_weights[key] = value

        return new_weights

    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, Path],
        config: Optional[MelRoFormerConfig] = None,
    ) -> "MelRoFormer":
        """Load a pre-trained model from a local path or HuggingFace repo.

        Config resolution order:
            1. The `config` argument if provided.
            2. A companion `<basename>.config.json` alongside the weights file
               (produced by `convert.py`).
            3. A `config.json` in the model directory.
            4. Raises ValueError — no `default()` is assumed.

        Args:
            model_path: Path to model directory or HuggingFace repo ID.
            config: Optional explicit config. If None, attempts to load a
                    companion config.json.

        Returns:
            Loaded MelRoFormer model.
        """
        import json

        from mlx_audio.utils import get_model_path

        path = Path(get_model_path(str(model_path)))

        # Locate the weights file
        weight_file: Optional[Path] = None
        # Prefer any .safetensors file (content-addressed outputs from convert.py
        # have names like `<basename>.<hash>.safetensors`)
        safetensors_files = sorted(path.glob("*.safetensors"))
        if safetensors_files:
            # Prefer specific known names if multiple exist
            priority = ["mel_roformer_vocals.safetensors", "weights.safetensors",
                       "model.safetensors"]
            for preferred in priority:
                candidate = path / preferred
                if candidate.exists():
                    weight_file = candidate
                    break
            if weight_file is None:
                weight_file = safetensors_files[0]

        if weight_file is None or not weight_file.exists():
            raise FileNotFoundError(
                f"No .safetensors file found in {path}"
            )

        # Resolve config
        if config is None:
            # Try companion config.json (same basename as weights)
            companion = weight_file.with_suffix(".config.json")
            if companion.exists():
                data = json.loads(companion.read_text())
                # Filter to fields the dataclass accepts
                from dataclasses import fields as _fields
                valid_keys = {f.name for f in _fields(MelRoFormerConfig)}
                filtered = {k: v for k, v in data.items() if k in valid_keys}
                config = MelRoFormerConfig(**filtered)
            else:
                # Try a plain config.json in the directory
                plain = path / "config.json"
                if plain.exists():
                    data = json.loads(plain.read_text())
                    from dataclasses import fields as _fields
                    valid_keys = {f.name for f in _fields(MelRoFormerConfig)}
                    filtered = {k: v for k, v in data.items() if k in valid_keys}
                    config = MelRoFormerConfig(**filtered)
                else:
                    raise ValueError(
                        f"No config provided and no companion config.json found at "
                        f"{weight_file.with_suffix('.config.json')} or {path / 'config.json'}. "
                        f"Pass an explicit preset, e.g. "
                        f"`MelRoFormer.from_pretrained(path, config=MelRoFormerConfig.kim_vocal_2())`."
                    )

        model = cls(config)

        weights = dict(mx.load(str(weight_file)))
        sanitized = model.sanitize(weights)
        model.load_weights(list(sanitized.items()), strict=False)
        return model
