"""MLX WaveFitNeXt latent decoder with frame-rate ConvNeXt generators.

Adapted from higurashi-ai/FastDialogueSidon (Apache-2.0) and
yukara-ikemiya/wavefit-pytorch (MIT, Copyright 2026 Yukara Ikemiya).
See README.md for upstream license and attribution links.
"""

import mlx.core as mx
from mlx import nn


def stft_features(waveform, window, hop_size, num_frames):
    """Centered, reflect-padded STFT, packed as real bins then inner imag bins."""
    n_fft = window.shape[0]
    length = waveform.shape[-1]
    if length <= n_fft // 2:
        raise ValueError("Waveform is too short for the vocoder's reflected STFT")
    indices = (
        mx.arange(num_frames)[:, None] * hop_size
        + mx.arange(n_fft)[None, :]
        - n_fft // 2
    )
    indices = mx.where(indices < 0, -indices, indices)
    indices = mx.where(indices >= length, 2 * length - 2 - indices, indices)
    frames = waveform[:, indices].astype(mx.float32) * window.astype(mx.float32)
    spectrum = mx.fft.rfft(frames, axis=-1)
    return mx.concatenate((spectrum.real, spectrum.imag[..., 1:-1]), axis=-1)


class ConvNeXtBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.hidden_dim
        self.dwconv = nn.Conv1d(
            dim, dim, config.kernel_size, padding=config.kernel_size // 2, groups=dim
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, config.intermediate_dim)
        self.pwconv2 = nn.Linear(config.intermediate_dim, dim)
        self.gamma = mx.full((dim,), 1e-6)

    def __call__(self, x):
        h = self.norm(self.dwconv(x))
        h = self.pwconv2(nn.gelu(self.pwconv1(h)))
        return x + self.gamma * h


class WaveNeXtGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        args = config.generator_args
        self.hop_size = config.sample_rate // (
            config.latent_rate * config.latent_upsample
        )
        self.spec_proj = nn.Conv1d(
            config.latent_dim + config.n_fft_stft, args.hidden_dim, 3, padding=1
        )
        self.input_norm = nn.LayerNorm(args.hidden_dim, eps=1e-6)
        self.blocks = [ConvNeXtBlock(args) for _ in range(args.num_layers)]
        self.final_norm = nn.LayerNorm(args.hidden_dim, eps=1e-6)
        self.linear1 = nn.Linear(args.hidden_dim, config.n_fft_stft)
        self.linear2 = nn.Linear(config.n_fft_stft, self.hop_size, bias=False)
        self.fft_win = 0.5 - 0.5 * mx.cos(
            2 * mx.pi * mx.arange(config.n_fft_stft) / config.n_fft_stft
        )

    def __call__(self, latents, previous=None):
        if previous is None:
            # The first iteration starts from exact silence. Its STFT contributes
            # zero, so only the latent channels of this convolution are needed.
            x = mx.conv1d(
                latents, self.spec_proj.weight[..., : latents.shape[-1]], padding=1
            )
            x = x + self.spec_proj.bias
        else:
            spec = stft_features(
                previous, self.fft_win, self.hop_size, latents.shape[1]
            )
            x = self.spec_proj(
                mx.concatenate((latents, spec.astype(latents.dtype)), -1)
            )
        x = self.input_norm(x)
        for block in self.blocks:
            x = block(x)
        x = self.linear2(self.linear1(self.final_norm(x)))
        return x.reshape(x.shape[0], -1)


class WaveFitNeXtDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.upsampler = (
            nn.ConvTranspose1d(
                config.latent_dim,
                config.latent_dim,
                kernel_size=2 * config.latent_upsample,
                stride=config.latent_upsample,
                padding=config.latent_upsample // 2,
            )
            if config.latent_upsample > 1
            else None
        )
        self.generator = [
            WaveNeXtGenerator(config) for _ in range(config.num_iteration)
        ]

    def __call__(self, latents):
        """[batch, latent frames, channels] -> [batch, audio samples]."""
        if self.upsampler is not None:
            latents = self.upsampler(latents)
        waveform = None
        for generator in self.generator:
            waveform = generator(latents, waveform)
            mx.eval(waveform)
        return waveform
