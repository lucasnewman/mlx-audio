"""HIFI-GAN"""

from typing import Dict, Optional, List, Tuple
import numpy as np
import mlx.core as mx
import mlx.nn as nn

from mlx_audio.utils import stft, istft

from .conv import WNConv1d, WNConvTranspose1d


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class Snake(nn.Module):
    def __init__(self, in_features: int, alpha: float = 1.0, alpha_logscale: bool = False):
        super().__init__()

        self.alpha_logscale = alpha_logscale

        self.alpha = (mx.zeros(in_features) if alpha_logscale else mx.ones(in_features)) * alpha

    def __call__(self, x: mx.array):
        alpha = self.alpha[None, :, None]
        if self.alpha_logscale:
            alpha = mx.exp(alpha)

        x += (1.0 / (alpha + 1e-9)) * mx.power(mx.sin(x * alpha), 2)

        return x


class ResBlock(nn.Module):
    """Residual block module in HiFiGAN/BigVGAN."""

    def __init__(
        self,
        channels: int = 512,
        kernel_size: int = 3,
        dilations: List[int] = [1, 3, 5],
    ):
        super(ResBlock, self).__init__()
        self.convs1 = []
        self.convs2 = []

        for dilation in dilations:
            self.convs1.append(
                WNConv1d(channels, channels, kernel_size, stride=1, dilation=dilation, padding=get_padding(kernel_size, dilation))
            )
            self.convs2.append(WNConv1d(channels, channels, kernel_size, stride=1, dilation=1, padding=get_padding(kernel_size, 1)))

        self.activations1 = [Snake(channels, alpha_logscale=False) for _ in range(len(self.convs1))]
        self.activations2 = [Snake(channels, alpha_logscale=False) for _ in range(len(self.convs2))]

    def __call__(self, x: mx.array) -> mx.array:
        for idx in range(len(self.convs1)):
            xt = self.activations1[idx](x)
            xt = self.convs1[idx](xt)
            xt = self.activations2[idx](xt)
            xt = self.convs2[idx](xt)
            x = xt + x
        return x


class SineGen(nn.Module):
    """Definition of sine generator
    SineGen(samp_rate, harmonic_num = 0,
            sine_amp = 0.1, noise_std = 0.003,
            voiced_threshold = 0,
            flag_for_pulse=False)
    samp_rate: sampling rate in Hz
    harmonic_num: number of harmonic overtones (default 0)
    sine_amp: amplitude of sine-wavefrom (default 0.1)
    noise_std: std of Gaussian noise (default 0.003)
    voiced_thoreshold: F0 threshold for U/V classification (default 0)
    flag_for_pulse: this SinGen is used inside PulseGen (default False)
    Note: when flag_for_pulse is True, the first time step of a voiced
        segment is always sin(np.pi) or cos(0)
    """

    def __init__(self, samp_rate, harmonic_num=0, sine_amp=0.1, noise_std=0.003, voiced_threshold=0):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0):
        # generate uv signal
        uv = (f0 > self.voiced_threshold).astype(mx.float32)
        return uv

    def __call__(self, f0):
        """
        :param f0: [B, 1, sample_len], Hz
        :return: [B, 1, sample_len]
        """
        F_mat = mx.zeros((f0.shape[0], self.harmonic_num + 1, f0.shape[-1]))
        for i in range(self.harmonic_num + 1):
            F_mat[:, i : i + 1, :] = f0 * (i + 1) / self.sampling_rate

        theta_mat = 2 * np.pi * (mx.cumsum(F_mat, axis=-1) % 1)

        # Sample uniform random phase
        phase_vec = mx.random.uniform(low=-np.pi, high=np.pi, shape=(f0.shape[0], self.harmonic_num + 1, 1))
        phase_vec = mx.where(mx.arange(self.harmonic_num + 1).reshape(1, -1, 1) == 0, 0, phase_vec)

        # generate sine waveforms
        sine_waves = self.sine_amp * mx.sin(theta_mat + phase_vec)

        # generate uv signal
        uv = self._f02uv(f0)

        # noise: for unvoiced should be similar to sine_amp
        #        std = self.sine_amp/3 -> max value ~ self.sine_amp
        # .       for voiced regions is self.noise_std
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * mx.random.normal(sine_waves.shape)

        # first: set the unvoiced part to 0 by uv
        # then: additive noise
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


class SourceModuleHnNSF(nn.Module):
    """SourceModule for hn-nsf
    SourceModule(sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0)
    sampling_rate: sampling_rate in Hz
    harmonic_num: number of harmonic above F0 (default: 0)
    sine_amp: amplitude of sine source signal (default: 0.1)
    add_noise_std: std of additive Gaussian noise (default: 0.003)
        note that amplitude of noise in unvoiced is decided
        by sine_amp
    voiced_threshold: threhold to set U/V given F0 (default: 0)
    Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
    F0_sampled (batchsize, length, 1)
    Sine_source (batchsize, length, 1)
    noise_source (batchsize, length 1)
    uv (batchsize, length, 1)
    """

    def __init__(self, sampling_rate, upsample_scale, harmonic_num=0, sine_amp=0.1, add_noise_std=0.003, voiced_threshod=0):
        super(SourceModuleHnNSF, self).__init__()

        self.sine_amp = sine_amp
        self.noise_std = add_noise_std

        # to produce sine waveforms
        self.l_sin_gen = SineGen(sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshod)

        # to merge source harmonics into a single excitation
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def __call__(self, x):
        """
        Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
        F0_sampled (batchsize, length, 1)
        Sine_source (batchsize, length, 1)
        noise_source (batchsize, length 1)
        """
        # source for harmonic branch
        sine_wavs, uv, _ = self.l_sin_gen(mx.transpose(x, (0, 2, 1)))
        sine_wavs = mx.transpose(sine_wavs, (0, 2, 1))
        uv = mx.transpose(uv, (0, 2, 1))
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))

        # source for noise branch, in the same shape as uv
        noise = mx.random.normal(uv.shape) * self.sine_amp / 3
        return sine_merge, noise, uv


class HiFTGenerator(nn.Module):
    """
    HiFTNet Generator: Neural Source Filter + ISTFTNet
    https://arxiv.org/abs/2309.09493
    """

    def __init__(
        self,
        in_channels: int = 80,
        base_channels: int = 512,
        nb_harmonics: int = 8,
        sampling_rate: int = 22050,
        nsf_alpha: float = 0.1,
        nsf_sigma: float = 0.003,
        nsf_voiced_threshold: float = 10,
        upsample_rates: List[int] = [8, 8],
        upsample_kernel_sizes: List[int] = [16, 16],
        istft_params: Dict[str, int] = {"n_fft": 16, "hop_len": 4},
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        source_resblock_kernel_sizes: List[int] = [7, 11],
        source_resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5]],
        lrelu_slope: float = 0.1,
        audio_limit: float = 0.99,
        f0_predictor: nn.Module = None,
    ):
        super(HiFTGenerator, self).__init__()

        self.out_channels = 1
        self.nb_harmonics = nb_harmonics
        self.sampling_rate = sampling_rate
        self.istft_params = istft_params
        self.lrelu_slope = lrelu_slope
        self.audio_limit = audio_limit

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.m_source = SourceModuleHnNSF(
            sampling_rate=sampling_rate,
            upsample_scale=np.prod(upsample_rates) * istft_params["hop_len"],
            harmonic_num=nb_harmonics,
            sine_amp=nsf_alpha,
            add_noise_std=nsf_sigma,
            voiced_threshod=nsf_voiced_threshold,
        )

        # F0 upsampling
        self.f0_upsample_scale = float(np.prod(upsample_rates) * istft_params["hop_len"])

        self.conv_pre = WNConv1d(in_channels, base_channels, 7, 1, padding=3)

        # Up
        self.ups = []
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                WNConvTranspose1d(
                    base_channels // (2**i),
                    base_channels // (2 ** (i + 1)),
                    k,
                    u,
                    padding=(k - u) // 2,
                )
            )

        # Down
        self.source_downs = []
        self.source_resblocks = []
        downsample_rates = [1] + upsample_rates[::-1][:-1]
        downsample_cum_rates = np.cumprod(downsample_rates)
        for i, (u, k, d) in enumerate(zip(downsample_cum_rates[::-1], source_resblock_kernel_sizes, source_resblock_dilation_sizes)):
            if u == 1:
                self.source_downs.append(nn.Conv1d(istft_params["n_fft"] + 2, base_channels // (2 ** (i + 1)), 1, 1))
            else:
                self.source_downs.append(nn.Conv1d(istft_params["n_fft"] + 2, base_channels // (2 ** (i + 1)), u * 2, u, padding=(u // 2)))

            self.source_resblocks.append(ResBlock(base_channels // (2 ** (i + 1)), k, d))

        self.resblocks = []
        for i in range(len(self.ups)):
            ch = base_channels // (2 ** (i + 1))
            for _, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(ch, k, d))

        self.conv_post = WNConv1d(ch, istft_params["n_fft"] + 2, 7, 1, padding=3)
        # from mlx.utils import tree_flatten
        # for k, v in tree_flatten(self.conv_post.parameters()):
        #     print(f"{k} has shape {v.shape}")
        # os.exit()

        self.stft_window = "hann"
        self.f0_predictor = f0_predictor

    def _stft(self, x):
        # x shape: (batch, 1, time) -> squeeze to (batch, time)
        x = x.squeeze(1)

        # Process each batch element separately since mlx_audio stft works on 1D signals
        batch_size = x.shape[0]
        results_real = []
        results_imag = []

        for i in range(batch_size):
            spec = stft(
                x[i],
                n_fft=self.istft_params["n_fft"],
                hop_length=self.istft_params["hop_len"],
                window=self.stft_window,
                center=False,
            )
            results_real.append(mx.real(spec))
            results_imag.append(mx.imag(spec))

        real = mx.stack(results_real, axis=0).transpose(0, 2, 1)
        imag = mx.stack(results_imag, axis=0).transpose(0, 2, 1)

        return real, imag

    def _istft(self, magnitude, phase):
        magnitude = mx.clip(magnitude, None, 1e2)
        real = magnitude * mx.cos(phase)
        imag = magnitude * mx.sin(phase)
        complex_spec = real + 1j * imag

        # Process each batch element
        batch_size = magnitude.shape[0]
        results = []

        for i in range(batch_size):
            # Transpose to match expected input shape for istft
            spec_i = complex_spec[i].transpose(1, 0)

            # Apply ISTFT
            signal = istft(spec_i, hop_length=self.istft_params["hop_len"], window=self.stft_window, center=False)
            results.append(signal)

        # Stack batch results
        return mx.stack(results, axis=0)

    def decode(self, x: mx.array, s: mx.array = None) -> mx.array:
        if s is None:
            s = mx.zeros((1, 1, 0))

        s_stft_real, s_stft_imag = self._stft(s)
        s_stft = mx.concatenate([s_stft_real, s_stft_imag], axis=1)

        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = nn.leaky_relu(x, self.lrelu_slope)
            x = self.ups[i](x)

            if i == self.num_upsamples - 1:
                # Reflection padding
                x = mx.pad(x, [(0, 0), (0, 0), (1, 0)], mode="edge")

            # fusion
            si = self.source_downs[i](s_stft)
            si = self.source_resblocks[i](si)
            x = x + si

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs = xs + self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = nn.leaky_relu(x, self.lrelu_slope)
        x = self.conv_post(x)
        magnitude = mx.exp(x[:, : self.istft_params["n_fft"] // 2 + 1, :])
        phase = mx.sin(x[:, self.istft_params["n_fft"] // 2 + 1 :, :])  # actually, sin is redundancy

        x = self._istft(magnitude, phase)
        x = mx.clip(x, -self.audio_limit, self.audio_limit)
        return x

    def __call__(
        self,
        batch: dict,
    ) -> Tuple[mx.array, mx.array]:
        speech_feat = mx.transpose(batch["speech_feat"], (0, 2, 1))
        # mel->f0
        f0 = self.f0_predictor(speech_feat)
        # f0->source
        # Upsample f0
        f0_up = self._upsample_f0(f0[:, None])
        s = mx.transpose(f0_up, (0, 2, 1))  # bs,n,t
        s, _, _ = self.m_source(s)
        s = mx.transpose(s, (0, 2, 1))
        # mel+source->speech
        generated_speech = self.decode(x=speech_feat, s=s)
        return generated_speech, f0

    def _upsample_f0(self, f0):
        """Upsample F0 using nearest neighbor interpolation"""
        # f0 shape: (batch, 1, time)
        batch, channels, time = f0.shape
        new_time = int(time * self.f0_upsample_scale)

        # Create indices for upsampling
        indices = mx.floor(mx.arange(new_time) / self.f0_upsample_scale).astype(mx.int32)
        indices = mx.clip(indices, 0, time - 1)

        # Gather values
        f0_up = mx.take(f0, indices, axis=2)

        return f0_up

    def inference(self, speech_feat: mx.array, cache_source: mx.array = None) -> Tuple[mx.array, mx.array]:
        if cache_source is None:
            cache_source = mx.zeros((1, 1, 0))

        # mel->f0
        f0 = self.f0_predictor(speech_feat)
        # f0->source
        f0_up = self._upsample_f0(f0[:, None])
        s = mx.transpose(f0_up, (0, 2, 1))  # bs,n,t
        s, _, _ = self.m_source(s)
        s = mx.transpose(s, (0, 2, 1))
        # use cache_source to avoid glitch
        if cache_source.shape[2] != 0:
            cache_len = min(cache_source.shape[2], s.shape[2])
            s = mx.concatenate([cache_source[:, :, :cache_len], s[:, :, cache_len:]], axis=2)
        generated_speech = self.decode(x=speech_feat, s=s)
        return generated_speech, s
