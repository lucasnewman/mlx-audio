"""Fluid Speech Pro by FluidVoice: Parakeet TDT with a frame-cleaning block.

Architecture and preprocessing follow the publisher's fluid_speech_mlx.py:
https://huggingface.co/FluidVoice-Labs/fluid-speech-pro-mlx
"""

from os import PathLike
from pathlib import Path
from typing import Callable

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.stt.models.parakeet.audio import log_mel_spectrogram
from mlx_audio.stt.models.parakeet.conformer import DwStridingSubsampling
from mlx_audio.stt.models.parakeet.parakeet import ParakeetTDT

from .config import ModelConfig

_CLEAN_LAYER_NAMES = (
    "feed_forward1.linear1",
    "feed_forward1.linear2",
    "feed_forward2.linear1",
    "feed_forward2.linear2",
    "self_attn.linear_q",
    "self_attn.linear_k",
    "self_attn.linear_v",
    "self_attn.linear_out",
    "self_attn.linear_pos",
    "conv.pointwise_conv1",
    "conv.pointwise_conv2",
)


class FrameCleaner(nn.Module):
    def __init__(self, width: int, dim: int, kernel: int):
        super().__init__()
        self.ln = nn.LayerNorm(width)
        self.down = nn.Linear(width, dim)
        self.mix = nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=dim)
        self.up = nn.Linear(dim, width)

    def __call__(self, x: mx.array) -> mx.array:
        return x + self.up(nn.gelu(self.mix(self.down(self.ln(x)))))


class FluidSubsampling(DwStridingSubsampling):
    def __init__(self, config: ModelConfig):
        super().__init__(config.encoder)
        spec = config.fluid_speech
        self.fv_insert = FrameCleaner(
            config.encoder.d_model, spec["insert_dim"], spec["insert_kernel"]
        )

    def __call__(self, x: mx.array, lengths: mx.array) -> tuple[mx.array, mx.array]:
        lengths = lengths.astype(mx.int32)
        x = x[..., None]
        # Mask before EVERY convolution, including pointwise convolutions.
        # Their biases otherwise leak padded frames into the extra block.
        for layer in self.conv:
            valid = mx.arange(x.shape[1])[None, :] < lengths[:, None]
            x = layer(mx.where(valid[:, :, None, None], x, 0))
            if isinstance(layer, nn.Conv2d) and layer.stride[0] > 1:
                lengths = (lengths + 1) // 2
        valid = mx.arange(x.shape[1])[None, :] < lengths[:, None]
        x = mx.where(valid[:, :, None, None], x, 0)
        x = self.out(x.transpose(0, 1, 3, 2).reshape(x.shape[0], x.shape[1], -1))
        # Crop only AFTER the residual block: the projected padded frame is
        # part of its right context in NeMo.
        x = self.fv_insert(x)
        return x[:, : int(mx.max(lengths))], lengths


class LowRankAdded(nn.Module):
    """Clean-mode update for a linear layer or a pointwise convolution."""

    def __init__(self, base: nn.Module, a: mx.array, b: mx.array):
        super().__init__()
        self.base = base
        self.a = a
        self.b = b

    def __call__(self, x: mx.array) -> mx.array:
        return self.base(x) + (x @ self.a.astype(x.dtype).T) @ self.b.astype(x.dtype).T


class Model(ParakeetTDT):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.config = config
        self.encoder.pre_encode = FluidSubsampling(config)
        self._clean_mode = False
        self._clean_layers = None
        self._clean_biases = None

    @property
    def sample_rate(self) -> int:
        return self.preprocessor_config.sample_rate

    def _log_mel_spectrogram(self, audio_data: mx.array) -> mx.array:
        if audio_data.ndim != 1 or audio_data.size == 0:
            raise ValueError("Expected a nonempty mono audio waveform")
        minimum = 2 * self.preprocessor_config.hop_length
        if audio_data.size < minimum:
            audio_data = mx.pad(audio_data, (0, minimum - audio_data.size))
        # Keep waveform and feature computation in fp32, including for
        # quantized checkpoints. Cast only after normalization.
        mel = log_mel_spectrogram(
            audio_data.astype(mx.float32), self.preprocessor_config
        )
        return mel.astype(self.encoder.pre_encode.conv[0].weight.dtype)

    def load_weights(self, weights, strict=True):
        if isinstance(weights, (str, PathLike)):
            weights = list(mx.load(str(weights)).items())
        # A missing frame cleaner must never silently run with random weights.
        return super().load_weights(weights, strict=True)

    @staticmethod
    def model_quant_predicate(path: str, module: nn.Module) -> bool:
        return isinstance(module, nn.Linear)

    def _prepare_clean_mode(self):
        path = Path(self.config.model_path or ".") / "clean-mode.safetensors"
        if not path.is_file():
            raise FileNotFoundError(f"Clean mode requires the adapter file: {path}")
        spec = self.config.fluid_speech.get("clean_mode", {})
        rank = spec.get("rank", 0)
        if rank <= 0:
            raise ValueError("Clean mode requires a positive adapter rank")
        scale = float(spec["alpha"]) / rank
        adapter = mx.load(str(path))
        names = [
            f"layers.{i}.{suffix}"
            for i in range(self.encoder_config.n_layers)
            for suffix in _CLEAN_LAYER_NAMES
        ]
        expected = {"decoder_input", "joint_hidden"}
        expected.update(f"{name}.{suffix}" for name in names for suffix in ("A", "B"))
        if set(adapter) != expected or not names:
            raise ValueError("Incomplete clean-mode adapter")
        layers = []
        for name in names:
            *parts, child = name.split(".")
            parent = self.encoder
            for part in parts:
                parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
            base = getattr(parent, child)
            a, b = adapter[f"{name}.A"], adapter[f"{name}.B"]
            if not isinstance(base, (nn.Linear, nn.QuantizedLinear, nn.Conv1d)):
                raise ValueError(f"Unsupported clean-mode layer: {name}")
            if isinstance(base, nn.Conv1d) and base.weight.shape[1] != 1:
                raise ValueError(f"Clean mode requires a pointwise convolution: {name}")
            inputs = base.weight.shape[-1]
            if isinstance(base, nn.QuantizedLinear):
                inputs = base.scales.shape[-1] * base.group_size
            if a.shape != (rank, inputs) or b.shape != (base.weight.shape[0], rank):
                raise ValueError(f"Invalid clean-mode adapter shape: {name}")
            layers.append((parent, child, base, LowRankAdded(base, a, b * scale)))
        lstm = self.decoder.prediction["dec_rnn"].lstm[0]
        decoder_input, joint_hidden = adapter["decoder_input"], adapter["joint_hidden"]
        if (
            decoder_input.shape != (lstm.Wx.shape[1],)
            or joint_hidden.shape != self.joint.pred.bias.shape
        ):
            raise ValueError("Invalid clean-mode decoder or joint input shape")
        lstm_bias = (
            lstm.bias.astype(mx.float32)
            + lstm.Wx.astype(mx.float32) @ decoder_input.astype(mx.float32)
        ).astype(lstm.bias.dtype)
        joint_bias = (
            self.joint.pred.bias.astype(mx.float32) + joint_hidden.astype(mx.float32)
        ).astype(self.joint.pred.bias.dtype)
        self._clean_layers = layers
        self._clean_biases = (lstm.bias, self.joint.pred.bias, lstm_bias, joint_bias)

    def set_clean_mode(self, enabled: bool):
        """Select verbatim or edited transcription without accumulating updates."""
        if enabled == self._clean_mode:
            return
        if self._clean_layers is None:
            self._prepare_clean_mode()
        for parent, child, base, adapted in self._clean_layers:
            setattr(parent, child, adapted if enabled else base)
        base_lstm, base_joint, clean_lstm, clean_joint = self._clean_biases
        self.decoder.prediction["dec_rnn"].lstm[0].bias = (
            clean_lstm if enabled else base_lstm
        )
        self.joint.pred.bias = clean_joint if enabled else base_joint
        # The compiled decoder captures the two changed biases.
        self._compiled_tdt_step = mx.compile(self._tdt_step)
        self._clean_mode = enabled

    def generate(
        self,
        audio: str | Path | mx.array,
        *,
        clean_mode: bool = False,
        dtype: mx.Dtype = mx.float32,
        chunk_duration: float | None = None,
        overlap_duration: float | None = None,
        chunk_callback: Callable | None = None,
        stream: bool = False,
        **kwargs,
    ):
        """Transcribe with timestamps and optional removal of fillers/repetitions.

        ``stream=True`` emits results from overlapping offline chunks; the
        encoder, including its frame cleaner, uses future context.
        """
        if stream:
            return self.stream_generate(
                audio,
                clean_mode=clean_mode,
                dtype=dtype,
                chunk_duration=5.0 if chunk_duration is None else chunk_duration,
                overlap_duration=1.0 if overlap_duration is None else overlap_duration,
                **kwargs,
            )
        self.set_clean_mode(clean_mode)
        return super().generate(
            audio,
            dtype=dtype,
            chunk_duration=chunk_duration,
            overlap_duration=overlap_duration,
            chunk_callback=chunk_callback,
            **kwargs,
        )

    def stream_generate(self, audio, *, clean_mode=False, dtype=mx.float32, **kwargs):
        self.set_clean_mode(clean_mode)
        yield from super().stream_generate(audio, dtype=dtype, **kwargs)
