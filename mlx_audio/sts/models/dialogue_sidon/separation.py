"""Shared offline input handling and speaker alignment for Sidon models."""

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx import nn

from mlx_audio.sts.models.base import STSOutput

from .frontend import resample


@dataclass
class SeparationResult:
    speakers: mx.array  # [2, samples], anonymous speaker slots
    sample_rate: int


def align_speakers(previous, current, overlap):
    """Match waveform channels by centered correlation in their overlap."""
    if overlap <= 0:
        return current
    a, b = previous[:, -overlap:], current[:, :overlap]
    a = a - a.mean(axis=-1, keepdims=True)
    b = b - b.mean(axis=-1, keepdims=True)
    norms = (
        mx.sqrt(mx.sum(a * a, axis=-1))[:, None]
        * mx.sqrt(mx.sum(b * b, axis=-1))[None, :]
    )
    correlations = mx.where(norms > 1e-8, (a @ b.T) / mx.maximum(norms, 1e-8), 0)
    swap = (
        correlations[0, 1] + correlations[1, 0]
        > correlations[0, 0] + correlations[1, 1]
    )
    return mx.where(swap, current[::-1], current)


class SeparationModel(nn.Module):
    output_suffix = "_separated"
    default_num_steps = 30

    @property
    def sample_rate(self):
        return self.config.sample_rate

    def _validate_num_steps(self, num_steps):
        raise NotImplementedError

    def generate(
        self,
        audio,
        sample_rate=None,
        *,
        num_steps=None,
        seed=None,
        chunk_seconds=20.0,
        overlap_seconds=5.0,
    ) -> STSOutput:
        """Separate audio into named speaker stems for the shared STS entrypoint."""
        if audio is None:
            raise ValueError("Input audio is required for separation")
        if isinstance(audio, (str, Path)):
            audio = Path(audio).expanduser()
        num_steps = self.default_num_steps if num_steps is None else num_steps
        result = self.separate(
            audio,
            sample_rate=sample_rate,
            num_steps=num_steps,
            seed=seed,
            chunk_seconds=chunk_seconds,
            overlap_seconds=overlap_seconds,
        )
        return STSOutput(
            sample_rate=result.sample_rate,
            audios={
                f"speaker_{index}": waveform
                for index, waveform in enumerate(result.speakers, 1)
            },
        )

    def separate(
        self,
        audio,
        sample_rate=None,
        *,
        num_steps=None,
        seed=None,
        chunk_seconds=20.0,
        overlap_seconds=5.0,
    ) -> SeparationResult:
        """Separate a file or array into two mono waveforms at 24 kHz.

        Arrays must be [samples] or [samples, channels], with sample_rate set.
        chunk_seconds=None enables whole-file inference. Long files use the
        published demo's channel alignment and waveform crossfade. A seed is
        local to this call and does not reset MLX's global random stream.
        """
        num_steps = self.default_num_steps if num_steps is None else num_steps
        self._validate_num_steps(num_steps)
        if chunk_seconds is not None and (
            not np.isfinite(chunk_seconds)
            or chunk_seconds <= 0
            or not np.isfinite(overlap_seconds)
            or not 0 <= overlap_seconds < chunk_seconds
        ):
            raise ValueError("Require chunk_seconds > overlap_seconds >= 0")
        if isinstance(audio, (str, Path)):
            from mlx_audio import audio_io

            audio, sample_rate = audio_io.read(str(audio))
        if sample_rate is None or int(sample_rate) != sample_rate or sample_rate <= 0:
            raise ValueError("A positive integer sample_rate is required for arrays")
        waveform = mx.array(audio).astype(mx.float32)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=-1)
        if waveform.ndim != 1 or waveform.shape[0] == 0:
            raise ValueError("Expected nonempty [samples] or [samples, channels] audio")
        if not mx.all(mx.isfinite(waveform)).item():
            raise ValueError("Audio must contain only finite values")
        output_length = max(
            1, round(waveform.shape[0] * self.sample_rate / sample_rate)
        )
        waveform = resample(waveform, int(sample_rate))
        total = waveform.shape[0]
        chunk = (
            max(1, int(chunk_seconds * 16000)) if chunk_seconds is not None else total
        )
        overlap = int(overlap_seconds * 16000) if chunk_seconds is not None else 0
        hop = chunk - overlap
        key = mx.random.key(seed) if seed is not None else None
        stitched = None
        previous_end = 0
        for start in range(0, total, hop):
            end = min(start + chunk, total)
            if key is not None:
                key, chunk_key = mx.random.split(key)
            else:
                chunk_key = None
            latent = self.predict_latents(waveform[start:end], num_steps, key=chunk_key)
            predicted = self.decode_latents(latent)[0].astype(mx.float32)
            target = max(1, round((end - start) * self.sample_rate / 16000))
            predicted = self._match_length(predicted, target)
            if stitched is None:
                stitched = predicted
            else:
                count = min(
                    max(0, round((previous_end - start) * self.sample_rate / 16000)),
                    stitched.shape[-1],
                    predicted.shape[-1],
                )
                if count:
                    predicted = align_speakers(stitched, predicted, count)
                    fade = mx.linspace(0, 1, count)[None, :]
                    blended = (
                        stitched[:, -count:] * (1 - fade) + predicted[:, :count] * fade
                    )
                    stitched = mx.concatenate(
                        (stitched[:, :-count], blended, predicted[:, count:]), axis=-1
                    )
                else:
                    stitched = mx.concatenate((stitched, predicted), axis=-1)
            previous_end = end
            mx.eval(stitched)
        return SeparationResult(
            self._match_length(stitched, output_length), self.sample_rate
        )

    @staticmethod
    def _match_length(audio, length):
        if audio.shape[-1] < length:
            return mx.pad(audio, ((0, 0), (0, length - audio.shape[-1])))
        return audio[:, :length]

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        from mlx_audio.sts.utils import load_model

        return load_model(path, **kwargs)
