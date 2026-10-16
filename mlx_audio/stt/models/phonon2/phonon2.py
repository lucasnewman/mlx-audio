"""Phonon-2 on the standard Parakeet graph, with exact packed encoder layers.

TDT scheduling adapted from fermion-research 0.2.11, Copyright 2026
Fermion Research, Apache-2.0. See LICENSE/NOTICE.
"""

from os import PathLike

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.stt.models.nemo.alignment import (
    AlignedToken,
    sentences_to_result,
    tokens_to_sentences,
)
from mlx_audio.stt.models.parakeet import tokenizer
from mlx_audio.stt.models.parakeet.parakeet import ParakeetTDT, ParakeetTDTArgs
from mlx_audio.utils import from_dict

from .packed import ENCODER_SUFFIXES, PackedFiveValueLinear


class Model(ParakeetTDT):
    def __init__(self, config):
        decoding = config.get("decoding", {})
        durations = decoding.get("durations")
        greedy = decoding.get("greedy") or {}
        maximum = greedy.get("max_symbols")
        if (
            decoding.get("model_type") != "tdt"
            or not isinstance(durations, list)
            or not durations
            or any(type(d) is not int or d < 0 for d in durations)
            or len(durations) != config["joint"]["num_extra_outputs"]
            or type(maximum) is not int
            or maximum <= 0
        ):
            raise ValueError(
                "Phonon-2 requires valid durations and a positive max_symbols"
            )
        super().__init__(from_dict(ParakeetTDTArgs, config))
        expected = {
            f"encoder.layers.{i}.{suffix}"
            for i in range(self.encoder_config.n_layers)
            for suffix in ENCODER_SUFFIXES
        }
        modules = config["phonon2_modules"]
        names = [r["name"] for r in modules]
        if len(names) != len(set(names)) or set(names) != expected:
            raise ValueError(
                "Phonon-2 must replace every encoder projection exactly once"
            )
        self._packed_modules = []
        for row in modules:
            parts = row["name"].split(".")
            parent = self
            for part in parts[:-1]:
                parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
            old = getattr(parent, parts[-1])
            outputs, inputs = row["shape"]
            shape = (
                (outputs, 1, inputs)
                if isinstance(old, nn.Conv1d)
                else (outputs, inputs)
            )
            if tuple(old.weight.shape) != shape:
                raise ValueError(f"Phonon-2 encoder shape mismatch: {row['name']}")
            layer = PackedFiveValueLinear(inputs, outputs, bias="bias" in old)
            setattr(parent, parts[-1], layer)
            self._packed_modules.append(layer)
        self._step = self._make_step()

    def load_weights(self, weights, strict=False):
        # Partial loads invalidate the source model's numerical contract.
        if isinstance(weights, (str, PathLike)):
            weights = list(mx.load(str(weights)).items())
        else:
            weights = list(weights)
        for name, value in weights:
            if name.endswith((".base_q", ".residual_q")) and value.dtype != mx.uint32:
                raise ValueError("Phonon-2 packed codes must remain uint32")
            if name.endswith((".row_lo", ".row_hi")) and value.dtype != mx.float16:
                raise ValueError("Phonon-2 row palettes must remain float16")
        result = super().load_weights(weights, strict=True)
        for module in self._packed_modules:
            module.materialize()
        return result

    def _make_step(self):
        durations = mx.array(self.durations, mx.int32)

        def step(feature, last, h, c, time, symbols, length):
            valid = time < length
            pred, dec, h2, c2 = self._tdt_step(feature, last.reshape(1, 1), h, c)
            duration = mx.take(durations, dec)
            emit = valid & (pred != self.blank_id)
            token = mx.where(emit, pred, -1)
            last = mx.where(emit, pred, last)
            h, c = mx.where(emit, h2, h), mx.where(emit, c2, c)
            symbols = symbols + 1
            advanced = time + duration
            if self.max_symbols is not None:
                hit = (duration == 0) & (symbols >= self.max_symbols)
                advanced = mx.where(hit, advanced + 1, advanced)
                symbols = mx.where((duration != 0) | hit, 0, symbols)
            else:
                symbols = mx.where(duration != 0, 0, symbols)
            new_time = mx.where(valid, advanced, time)
            return (
                token,
                time,
                duration,
                new_time,
                last,
                h,
                c,
                symbols,
                mx.minimum(new_time, length - 1),
            )

        return mx.compile(step)

    def decode(self, mel):
        if mel.ndim == 2:
            mel = mel[None]
        features, lengths = self.encoder(mel)
        mx.eval(features, lengths)
        results = []
        sub = self.encoder_config.subsampling_factor
        sr, hop = (
            self.preprocessor_config.sample_rate,
            self.preprocessor_config.hop_length,
        )
        for b in range(mel.shape[0]):
            length = int(lengths[b])
            if length <= 0:
                results.append(sentences_to_result([]))
                continue
            h, c = self._make_initial_decoder_state(1, features.dtype)
            time, last, symbols = mx.array(0), mx.array(self.blank_id), mx.array(0)
            index, limit = mx.array([0]), mx.array(length)
            hypothesis = []
            host_time = 0
            while host_time < length:
                tokens, starts, durations = [], [], []
                for _ in range(16):
                    frame = mx.take(features[b : b + 1], index, axis=1)
                    token, at, duration, time, last, h, c, symbols, next_index = (
                        self._step(frame, last, h, c, time, symbols, limit)
                    )
                    index = next_index.reshape(1)
                    tokens.append(token)
                    starts.append(at)
                    durations.append(duration)
                values = mx.stack(tokens + starts + durations + [time]).tolist()
                host_time = values[-1]
                for i in range(16):
                    token = values[i]
                    if token < 0 or tokenizer.is_special_token(token, self.vocabulary):
                        continue
                    hypothesis.append(
                        AlignedToken(
                            token,
                            text=tokenizer.decode([token], self.vocabulary),
                            start=values[16 + i] * sub / sr * hop,
                            duration=values[32 + i] * sub / sr * hop,
                        )
                    )
            results.append(sentences_to_result(tokens_to_sentences(hypothesis)))
        return results
