# Adapted from higurashi-ai/Haru (Apache-2.0).
# https://github.com/higurashi-ai/Haru/tree/main/haru_mlx

"""A live full-duplex conversation: one user frame in, one model frame out.

    session = DuplexSession(model, system_prompt="...", reference=voice_clip)
    frame = session.start()              # model frame 0
    while True:
        frame = session.step(user_pcm)   # 3200 samples in, 3200 samples out

Frame ``t`` of the model comes from the hidden state at the previous position:
its text token from the output embedding and its audio latent from the MeanFlow
head, decoded to audio for playback. When user frame ``t`` arrives, the position
``text[t] + model_audio[t] + user_audio[t]`` runs and yields frame ``t + 1``.
Frame 0 comes from the end of the prompt, so the model stays exactly one frame
ahead and the frame length (133 ms) is the only structural latency.

A live client can spread the work of a frame over its 133 ms: ``feed()`` takes
microphone audio of any length as it arrives and runs most of the audio encoder
on it, and ``step()`` without an argument finishes the frame once
``frames_ready`` is positive.

Use each session from a single thread; each conversation owns its LM and codec state.
"""

from __future__ import annotations

import os
import time
from dataclasses import replace

import mlx.core as mx
import numpy as np

from .codec import CodecStream
from .frames import DuplexConfig, Frame, PendingFrame
from .model import Model
from .prompt import DEFAULT_SYSTEM_PROMPT, build_prompt
from .sampling import DryState, StreamDetokenizer, sample_token

__all__ = ["DuplexConfig", "DuplexSession", "Frame", "PendingFrame"]

_PREFILL_CHUNK = 512


class DuplexSession:
    def __init__(
        self,
        model: Model,
        system_prompt: str | None = None,
        reference: np.ndarray | None = None,
        config: DuplexConfig | None = None,
    ):
        """``reference``: mono 24 kHz clip of the voice the model should speak with."""
        self.model = model
        self.config = config or DuplexConfig()
        self.system_prompt = (
            DEFAULT_SYSTEM_PROMPT if system_prompt is None else system_prompt
        )
        cfg = model.config
        self.frame_samples = cfg.frame_samples
        self.pad_id, self.eop_id = cfg.pad_token_id, cfg.eop_token_id
        self._silent = {self.pad_id, self.eop_id}
        self._stream_ids = mx.array([self.pad_id, self.eop_id], dtype=mx.int32)
        self._banned = sorted(
            self._silent
            | ({cfg.eos_token_id} if cfg.eos_token_id is not None else set())
        )

        self.reference = None
        if reference is not None and len(reference):
            limit = int(self.config.reference_seconds * cfg.sample_rate)
            self.reference = np.asarray(reference, dtype=np.float32).reshape(-1)[
                -limit:
            ]

        self.index = 0
        self.prompt_length = 0
        self._key = mx.random.key(0)
        self._cache = None
        self._codec = CodecStream(model.codec)
        self._decode_stream = None
        self._pending_frame: PendingFrame | None = None
        self._token: mx.array | None = None
        self._latent: mx.array | None = None
        self._force_text = 0
        self._dry = (
            DryState(self.config.dry, set(self._banned)) if self.config.dry else None
        )
        self._detokenizer = StreamDetokenizer(model.tokenizer, self._silent)

    def _next_key(self) -> mx.array:
        """A fresh random key; the session never touches the global random state."""
        self._key, key = mx.random.split(self._key)
        return key

    # ------------------------------------------------------------- prompt

    def warmup(self) -> None:
        """Compile live frame shapes using separate state before opening audio I/O."""
        warm = DuplexSession(
            self.model,
            config=replace(self.config, seed=0, max_frames=2, speak_first=False),
        )
        warm.start_async().resolve()
        # The microphone encoder uses whole, half, quarter, ... frame chunks.
        hop = self.model.codec.front_hop
        steps = self.frame_samples // hop
        while steps:
            warm.feed(np.zeros(steps * hop, dtype=np.float32))
            steps //= 2
        warm.step_async().resolve()
        warm.synchronize_decoder()

    def start(self) -> Frame:
        """Run the prompt (system prompt + voice reference) and produce frame 0."""
        return self.start_async().resolve()

    def start_async(self) -> PendingFrame:
        """Produce frame 0's latent and schedule PCM on the decoder stream."""
        self.synchronize_decoder()
        started = time.perf_counter()
        model, cfg = self.model, self.config
        seed = int.from_bytes(os.urandom(4), "little") if cfg.seed is None else cfg.seed
        self._key = mx.random.key(seed)
        self._codec.reset()
        self._dry = DryState(cfg.dry, set(self._banned)) if cfg.dry else None
        self._detokenizer = StreamDetokenizer(model.tokenizer, self._silent)
        self._cache = model.new_cache()
        self._pending_frame = None

        prompt = build_prompt(
            model.config,
            model.codec,
            model.tokenizer,
            self.system_prompt,
            self.reference,
            cfg.speak_first,
            self._next_key(),
        )
        self._force_text = 1 if cfg.speak_first else 0
        n = prompt.length

        hidden = None
        for s in range(0, n, _PREFILL_CHUNK):
            e = min(n, s + _PREFILL_CHUNK)
            span = slice(s, e)
            embeddings = model.embed(
                prompt.tokens[None, span],
                prompt.model_latents[None, span],
                prompt.user_latents[None, span],
                prompt.model_mask[None, span],
                prompt.user_mask[None, span],
            )
            hidden = model(embeddings, self._cache)[:, -1]
            mx.eval(hidden, self._cache_arrays())
        self.prompt_length = n
        self.index = 0
        # Encoding the reference and running the prompt leave large freed
        # buffers in MLX's pool; a frame never reuses buffers that large.
        mx.clear_cache()
        return self._emit(hidden, started)

    def synchronize_decoder(self) -> None:
        """Finish pending decoder work on this session's thread."""
        if self._decode_stream is not None:
            mx.synchronize(self._decode_stream)

    # ------------------------------------------------------------- frames

    def feed(self, user_audio: np.ndarray) -> None:
        """Take user audio of any length (float32, 24 kHz) as it arrives."""
        self._codec.feed(mx.array(np.asarray(user_audio, dtype=np.float32).reshape(-1)))
        mx.eval(self._codec.encoder_arrays())

    @property
    def frames_ready(self) -> int:
        """User frames fed and not yet consumed by ``step()``."""
        return self._codec.frames_ready

    def step(self, user_audio: np.ndarray | None = None) -> Frame:
        """Consume one user frame (3200 samples, float32, 24 kHz); return the next model frame.

        Leave ``user_audio`` out when the frame's audio went through ``feed()``.
        """
        return self.step_async(user_audio).resolve()

    def step_async(self, user_audio: np.ndarray | None = None) -> PendingFrame:
        """Schedule generation without reading the current frame's token or PCM.

        Resolve returned frames in order. Keep the number of unresolved frames
        bounded to avoid accumulating playback latency and GPU allocations.
        """
        if self._latent is None:
            raise RuntimeError("call start() first")
        if self.index >= self.config.max_frames:
            raise RuntimeError(
                "session has reached max_frames; call start() to restart"
            )
        started = time.perf_counter()
        model = self.model
        if user_audio is not None:
            audio = np.asarray(user_audio, dtype=np.float32).reshape(-1)
            if audio.shape[0] != self.frame_samples:
                raise ValueError(
                    f"a user frame is {self.frame_samples} samples, got {audio.shape[0]}"
                )
            self._codec.feed(mx.array(audio))
        user_latent = model.codec.sample(self._codec.pop(), self._next_key())
        embeddings = model.embed(self._token[None], self._latent[None], user_latent)
        hidden = model(embeddings, self._cache)[:, -1]
        return self._emit(hidden, started)

    def _emit(self, hidden: mx.array, started: float) -> PendingFrame:
        """Sample the next frame's text token and audio from ``hidden [1, H]``."""
        model, cfg = self.model, self.config
        if self._pending_frame is not None:
            # Run this backbone while reading the previous token for DRY.
            mx.async_eval(hidden, self._cache_arrays(), self._codec.encoder_arrays())
            previous = self._pending_frame
            if self._dry is not None:
                self._dry.update(previous.token)
            # Finalize text in generation order even if callers defer resolution.
            _ = previous.text
            self._pending_frame = None
        logits = model.logits(hidden)
        if self._dry is not None:
            self._dry.apply(logits)
        logits[:, self._stream_ids] = model.logits_fp32(hidden, self._stream_ids)
        if cfg.eop_bias:
            logits[:, self.eop_id] = logits[:, self.eop_id] + cfg.eop_bias
        if self._force_text > 0:
            logits[:, mx.array(self._banned, dtype=mx.int32)] = -mx.inf
            self._force_text -= 1
        token = sample_token(
            logits, cfg.temperature, cfg.top_k, cfg.top_p, self._next_key()
        )

        noise = mx.random.normal(
            (1, model.config.latent_dim), key=self._next_key()
        ).astype(model.dtype)
        latent = model.unscale_latents(
            model.head.sample(
                hidden, noise, cfg.guidance_scale, steps=cfg.meanflow_steps
            )
        )
        mx.async_eval(token, latent, self._cache_arrays(), self._codec.encoder_arrays())
        if self._decode_stream is None:
            self._decode_stream = mx.new_stream(mx.default_device())
        with mx.stream(self._decode_stream):
            audio = mx.clip(self._codec.decode(latent[None])[0], -1.0, 1.0)
            mx.async_eval(audio, self._codec.decoder_state)

        self._token, self._latent = token, latent
        frame = PendingFrame(
            audio=audio,
            _token_data=token,
            _detokenizer=self._detokenizer,
            index=self.index,
            seconds=time.perf_counter() - started,
            started=started,
        )
        self._pending_frame = frame
        self.index += 1
        frame.done = self.index >= cfg.max_frames
        return frame

    def _cache_arrays(self) -> list[mx.array]:
        return [
            array
            for layer_cache in self._cache
            for array in layer_cache.state
            if array is not None
        ]
