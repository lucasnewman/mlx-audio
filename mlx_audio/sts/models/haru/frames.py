# Adapted from higurashi-ai/Haru (Apache-2.0).
# https://github.com/higurashi-ai/Haru/tree/main/haru_mlx

"""Settings of a conversation and the frame it produces every 133 ms."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

from .sampling import DryConfig

if TYPE_CHECKING:
    import mlx.core as mx

    from .sampling import StreamDetokenizer


@dataclass
class DuplexConfig:
    guidance_scale: float = 1.5  # audio guidance (valid range 1-2)
    temperature: float = 0.7
    top_k: int = 25
    top_p: float = 1.0
    dry: DryConfig | None = field(default_factory=DryConfig)
    eop_bias: float = (
        0.0  # added to the EOP logit: > 0 makes the model start speaking sooner
    )
    speak_first: bool = False  # the model opens the conversation
    max_frames: int = 27_000  # 60 minutes
    reference_seconds: float = 180.0
    seed: int | None = None
    meanflow_steps: int | None = None  # None uses the checkpoint's head.num_steps

    def __post_init__(self):
        if self.max_frames < 1 or self.reference_seconds <= 0:
            raise ValueError("max_frames and reference_seconds must be positive")
        if not 1.0 <= self.guidance_scale <= 2.0:
            raise ValueError("guidance_scale must be between 1 and 2")
        if not 0 < self.top_p <= 1 or self.top_k < 0:
            raise ValueError("top_p must be in (0, 1] and top_k must be nonnegative")
        if self.meanflow_steps is not None and (
            type(self.meanflow_steps) is not int or self.meanflow_steps < 1
        ):
            raise ValueError("meanflow_steps must be a positive integer")


@dataclass
class Frame:
    audio: np.ndarray  # float32 [3200] at 24 kHz
    token: int
    text: str  # new text of this frame, "" if none
    index: int
    seconds: float  # compute time
    done: bool = False


@dataclass
class PendingFrame:
    """Scheduled generation whose token and PCM may still be evaluating.

    Resolve on the session's thread before handing PCM to an audio callback.
    ``seconds`` measures time through decoder dispatch; the resolved
    frame measures wall time through resolution, including any time held.
    """

    audio: mx.array = field(repr=False)
    _token_data: mx.array = field(repr=False)
    _detokenizer: StreamDetokenizer = field(repr=False)
    index: int
    seconds: float
    started: float
    done: bool = False

    @cached_property
    def token(self) -> int:
        """Read the token on demand; accessing this may wait for the backbone."""
        return int(self._token_data.item())

    @cached_property
    def text(self) -> str:
        return self._detokenizer.push(self.token)

    def resolve(self) -> Frame:
        audio = np.array(self.audio, dtype=np.float32)
        return Frame(
            audio=audio,
            token=self.token,
            text=self.text,
            index=self.index,
            seconds=time.perf_counter() - self.started,
            done=self.done,
        )
