"""Tests for the Ming-3.0 attention mask builder.

Covers the two defects that made ``Ming-3.0-tts-0.5B`` unusable on Apple
Silicon: the additive mask was hard-coded to float32 while the activations are
bfloat16, and a disabled sliding window masked out every attention row.
"""

import mlx.core as mx
import pytest

from mlx_audio.tts.models.bailingmm.bailingmm import MingQwen2Attention

# The mask uses a finite negative sentinel that overflows to -inf in
# low-precision dtypes, so compare against a threshold instead of -inf.
MASKED = -1e30


def _build(mask, q_len, kv_len, sliding_window, dtype, cache_offset=0):
    return MingQwen2Attention._build_sliding_additive_mask(
        mask=mask,
        q_len=q_len,
        kv_len=kv_len,
        cache_offset=cache_offset,
        sliding_window=sliding_window,
        dtype=dtype,
    )


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
def test_mask_dtype_follows_query_dtype(dtype):
    """The mask must be built in the dtype of the query activations."""
    mask = _build("causal", 4, 4, 0, dtype)
    assert mask.dtype == dtype


def test_disabled_sliding_window_keeps_causal_structure():
    """sliding_window=0 must not mask out every position.

    Regression: ``sw_allowed = k_pos > (q_pos - 0)`` is False for every
    causal position, so the sliding mask used to become all -3.4e38 and
    every attention row was masked out.
    """
    mask = _build("causal", 4, 4, 0, mx.bfloat16)
    values = mask[0, 0].tolist()

    for row in range(4):
        for col in range(4):
            if col <= row:
                assert values[row][col] == 0.0, f"({row},{col}) should be visible"
            else:
                assert values[row][col] < MASKED, f"({row},{col}) should be masked"


def test_enabled_sliding_window_still_applies():
    """A positive sliding window keeps restricting the visible range."""
    mask = _build("causal", 4, 4, 2, mx.float32)
    values = mask[0, 0].tolist()

    # With window=2, row 3 can only see columns 2 and 3.
    assert values[3][0] < MASKED
    assert values[3][1] < MASKED
    assert values[3][2] == 0.0
    assert values[3][3] == 0.0


def test_single_token_without_mask_returns_none():
    """A single decode step attends to everything, so no mask is needed."""
    assert _build(None, 1, 4, 0, mx.bfloat16, cache_offset=3) is None
