import unittest

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.stt.models.qwen3_asr.qwen3_asr import _rope_safe


class TestRopeSafe(unittest.TestCase):
    """Regression test for the mx.fast.rope batched single-token bug.

    nn.RoPE on a (B, heads, 1, dim) tensor with B > 1 corrupts every row but
    the first, which silently breaks batched single-token decode. _rope_safe
    must return identical outputs for identical batch rows and match the
    single-row reference exactly.
    """

    def test_batched_single_token_matches_single_row(self):
        rope = nn.RoPE(128, traditional=False, base=1_000_000.0)
        row = mx.random.normal((1, 16, 1, 128))
        batched = mx.concatenate([row, row], axis=0)  # two identical rows

        ref = rope(row, offset=300)
        out = _rope_safe(rope, batched, 300)

        self.assertEqual(float(mx.max(mx.abs(out[0] - out[1]))), 0.0)
        self.assertEqual(float(mx.max(mx.abs(out[0] - ref[0]))), 0.0)

    def test_multi_token_unchanged(self):
        rope = nn.RoPE(128, traditional=False, base=1_000_000.0)
        x = mx.random.normal((2, 16, 4, 128))
        self.assertTrue(
            mx.allclose(_rope_safe(rope, x, 300), rope(x, offset=300)).item()
        )


if __name__ == "__main__":
    unittest.main()
