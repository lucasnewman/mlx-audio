import os
import unittest
from pathlib import Path

import mlx.core as mx

from mlx_audio.codec.models.stepaudio2 import (
    DiT,
    StepAudio2CAMPPlus,
    StepAudio2HiFTGenerator,
)
from mlx_audio.codec.models.stepaudio2.convert import (
    load_campplus_weights,
    load_torch_weights,
    sanitize_flow_weights,
    sanitize_hift_weights,
)
from mlx_audio.codec.models.stepaudio2.flow import CausalMaskedDiffWithXvec


class TestStepAudio2Codec(unittest.TestCase):
    def test_tiny_dit_forward_shape(self):
        model = DiT(
            in_channels=8,
            out_channels=2,
            mlp_ratio=2.0,
            depth=1,
            num_heads=2,
            head_dim=2,
            hidden_size=4,
        )
        x = mx.zeros((1, 2, 3))
        mu = mx.zeros((1, 2, 3))
        spks = mx.zeros((1, 2))
        cond = mx.zeros((1, 2, 3))
        t = mx.array([0.5])
        mask = mx.ones((1, 1, 3), dtype=mx.bool_)
        out = model(x, mask, mu, t, spks, cond)
        self.assertEqual(out.shape, (1, 2, 3))

    @unittest.skipUnless(
        os.environ.get("STEPAUDIO2_ASSETS"),
        "set STEPAUDIO2_ASSETS to a token2wav asset directory",
    )
    def test_reference_assets_strict_load(self):
        assets = Path(os.environ["STEPAUDIO2_ASSETS"])

        flow = CausalMaskedDiffWithXvec()
        flow_weights = sanitize_flow_weights(
            flow, load_torch_weights(assets / "flow.pt")
        )
        flow.load_weights(list(flow_weights.items()), strict=True)

        hift = StepAudio2HiFTGenerator()
        hift_weights = sanitize_hift_weights(
            hift, load_torch_weights(assets / "hift.pt")
        )
        hift.load_weights(list(hift_weights.items()), strict=True)

        speaker = StepAudio2CAMPPlus()
        load_campplus_weights(speaker, assets / "campplus.onnx", strict=True)

        mx.eval(flow.parameters(), hift.parameters(), speaker.parameters())
