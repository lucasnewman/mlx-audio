from __future__ import annotations

import mlx.core as mx
import numpy as np


def test_delta_equals_scaled_BA():
    from mlx_audio.stt.models.mega_asr.lora import build_deltas

    A = mx.random.normal((8, 16))
    B = mx.random.normal((32, 8))
    scaling = 1.5

    deltas = build_deltas({"m.proj": {"A": A, "B": B, "scaling": scaling}})

    exp = scaling * (np.array(B) @ np.array(A))
    got = np.array(deltas["m.proj"])

    assert got.shape == (32, 16)
    assert np.allclose(got, exp, atol=1e-5)


def test_delta_shape_is_out_by_in_matching_linear_weight():
    from mlx_audio.stt.models.mega_asr.lora import build_deltas

    A = mx.random.normal((4, 10))
    B = mx.random.normal((20, 4))

    deltas = build_deltas({"a.b.c": {"A": A, "B": B, "scaling": 1.0}})

    assert deltas["a.b.c"].shape == (20, 10)


def test_build_deltas_multiple_modules():
    from mlx_audio.stt.models.mega_asr.lora import build_deltas

    adapter = {
        "audio_tower.layers.0.self_attn.q_proj": {
            "A": mx.random.normal((8, 16)),
            "B": mx.random.normal((16, 8)),
            "scaling": 1.0,
        },
        "model.layers.0.mlp.down_proj": {
            "A": mx.random.normal((8, 32)),
            "B": mx.random.normal((12, 8)),
            "scaling": 1.0,
        },
    }

    deltas = build_deltas(adapter)

    assert set(deltas) == set(adapter)
    assert deltas["audio_tower.layers.0.self_attn.q_proj"].shape == (16, 16)
    assert deltas["model.layers.0.mlp.down_proj"].shape == (12, 32)
