from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx_audio.stt.models.mega_asr.convert_lora import LoraModule


class _SelfAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(16, 32)


class _AudioLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _SelfAttn()


class _AudioTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [_AudioLayer()]
        self.conv_out = nn.Linear(20, 8, bias=False)


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj = nn.Linear(24, 12, bias=False)


class _TextLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _MLP()


class _TextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [_TextLayer()]


class _StubModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_tower = _AudioTower()
        self.model = _TextModel()


def _build_stub_and_adapter() -> tuple[_StubModel, dict[str, LoraModule]]:
    model = _StubModel()
    r = 4
    adapter: dict[str, LoraModule] = {
        "audio_tower.layers.0.self_attn.q_proj": {
            "A": mx.random.normal((r, 16)),
            "B": mx.random.normal((32, r)),
            "scaling": 1.0,
        },
        "audio_tower.conv_out": {
            "A": mx.random.normal((r, 20)),
            "B": mx.random.normal((8, r)),
            "scaling": 0.5,
        },
        "model.layers.0.mlp.down_proj": {
            "A": mx.random.normal((r, 24)),
            "B": mx.random.normal((12, r)),
            "scaling": 2.0,
        },
    }
    return model, adapter


def test_resolve_linear_returns_correct_leaf():
    from mlx_audio.stt.models.mega_asr.lora import resolve_linear

    model, _ = _build_stub_and_adapter()

    q = resolve_linear(model, "audio_tower.layers.0.self_attn.q_proj")
    assert q is model.audio_tower.layers[0].self_attn.q_proj
    assert isinstance(q, nn.Linear)

    d = resolve_linear(model, "model.layers.0.mlp.down_proj")
    assert d is model.model.layers[0].mlp.down_proj
    assert isinstance(d, nn.Linear)


def test_apply_remove_roundtrip():
    from mlx_audio.stt.models.mega_asr.lora import (
        apply_deltas,
        build_deltas,
        remove_deltas,
        resolve_linear,
    )

    model, adapter = _build_stub_and_adapter()
    deltas = build_deltas(adapter)
    paths = list(adapter)

    base = {p: np.array(resolve_linear(model, p).weight) for p in paths}

    apply_deltas(model, deltas)
    for p in paths:
        after = np.array(resolve_linear(model, p).weight)
        assert not np.allclose(after, base[p])

    remove_deltas(model, deltas)
    for p in paths:
        restored = np.array(resolve_linear(model, p).weight)
        assert np.allclose(restored, base[p], atol=1e-4)


def test_double_apply_is_guarded():
    from mlx_audio.stt.models.mega_asr.lora import apply_deltas, build_deltas

    model, adapter = _build_stub_and_adapter()
    deltas = build_deltas(adapter)

    apply_deltas(model, deltas)
    with pytest.raises(RuntimeError):
        apply_deltas(model, deltas)


def test_remove_when_inactive_is_guarded():
    from mlx_audio.stt.models.mega_asr.lora import build_deltas, remove_deltas

    model, adapter = _build_stub_and_adapter()
    deltas = build_deltas(adapter)

    with pytest.raises(RuntimeError):
        remove_deltas(model, deltas)


def test_fp16_weight_keeps_dtype_and_accumulates_in_fp32():
    from mlx_audio.stt.models.mega_asr.lora import (
        apply_deltas,
        build_deltas,
        resolve_linear,
    )

    model, adapter = _build_stub_and_adapter()
    path = "audio_tower.layers.0.self_attn.q_proj"
    leaf = resolve_linear(model, path)
    leaf.weight = leaf.weight.astype(mx.float16)
    base_fp32 = np.array(leaf.weight.astype(mx.float32))

    deltas = build_deltas(adapter)
    apply_deltas(model, deltas)

    out = resolve_linear(model, path).weight
    assert out.dtype == mx.float16
    expected = base_fp32 + np.array(deltas[path])
    assert np.allclose(np.array(out.astype(mx.float32)), expected, atol=1e-2)
