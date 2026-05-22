from __future__ import annotations

from collections.abc import Mapping

import mlx.core as mx
import mlx.nn as nn

from .convert_lora import LoraModule

_LORA_ACTIVE_ATTR = "_lora_active"
_FP32_ACCUM_DTYPES = (mx.float16, mx.bfloat16)


def build_deltas(adapter: Mapping[str, LoraModule]) -> dict[str, mx.array]:
    deltas: dict[str, mx.array] = {}
    for path, module in adapter.items():
        a = module["A"].astype(mx.float32)
        b = module["B"].astype(mx.float32)
        scaling = float(module["scaling"])
        delta = scaling * (b @ a)
        assert delta.shape == (b.shape[0], a.shape[1])
        deltas[path] = delta
    return deltas


def resolve_linear(model: nn.Module, path: str) -> nn.Linear:
    node: object = model
    for segment in path.split("."):
        if segment.isdigit():
            if not isinstance(node, (list, tuple)):
                raise TypeError(
                    f"path {path!r}: segment {segment!r} indexes a non-sequence "
                    + type(node).__name__
                )
            node = node[int(segment)]
        else:
            node = getattr(node, segment)
    if not isinstance(node, nn.Linear):
        raise TypeError(
            f"path {path!r} resolves to {type(node).__name__}, expected nn.Linear"
        )
    return node


def _accumulate(leaf: nn.Linear, delta: mx.array, sign: float) -> None:
    weight = leaf.weight
    if weight.shape != delta.shape:
        raise ValueError(
            f"delta shape {tuple(delta.shape)} does not match "
            + f"weight shape {tuple(weight.shape)}"
        )
    signed = delta if sign > 0 else -delta
    if weight.dtype in _FP32_ACCUM_DTYPES:
        updated = (weight.astype(mx.float32) + signed.astype(mx.float32)).astype(
            weight.dtype
        )
    else:
        updated = weight + signed.astype(weight.dtype)
    leaf.weight = updated


def apply_deltas(model: nn.Module, deltas: Mapping[str, mx.array]) -> None:
    if getattr(model, _LORA_ACTIVE_ATTR, False):
        raise RuntimeError("LoRA deltas already applied; remove before re-applying")
    for path, delta in deltas.items():
        _accumulate(resolve_linear(model, path), delta, 1.0)
    setattr(model, _LORA_ACTIVE_ATTR, True)


def remove_deltas(model: nn.Module, deltas: Mapping[str, mx.array]) -> None:
    if not getattr(model, _LORA_ACTIVE_ATTR, False):
        raise RuntimeError("LoRA deltas are not applied; nothing to remove")
    for path, delta in deltas.items():
        _accumulate(resolve_linear(model, path), delta, -1.0)
    setattr(model, _LORA_ACTIVE_ATTR, False)
