from __future__ import annotations

from collections.abc import Mapping

import mlx.core as mx

from .convert_lora import LoraModule


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
