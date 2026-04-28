import re
from pathlib import Path
from typing import Dict

import mlx.core as mx
from mlx.utils import tree_flatten


def materialize_weight_norm(g: mx.array, v: mx.array) -> mx.array:
    axes = tuple(range(1, v.ndim))
    v_norm = mx.sqrt(mx.sum(v * v, axis=axes, keepdims=True))
    return g * v / (v_norm + 1e-12)


def _merge_weight_norm(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    pairs: dict[str, dict[str, mx.array]] = {}
    merged = {}
    for key, value in weights.items():
        if ".parametrizations.weight.original0" in key:
            base = key.replace(".parametrizations.weight.original0", ".weight")
            pairs.setdefault(base, {})["g"] = value
        elif ".parametrizations.weight.original1" in key:
            base = key.replace(".parametrizations.weight.original1", ".weight")
            pairs.setdefault(base, {})["v"] = value
        else:
            merged[key] = value
    for key, pair in pairs.items():
        if "g" in pair and "v" in pair:
            merged[key] = materialize_weight_norm(pair["g"], pair["v"])
    return merged


def load_torch_weights(path: str | Path) -> Dict[str, mx.array]:
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    return {k: mx.array(v.detach().cpu().numpy()) for k, v in state.items()}


def sanitize_flow_weights(model, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    curr = dict(tree_flatten(model.parameters()))
    sanitized = {}
    for key, value in _merge_weight_norm(weights).items():
        if "num_batches_tracked" in key:
            continue

        new_key = key
        new_key = new_key.replace(".embed.out.0.", ".embed.linear.")
        new_key = new_key.replace(".embed.out.1.", ".embed.norm.")
        new_key = new_key.replace(".up_embed.out.0.", ".up_embed.linear.")
        new_key = new_key.replace(".up_embed.out.1.", ".up_embed.norm.")

        if "weight" in new_key and value.ndim == 3:
            if new_key in curr and value.shape != curr[new_key].shape:
                value = mx.swapaxes(value, 1, 2)

        if new_key in curr:
            sanitized[new_key] = value
    for key in (
        "decoder.rand_noise",
        "encoder.embed.pos_enc.pe",
        "encoder.up_embed.pos_enc.pe",
    ):
        if key in curr and key not in sanitized:
            sanitized[key] = curr[key]
    return sanitized


def sanitize_hift_weights(model, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    curr = dict(tree_flatten(model.parameters()))
    sanitized = {}
    for key, value in _merge_weight_norm(weights).items():
        if "num_batches_tracked" in key:
            continue
        new_key = key.removeprefix("generator.")
        new_key = re.sub(
            r"f0_predictor\.condnet\.([02468])\.",
            lambda m: f"f0_predictor.condnet.{int(m.group(1)) // 2}.",
            new_key,
        )

        if "weight" in new_key and value.ndim == 3:
            if new_key in curr and value.shape != curr[new_key].shape:
                if new_key.startswith("ups."):
                    value = mx.transpose(value, (1, 2, 0))
                else:
                    value = mx.swapaxes(value, 1, 2)

        if new_key in curr:
            sanitized[new_key] = value
    if "stft_window" in curr and "stft_window" not in sanitized:
        sanitized["stft_window"] = curr["stft_window"]
    return sanitized


def load_flow_weights(model, path: str | Path, strict: bool = True) -> None:
    path = Path(path)
    if path.suffix == ".safetensors":
        weights = mx.load(str(path), format="safetensors")
    else:
        weights = sanitize_flow_weights(model, load_torch_weights(path))
    model.load_weights(list(weights.items()), strict=strict)


def load_hift_weights(model, path: str | Path, strict: bool = True) -> None:
    path = Path(path)
    if path.suffix == ".safetensors":
        weights = mx.load(str(path), format="safetensors")
    else:
        weights = sanitize_hift_weights(model, load_torch_weights(path))
    model.load_weights(list(weights.items()), strict=strict)


def convert_stepaudio2_assets(input_dir: str | Path, output_dir: str | Path) -> None:
    from .flow import CausalMaskedDiffWithXvec
    from .hift import StepAudio2HiFTGenerator

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    flow = CausalMaskedDiffWithXvec()
    flow_weights = sanitize_flow_weights(
        flow, load_torch_weights(input_dir / "flow.pt")
    )
    mx.save_safetensors(str(output_dir / "flow.safetensors"), flow_weights)

    hift = StepAudio2HiFTGenerator()
    hift_weights = sanitize_hift_weights(
        hift, load_torch_weights(input_dir / "hift.pt")
    )
    mx.save_safetensors(str(output_dir / "hift.safetensors"), hift_weights)
