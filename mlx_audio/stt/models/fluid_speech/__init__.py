import json
import shutil
from pathlib import Path

import mlx.core as mx

from .config import ModelConfig
from .fluid_speech import Model

__all__ = ["Model", "ModelConfig"]


def load_source_weights(model_path):
    """Select one checkpoint; the release also holds 8-bit weights and adapters."""
    model_path = Path(model_path)
    index = model_path / "model.safetensors.index.json"
    if index.is_file():
        files = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        if any(Path(name).name != name for name in files):
            raise ValueError(
                "Fluid Speech weight shards must be in the model directory"
            )
    elif (model_path / "model.safetensors").is_file():
        files = ["model.safetensors"]
    else:
        raise FileNotFoundError(f"No model.safetensors or weight index in {model_path}")
    weights = {}
    for name in files:
        weights.update(mx.load(str(model_path / name)))
    return weights


# The source already uses MLX layouts; select the same files during inference
# and conversion. Other families may have conversion-only source loaders.
load_runtime_weights = load_source_weights


def copy_supporting_files(source, destination):
    adapter = Path(source) / "clean-mode.safetensors"
    if adapter.is_file():
        shutil.copy2(adapter, Path(destination) / adapter.name)


def write_model_card(path, source, repo, config):
    from .model_card import model_card

    (Path(path) / "README.md").write_text(model_card(source, repo))
