"""Convert Stability's MLX export of Stable Audio 3 to the mlx-audio layout."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import mlx.core as mx

from mlx_audio.utils import get_model_path

from .stable_audio_3 import TEXT_ENCODER_DIR

OPTIMIZED_REPO = "stabilityai/stable-audio-3-optimized"
OPTIMIZED_REVISION = "30cb2644da54bfa63275d1bc7f1fa67fc718680f"
DIT_FILE = "dit_medium_f16.npz"
DECODER_FILE = "same_l_decoder_f32.npz"
BASE_REPO = "stabilityai/stable-audio-3-medium-base"
BASE_REVISION = "c397d9a6912dc8b1f71a80e7b9673f4d307e7f8c"
MODEL_TYPE = "stable_audio_3"
TEXT_ENCODER_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
)

CONDITIONER_KEYS = {
    "cond.padding_embedding": "conditioner.conditioners.prompt.padding_embedding",
    "cond.seconds_total_weight": "conditioner.conditioners.seconds_total.embedder.embedding.1.weight",
    "cond.seconds_total_bias": "conditioner.conditioners.seconds_total.embedder.embedding.1.bias",
}


def dit_key(key: str) -> str:
    if key.startswith("cond."):
        return CONDITIONER_KEYS[key]
    key = re.sub(r"(_norm)\.weight$", r"\1.gamma", key)
    key = key.replace(".to_local_embed.seq.", ".to_local_embed.")
    return "model.model." + key


def decoder_key(key: str) -> str:
    if key == "running_std":
        return "pretransform.model.bottleneck.running_std"
    if key.startswith("project_in."):
        return "pretransform.model.decoder.layers.1." + key[len("project_in.") :]
    key = key.replace("blocks.", "transformers.").replace(".attn.", ".self_attn.")
    key = key.replace(".ff.glu_proj.", ".ff.ff.0.proj.")
    key = key.replace(".ff.proj_out.", ".ff.ff.2.")
    return "pretransform.model.decoder.layers.3." + key


def convert_weights(dit: dict, decoder: dict) -> dict:
    weights = {dit_key(name): value for name, value in dit.items()}
    for name, value in decoder.items():
        if name == "mapping.weight" and value.ndim == 3:
            value = value.transpose(0, 2, 1)
        weights[decoder_key(name)] = value
    return weights


def convert(
    mlx_path: str | Path,
    source: str = OPTIMIZED_REPO,
    revision: str | None = OPTIMIZED_REVISION,
    base: str | None = None,
) -> None:
    source_dir = get_model_path(
        source,
        revision=revision,
        allow_patterns=[f"MLX/{DIT_FILE}", f"MLX/{DECODER_FILE}"],
    )
    base_dir = get_model_path(
        base or BASE_REPO,
        revision=None if base else BASE_REVISION,
        allow_patterns=[
            "model_config.json",
            *(f"{TEXT_ENCODER_DIR}/{n}" for n in TEXT_ENCODER_FILES),
        ],
    )

    out = Path(mlx_path)
    out.mkdir(parents=True, exist_ok=True)
    weights = convert_weights(
        mx.load(str(source_dir / "MLX" / DIT_FILE)),
        mx.load(str(source_dir / "MLX" / DECODER_FILE)),
    )
    mx.save_safetensors(
        str(out / "model.safetensors"), weights, metadata={"format": "mlx"}
    )

    config = json.loads((base_dir / "model_config.json").read_text())
    config["model_type"] = MODEL_TYPE
    for name in ("config.json", "model_config.json"):
        (out / name).write_text(json.dumps(config, indent=2))

    text_dir = out / TEXT_ENCODER_DIR
    text_dir.mkdir(exist_ok=True)
    for name in TEXT_ENCODER_FILES:
        shutil.copyfile(base_dir / TEXT_ENCODER_DIR / name, text_dir / name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-path", required=True, help="Output directory")
    parser.add_argument(
        "--source",
        default=OPTIMIZED_REPO,
        help="Stability optimized repo id or a local directory with an MLX/ folder",
    )
    parser.add_argument("--revision", default=OPTIMIZED_REVISION)
    parser.add_argument(
        "--base",
        default=None,
        help="Repo id or local directory with model_config.json and the text encoder",
    )
    args = parser.parse_args()
    convert(args.mlx_path, args.source, args.revision, args.base)


if __name__ == "__main__":
    main()
