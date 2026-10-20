import argparse
import json
import re
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


def load_source_weights(path: Path, weights_name: str | None = None) -> dict:
    if weights_name is None and (path / "config.json").is_file():
        weights_name = json.loads((path / "config.json").read_text()).get(
            "weights_name"
        )
    # Voice embeddings are also safetensors; never merge them into model weights.
    preferred = [weights_name, "model.safetensors"]
    for name in preferred:
        if name and (path / name).is_file():
            return mx.load(str(path / name))
    files = sorted(path.glob("model*.safetensors"))
    if files:
        return {k: v for file in files for k, v in mx.load(str(file)).items()}
    files = sorted(path.glob("*.gguf"))
    if len(files) == 1:
        return mx.load(str(files[0]))
    # Load NPZ and custom-named safetensors shards, excluding voice embeddings.
    weights = {}
    for file in sorted([*path.glob("*.safetensors"), *path.glob("*.npz")]):
        shard = mx.load(str(file))
        if any(key.startswith(("flow_lm.", "mimi.")) for key in shard):
            weights.update(shard)
    if weights:
        return weights
    raise FileNotFoundError(
        f"Expected model safetensors or one GGUF checkpoint in {path}."
    )


def _seanet_names(prefix, cfg, encoder):
    """Map the upstream Sequential indices to the MLX SEANet modules."""
    names = {f"{prefix}.model.0": f"{prefix}.init_conv1d"}
    index = 1
    for layer in range(len(cfg.ratios)):
        dest = f"{prefix}.layers.{layer}"
        if not encoder:
            names[f"{prefix}.model.{index + 1}"] = f"{dest}.upsample"
            index += 2
        for residual in range(cfg.n_residual_layers):
            for block in range(2):
                names[f"{prefix}.model.{index}.block.{2 * block + 1}"] = (
                    f"{dest}.residuals.{residual}.block.{block}"
                )
            index += 1
        if encoder:
            names[f"{prefix}.model.{index + 1}"] = f"{dest}.downsample"
            index += 2
    names[f"{prefix}.model.{index + 1}"] = f"{prefix}.final_conv1d"
    return names


def convert_weights(model, weights):
    upstream = any(
        ".decoder.model." in k or ".condition_provider." in k for k in weights
    )
    if not upstream:
        model.sanitize(weights)
        return weights

    names = {}
    for kind in ("encoder", "decoder"):
        names.update(
            _seanet_names(f"mimi.{kind}", model.config.mimi.seanet, kind == "encoder")
        )
    names = sorted(names.items(), key=lambda item: -len(item[0]))
    target_keys = dict(tree_flatten(model.parameters()))

    result = {}
    for key, value in weights.items():
        if any(
            part in key
            for part in (
                ".w_s_t.",
                ".quantizer.vq.",
                ".quantizer.logvar_proj.",
            )
        ):
            continue
        key = key.replace(
            "flow_lm.condition_provider.conditioners.speaker_wavs.output_proj.weight",
            "speaker_proj_weight",
        )
        key = key.replace("flow_lm.speaker_proj_weight", "speaker_proj_weight")
        key = key.replace(
            "flow_lm.condition_provider.conditioners.transcript_in_segment.",
            "flow_lm.conditioner.",
        )
        key = key.replace(
            "flow_lm.condition_provider.conditioners.num_speakers.",
            "flow_lm.num_speakers.",
        )
        key = key.replace("flow_lm.backbone.", "flow_lm.transformer.")
        key = key.replace("flow_lm.flow.", "flow_lm.flow_net.")
        key = key.replace("mimi.model.", "mimi.")
        for source, target in names:
            if key.startswith(source + "."):
                key = target + key[len(source) :]
                break
        key = key.replace(".in_proj_weight", ".in_proj.weight")
        key = re.sub(
            r"(\.(?:encoder|decoder)_transformer\.transformer\.layers\.\d+)\.(linear[12])\.",
            r"\1.gating.\2.",
            key,
        )
        if key not in target_keys:
            key = re.sub(r"\.(conv|convtr)\.(weight|bias)$", r".\1.\1.\2", key)
        if value.ndim == 3 and key.endswith(".weight"):
            if ".convtr." in key:
                # Torch [in, out/groups, kernel] -> MLX [out, kernel, in/groups].
                value = (
                    value.transpose(0, 2, 1)
                    if key.startswith("mimi.upsample.")
                    else value.transpose(1, 2, 0)
                )
            else:
                value = value.transpose(0, 2, 1)
        result[key] = value

    model.sanitize(result)
    return result


def _prepare_quantized_modules(model, weights):
    # mx.load exposes GGUF Q8_0 matrices as MLX affine packed weights. Keep them
    # quantized; derive their packing from the target matrix and scale shapes.
    def quantized(path, module):
        if not hasattr(module, "to_quantized") or isinstance(
            module, nn.QuantizedLinear
        ):
            return False
        weight = weights.get(path + ".weight")
        scales = weights.get(path + ".scales")
        if weight is None or scales is None or weight.dtype != mx.uint32:
            return False
        width = module.weight.shape[-1]
        return {
            "bits": weight.shape[-1] * 32 // width,
            "group_size": width // scales.shape[-1],
        }

    if model.config.quantization is None:
        nn.quantize(model, class_predicate=quantized)


def convert(
    hf_path: str,
    mlx_path: str = "phonon-tts",
    quantize: bool = False,
    q_group_size: int = 32,
    q_bits: int = 8,
    dtype: str | None = None,
    dequantize: bool = False,
):
    """Preserve source precision and packing unless explicitly overridden."""
    from mlx_audio.convert import copy_model_files, load_config
    from mlx_audio.lm.convert import dequantize_model, save_config, save_model
    from mlx_audio.utils import apply_quantization, get_model_path

    from .pocket_tts import Model, ModelConfig

    if quantize and dequantize:
        raise ValueError("Choose either quantize or dequantize, not both.")
    if dtype not in (None, "float16", "bfloat16", "float32"):
        raise ValueError(f"Unsupported dtype: {dtype}")
    source = get_model_path(
        hf_path,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.gguf",
            "*.model",
            "*.npz",
        ],
    )
    destination = Path(mlx_path)
    if source.resolve() == destination.resolve():
        raise ValueError("Source and destination must be different directories.")
    config = load_config(source)
    model = Model(ModelConfig.from_dict({**config, "model_path": str(source)}))
    weights = convert_weights(model, load_source_weights(source))
    _prepare_quantized_modules(model, weights)
    apply_quantization(model, config, weights)
    model.load_weights(list(weights.items()), strict=True)
    if dtype is not None:
        model.load_weights(
            [
                (
                    key,
                    (
                        value.astype(getattr(mx, dtype))
                        if mx.issubdtype(value.dtype, mx.floating)
                        else value
                    ),
                )
                for key, value in tree_flatten(model.parameters())
            ]
        )
    if quantize:
        # Existing packed matrices retain their original format and values.
        nn.quantize(
            model,
            group_size=q_group_size,
            bits=q_bits,
            class_predicate=lambda _, module: isinstance(
                module, (nn.Linear, nn.Embedding)
            )
            and module.weight.shape[-1] % q_group_size == 0,
        )
    if dequantize:
        model = dequantize_model(model)
    packed_layers = {
        name: {
            "group_size": module.group_size,
            "bits": module.bits,
            "mode": module.mode,
        }
        for name, module in model.named_modules()
        if isinstance(module, (nn.QuantizedLinear, nn.QuantizedEmbedding))
    }
    config.pop("quantization", None)
    config.pop("quantization_config", None)
    if packed_layers:
        defaults = min(packed_layers.values(), key=lambda spec: spec["group_size"])
        config["quantization"] = {**defaults, **packed_layers}
    config["model_type"] = "pocket_tts"
    config["weights_name"] = "model.safetensors"
    config["mimi"]["encoder_enabled"] = model.config.mimi.encoder_enabled
    config.pop("model_path", None)
    destination.mkdir(parents=True, exist_ok=True)
    copy_model_files(source, destination)
    save_model(destination, model, donate_model=True)
    save_config(config, destination / "config.json")
    print(f"Saved Pocket TTS model to {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf-path", required=True, help="Source directory or Hugging Face repository"
    )
    parser.add_argument("--mlx-path", default="phonon-tts")
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Quantize remaining eligible floating-point layers",
    )
    parser.add_argument("--q-bits", type=int, default=8)
    parser.add_argument("--q-group-size", type=int, default=32)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--dequantize", action="store_true")
    convert(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
