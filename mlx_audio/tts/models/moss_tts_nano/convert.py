from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download

DEFAULT_TTS_REPO = "OpenMOSS-Team/MOSS-TTS-Nano"
DEFAULT_AUDIO_TOKENIZER_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"
DTYPES = {
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
    "float32": mx.float32,
}


def _resolve_source(path_or_repo: str, allow_patterns: list[str]) -> Path:
    path = Path(path_or_repo).expanduser()
    if path.exists():
        return path.resolve()
    return Path(
        snapshot_download(
            path_or_repo,
            allow_patterns=allow_patterns,
        )
    )


def _copy_existing(src: Path, dst: Path, names: list[str]) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in names:
        source = src / name
        if source.exists():
            shutil.copy2(source, dst / name)


def _torch_state_dict_to_mlx_safetensors(
    source_path: Path,
    output_path: Path,
    dtype: mx.Dtype,
) -> None:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Converting the upstream MOSS-TTS-Nano pytorch_model.bin requires torch."
        ) from exc

    state = torch.load(str(source_path), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    weights = {}
    for key, tensor in state.items():
        if not hasattr(tensor, "detach"):
            continue
        tensor = tensor.detach().cpu()
        if tensor.is_floating_point():
            # NumPy cannot materialize torch.bfloat16 directly. Cast through
            # float32, then let MLX apply the requested output dtype.
            tensor = tensor.float()
        weights[key] = mx.array(tensor.numpy()).astype(dtype)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(output_path), weights)


def _safetensors_to_mlx_safetensors(
    source_dir: Path,
    output_path: Path,
    dtype: mx.Dtype,
) -> None:
    index_path = source_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        filenames = sorted(set(index.get("weight_map", {}).values()))
    else:
        filenames = [path.name for path in sorted(source_dir.glob("*.safetensors"))]
    if not filenames:
        raise FileNotFoundError(f"No safetensors files found in {source_dir}")

    weights = {}
    for filename in filenames:
        for key, tensor in mx.load(str(source_dir / filename)).items():
            weights[key] = tensor.astype(dtype)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(output_path), weights)


def convert(
    *,
    tts_source: str = DEFAULT_TTS_REPO,
    audio_tokenizer_source: str = DEFAULT_AUDIO_TOKENIZER_REPO,
    output_dir: str | Path,
    dtype: str = "float16",
) -> Path:
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {sorted(DTYPES)}")
    target_dtype = DTYPES[dtype]
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    tts_dir = _resolve_source(
        tts_source,
        allow_patterns=[
            "*.json",
            "*.model",
            "*.py",
            "*.bin",
            "*.txt",
            "*.md",
        ],
    )
    codec_dir = _resolve_source(
        audio_tokenizer_source,
        allow_patterns=["*.json", "*.safetensors", "*.index.json", "*.md"],
    )

    _copy_existing(
        tts_dir,
        output,
        [
            "config.json",
            "tokenizer.model",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "README.md",
        ],
    )
    _copy_existing(
        codec_dir,
        output / "audio_tokenizer",
        [
            "config.json",
            "README.md",
        ],
    )

    config_path = output / "config.json"
    config = json.loads(config_path.read_text())
    config["model_type"] = "moss_tts_nano"
    config["torch_dtype"] = dtype
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    _torch_state_dict_to_mlx_safetensors(
        tts_dir / "pytorch_model.bin",
        output / "model.safetensors",
        target_dtype,
    )
    _safetensors_to_mlx_safetensors(
        codec_dir,
        output / "audio_tokenizer" / "model.safetensors",
        target_dtype,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MOSS-TTS-Nano to MLX format.")
    parser.add_argument("--tts-source", default=DEFAULT_TTS_REPO)
    parser.add_argument(
        "--audio-tokenizer-source", default=DEFAULT_AUDIO_TOKENIZER_REPO
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="float16")
    args = parser.parse_args()
    output = convert(
        tts_source=args.tts_source,
        audio_tokenizer_source=args.audio_tokenizer_source,
        output_dir=args.output_dir,
        dtype=args.dtype,
    )
    print(f"Converted MOSS-TTS-Nano model written to {output}")


if __name__ == "__main__":
    main()
