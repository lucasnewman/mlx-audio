"""Convert Reverb's nested FP32 checkpoint and bundle its Higgs v2 codec.

python -m mlx_audio.tts.models.reverb.convert --output ./reverb-mlx
"""

import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten

from .config import ModelConfig
from .reverb import Model

SOURCE_REPO = "software-mansion/Reverb-TTS"
CODEC_REPO = "eustlb/higgs-audio-v2-tokenizer"
SOURCE_PATTERNS = [
    "README.md",
    "model/safetensors/fp32/*",
    "model/tokenizer*.json",
    "model/chat_template.jinja",
    "voices/*.json",
]
CODEC_PATTERNS = [
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "README.md",
]


def convert(
    model, output, *, codec=CODEC_REPO, dtype="bfloat16", bits=None, revision=None
):
    output = Path(output).expanduser()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}")
    source = Path(model).expanduser()
    if not source.is_dir():
        source = Path(
            snapshot_download(model, revision=revision, allow_patterns=SOURCE_PATTERNS)
        )
    weights_dir = source / "model" / "safetensors" / "fp32"
    config = json.loads((weights_dir / "config.json").read_text())
    if "echo" not in config:
        raise ValueError("Expected a Reverb checkpoint with an echo configuration.")
    config["model_type"] = "reverb"
    codec_path = Path(codec).expanduser()
    if not codec_path.is_dir():
        codec_path = Path(snapshot_download(codec, allow_patterns=CODEC_PATTERNS))
    for name in ("config.json", "model.safetensors"):
        if not (codec_path / name).is_file():
            raise FileNotFoundError(f"Missing codec file: {codec_path / name}")
    target_dtype = {
        "float32": mx.float32,
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
    }[dtype]
    instance = Model(ModelConfig.from_dict(config))
    weights = instance.sanitize(mx.load(str(weights_dir / "model.safetensors")))
    instance.load_weights(
        [(key, value.astype(target_dtype)) for key, value in weights.items()],
        strict=True,
    )
    config["architectures"] = ["ReverbModel"]
    config["dtype"] = dtype
    config["sample_rate"] = 24000
    if bits is not None:
        nn.quantize(instance, group_size=64, bits=bits)
        config["quantization"] = {"group_size": 64, "bits": bits}
    output.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(output / "model.safetensors"), dict(tree_flatten(instance.parameters()))
    )
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        shutil.copy2(source / "model" / name, output / name)
    shutil.copytree(source / "voices", output / "voices")
    (output / "audio_tokenizer").mkdir()
    for name in CODEC_PATTERNS:
        path = codec_path / name
        if path.exists():
            shutil.copy2(path, output / "audio_tokenizer" / name)
    if (source / "README.md").exists():
        shutil.copy2(source / "README.md", output / "README.upstream.md")
    (output / "README.md").write_text(
        "---\nlicense: cc-by-nc-4.0\nlibrary_name: mlx-audio\n"
        f"base_model: {SOURCE_REPO}\npipeline_tag: text-to-speech\n---\n\n"
        "# Reverb for MLX Audio\n\n"
        f"Converted from [{SOURCE_REPO}](https://huggingface.co/{SOURCE_REPO}). "
        "Weights are CC BY-NC 4.0; see README.upstream.md for attribution.\n\n"
        "Includes the Higgs Audio v2 codec; inference and voice cloning use MLX.\n\n"
        "```python\nfrom mlx_audio.tts.utils import load\n"
        "from mlx_audio.audio_io import write\n\n"
        f'model = load("{output}")\n'
        'result = next(model.generate("Hello from Reverb on Apple Silicon."))\n'
        'write("speech.wav", result.audio, result.sample_rate)\n```\n'
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=SOURCE_REPO)
    parser.add_argument("--output", required=True)
    parser.add_argument("--codec", default=CODEC_REPO)
    parser.add_argument("--revision")
    parser.add_argument(
        "--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16"
    )
    parser.add_argument("--bits", choices=[4, 8], type=int)
    args = parser.parse_args()
    print(convert(**vars(args)))


if __name__ == "__main__":
    main()
