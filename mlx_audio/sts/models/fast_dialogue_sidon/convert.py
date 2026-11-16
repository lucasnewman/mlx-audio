"""Save a standalone MLX FastDialogueSidon checkpoint (no PyTorch required)."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

SOURCE_REPO = "higurashi-ai/FastDialogueSidon"
SOURCE_REVISION = "daa52c2df05813cb7295cd77d85cd485b810fdc6"


def model_card(source_repo, revision, dtype):
    return f"""---
license: apache-2.0
library_name: mlx-audio
tags:
- mlx
- audio-to-audio
- speech-separation
base_model: {source_repo}
---

# FastDialogueSidon ({dtype}, MLX)

One-step separation and restoration of two-speaker recordings on Apple Silicon.
Outputs two mono tracks at 24 kHz. Requires MLX Audio with FastDialogueSidon support.
Source: [{source_repo}](https://huggingface.co/{source_repo}/tree/{revision}).

```python
from mlx_audio.sts import load
from mlx_audio.audio_io import write

model = load("/path/to/this/model")
result = model.separate("dialogue.wav", seed=0)
for i, speaker in enumerate(result.speakers, 1):
    write(f"speaker_{{i}}.wav", speaker, result.sample_rate)
```

Long recordings use 20-second chunks with 5-second overlap and speaker alignment.
Set `chunk_seconds` and `overlap_seconds` to adjust these; `chunk_seconds=None`
processes the entire recording with higher memory use. The model always uses
one CTM step. Tracks are anonymous speaker slots and may switch after silence.

Code and weights: [Apache-2.0](https://github.com/higurashi-ai/FastDialogueSidon/blob/main/LICENSE).
FastDialogueSidon by Rikuto Kotoge (higurashi-ai), based on SaruLab's DialogueSidon.
The WaveFitNeXt decoder derives from Yukara Ikemiya's MIT-licensed wavefit-pytorch;
see the upstream [third-party notices](https://github.com/higurashi-ai/FastDialogueSidon/blob/main/THIRD_PARTY_NOTICES.md).
"""


def convert(
    hf_path=SOURCE_REPO,
    mlx_path="output/FastDialogueSidon",
    dtype="float32",
    revision=None,
):
    from mlx_audio.sts import load

    if dtype not in ("float32", "float16", "bfloat16"):
        raise ValueError("dtype must be float32, float16, or bfloat16")
    destination = Path(mlx_path).expanduser()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {destination}")
    if revision is None and str(hf_path) == SOURCE_REPO:
        revision = SOURCE_REVISION
    model = load(hf_path, strict=True, lazy=True, revision=revision)
    weights = {
        key: value.astype(
            mx.float32
            if key.startswith("latent_norm_") or key.endswith(".fft_win")
            else getattr(mx, dtype)
        )
        for key, value in tree_flatten(model.parameters())
    }
    mx.eval(weights)
    config = asdict(model.config)
    config.pop("model_path", None)
    config["weight_format"] = "mlx"
    config["torch_dtype"] = dtype
    config["source_revision"] = revision or config["source_revision"]
    if not Path(hf_path).is_dir():
        config["source_repo"] = str(hf_path)
    destination.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(destination / "model.safetensors"), weights)
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (destination / "README.md").write_text(
        model_card(config["source_repo"], config["source_revision"] or "main", dtype)
    )
    print(f"Saved {len(weights)} tensors to {destination}")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-path", default=SOURCE_REPO)
    parser.add_argument("--mlx-path", default="output/FastDialogueSidon")
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="float32"
    )
    parser.add_argument("--revision")
    convert(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
