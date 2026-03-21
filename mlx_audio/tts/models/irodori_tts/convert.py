#!/usr/bin/env python3
"""
Convert Irodori-TTS weights from PyTorch safetensors to MLX format.

Downloads the original weights from Hugging Face, applies key remapping via
the model's sanitize() method, casts to fp16, and saves as MLX-compatible
safetensors alongside a config.json.

Converts Aratako/Irodori-TTS-500M from HuggingFace to MLX fp16 format.

Usage:
    # Convert to fp16
    python convert.py

    # Specify custom output directory
    python convert.py --output-dir ./Irodori-TTS-500M-fp16

    # Upload to Hugging Face
    python convert.py --upload-repo mlx-community/Irodori-TTS-500M-fp16

    # Dry-run (convert + write files but skip upload)
    python convert.py --upload-repo mlx-community/Irodori-TTS-500M-fp16 --dry-run

Requirements (conversion only):
    pip install torch safetensors huggingface_hub

After conversion, only mlx-audio is needed for inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np


# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------

_MODEL_CONFIGS: dict[str, dict] = {
    "500m": {
        "hf_repo": "Aratako/Irodori-TTS-500M",
        "model_config": {
            "model_type": "irodori_tts",
            "sample_rate": 48000,
            "max_text_length": 256,
            "max_speaker_latent_length": 6400,
            "audio_downsample_factor": 1920,
            "dacvae_repo": "Aratako/Irodori-TTS-500M",
            "dit": {
                "latent_dim": 128,
                "latent_patch_size": 1,
                "model_dim": 1280,
                "num_layers": 12,
                "num_heads": 20,
                "mlp_ratio": 2.875,
                "text_mlp_ratio": 2.6,
                "speaker_mlp_ratio": 2.6,
                "text_vocab_size": 99574,
                "text_tokenizer_repo": "llm-jp/llm-jp-3-150m",
                "text_add_bos": True,
                "text_dim": 512,
                "text_layers": 10,
                "text_heads": 8,
                "speaker_dim": 768,
                "speaker_layers": 8,
                "speaker_heads": 12,
                "speaker_patch_size": 1,
                "timestep_embed_dim": 512,
                "adaln_rank": 192,
                "norm_eps": 1e-5,
            },
            "sampler": {
                "num_steps": 40,
                "cfg_scale_text": 3.0,
                "cfg_scale_speaker": 5.0,
                "cfg_guidance_mode": "independent",
                "cfg_min_t": 0.5,
                "cfg_max_t": 1.0,
                "truncation_factor": None,
                "rescale_k": None,
                "rescale_sigma": None,
                "context_kv_cache": True,
                "speaker_kv_scale": None,
                "speaker_kv_min_t": 0.9,
                "speaker_kv_max_layers": None,
                "sequence_length": 750,
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def download_weights(repo_id: str, cache_dir: Path) -> Path:
    from huggingface_hub import snapshot_download

    print(f"Downloading {repo_id} from Hugging Face...")
    ckpt_dir = Path(
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=["*.safetensors", "*.json", "*.yaml"],
            cache_dir=cache_dir,
        )
    )
    print(f"Downloaded to: {ckpt_dir}")
    return ckpt_dir


# ---------------------------------------------------------------------------
# Load / save helpers
# ---------------------------------------------------------------------------


def load_safetensors(path: Path) -> Dict[str, np.ndarray]:
    """Load a safetensors file and return numpy arrays (fp32)."""
    try:
        # prefer safetensors-native loader (no torch required)
        from safetensors import safe_open

        result = {}
        with safe_open(str(path), framework="numpy") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                # bfloat16 has no numpy equivalent; cast to float32 first
                if tensor.dtype.name in ("float16", "bfloat16"):
                    tensor = tensor.astype(np.float32)
                result[key] = tensor
        return result
    except Exception:
        # fall back to torch loader
        import torch
        from safetensors.torch import load_file

        state_dict = load_file(str(path))
        return {k: v.to(torch.float32).cpu().numpy() for k, v in state_dict.items()}


def numpy_to_mlx(weights: Dict[str, np.ndarray]):
    import mlx.core as mx

    return {k: mx.array(v) for k, v in weights.items()}


def mlx_to_numpy_fp16(weights: Dict) -> Dict[str, np.ndarray]:
    import mlx.core as mx

    out = {}
    for k, v in weights.items():
        arr = np.array(v)
        if arr.dtype in (np.float32, np.float64):
            arr = arr.astype(np.float16)
        out[k] = arr
    return out


def save_safetensors(weights: Dict[str, np.ndarray], path: Path) -> None:
    from safetensors.numpy import save_file

    clean = {}
    for k, v in weights.items():
        if not isinstance(v, np.ndarray):
            v = np.array(v)
        if v.dtype == np.float64:
            v = v.astype(np.float32)
        clean[k] = v

    save_file(clean, str(path))
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"Saved: {path}  ({len(clean)} tensors, {size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# DACVAE conversion  (facebook/dacvae-watermarked → MLX)
# ---------------------------------------------------------------------------

_DACVAE_CONFIG = {
    "encoder_dim": 64,
    "encoder_rates": [2, 8, 10, 12],
    "latent_dim": 1024,
    "decoder_dim": 1536,
    "decoder_rates": [12, 10, 8, 2],
    "n_codebooks": 16,
    "codebook_size": 1024,
    "codebook_dim": 128,
    "quantizer_dropout": False,
    "sample_rate": 48000,
    "mean": 0.0,
    "std": 1.0,
}


def _map_residual_unit(name: str, prefix: str) -> str:
    """PyTorch encoder ResidualUnit: block.{0-3} → act1/conv1/act2/conv2."""
    suffix = name[len(prefix):]
    for i, attr in enumerate(["act1.", "conv1.", "act2.", "conv2."]):
        if suffix.startswith(f"block.{i}."):
            return prefix + attr + suffix[8:]
    return name


def _map_decoder_residual_unit(name: str, prefix: str) -> str:
    """PyTorch decoder ResidualUnit: block.{0-3} → act1/conv1/act2/conv2."""
    suffix = name[len(prefix):]
    mapping = {"block.0.": "act1.", "block.1.": "conv1.", "block.2.": "act2.", "block.3.": "conv2."}
    for old, new in mapping.items():
        if suffix.startswith(old):
            return prefix + new + suffix[len(old):]
    return name


def _remap_dacvae_key(k: str) -> str:
    """Map a facebook/dacvae-watermarked state_dict key to mlx-audio DACVAE key."""
    # Quantizer
    if k.startswith("quantizer.in_proj."):
        return k.replace("quantizer.in_proj.", "quantizer_in_proj.")
    if k.startswith("quantizer.out_proj."):
        return k.replace("quantizer.out_proj.", "quantizer_out_proj.")

    # Encoder: block.0 → conv_in
    if k.startswith("encoder.block.0."):
        return k.replace("encoder.block.0.", "encoder.conv_in.")
    # Encoder: block.{1-4} → blocks.{0-3}
    for enc_idx in range(1, 5):
        blk_idx = enc_idx - 1
        for res_idx in range(3):
            old = f"encoder.block.{enc_idx}.block.{res_idx}."
            new = f"encoder.blocks.{blk_idx}.res{res_idx + 1}."
            if k.startswith(old):
                return _map_residual_unit(k.replace(old, new), new)
        old = f"encoder.block.{enc_idx}.block.3."
        if k.startswith(old):
            return k.replace(old, f"encoder.blocks.{blk_idx}.snake.")
        old = f"encoder.block.{enc_idx}.block.4."
        if k.startswith(old):
            return k.replace(old, f"encoder.blocks.{blk_idx}.conv.")
    # Encoder: block.5 → snake_out, block.6 → conv_out
    if k.startswith("encoder.block.5."):
        return k.replace("encoder.block.5.", "encoder.snake_out.")
    if k.startswith("encoder.block.6."):
        return k.replace("encoder.block.6.", "encoder.conv_out.")

    # Decoder: model.0 → conv_in
    if k.startswith("decoder.model.0."):
        return k.replace("decoder.model.0.", "decoder.conv_in.")
    # Decoder: model.{1-4}.block.{N} → blocks.{0-3}.block_{N}
    for dec_idx in range(1, 5):
        blk_idx = dec_idx - 1
        for block_num in [0, 1, 3, 4, 5, 6, 7, 8, 11]:
            old = f"decoder.model.{dec_idx}.block.{block_num}."
            new = f"decoder.blocks.{blk_idx}.block_{block_num}."
            if k.startswith(old):
                k2 = k.replace(old, new)
                if block_num in [4, 5, 6, 7, 8]:
                    k2 = _map_decoder_residual_unit(k2, new)
                return k2
    # Decoder wm_model final layers
    if k.startswith("decoder.wm_model.encoder_block.pre.0."):
        return k.replace("decoder.wm_model.encoder_block.pre.0.", "decoder.snake_out.")
    if k.startswith("decoder.wm_model.encoder_block.pre.1."):
        return k.replace("decoder.wm_model.encoder_block.pre.1.", "decoder.conv_out.")
    # wm_model pre/post block numbering
    for block in ["encoder_block", "decoder_block"]:
        for prefix in ["pre", "post"]:
            for idx in range(4):
                old = f".{block}.{prefix}.{idx}."
                new = f".{block}.{prefix}_{idx}."
                if old in k:
                    k = k.replace(old, new)

    # LSTM weights: weight_ih_lN → lstm.layers.N.Wx, weight_hh_lN → lstm.layers.N.Wh
    import re
    k = re.sub(r"\.lstm\.weight_ih_l(\d+)$", lambda m: f".lstm.layers.{m.group(1)}.Wx", k)
    k = re.sub(r"\.lstm\.weight_hh_l(\d+)$", lambda m: f".lstm.layers.{m.group(1)}.Wh", k)

    return k


def convert_dacvae(output_dir: Path, cache_dir: Path) -> None:
    """
    Download facebook/dacvae-watermarked and convert to MLX safetensors.
    Saves model.safetensors + config.json into output_dir/dacvae/.
    Requires torch (CPU-only is fine).
    """
    import torch
    from huggingface_hub import hf_hub_download

    dacvae_dir = output_dir / "dacvae"
    dacvae_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading facebook/dacvae-watermarked...")
    pth_path = hf_hub_download(
        repo_id="facebook/dacvae-watermarked",
        filename="weights.pth",
        cache_dir=cache_dir,
    )
    ckpt = torch.load(pth_path, map_location="cpu", weights_only=False)
    pt_sd = ckpt["state_dict"]
    print(f"  Loaded {len(pt_sd)} tensors from weights.pth")

    import re

    # First pass: collect LSTM biases to combine (bias_ih + bias_hh → bias)
    lstm_biases: Dict[str, Dict[str, np.ndarray]] = {}
    lstm_bias_keys: set[str] = set()
    for k, v in pt_sd.items():
        m = re.search(r"^(.+\.lstm)\.bias_(ih|hh)_l(\d+)$", k)
        if m:
            base, kind, layer = m.group(1), m.group(2), m.group(3)
            lstm_biases.setdefault((base, layer), {})[kind] = v.float().numpy()
            lstm_bias_keys.add(k)

    remapped: Dict[str, np.ndarray] = {}
    # Emit combined LSTM biases (bias_ih_lN + bias_hh_lN → layers.N.bias)
    for (base, layer), biases in lstm_biases.items():
        if "ih" in biases and "hh" in biases:
            # Use a sentinel suffix so _remap_dacvae_key applies wm_model renaming,
            # then swap sentinel for the real MLX LSTM bias key.
            sentinel = f"{base}.__bias_sentinel_l{layer}"
            remapped_sentinel = _remap_dacvae_key(sentinel)
            new_k = re.sub(
                r"\.__bias_sentinel_l(\d+)$",
                r".layers.\1.bias",
                remapped_sentinel,
            )
            remapped[new_k] = biases["ih"] + biases["hh"]

    for k, v in pt_sd.items():
        if k in lstm_bias_keys:
            continue
        new_k = _remap_dacvae_key(k)
        remapped[new_k] = v.float().numpy()

    # Transpose Conv/Snake 3D weights to MLX layout.
    # PyTorch Conv1d:      (out, in, k)  → MLX (out, k, in)  → transpose (0,2,1)
    # PyTorch ConvTrans1d: (in, out, k)  → MLX (out, k, in)  → transpose (1,2,0)
    # Use the reference MLX model to determine the expected shape for each key.
    from mlx_audio.codec.models.dacvae import DACVAE as _DACVAE, DACVAEConfig as _DACVAEConfig
    from mlx.utils import tree_flatten as _tree_flatten
    _ref = _DACVAE(_DACVAEConfig())
    _mlx_shapes = {k: tuple(v.shape) for k, v in _tree_flatten(_ref.parameters())}

    for k, v in remapped.items():
        if k not in _mlx_shapes or v.ndim != 3 or v.shape == _mlx_shapes[k]:
            continue
        mlx_s = _mlx_shapes[k]
        if v.shape[0] == mlx_s[0]:          # Conv1d / Snake: (a,b,c) → (a,c,b)
            remapped[k] = v.transpose(0, 2, 1)
        elif v.shape[1] == mlx_s[0]:        # ConvTranspose1d: (a,b,c) → (b,c,a)
            remapped[k] = v.transpose(1, 2, 0)

    print("Saving dacvae/model.safetensors...")
    save_safetensors(remapped, dacvae_dir / "model.safetensors")

    print("Saving dacvae/config.json...")
    with open(dacvae_dir / "config.json", "w") as f:
        json.dump(_DACVAE_CONFIG, f, indent=2)
    print(f"  Saved: {dacvae_dir / 'config.json'}")


# ---------------------------------------------------------------------------
# README generation
# ---------------------------------------------------------------------------


def generate_readme(output_dir: Path, upload_repo: str, hf_repo: str) -> None:
    try:
        from mlx_audio.version import __version__

        version_str = f"mlx-audio version **{__version__}**"
    except ImportError:
        version_str = "mlx-audio"

    card = f"""---
library_name: mlx-audio
base_model:
- {hf_repo}
tags:
- mlx
- japanese
pipeline_tag: text-to-speech
---

# {upload_repo}

This model was converted to MLX format from [{hf_repo}](https://huggingface.co/{hf_repo}) using [{version_str}](https://github.com/Blaizzy/mlx-audio).

## Use with mlx-audio

```bash
pip install -U mlx-audio
```

### Command line

```bash
mlx_audio.tts.generate --model {upload_repo} --text "こんにちは、Irodori TTSのMLX版です。" --ref_audio reference.wav
```

### Python

```python
from mlx_audio.tts.generate import generate_audio

generate_audio(
    text="こんにちは、Irodori TTSのMLX版です。",
    model="{upload_repo}",
    ref_audio="reference.wav",
    file_prefix="output",
)
```
"""
    readme_path = output_dir / "README.md"
    readme_path.write_text(card)
    print(f"Created: {readme_path}")


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


def upload_to_hub(output_dir: Path, upload_repo: str) -> None:
    from huggingface_hub import HfApi

    print(f"\nUploading to {upload_repo}...")
    api = HfApi()
    api.create_repo(repo_id=upload_repo, exist_ok=True)
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=upload_repo,
        repo_type="model",
    )
    print(f"Upload complete: https://huggingface.co/{upload_repo}")


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------


def convert(
    output_dir: Path | None = None,
    cache_dir: Path | None = None,
    upload_repo: str | None = None,
    dry_run: bool = False,
) -> None:
    """
    Convert Irodori-TTS-500M PyTorch weights to MLX fp16 safetensors.

    Args:
        output_dir:  destination directory (default: ./Irodori-TTS-500M-fp16)
        cache_dir:   HuggingFace download cache (default: ~/.cache/irodori-convert)
        upload_repo: HuggingFace repo to upload result to
        dry_run:     write files but skip upload
    """
    cfg = _MODEL_CONFIGS["500m"]
    hf_repo: str = cfg["hf_repo"]
    model_config: dict = cfg["model_config"]

    if output_dir is None:
        slug = hf_repo.split("/")[-1]
        output_dir = Path(f"./{slug}-fp16")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "irodori-convert"

    # 1. Download
    ckpt_dir = download_weights(hf_repo, cache_dir)

    # 2. Locate safetensors weight file
    sf_files = sorted(ckpt_dir.glob("*.safetensors"))
    # Prefer "model.safetensors"; exclude any DACVAE / codec files
    model_files = [
        f for f in sf_files if "dac" not in f.stem.lower() and "vae" not in f.stem.lower()
    ]
    if not model_files:
        model_files = sf_files
    if not model_files:
        raise FileNotFoundError(
            f"No .safetensors files found in {ckpt_dir}. "
            "Check the downloaded checkpoint."
        )

    # If multiple shards (model-00001-of-NNNNN.safetensors), merge them
    shard_files = sorted(ckpt_dir.glob("model-*-of-*.safetensors"))
    if shard_files:
        print(f"Found {len(shard_files)} weight shards — merging...")
        raw_weights: Dict[str, np.ndarray] = {}
        for shard in shard_files:
            print(f"  Loading {shard.name}...")
            raw_weights.update(load_safetensors(shard))
    else:
        weight_file = model_files[0]
        print(f"Loading weights from {weight_file.name}...")
        raw_weights = load_safetensors(weight_file)

    print(f"Loaded {len(raw_weights)} tensors")

    # 3. Apply sanitize() key remapping
    print("Remapping weight keys via sanitize()...")
    from mlx_audio.tts.models.irodori_tts.irodori_tts import Model
    from mlx_audio.tts.models.irodori_tts.config import ModelConfig

    # Build a minimal model instance to call sanitize (no weights loaded)
    mc = ModelConfig.from_dict(model_config)
    dummy_model = Model.__new__(Model)
    dummy_model.config = mc

    weights_mx = numpy_to_mlx(raw_weights)
    weights_mx = dummy_model.sanitize(weights_mx)
    print(f"After remapping: {len(weights_mx)} tensors")

    # 4. Cast to fp16
    print("Casting to fp16...")
    weights_fp16 = mlx_to_numpy_fp16(weights_mx)

    # 5. Save model.safetensors
    print("\nSaving model.safetensors...")
    save_safetensors(weights_fp16, output_dir / "model.safetensors")

    # 6. Save config.json
    print("Saving config.json...")
    with open(output_dir / "config.json", "w") as f:
        json.dump(model_config, f, indent=2)
    print(f"Saved: {output_dir / 'config.json'}")

    # 7. Convert DACVAE (facebook/dacvae-watermarked → dacvae/)
    print("\nConverting DACVAE codec...")
    try:
        convert_dacvae(output_dir, cache_dir)
    except ImportError:
        print("  WARNING: torch not available — skipping DACVAE conversion.")
        print("  Run with torch installed to include the DACVAE codec.")

    # 8. README (when uploading)
    if upload_repo:
        print("Generating README.md...")
        generate_readme(output_dir, upload_repo, hf_repo)

    # Summary
    print(f"\nConversion complete. Output: {output_dir}")
    print("Files:")
    for p in sorted(output_dir.iterdir()):
        if p.is_dir():
            size_mb = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1024**2
            print(f"  {p.name}/: {size_mb:.1f} MB")
        else:
            print(f"  {p.name}: {p.stat().st_size / 1024**2:.1f} MB")

    # 9. Upload
    if upload_repo and not dry_run:
        upload_to_hub(output_dir, upload_repo)
    elif upload_repo:
        print(f"\nDry-run: skipping upload to {upload_repo}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Irodori-TTS PyTorch weights to MLX format"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: ./Irodori-TTS-500M-fp16)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Download cache directory (default: ~/.cache/irodori-convert)",
    )
    parser.add_argument(
        "--upload-repo",
        type=str,
        default=None,
        help="Hugging Face repo to upload to (e.g., mlx-community/Irodori-TTS-500M-fp16)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Convert and write files, but skip upload",
    )
    args = parser.parse_args()

    convert(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        upload_repo=args.upload_repo,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
