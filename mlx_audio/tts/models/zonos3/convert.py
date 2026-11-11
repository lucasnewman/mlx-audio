"""Convert Zyphra's published inference GGUFs to standalone MLX safetensors.

GGUF support is needed only here: ``pip install gguf``. Q4_0 / Q8_0 tensors
are repacked exactly into MLX affine blocks; no PyTorch is required.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import numpy as np

from .config import LMConfig
from .weights import encode_name

ARCH = "zonos3"
_GDN_PATTERN = re.compile(r"gdn(\d+)_global1")


def _key(name):
    return f"{ARCH}.{name}"


def read_lm_config(f, tokenizer_path):
    cls = LMConfig
    if f.arch != ARCH:
        raise ValueError(f"{f.path}: general.architecture is {f.arch!r}, not {ARCH!r}")
    for key, wanted in (
        ("residual.type", "full_attnres"),
        ("moe.balancing", "quantile"),
    ):
        if f.meta(_key(key)) != wanted:
            raise ValueError(
                f"{f.path}: {_key(key)} = {f.meta(_key(key))!r}, expected {wanted!r}"
            )
    pattern = _GDN_PATTERN.fullmatch(f.meta(_key("attention.pattern")))
    if pattern is None:
        raise ValueError(
            f"{f.path}: unsupported attention pattern {f.meta(_key('attention.pattern'))!r}"
        )
    names = tuple(f.meta(_key("condition.names")))
    units = f.meta(_key("condition.units"))
    text_vocab = int(f.meta(_key("text_vocab")))
    spec = json.loads(f.meta(_key("text_tokenizer_spec")))
    spec["path"] = str(tokenizer_path)
    span_end = f.meta(_key("span_end_id"), None)
    condition_count = int(f.meta(_key("condition.count")))
    counts = f.meta(
        _key("condition.bucket_counts"), None
    )  # legacy GGUFs: one u32 condition.bucket_count
    counts = (
        tuple(int(n) for n in counts)
        if counts is not None
        else (int(f.meta(_key("condition.bucket_count"))),) * condition_count
    )
    return cls(
        n_layers=int(f.meta(_key("block_count"))),
        max_seqlen=int(f.meta(_key("context_length"))),
        dim=int(f.meta(_key("embedding_length"))),
        head_dim=int(f.meta(_key("head_dim"))),
        n_heads=int(f.meta(_key("attention.head_count"))),
        n_kv_heads=int(f.meta(_key("attention.head_count_kv"))),
        ffn_hidden=int(f.meta(_key("feed_forward_length"))),
        n_experts=int(f.meta(_key("expert_count"))),
        moe_topk=int(f.meta(_key("expert_used_count_per_layer"))[1]),
        n_codebooks=int(f.meta(_key("n_codebooks"))),
        codebook_size=int(f.meta(_key("codebook_size"))),
        audio_vocab=int(f.meta(_key("audio_vocab"))),
        eoa_id=int(f.meta(_key("eoa_id"))),
        audio_pad_id=int(f.meta(_key("audio_pad_id"))),
        span_end_id=None if span_end is None else int(span_end),
        text_vocab=text_vocab,
        text_extra_tokens=f.shape("text_embd.weight")[0] - text_vocab - 1,
        gdn_heads=int(f.meta(_key("gdn.head_count"))),
        gdn_head_dim=int(f.meta(_key("gdn.head_dim"))),
        gdn_kernel=int(f.meta(_key("gdn.conv_kernel"))),
        speaker_embedding_dim=int(f.meta(_key("speaker.embedding_length"))),
        condition_count=condition_count,
        condition_bucket_counts=counts,
        condition_names=names,
        condition_units=dict(zip(names, units)),
        condition_edges={
            n: tuple(float(v) for v in f.meta(_key(f"condition.edges.{n}")))
            for n in names
        },
        rope_theta=float(f.meta(_key("rope.freq_base"))),
        norm_eps=float(f.meta(_key("attention.layer_norm_rms_epsilon"))),
        qk_eps=float(f.meta(_key("attention.qk_norm_epsilon"))),
        loss_softcap=float(f.meta(_key("logit_softcap"))),
        gdn_period=int(pattern.group(1)) + 1,
        silence_frame=tuple(int(v) for v in f.meta(_key("silence_frame"))),
        text_tokenizer_spec=spec,
        tag_cfg_trained=bool(f.meta(_key("tag_cfg_trained"), False)),
    )


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _lm_matrix(name, shape):
    return not (
        len(shape) == 1
        or name.endswith(".gdn_conv.weight")
        or name == "attnres_q.weight"
        or name.startswith(("audio_embd.", "text_embd.", "cond_embd."))
    )


def _codec_matrix(name, shape):
    return (
        len(shape) == 2
        and name.endswith(".weight")
        and ".dw." not in name
        and "norm" not in name
        and not name.startswith(("sem.", "res."))
        and name != "head.mixer.weight"
    )


def convert_component(path, out, kind, dtype=None):
    try:
        from .gguf import GGUFFile
    except ModuleNotFoundError as error:
        if error.name == "gguf":
            raise ImportError(
                "ZONOS3 conversion needs gguf: pip install gguf"
            ) from error
        raise

    f = GGUFFile(path)
    out.mkdir(parents=True, exist_ok=True)
    specs, weights, host = {}, {}, {}
    matrix = {"lm": _lm_matrix, "codec": _codec_matrix, "speaker": lambda n, s: False}[
        kind
    ]
    for name in f.names():
        shape = f.shape(name)
        key = encode_name(name)
        spec = {"shape": list(shape), "matrix": matrix(name, shape)}
        if spec["matrix"]:
            w = f.weight(name)
            spec.update(dtype=str(w.w.dtype).removeprefix("mlx.core."), bits=w.bits)
            value = (
                w.w
                if w.bits is not None or dtype is None
                else w.w.astype(getattr(mx, dtype))
            )
            spec["dtype"] = str(value.dtype).removeprefix("mlx.core.")
            weights[f"matrices.{key}.weight"] = value
            if w.scales is not None:
                weights[f"matrices.{key}.scales"] = w.scales
                weights[f"matrices.{key}.biases"] = w.biases
        else:
            value = f.f32(name)
            spec["dtype"] = "float32"
            weights[f"tables.{key}"] = value
            if kind == "speaker" and name.startswith("frontend."):
                host[name] = f.array(name)
        specs[name] = spec
    metadata = {k: _plain(f.meta(k)) for k in f.reader.fields if k != "GGUF.version"}
    config = {"metadata": metadata, "tensors": specs}
    if kind == "lm":
        tokenizer = f.meta(_key("tokenizer.json"))
        (out / "tokenizer.json").write_text(tokenizer, encoding="utf-8")
        config = {
            "model_type": "zonos3",
            "backbone": asdict(read_lm_config(f, "tokenizer.json")),
            "tensors": specs,
        }
        weights = {"backbone.params." + k: v for k, v in weights.items()}
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    mx.save_safetensors(
        str(out / "model.safetensors"), weights, metadata={"format": "mlx"}
    )
    if host:
        np.savez(out / "frontend.npz", **host)
    return config


def convert(
    hf_path, mlx_path, revision=None, decoder="keep2", speaker_encoder=True, dtype=None
):
    from huggingface_hub import snapshot_download

    source = Path(hf_path).expanduser()
    decoder_file = "zac-keep2.gguf" if decoder == "keep2" else "zac.gguf"
    if not source.exists():
        source = Path(
            snapshot_download(
                hf_path,
                revision=revision,
                allow_patterns=[
                    "gguf/zonos3-ptq.gguf",
                    f"gguf/{decoder_file}",
                    *(["gguf/spk-encoder.gguf"] if speaker_encoder else []),
                    "gguf/voices/*.npy",
                ],
            )
        )
        revision = source.name
    root = source / "gguf" if (source / "gguf").exists() else source
    required = ["zonos3-ptq.gguf", decoder_file]
    if speaker_encoder:
        required.append("spk-encoder.gguf")
    for name in required:
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)
    out = Path(mlx_path).expanduser()
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"output directory must be empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    convert_component(root / "zonos3-ptq.gguf", out, "lm", dtype=dtype)
    convert_component(root / decoder_file, out / "codec", "codec", dtype=dtype)
    if speaker_encoder:
        convert_component(root / "spk-encoder.gguf", out / "speaker_encoder", "speaker")
    voices = {}
    voice_dir = out / "voices"
    for path in sorted((root / "voices").glob("*.embedding.npy")):
        voice_dir.mkdir(exist_ok=True)
        name = path.name.removesuffix(".npy") + ".npz"
        np.savez(voice_dir / name, embedding=np.load(path, allow_pickle=False))
        voices[path.name.split("_")[0].lower()] = name
    (out / "voices.json").write_text(json.dumps(voices, indent=2) + "\n")
    model_card = Path(__file__).with_name("README.md").read_text()
    (out / "README.md").write_text(
        "---\nlicense: apache-2.0\nlibrary_name: mlx-audio\npipeline_tag: text-to-speech\ntags:\n  - mlx\nbase_model: Zyphra/ZONOS3-Mini\n---\n\n"
        + model_card
    )
    (out / "conversion.json").write_text(
        json.dumps(
            {
                "source": str(hf_path),
                "revision": revision,
                "decoder": decoder_file,
                "backbone": "zonos3-ptq.gguf",
                "quantization": "Q4_0/Q8_0 repacked without requantization",
                "reference_code": "https://github.com/Zyphra/ZONOS3-Mini/tree/6ff97cac0c684732de5f2bc6e3671aa783c15054",
            },
            indent=2,
        )
        + "\n"
    )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-path", default="Zyphra/ZONOS3-Mini")
    parser.add_argument("--mlx-path", required=True)
    parser.add_argument("--revision")
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        help="Storage dtype of dense matrices; defaults to source precision",
    )
    parser.add_argument("--decoder", choices=["keep2", "full"], default="keep2")
    parser.add_argument("--no-speaker-encoder", action="store_true")
    args = parser.parse_args()
    path = convert(
        args.hf_path,
        args.mlx_path,
        args.revision,
        args.decoder,
        not args.no_speaker_encoder,
        args.dtype,
    )
    print(f"Saved MLX model to {path}")


if __name__ == "__main__":
    main()
