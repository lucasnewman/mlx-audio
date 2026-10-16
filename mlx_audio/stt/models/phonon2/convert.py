"""Convert official Phonon-2 weights to lossless MLX packed planes.

python -m mlx_audio.stt.models.phonon2.convert \
    --hf-path FermionResearch/Phonon-2 --mlx-path ./phonon2-mlx

Uses NumPy/MLX only. Packed codes and fp16 row palettes are preserved;
remaining tables use bf16, matching Fermion's packed MLX runtime.

Container parsing adapted from fermion-research 0.2.11, Copyright 2026
Fermion Research, Apache-2.0. Streaming records and explicit format checks
added.
"""

import argparse
import json
import math
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import mlx.core as mx
import numpy as np
from huggingface_hub import hf_hub_download

from mlx_audio.stt.models.parakeet.redux import map_weight_name
from mlx_audio.stt.models.phonon.transport import _extract_archive, _sha256_file
from mlx_audio.utils import get_model_path

from .mapping import hf_to_mlx
from .packed import pack_planes

FORMAT = "fermion-five-value-parakeet-v1"


def five_value(blob, shape):
    if len(shape) != 2:
        raise ValueError("Five-value records must be matrices")
    rows, cols = shape
    n = rows * ((cols + 4) // 5)
    if len(blob) < n + rows * 4:
        raise ValueError("Truncated five-value record")
    encoded = np.frombuffer(blob, np.uint8, count=n).reshape(rows, -1)
    if np.any(encoded > 242):
        raise ValueError("Invalid base-3 byte")
    codes = (encoded[..., None].astype(np.uint16) // np.array([1, 3, 9, 27, 81])) % 3
    sign = codes.reshape(rows, -1)[:, :cols].astype(np.int8) - 1
    mask = sign != 0
    count = int(mask.sum())
    nbits = (count + 7) // 8
    if len(blob) != n + nbits + 4 * rows:
        raise ValueError("Five-value record length disagrees with its shape")
    hi_mask = np.zeros(shape, dtype=bool)
    hi_mask[mask] = np.unpackbits(
        np.frombuffer(blob, np.uint8, count=nbits, offset=n), bitorder="little"
    )[:count]
    lo = np.frombuffer(blob, "<f2", count=rows, offset=n + nbits).copy()
    hi = np.frombuffer(blob, "<f2", count=rows, offset=n + nbits + rows * 2).copy()
    if (
        not np.all(np.isfinite(lo))
        or not np.all(np.isfinite(hi))
        or np.any(lo < 0)
        or np.any(hi < lo)
    ):
        raise ValueError("Invalid five-value palettes")
    return sign, hi_mask, lo, hi


def dense_record(blob, shape, kind):
    total = math.prod(shape)
    if kind == "fp16":
        if len(blob) != 2 * total:
            raise ValueError("Invalid fp16 record length")
        return np.frombuffer(blob, "<f2").reshape(shape).copy()
    rows = shape[0]
    if kind not in ("int6", "int8"):
        raise ValueError(f"Unsupported Phonon-2 record kind: {kind}")
    body = (total + 3) // 4 * 3 if kind == "int6" else total
    if len(blob) != body + 2 * rows:
        raise ValueError("Invalid integer record length")
    scales = np.frombuffer(blob, "<f2", count=rows, offset=body).astype(np.float32)
    if kind == "int8":
        q = np.frombuffer(blob, np.int8, count=total).astype(np.float32)
    else:
        b = np.frombuffer(blob, np.uint8, count=body).reshape(-1, 3).astype(np.uint32)
        words = b[:, 0] | b[:, 1] << 8 | b[:, 2] << 16
        q = ((words[:, None] >> np.array([0, 6, 12, 18], np.uint32)) & 63).reshape(-1)[
            :total
        ].astype(np.float32) - 32
    return (q.reshape(rows, -1) * scales[:, None]).reshape(shape)


def records(path):
    """Yield validated (entry, payload) records; never expand the entire encoder."""
    with open(path, "rb") as f:
        size = f.seek(0, 2)
        f.seek(0)
        raw = f.read(8)
        length = int.from_bytes(raw, "little")
        if len(raw) != 8 or not 0 < length <= min(size - 8, 16 << 20):
            raise ValueError("Invalid Phonon-2 container header length")
        header = json.loads(f.read(length))
        if (
            not isinstance(header, dict)
            or header.get("format") != FORMAT
            or not isinstance(header.get("index"), list)
        ):
            raise ValueError("Unsupported Phonon-2 container format")
        names = set()
        for entry in header["index"]:
            if not isinstance(entry, dict):
                raise ValueError("Invalid Phonon-2 container record")
            name, kind, shape, count = (entry.get(k) for k in ("n", "k", "shape", "b"))
            if (
                not isinstance(name, str)
                or not name
                or name in names
                or not isinstance(shape, list)
                or (not shape and kind != "fp16")
                or any(type(d) is not int or d <= 0 for d in shape)
                or type(count) is not int
                or count <= 0
                or count > size - f.tell()
            ):
                raise ValueError(
                    f"Invalid or duplicate Phonon-2 container record: {name!r}"
                )
            names.add(name)
            blob = f.read(count)
            yield entry, (
                five_value(blob, shape)
                if kind == "five_value"
                else dense_record(blob, shape, kind)
            )
        if f.read(1):
            raise ValueError("Trailing bytes in Phonon-2 container")


@contextmanager
def unpack_source(model_path, *, source=None, revision=None, force_download=False):
    """Yield a verified source directory without modifying the Hub snapshot."""
    model_path = Path(model_path).expanduser().resolve()
    outer = json.loads((model_path / "config.json").read_text())
    artifact = outer.get("artifact")
    archive = None
    if "artifact" in outer:
        if not isinstance(artifact, dict):
            raise ValueError("Invalid Phonon-2 artifact metadata")
        filename, digest = artifact.get("filename", ""), artifact.get("sha256", "")
        if (
            not isinstance(filename, str)
            or not isinstance(digest, str)
            or Path(filename).name != filename
            or "\\" in filename
            or not filename.endswith(".bps.tar.zst")
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
            or type(artifact.get("unpacked_bytes")) is not int
            or artifact["unpacked_bytes"] <= 0
        ):
            raise ValueError("Invalid Phonon-2 artifact metadata")
        archive = model_path / filename
        if not archive.is_file() or (source is not None and force_download):
            repo = source or outer.get("model_id")
            if not repo:
                raise FileNotFoundError(f"Phonon-2 archive not found: {archive}")
            archive = Path(
                hf_hub_download(
                    repo, filename, revision=revision, force_download=force_download
                )
            )
        if _sha256_file(archive) != digest:
            raise ValueError("Phonon-2 archive SHA-256 mismatch")
    else:
        container = model_path / "model.fermion"
        digest = _sha256_file(container)
        manifest = model_path / "packed_manifest.json"
        if manifest.is_file():
            metadata = json.loads(manifest.read_text())
            if not isinstance(metadata, dict) or metadata.get("format") != FORMAT:
                raise ValueError("Invalid Phonon-2 packed manifest")
            expected = metadata.get("container_sha256")
            if expected and expected != digest:
                raise ValueError("Phonon-2 container SHA-256 mismatch")

    if archive is None:
        yield model_path
    else:
        with tempfile.TemporaryDirectory(prefix="phonon2-source-") as temporary:
            extracted = Path(temporary)
            _extract_archive(archive, extracted, artifact["unpacked_bytes"])
            yield extracted


def _convert_weights(source, destination):
    config = json.loads((source / "config.json").read_text())
    dense, weights, modules = {}, {}, []
    for entry, payload in records(source / "model.fermion"):
        name = entry["n"]
        if entry["k"] != "five_value":
            dense[name] = payload
            continue
        sign, mask, lo, hi = payload
        mapped = map_weight_name(name)
        a, b = pack_planes(sign, mask)
        modules.append({"name": mapped, "shape": entry["shape"]})
        for suffix, array in (
            ("base_q", a),
            ("residual_q", b),
            ("row_lo", lo),
            ("row_hi", hi),
        ):
            weights[f"{mapped}.{suffix}"] = mx.array(array)
        # Do not retain encoder-sized unpacking intermediates until the end.
        mx.eval(list(weights.values())[-4:])
    for name, array in hf_to_mlx(dense).items():
        weights[name] = mx.array(array).astype(mx.bfloat16)
    config.update(
        model_type="phonon2",
        phonon2_modules=modules,
        phonon2_conversion_version=1,
    )
    mx.save_safetensors(str(destination / "model.safetensors"), weights)
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def convert(
    hf_path: str,
    mlx_path: str,
    *,
    revision: str | None = None,
    force_download: bool = False,
) -> Path:
    """Write a standalone MLX checkpoint; never overwrite a nonempty directory."""
    destination = Path(mlx_path).expanduser().resolve()
    local_source = Path(hf_path).expanduser()
    if destination == local_source.resolve():
        raise ValueError("The output directory must differ from the source checkpoint.")
    if destination.exists() and (
        not destination.is_dir() or any(destination.iterdir())
    ):
        raise ValueError(
            f"Output already exists and is not an empty directory: {destination}"
        )
    model_path = get_model_path(
        hf_path,
        revision=revision,
        force_download=force_download,
        allow_patterns=["*.json", "*.txt", "NOTICE", "README.md"],
    )
    remote = None if local_source.exists() else hf_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".phonon2-", dir=destination.parent
    ) as temp:
        staged = Path(temp) / "model"
        staged.mkdir()
        with unpack_source(
            model_path,
            source=remote,
            revision=revision,
            force_download=force_download,
        ) as source:
            _convert_weights(source, staged)
            for name in (
                "LICENSE-WEIGHTS-CC-BY-4.0.txt",
                "LICENSE-CODE-Apache-2.0.txt",
                "NOTICE",
            ):
                for directory in (source, model_path):
                    if (directory / name).is_file():
                        shutil.copyfile(directory / name, staged / name)
                        break
        (staged / "README.md").write_text(
            "---\nlicense: cc-by-4.0\nlibrary_name: mlx-audio\n"
            "pipeline_tag: automatic-speech-recognition\n"
            "base_model: FermionResearch/Phonon-2\ntags: [mlx, mlx-audio, phonon2]\n"
            "---\n\n# Phonon-2 for MLX\n\n"
            "Converted from FermionResearch/Phonon-2. Five-value encoder weights "
            "are preserved in two MLX 2-bit planes; dense tables use bf16.\n\n"
            "```python\nfrom mlx_audio.stt import load\n"
            'model = load("path/to/phonon2-mlx")\n'
            'result = model.generate("audio.wav", chunk_duration=30, overlap_duration=2)\n'
            "print(result.text)\n```\n\n"
            "English transcription and token timestamps; overlapping chunk output "
            "is not live-input streaming. Weights are CC-BY-4.0; retain the accompanying notices.\n"
        )
        # Recheck after conversion so a competing writer cannot be overwritten.
        if destination.exists():
            destination.rmdir()
        staged.replace(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-path", default="FermionResearch/Phonon-2")
    parser.add_argument("--mlx-path", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--force-download", action="store_true")
    print(convert(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
