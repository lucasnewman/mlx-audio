"""Verify the official Phonon-2 artifact and cache ordinary MLX weights."""

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import mlx.core as mx
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import WeakFileLock

from mlx_audio.stt.models.parakeet.redux import map_weight_name
from mlx_audio.stt.models.phonon.transport import (
    _default_cache_root,
    _extract_archive,
    _sha256_file,
)

from .container import FORMAT, records
from .mapping import hf_to_mlx
from .packed import pack_planes

CACHE_VERSION = 1


def is_phonon2_model(path, config):
    if config.get("model_type") == "parakeet_tdt_five_value":
        return True
    manifest = Path(path) / "packed_manifest.json"
    if not (Path(path) / "model.fermion").is_file() or not manifest.is_file():
        return False
    try:
        metadata = json.loads(manifest.read_text())
    except (OSError, ValueError):
        return False
    return isinstance(metadata, dict) and metadata.get("format") == FORMAT


def _convert(source, destination):
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
        phonon2_cache_version=CACHE_VERSION,
    )
    config["preprocessor"]["compute_in_float32"] = True
    mx.save_safetensors(str(destination / "model.safetensors"), weights)
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def _complete(path):
    try:
        marker = json.loads((path / "conversion.json").read_text())
        config = json.loads((path / "config.json").read_text())
        return (
            isinstance(marker, dict)
            and isinstance(config, dict)
            and marker.get("version") == CACHE_VERSION
            and config.get("model_type") == "phonon2"
            and marker.get("config_sha256") == _sha256_file(path / "config.json")
            and marker.get("weights_sha256") == _sha256_file(path / "model.safetensors")
        )
    except (OSError, ValueError):
        return False


def prepare_model_path(model_path, *, source=None, revision=None, force_download=False):
    """Accept a Hub snapshot or an unpacked official directory; never modify it."""
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
        digest = hashlib.sha256(
            (digest + _sha256_file(model_path / "config.json")).encode()
        ).hexdigest()

    destination = _default_cache_root() / "phonon2" / f"v{CACHE_VERSION}-{digest}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with WeakFileLock(str(destination) + ".lock"):
        if not force_download and _complete(destination):
            return destination
        temporary = Path(tempfile.mkdtemp(prefix=".convert-", dir=destination.parent))
        try:
            input_path = model_path
            if archive is not None:
                input_path = temporary / "source"
                _extract_archive(archive, input_path, artifact["unpacked_bytes"])
            _convert(input_path, temporary)
            marker = {
                "version": CACHE_VERSION,
                "config_sha256": _sha256_file(temporary / "config.json"),
                "weights_sha256": _sha256_file(temporary / "model.safetensors"),
            }
            (temporary / "conversion.json").write_text(json.dumps(marker) + "\n")
            for name in (
                "LICENSE-WEIGHTS-CC-BY-4.0.txt",
                "LICENSE-CODE-Apache-2.0.txt",
                "NOTICE",
            ):
                for directory in (input_path, model_path):
                    if (directory / name).is_file():
                        shutil.copyfile(directory / name, temporary / name)
                        break
            if archive is not None:
                shutil.rmtree(input_path)
            if destination.exists():
                shutil.rmtree(destination)
            temporary.replace(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return destination
