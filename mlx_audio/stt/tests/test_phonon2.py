"""Offline format, standard-loader, and inference regression tests for Phonon-2."""

import hashlib
import io
import json
import re
import tarfile

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_audio.registry import classify_model
from mlx_audio.stt import load
from mlx_audio.stt import utils as stt_utils
from mlx_audio.stt.models.parakeet.audio import PreprocessArgs, log_mel_spectrogram
from mlx_audio.stt.models.parakeet.parakeet import ParakeetTDT, ParakeetTDTArgs
from mlx_audio.stt.models.phonon2 import Model
from mlx_audio.stt.models.phonon2.container import FORMAT, dense_record, records
from mlx_audio.stt.models.phonon2.packed import (
    ENCODER_SUFFIXES,
    PackedFiveValueLinear,
    pack_planes,
)
from mlx_audio.stt.models.phonon2.transport import prepare_model_path
from mlx_audio.utils import from_dict


def write_container(path, rows):
    header = {
        "format": FORMAT,
        "index": [dict(n=n, k=k, shape=list(s), b=len(b)) for n, k, s, b in rows],
    }
    data = json.dumps(header).encode()
    path.write_bytes(
        len(data).to_bytes(8, "little") + data + b"".join(r[3] for r in rows)
    )


def pack_record(sign, mask, lo, hi):
    codes = (sign + 1).astype(np.uint8)
    codes = np.pad(codes, ((0, 0), (0, (-codes.shape[1]) % 5)), constant_values=1)
    trits = np.sum(
        codes.reshape(sign.shape[0], -1, 5) * np.array([1, 3, 9, 27, 81]), axis=-1
    ).astype(np.uint8)
    return (
        trits.tobytes()
        + np.packbits(mask[sign != 0], bitorder="little").tobytes()
        + lo.astype("<f2").tobytes()
        + hi.astype("<f2").tobytes()
    )


def hf_name(name, value):
    replacements = (
        ("encoder.pre_encode.conv.", "encoder.subsampling.layers."),
        ("encoder.pre_encode.out.", "encoder.subsampling.linear."),
        (".conv.batch_norm.", ".conv.norm."),
        (".self_attn.linear_q.", ".self_attn.q_proj."),
        (".self_attn.linear_k.", ".self_attn.k_proj."),
        (".self_attn.linear_v.", ".self_attn.v_proj."),
        (".self_attn.linear_out.", ".self_attn.o_proj."),
        (".self_attn.linear_pos.", ".self_attn.relative_k_proj."),
        (".self_attn.pos_bias_u", ".self_attn.bias_u"),
        (".self_attn.pos_bias_v", ".self_attn.bias_v"),
        ("decoder.prediction.embed.", "decoder.embedding."),
        ("joint.enc.", "encoder_projector."),
        ("joint.pred.", "decoder.decoder_projector."),
        ("joint.joint_net.2.", "joint.head."),
    )
    original = name
    for old, new in replacements:
        name = name.replace(old, new)
    match = re.fullmatch(r"decoder.prediction.dec_rnn.lstm.(\d+).(Wx|Wh|bias)", name)
    if match:
        layer, part = match.groups()
        if part == "bias":
            return [
                (f"decoder.lstm.bias_ih_l{layer}", value),
                (f"decoder.lstm.bias_hh_l{layer}", np.zeros_like(value)),
            ]
        name = f"decoder.lstm.weight_{'ih' if part == 'Wx' else 'hh'}_l{layer}"
    if original.startswith("encoder.pre_encode.conv.") and value.ndim == 4:
        value = value.transpose(0, 3, 1, 2)
    elif ".conv." in original and original.endswith(".weight") and value.ndim == 3:
        value = value.transpose(0, 2, 1)
    return [(name, value)]


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("MLX_AUDIO_CACHE_DIR", str(tmp_path / "cache"))
    config = {
        "preprocessor": dict(
            sample_rate=16000,
            normalize="per_feature",
            window_size=0.025,
            window_stride=0.01,
            window="hann",
            features=8,
            n_fft=512,
            dither=0,
        ),
        "encoder": dict(
            feat_in=8,
            n_layers=1,
            d_model=128,
            n_heads=4,
            ff_expansion_factor=2,
            subsampling_factor=2,
            self_attention_model="rel_pos",
            subsampling="dw_striding",
            conv_kernel_size=3,
            subsampling_conv_channels=4,
            pos_emb_max_len=128,
            use_bias=False,
        ),
        "decoder": dict(
            blank_as_pad=True,
            vocab_size=8,
            prednet=dict(pred_hidden=16, pred_rnn_layers=1),
        ),
        "joint": dict(
            num_classes=8,
            vocabulary=[f"▁word{i}" for i in range(8)],
            num_extra_outputs=3,
            jointnet=dict(
                joint_hidden=16, activation="relu", encoder_hidden=128, pred_hidden=16
            ),
        ),
        "decoding": dict(
            model_type="tdt", durations=[0, 1, 2], greedy=dict(max_symbols=4)
        ),
    }
    mx.random.seed(7)
    model = ParakeetTDT(from_dict(ParakeetTDTArgs, config))
    rows = []
    packed = {f"encoder.layers.0.{s}.weight" for s in ENCODER_SUFFIXES}
    for name, weight in tree_flatten(model.parameters()):
        value = np.asarray(weight.astype(mx.float32))
        for mapped, array in hf_name(name, value):
            if name in packed:
                shape = (array.shape[0], int(np.prod(array.shape[1:])))
                sign = np.resize(np.array([-1, 0, 1], np.int8), shape)
                mask = np.resize(
                    np.array([True, False, False, False, False, True]), shape
                ) & (sign != 0)
                lo, hi = np.full(shape[0], 0.125), np.full(shape[0], 0.375)
                rows.append(
                    (mapped[:-7], "five_value", shape, pack_record(sign, mask, lo, hi))
                )
            else:
                # Force the blank token and duration 1 so synthetic audio terminates deterministically.
                if mapped == "joint.head.weight":
                    array = np.zeros_like(array)
                if mapped == "joint.head.bias":
                    array = np.zeros_like(array)
                    array[8] = array[10] = 20
                rows.append(
                    (mapped, "fp16", array.shape, array.astype("<f2").tobytes())
                )
    rows.append(
        (
            "encoder.layers.0.conv.norm.num_batches_tracked",
            "fp16",
            (),
            np.float16(0).tobytes(),
        )
    )
    source = tmp_path / "unpacked"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(config))
    write_container(source / "model.fermion", rows)
    (source / "packed_manifest.json").write_text(
        json.dumps(
            {
                "format": FORMAT,
                "container_sha256": hashlib.sha256(
                    (source / "model.fermion").read_bytes()
                ).hexdigest(),
            }
        )
    )
    return source


def archive_checkpoint(source, root):
    import zstandard

    root.mkdir()
    files = {
        name: (source / name).read_bytes()
        for name in ("config.json", "model.fermion", "packed_manifest.json")
    }
    for name in ("NOTICE", "LICENSE-WEIGHTS-CC-BY-4.0.txt"):
        if (source / name).is_file():
            files[name] = (source / name).read_bytes()
    manifest = {
        "release_format": "phonon-byteplane-tar-zstd-v1",
        "files": [
            {
                "path": n,
                "original_bytes": len(b),
                "original_sha256": hashlib.sha256(b).hexdigest(),
            }
            for n, b in files.items()
        ],
    }
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as tar:
        for name, blob in {
            "bps_manifest.json": json.dumps(manifest).encode(),
            **files,
        }.items():
            info = tarfile.TarInfo(name)
            info.size = len(blob)
            tar.addfile(info, io.BytesIO(blob))
    archive = zstandard.ZstdCompressor().compress(stream.getvalue())
    (root / "phonon-2.bps.tar.zst").write_bytes(archive)
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "parakeet_tdt_five_value",
                "artifact": {
                    "filename": "phonon-2.bps.tar.zst",
                    "sha256": hashlib.sha256(archive).hexdigest(),
                    "unpacked_bytes": sum(map(len, files.values())),
                },
            }
        )
    )
    return root


def test_standard_local_loader_and_cache(checkpoint, monkeypatch):
    from mlx_audio.stt.models.parakeet import parakeet

    original_frontend = parakeet.log_mel_spectrogram
    model = load(checkpoint)
    assert isinstance(model, Model)
    assert parakeet.log_mel_spectrogram is original_frontend
    assert len(model._packed_modules) == 11
    assert model.preprocessor_config.compute_in_float32
    assert classify_model("parakeet_tdt_five_value") == "stt"
    result = model.generate(mx.zeros(16000))
    assert result.text == ""
    materialized = prepare_model_path(checkpoint)
    mtime = (materialized / "model.safetensors").stat().st_mtime_ns
    again = load(str(checkpoint), lazy=True)
    assert isinstance(again, Model)
    assert (materialized / "model.safetensors").stat().st_mtime_ns == mtime
    assert isinstance(load(materialized), Model)
    weights = mx.load(str(materialized / "model.safetensors"))
    weights.pop(next(iter(weights)))
    with pytest.raises(ValueError):
        again.load_weights(list(weights.items()), strict=False)


def test_hub_loader_download_forwards_revision(checkpoint, tmp_path, monkeypatch):
    from mlx_audio.stt.models.phonon2 import transport

    root = archive_checkpoint(checkpoint, tmp_path / "snapshot")
    archive = root / "phonon-2.bps.tar.zst"
    download = tmp_path / "download.bps.tar.zst"
    archive.rename(download)
    calls = []

    def get_path(repo, **kwargs):
        assert repo == "FermionResearch/Phonon-2"
        return root

    def fetch(repo, filename, **kwargs):
        calls.append((repo, filename, kwargs))
        return str(download)

    monkeypatch.setattr(stt_utils, "get_model_path", get_path)
    monkeypatch.setattr(transport, "hf_hub_download", fetch)
    assert isinstance(
        load("FermionResearch/Phonon-2", revision="pinned", force_download=True), Model
    )
    assert calls == [
        (
            "FermionResearch/Phonon-2",
            "phonon-2.bps.tar.zst",
            {"revision": "pinned", "force_download": True},
        )
    ]


def test_archive_checksum_and_cache_repair(checkpoint, tmp_path):
    root = archive_checkpoint(checkpoint, tmp_path / "snapshot")
    materialized = prepare_model_path(root)
    (materialized / "model.safetensors").write_bytes(b"corrupt")
    assert isinstance(load(root), Model)
    archive = root / "phonon-2.bps.tar.zst"
    archive.write_bytes(archive.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="SHA-256"):
        load(root)


def test_packed_forward_matches_independent_dense_math():
    sign = np.resize(np.array([-1, 0, 1], np.int8), (3, 128))
    mask = (np.indices(sign.shape)[1] % 2 == 0) & (sign != 0)
    a, b = pack_planes(sign, mask)
    layer = PackedFiveValueLinear(128, 3, bias=True)
    layer.base_q, layer.residual_q = mx.array(a), mx.array(b)
    lo = np.array([0.1, 0.2, 0.3], np.float16)
    hi = np.array([0.4, 0.5, 0.6], np.float16)
    layer.row_lo, layer.row_hi = mx.array(lo), mx.array(hi)
    layer.bias = mx.array([1.0, 2.0, 3.0])
    layer.materialize()
    x = np.random.default_rng(8).normal(size=(2, 128)).astype(np.float32)
    dense = sign * np.where(mask, hi[:, None], lo[:, None]).astype(np.float32)
    np.testing.assert_allclose(
        np.asarray(layer(mx.array(x))),
        x @ dense.T + np.array([1.0, 2.0, 3.0]),
        atol=2e-5,
    )


@pytest.mark.parametrize("kind,bits", [("int6", 6), ("int8", 8)])
def test_integer_tables(kind, bits):
    q = np.array([-31, -1, 0, 31, -2, 2, 3, 4], dtype=np.int8)
    if bits == 6:
        u = (q.astype(np.int32) + 32).astype(np.uint32).reshape(-1, 4)
        words = np.sum(u << np.array([0, 6, 12, 18]), axis=1)
        body = (
            ((words[:, None] >> np.array([0, 8, 16])) & 255).astype(np.uint8).tobytes()
        )
    else:
        body = q.tobytes()
    blob = body + np.array([0.5, 0.25], dtype="<f2").tobytes()
    np.testing.assert_array_equal(
        dense_record(blob, (2, 4), kind), q.reshape(2, 4) * [[0.5], [0.25]]
    )


@pytest.mark.parametrize(
    "mutation", ["truncated", "trailing", "duplicate", "bad_format", "huge_header"]
)
def test_invalid_container_rejected(tmp_path, mutation):
    path = tmp_path / "model.fermion"
    row = ("x", "fp16", (1,), np.float16(1).tobytes())
    write_container(path, [row, row] if mutation == "duplicate" else [row])
    data = path.read_bytes()
    if mutation == "truncated":
        data = data[:-1]
    elif mutation == "trailing":
        data += b"x"
    elif mutation == "bad_format":
        data = data.replace(FORMAT.encode(), b"x" * len(FORMAT))
    elif mutation == "huge_header":
        data = (2**63).to_bytes(8, "little") + data[8:]
    path.write_bytes(data)
    with pytest.raises(ValueError):
        list(records(path))


def test_fp32_frontend_is_opt_in_and_finite():
    args = PreprocessArgs(
        16000, "per_feature", 0.025, 0.01, "hann", 8, 512, 0, compute_in_float32=True
    )
    x = mx.random.normal((36 * 16000,)).astype(mx.float16)
    mel = log_mel_spectrogram(x, args)
    ref_args = PreprocessArgs(16000, "per_feature", 0.025, 0.01, "hann", 8, 512, 0)
    reference = log_mel_spectrogram(x.astype(mx.float32), ref_args).astype(mx.float16)
    assert bool(mx.all(mx.isfinite(mel)))
    np.testing.assert_array_equal(np.asarray(mel), np.asarray(reference))
    assert not ref_args.compute_in_float32
    with pytest.raises(ValueError, match="nonempty"):
        log_mel_spectrogram(mx.zeros(0), args)


def test_decoder_matches_stock_control_flow_including_batch_and_padding(checkpoint):
    model = load(checkpoint)
    # Frame 0 emits blank and must preserve LSTM state. A partial final
    # 16-step block must not emit tokens from clamped frames past the end.
    model.encoder = lambda mel: (
        mx.array([[[0.0], [1.0], [1.0]], [[0.0], [1.0], [1.0]]]),
        mx.array([3, 2]),
    )

    def step(feature, token, h, c):
        pred = mx.where(feature[0, 0, 0] == 0, model.blank_id, 3).astype(mx.int32)
        return pred, mx.array(1), h + 1, c + 1

    model._tdt_step = step
    model._compiled_tdt_step = mx.compile(step)
    model._step = model._make_step()
    mel = mx.zeros((2, 6, 8))
    stock = ParakeetTDT.decode(model, mel)
    compiled = model.decode(mel)

    def tokens(result):
        return [(t.id, t.start, t.duration) for s in result.sentences for t in s.tokens]

    assert [tokens(r) for r in stock] == [tokens(r) for r in compiled]
    assert [len(tokens(r)) for r in compiled] == [2, 1]


def test_chunk_streaming_uses_instance_frontend(checkpoint):
    model = load(checkpoint)
    outputs = list(
        model.generate(
            mx.zeros(16000), stream=True, chunk_duration=0.5, overlap_duration=0.1
        )
    )
    assert outputs and outputs[-1].is_final
    assert outputs[-1].audio_position == 1


def test_missing_projection_and_wrong_shape_rejected(checkpoint):
    converted = prepare_model_path(checkpoint)
    config = json.loads((converted / "config.json").read_text())
    config["phonon2_modules"].pop()
    with pytest.raises(ValueError, match="every encoder"):
        Model(config)
    config = json.loads((converted / "config.json").read_text())
    config["phonon2_modules"][0]["shape"] = [1, 128]
    with pytest.raises(ValueError, match="shape mismatch"):
        Model(config)


@pytest.mark.parametrize("bad", ["base3", "palette", "width", "zero_high"])
def test_malformed_packed_record_rejected(tmp_path, bad):
    path = tmp_path / "model.fermion"
    sign = np.ones((1, 128), np.int8)
    mask = np.zeros_like(sign, dtype=bool)
    blob = pack_record(sign, mask, np.array([0.1]), np.array([0.3]))
    if bad == "base3":
        blob = b"\xff" + blob[1:]
    elif bad == "palette":
        blob = blob[:-2] + np.float16(-1).tobytes()
    elif bad == "width":
        with pytest.raises(ValueError):
            pack_planes(sign[:, :127], mask[:, :127])
        return
    else:
        with pytest.raises(ValueError):
            pack_planes(np.zeros_like(sign), np.ones_like(mask))
        return
    write_container(
        path, [("encoder.layers.0.self_attn.q_proj", "five_value", sign.shape, blob)]
    )
    with pytest.raises(ValueError):
        list(records(path))


@pytest.mark.parametrize("samples", [1, 319])
def test_short_audio_and_chunk_tail(checkpoint, samples):
    model = load(checkpoint)
    assert model.generate(mx.zeros(samples)).text == ""
    assert (
        model.generate(
            mx.zeros(8000 + samples), chunk_duration=0.5, overlap_duration=0
        ).text
        == ""
    )
    chunks = list(
        model.generate(
            mx.zeros(8000 + samples),
            stream=True,
            chunk_duration=0.5,
            overlap_duration=0,
        )
    )
    assert chunks[-1].is_final
    assert chunks[-1].audio_position == (8000 + samples) / 16000


@pytest.mark.parametrize("value", [[], None, "broken"])
def test_corrupt_cache_marker_is_rebuilt(checkpoint, value):
    converted = prepare_model_path(checkpoint)
    (converted / "conversion.json").write_text(json.dumps(value))
    assert isinstance(load(checkpoint), Model)


@pytest.mark.parametrize(
    "artifact",
    [None, [], "bad", {"filename": None}, {"filename": "x.bps.tar.zst", "sha256": 3}],
)
def test_invalid_artifact_metadata(checkpoint, artifact):
    (checkpoint / "config.json").write_text(json.dumps({"artifact": artifact}))
    with pytest.raises(ValueError, match="artifact metadata"):
        prepare_model_path(checkpoint)


@pytest.mark.parametrize("header", [[], {"format": FORMAT, "index": [None]}])
def test_malformed_json_container_structure(tmp_path, header):
    blob = json.dumps(header).encode()
    path = tmp_path / "model.fermion"
    path.write_bytes(len(blob).to_bytes(8, "little") + blob)
    with pytest.raises(ValueError, match="Phonon-2"):
        list(records(path))


@pytest.mark.parametrize("manifest", ["broken json", "[]", "null"])
def test_detector_ignores_invalid_unrelated_manifest(checkpoint, manifest):
    from mlx_audio.stt.models.phonon2.transport import is_phonon2_model

    (checkpoint / "packed_manifest.json").write_text(manifest)
    assert not is_phonon2_model(checkpoint, {})


@pytest.mark.parametrize("suffix", ["base_q", "residual_q", "row_lo", "row_hi"])
def test_weight_cast_rejected(checkpoint, suffix):
    model = load(checkpoint)
    weights = mx.load(str(prepare_model_path(checkpoint) / "model.safetensors"))
    name = next(k for k in weights if k.endswith("." + suffix))
    weights[name] = weights[name].astype(mx.float32)
    with pytest.raises(ValueError, match="must remain"):
        model.load_weights(weights.items())


@pytest.mark.parametrize(
    "maximum,durations",
    [(None, [0, 1, 2]), (0, [0, 1, 2]), (4, [0, -1, 2]), (4, [0, 1])],
)
def test_invalid_decoder_config_rejected(checkpoint, maximum, durations):
    config = json.loads((prepare_model_path(checkpoint) / "config.json").read_text())
    config["decoding"]["greedy"]["max_symbols"] = maximum
    config["decoding"]["durations"] = durations
    with pytest.raises(ValueError, match="durations"):
        Model(config)


def test_generic_conversion_guard(checkpoint):
    from mlx_audio.convert import Domain, get_model_type
    from mlx_audio.stt.models.phonon2 import prepare_config

    assert (
        get_model_type(
            {"model_type": "parakeet_tdt_five_value"}, checkpoint, Domain.STT
        )
        == "phonon2"
    )
    for path in (checkpoint, prepare_model_path(checkpoint)):
        config = json.loads((path / "config.json").read_text())
        with pytest.raises(ValueError, match="mlx_audio.stt.load"):
            prepare_config(config, path)


def test_archive_notices_survive_conversion(checkpoint, tmp_path):
    for name in ("NOTICE", "LICENSE-WEIGHTS-CC-BY-4.0.txt"):
        (checkpoint / name).write_text("archive-only attribution")
    root = archive_checkpoint(checkpoint, tmp_path / "snapshot")
    converted = prepare_model_path(root)
    for name in ("NOTICE", "LICENSE-WEIGHTS-CC-BY-4.0.txt"):
        assert (converted / name).read_text() == "archive-only attribution"
