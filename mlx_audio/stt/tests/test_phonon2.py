"""Offline format, standard-loader, and inference regression tests for Phonon-2."""

import hashlib
import io
import json
import re
import tarfile
from dataclasses import asdict

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_audio.registry import classify_model
from mlx_audio.stt import load
from mlx_audio.stt.models.parakeet.audio import PreprocessArgs, log_mel_spectrogram
from mlx_audio.stt.models.parakeet.parakeet import ParakeetTDT, ParakeetTDTArgs
from mlx_audio.stt.models.phonon2 import Model, ModelConfig
from mlx_audio.stt.models.phonon2.convert import FORMAT, convert, dense_record, records
from mlx_audio.stt.models.phonon2.packed import (
    ENCODER_SUFFIXES,
    PackedFiveValueLinear,
    pack_planes,
)
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
def source_checkpoint(tmp_path):
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


@pytest.fixture
def checkpoint(source_checkpoint, tmp_path):
    return convert(str(source_checkpoint), str(tmp_path / "mlx-model"))


def archive_checkpoint(source, root):
    import zstandard

    root.mkdir()
    files = {
        name: (source / name).read_bytes()
        for name in ("config.json", "model.fermion", "packed_manifest.json")
    }
    for name in (
        "NOTICE",
        "LICENSE-WEIGHTS-CC-BY-4.0.txt",
        "LICENSE-CODE-Apache-2.0.txt",
    ):
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


def test_standard_local_loader(checkpoint, monkeypatch):
    import importlib

    from mlx_audio.stt.models.parakeet import parakeet

    converter = importlib.import_module("mlx_audio.stt.models.phonon2.convert")

    def unexpected_conversion(*args, **kwargs):
        pytest.fail("Runtime loading must not extract or convert model archives")

    monkeypatch.setattr(converter, "unpack_source", unexpected_conversion)
    original_frontend = parakeet.log_mel_spectrogram
    model = load(checkpoint)
    assert isinstance(model, Model)
    assert isinstance(model.config, ModelConfig)
    assert parakeet.log_mel_spectrogram is original_frontend
    assert len(model._packed_modules) == 11
    assert classify_model("phonon2") == "stt"
    assert model.generate(mx.zeros(16000)).text == ""
    config = json.loads((checkpoint / "config.json").read_text())
    assert PreprocessArgs(**config["preprocessor"]) == model.preprocessor_config
    again = load(str(checkpoint), lazy=True)
    assert isinstance(again, Model)
    weights = mx.load(str(checkpoint / "model.safetensors"))
    weights.pop(next(iter(weights)))
    with pytest.raises(ValueError):
        again.load_weights(list(weights.items()), strict=False)


def test_hub_converter_forwards_revision(source_checkpoint, tmp_path, monkeypatch):
    import importlib

    converter = importlib.import_module("mlx_audio.stt.models.phonon2.convert")

    root = archive_checkpoint(source_checkpoint, tmp_path / "snapshot")
    archive = root / "phonon-2.bps.tar.zst"
    download = tmp_path / "download.bps.tar.zst"
    archive.rename(download)
    calls = []

    def get_path(repo, **kwargs):
        assert repo == "FermionResearch/Phonon-2"
        assert kwargs["revision"] == "pinned"
        assert kwargs["force_download"]
        return root

    def fetch(repo, filename, **kwargs):
        calls.append((repo, filename, kwargs))
        return str(download)

    monkeypatch.setattr(converter, "get_model_path", get_path)
    monkeypatch.setattr(converter, "hf_hub_download", fetch)
    output = convert(
        "FermionResearch/Phonon-2",
        str(tmp_path / "mlx"),
        revision="pinned",
        force_download=True,
    )
    assert isinstance(load(output), Model)
    assert calls == [
        (
            "FermionResearch/Phonon-2",
            "phonon-2.bps.tar.zst",
            {"revision": "pinned", "force_download": True},
        )
    ]


def test_archive_checksum_and_source_preserved(source_checkpoint, tmp_path):
    root = archive_checkpoint(source_checkpoint, tmp_path / "snapshot")
    source_hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()
    }
    output = convert(str(root), str(tmp_path / "mlx"))
    assert isinstance(load(output), Model)
    assert source_hashes == {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()
    }
    archive = root / "phonon-2.bps.tar.zst"
    archive.write_bytes(archive.read_bytes() + b"corrupt")
    target = tmp_path / "failed"
    with pytest.raises(ValueError, match="SHA-256"):
        convert(str(root), str(target))
    assert not target.exists()


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


def test_model_config_round_trip_and_nested_types(checkpoint):
    path = checkpoint
    raw = json.loads((path / "config.json").read_text())
    raw["model_path"] = str(path)
    raw["unrelated_metadata"] = "ignored"
    original = json.dumps(raw, sort_keys=True)
    config = ModelConfig.from_dict(raw)
    parakeet = from_dict(ParakeetTDTArgs, raw)
    for name in ("preprocessor", "encoder", "decoder", "joint", "decoding"):
        assert getattr(config, name) == getattr(parakeet, name)
    assert config.model_type == "phonon2"
    assert config.model_path == str(path)
    assert config.phonon2_conversion_version == raw["phonon2_conversion_version"]
    assert config.phonon2_modules == raw["phonon2_modules"]
    assert ModelConfig.from_dict(asdict(config)) == config
    assert json.dumps(raw, sort_keys=True) == original


def test_inherited_decoder_handles_batch_lengths_and_blank_state(checkpoint):
    model = load(checkpoint)
    assert Model.decode is ParakeetTDT.decode
    model.encoder = lambda mel: (
        mx.array([[[0.0], [1.0], [1.0]], [[0.0], [1.0], [1.0]]]),
        mx.array([3, 2]),
    )
    states = []

    def step(feature, token, h, c):
        states.append(int(h[0, 0, 0]))
        pred = mx.where(feature[0, 0, 0] == 0, model.blank_id, 3).astype(mx.int32)
        return pred, mx.array(1), h + 1, c + 1

    model._compiled_tdt_step = step
    results = model.decode(mx.zeros((2, 6, 8)))
    tokens = [[t for s in result.sentences for t in s.tokens] for result in results]
    assert [len(batch) for batch in tokens] == [2, 1]
    assert [[t.id for t in batch] for batch in tokens] == [[3, 3], [3]]
    assert states == [0, 0, 1, 0, 0]
    assert [[t.start for t in batch] for batch in tokens] == [[0.02, 0.04], [0.02]]


def test_inherited_decoder_bounds_zero_duration_emissions(checkpoint):
    model = load(checkpoint)
    model.encoder = lambda mel: (mx.zeros((1, 2, 1)), mx.array([2]))
    model._compiled_tdt_step = lambda feature, token, h, c: (
        mx.array(3),
        mx.array(0),
        h + 1,
        c + 1,
    )
    result = model.decode(mx.zeros((1, 6, 8)))[0]
    tokens = [t for sentence in result.sentences for t in sentence.tokens]
    assert len(tokens) == 2 * model.max_symbols
    assert [t.start for t in tokens] == [0.0] * model.max_symbols + [
        0.02
    ] * model.max_symbols


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
    converted = checkpoint
    config = json.loads((converted / "config.json").read_text())
    config["phonon2_modules"].pop()
    with pytest.raises(ValueError, match="every encoder"):
        Model(ModelConfig.from_dict(config))
    config = json.loads((converted / "config.json").read_text())
    config["phonon2_modules"][0]["shape"] = [1, 128]
    with pytest.raises(ValueError, match="shape mismatch"):
        Model(ModelConfig.from_dict(config))


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


def test_converter_refuses_overwrite_and_source(source_checkpoint, checkpoint):
    original = (checkpoint / "model.safetensors").read_bytes()
    with pytest.raises(ValueError, match="not an empty directory"):
        convert(str(source_checkpoint), str(checkpoint))
    assert (checkpoint / "model.safetensors").read_bytes() == original
    with pytest.raises(ValueError, match="differ"):
        convert(str(source_checkpoint), str(source_checkpoint))


def test_converter_accepts_empty_destination(source_checkpoint, tmp_path):
    output = tmp_path / "empty"
    output.mkdir()
    assert isinstance(load(convert(str(source_checkpoint), str(output))), Model)


@pytest.mark.parametrize(
    "artifact",
    [None, [], "bad", {"filename": None}, {"filename": "x.bps.tar.zst", "sha256": 3}],
)
def test_invalid_artifact_metadata(source_checkpoint, tmp_path, artifact):
    (source_checkpoint / "config.json").write_text(json.dumps({"artifact": artifact}))
    with pytest.raises(ValueError, match="artifact metadata"):
        convert(str(source_checkpoint), str(tmp_path / "failed"))


@pytest.mark.parametrize("header", [[], {"format": FORMAT, "index": [None]}])
def test_malformed_json_container_structure(tmp_path, header):
    blob = json.dumps(header).encode()
    path = tmp_path / "model.fermion"
    path.write_bytes(len(blob).to_bytes(8, "little") + blob)
    with pytest.raises(ValueError, match="Phonon-2"):
        list(records(path))


@pytest.mark.parametrize("manifest", ["broken json", "[]", "null"])
def test_invalid_source_manifest(source_checkpoint, tmp_path, manifest):
    (source_checkpoint / "packed_manifest.json").write_text(manifest)
    with pytest.raises(ValueError):
        convert(str(source_checkpoint), str(tmp_path / "failed"))


@pytest.mark.parametrize("suffix", ["base_q", "residual_q", "row_lo", "row_hi"])
def test_weight_cast_rejected(checkpoint, suffix):
    model = load(checkpoint)
    weights = mx.load(str(checkpoint / "model.safetensors"))
    name = next(k for k in weights if k.endswith("." + suffix))
    weights[name] = weights[name].astype(mx.float32)
    with pytest.raises(ValueError, match="must remain"):
        model.load_weights(weights.items())


@pytest.mark.parametrize(
    "maximum,durations",
    [(None, [0, 1, 2]), (0, [0, 1, 2]), (4, [0, -1, 2]), (4, [0, 1])],
)
def test_invalid_decoder_config_rejected(checkpoint, maximum, durations):
    config = json.loads((checkpoint / "config.json").read_text())
    config["decoding"]["greedy"]["max_symbols"] = maximum
    config["decoding"]["durations"] = durations
    with pytest.raises(ValueError, match="durations"):
        Model(ModelConfig.from_dict(config))


def test_generic_conversion_guard(checkpoint):
    from mlx_audio.convert import Domain, get_model_type
    from mlx_audio.stt.models.phonon2 import prepare_config

    assert (
        get_model_type(
            {"model_type": "parakeet_tdt_five_value"}, checkpoint, Domain.STT
        )
        == "phonon2"
    )
    config = json.loads((checkpoint / "config.json").read_text())
    with pytest.raises(ValueError, match="phonon2.convert"):
        prepare_config(config, checkpoint)


def test_archive_notices_survive_conversion(source_checkpoint, tmp_path):
    for name in (
        "NOTICE",
        "LICENSE-WEIGHTS-CC-BY-4.0.txt",
        "LICENSE-CODE-Apache-2.0.txt",
    ):
        (source_checkpoint / name).write_text("archive-only attribution")
    root = archive_checkpoint(source_checkpoint, tmp_path / "snapshot")
    output = convert(str(root), str(tmp_path / "mlx"))
    for name in (
        "NOTICE",
        "LICENSE-WEIGHTS-CC-BY-4.0.txt",
        "LICENSE-CODE-Apache-2.0.txt",
    ):
        assert (output / name).read_text() == "archive-only attribution"


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
def test_phonon2_frontend_matches_fp32_reference(checkpoint, dtype):
    model = load(checkpoint)
    assert not hasattr(model.preprocessor_config, "compute_in_float32")
    waveform = mx.random.normal((36 * 16000,)).astype(dtype)
    expected = log_mel_spectrogram(
        waveform.astype(mx.float32), model.preprocessor_config
    ).astype(dtype)
    actual = model._log_mel_spectrogram(waveform)
    assert bool(mx.all(mx.isfinite(actual)))
    np.testing.assert_array_equal(
        np.asarray(actual.astype(mx.float32)), np.asarray(expected.astype(mx.float32))
    )
    with pytest.raises(ValueError, match="nonempty"):
        model._log_mel_spectrogram(mx.zeros(0))
    with pytest.raises(ValueError, match="mono"):
        model._log_mel_spectrogram(mx.zeros((2, 16000)))


def test_all_generation_paths_use_local_frontend(checkpoint, monkeypatch):
    model = load(checkpoint)
    calls = []
    frontend = model._log_mel_spectrogram

    def record(audio):
        calls.append(audio.size)
        return frontend(audio)

    monkeypatch.setattr(model, "_log_mel_spectrogram", record)
    model.generate(mx.zeros(8001), chunk_duration=0.5, overlap_duration=0)
    assert calls == [8000, 1]
    calls.clear()
    list(
        model.generate(
            mx.zeros(8001), stream=True, chunk_duration=0.5, overlap_duration=0
        )
    )
    assert calls == [8000, 1]
    calls.clear()
    model.generate(mx.zeros(1))
    assert calls == [1]
