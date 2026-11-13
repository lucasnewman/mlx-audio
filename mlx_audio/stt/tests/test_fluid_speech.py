"""Offline coverage for Fluid Speech's frontend, adapters, and checkpoint layout."""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten
from scipy.special import erf

from mlx_audio.registry import classify_model, model_type_from_config
from mlx_audio.stt import load
from mlx_audio.stt.models.fluid_speech import Model, ModelConfig, load_source_weights
from mlx_audio.stt.models.fluid_speech.fluid_speech import FrameCleaner, LowRankAdded
from mlx_audio.utils import mel_filters


@pytest.fixture
def config():
    return {
        "target": "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel",
        "preprocessor": {
            "sample_rate": 16000,
            "normalize": "per_feature",
            "window_size": 0.025,
            "window_stride": 0.01,
            "window": "hann",
            "features": 16,
            "n_fft": 512,
            "dither": 0.0,
        },
        "encoder": {
            "feat_in": 16,
            "n_layers": 1,
            "d_model": 64,
            "n_heads": 4,
            "ff_expansion_factor": 2,
            "subsampling_factor": 8,
            "self_attention_model": "rel_pos",
            "subsampling": "dw_striding",
            "conv_kernel_size": 3,
            "subsampling_conv_channels": 8,
            "pos_emb_max_len": 128,
            "use_bias": False,
        },
        "decoder": {
            "blank_as_pad": True,
            "vocab_size": 5,
            "prednet": {"pred_hidden": 32, "pred_rnn_layers": 2},
        },
        "joint": {
            "num_classes": 5,
            "num_extra_outputs": 5,
            "vocabulary": ["<unk>", "▁hello", "▁world", ".", "!"],
            "jointnet": {
                "joint_hidden": 64,
                "activation": "relu",
                "encoder_hidden": 64,
                "pred_hidden": 32,
            },
        },
        "decoding": {
            "model_type": "tdt",
            "durations": [0, 1, 2, 3, 4],
            "greedy": {"max_symbols": 4},
        },
        "fluid_speech": {
            "insert": "pre",
            "insert_dim": 64,
            "insert_kernel": 15,
            "clean_mode": {"rank": 2, "alpha": 4.0},
        },
    }


@pytest.fixture
def checkpoint(tmp_path, config):
    mx.random.seed(7)
    model = Model(ModelConfig.from_dict(config))
    model.eval()
    path = tmp_path / "arbitrary-folder"
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config))
    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(path / "model.safetensors"), weights)
    # Independent list of adapter targets from the published checkpoint.
    suffixes = [
        "feed_forward1.linear1",
        "feed_forward1.linear2",
        "feed_forward2.linear1",
        "feed_forward2.linear2",
        "self_attn.linear_q",
        "self_attn.linear_k",
        "self_attn.linear_v",
        "self_attn.linear_out",
        "self_attn.linear_pos",
        "conv.pointwise_conv1",
        "conv.pointwise_conv2",
    ]
    rng = np.random.default_rng(3)
    adapter = {}
    for suffix in suffixes:
        name = f"layers.0.{suffix}"
        weight = weights[f"encoder.{name}.weight"]
        adapter[name + ".A"] = mx.array(
            rng.normal(0, 0.01, (2, weight.shape[-1])).astype(np.float32)
        )
        adapter[name + ".B"] = mx.array(
            rng.normal(0, 0.01, (weight.shape[0], 2)).astype(np.float32)
        )
    adapter["decoder_input"] = mx.full((32,), 0.1)
    adapter["joint_hidden"] = mx.full((64,), 0.2)
    mx.save_safetensors(str(path / "clean-mode.safetensors"), adapter)
    # An alternate full checkpoint must never overwrite the selected weights.
    mx.save_safetensors(str(path / "model.8bit.safetensors"), {"wrong": mx.ones((1,))})
    return path, weights, adapter


@pytest.mark.parametrize("model_type", [None, "parakeet", "fluid_speech"])
def test_metadata_detection(config, model_type):
    if model_type:
        config["model_type"] = model_type
    family = model_type_from_config(config)
    assert family == "fluid_speech"
    assert classify_model(family) == "stt"
    from mlx_audio.convert import Domain, detect_model_domain, get_model_type

    path = Path("/tmp/arbitrary-folder")
    assert detect_model_domain(config, path) == Domain.STT
    assert get_model_type(config, path, Domain.STT) == "fluid_speech"


@pytest.mark.parametrize(
    "field,value", [("insert", "post"), ("insert_dim", 0), ("insert_kernel", 4)]
)
def test_invalid_frame_cleaner(config, field, value):
    config["fluid_speech"][field] = value
    with pytest.raises(ValueError, match="pre-encoder"):
        ModelConfig.from_dict(config)


def test_frame_cleaner_matches_independent_numpy():
    mx.random.seed(8)
    model = FrameCleaner(8, 4, 3)
    x = np.random.default_rng(4).normal(size=(2, 9, 8)).astype(np.float32)
    w = {k: np.asarray(v) for k, v in tree_flatten(model.parameters())}
    h = (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5)
    h = h * w["ln.weight"] + w["ln.bias"]
    h = h @ w["down.weight"].T + w["down.bias"]
    padded = np.pad(h, ((0, 0), (1, 1), (0, 0)))
    conv = (
        sum(padded[:, i : i + 9] * w["mix.weight"][:, i, 0] for i in range(3))
        + w["mix.bias"]
    )
    h = conv * 0.5 * (1 + erf(conv / np.sqrt(2)))
    expected = x + h @ w["up.weight"].T + w["up.bias"]
    # Avoid GPU matmul precision differences in the independent formula check.
    with mx.stream(mx.cpu):
        actual = np.asarray(model(mx.array(x)))
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("samples", [320, 1600, 1601, 2720])
def test_frontend_matches_nemo_formula(config, samples):
    model = Model(ModelConfig.from_dict(config))
    audio = np.random.default_rng(5).normal(0, 0.1, samples).astype(np.float32)
    x = np.concatenate([audio[:1], audio[1:] - 0.97 * audio[:-1]])
    x = np.pad(x, (256, 256))
    valid = samples // 160
    indices = np.arange(valid + 1)[:, None] * 160 + np.arange(512)[None, :]
    window = np.pad(np.hanning(400), (56, 56))
    power = np.abs(np.fft.rfft(x[indices] * window, axis=-1)) ** 2
    filters = np.asarray(mel_filters(16000, 512, 16, norm="slaney", mel_scale="slaney"))
    mel = np.log(power @ filters.T + 2.0**-24)
    real = mel[:valid]
    expected = (mel - real.mean(0)) / (real.std(0, ddof=1) + 1e-5)
    expected[valid:] = 0
    with mx.stream(mx.cpu):
        actual = np.asarray(model._log_mel_spectrogram(mx.array(audio)))
    np.testing.assert_allclose(actual[0], expected, atol=2e-5, rtol=2e-5)
    assert np.all(actual[0, -1] == 0)


def test_subsampling_masks_padding_and_keeps_frame_cleaner_context(config):
    model = Model(ModelConfig.from_dict(config))
    pre = model.encoder.pre_encode
    pre.eval()
    # 16 real frames plus NeMo's extra frame -> 3 subsampled frames,
    # of which only 2 survive cropping. Its projection bias still feeds the
    # frame cleaner, so cropping before that block would change the answer.
    x = mx.random.normal((1, 17, 16))
    lengths = mx.array([16])
    actual, result_lengths = pre(x, lengths)
    poisoned = mx.concatenate([x[:, :16], mx.full((1, 1, 16), 10000.0)], axis=1)
    np.testing.assert_allclose(actual, pre(poisoned, lengths)[0], atol=1e-6)
    stages = []
    original = pre.fv_insert

    class RecordingCleaner(nn.Module):
        def __call__(self, frames):
            stages.append(frames)
            return original(frames)

    pre.fv_insert = RecordingCleaner()
    pre(x, lengths)
    assert stages[0].shape[1] == 3
    assert result_lengths.tolist() == [2]
    assert actual.shape == (1, 2, 64)
    np.testing.assert_allclose(actual, original(stages[0])[:, :2], atol=1e-6)
    assert not np.allclose(actual, original(stages[0][:, :2]))


def test_direct_load_selects_dense_weights_and_requires_frame_cleaner(checkpoint):
    path, weights, _ = checkpoint
    model = load(path)
    assert isinstance(model, Model)
    assert set(load_source_weights(path)) == set(weights)
    weights.pop("encoder.pre_encode.fv_insert.mix.weight")
    mx.save_safetensors(str(path / "model.safetensors"), weights)
    with pytest.raises(ValueError, match="fv_insert.mix.weight"):
        load(path, strict=False)


def test_sharded_checkpoint_ignores_adapter_and_other_precision(checkpoint):
    path, weights, _ = checkpoint
    (path / "model.safetensors").unlink()
    mx.save_safetensors(str(path / "model-00001-of-00001.safetensors"), weights)
    (path / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {key: "model-00001-of-00001.safetensors" for key in weights}}
        )
    )
    assert isinstance(load(path, strict=True), Model)


@pytest.mark.parametrize("quantized", [False, True])
def test_clean_mode_matches_folded_update_and_is_reversible(checkpoint, quantized):
    path, _, adapter = checkpoint
    model = load(path)
    if quantized:
        nn.quantize(
            model,
            group_size=32,
            bits=8,
            class_predicate=lambda p, m: model.model_quant_predicate(p, m)
            and m.weight.shape[-1] % 32 == 0,
        )
    base = model.encoder.layers[0].feed_forward1.linear1
    x = mx.random.normal((1, 4, 64))
    expected = (
        base(x)
        + x
        @ (
            2
            * adapter["layers.0.feed_forward1.linear1.B"]
            @ adapter["layers.0.feed_forward1.linear1.A"]
        ).T
    )
    before = dict(tree_flatten(model.parameters()))
    lstm = model.decoder.prediction["dec_rnn"].lstm[0]
    expected_bias = lstm.bias + lstm.Wx @ adapter["decoder_input"]
    model.set_clean_mode(True)
    np.testing.assert_allclose(
        model.encoder.layers[0].feed_forward1.linear1(x), expected, rtol=2e-5, atol=2e-6
    )
    np.testing.assert_allclose(lstm.bias, expected_bias, atol=1e-6)
    adapted = model.encoder.layers[0].feed_forward1.linear1
    assert isinstance(adapted, LowRankAdded)
    model.set_clean_mode(True)
    assert model.encoder.layers[0].feed_forward1.linear1 is adapted
    model.set_clean_mode(False)
    assert model.encoder.layers[0].feed_forward1.linear1 is base
    for key, value in tree_flatten(model.parameters()):
        np.testing.assert_array_equal(value, before[key])
    model.set_clean_mode(True)
    np.testing.assert_allclose(
        model.encoder.layers[0].feed_forward1.linear1(x), expected, rtol=2e-5, atol=2e-6
    )


def test_missing_clean_adapter_fails_before_mutating_model(checkpoint):
    path, _, adapter = checkpoint
    del adapter["layers.0.self_attn.linear_q.B"]
    mx.save_safetensors(str(path / "clean-mode.safetensors"), adapter)
    model = load(path)
    before = dict(tree_flatten(model.parameters()))
    with pytest.raises(ValueError, match="Incomplete"):
        model.set_clean_mode(True)
    assert not model._clean_mode
    assert set(dict(tree_flatten(model.parameters()))) == set(before)
    (path / "clean-mode.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="adapter"):
        model.set_clean_mode(True)


def test_generation_and_streaming_dispatch(checkpoint):
    path, _, _ = checkpoint
    model = load(path)
    # Make the greedy decoder emit blanks and advance one frame per step.
    head = model.joint.joint_net[-1]
    head.weight = mx.zeros_like(head.weight)
    head.bias = mx.zeros_like(head.bias)
    head.bias[5] = 10
    head.bias[7] = 10
    audio = mx.zeros((1600,))
    assert model.generate(audio).text == ""
    assert model.generate(audio, clean_mode=True).text == ""
    chunks = list(model.generate(audio, stream=True, clean_mode=False))
    assert len(chunks) == 1 and chunks[0].is_final
    assert not model._clean_mode
    assert model.generate(mx.zeros((16,))).text == ""
    with pytest.raises(ValueError, match="nonempty mono"):
        model.generate(mx.zeros((0,)))


@pytest.mark.parametrize("bits", [4, 8])
def test_conversion_preserves_clean_mode_and_reloads(checkpoint, tmp_path, bits):
    from mlx_audio.convert import convert

    source, _, _ = checkpoint
    destination = tmp_path / f"converted-{bits}"
    convert(
        str(source),
        str(destination),
        quantize=True,
        q_group_size=32,
        q_bits=bits,
        dtype="bfloat16",
        model_domain="stt",
    )
    assert (destination / "clean-mode.safetensors").is_file()
    assert not (destination / "model.8bit.safetensors").exists()
    model = load(destination, strict=True)
    assert isinstance(model.encoder.layers[0].feed_forward1.linear1, nn.QuantizedLinear)
    model.set_clean_mode(True)
    features, lengths = model.encoder(model._log_mel_spectrogram(mx.zeros((1600,))))
    assert bool(mx.all(mx.isfinite(features)))
    assert lengths.tolist() == [2]
    assert "Fluid Speech Pro by FluidVoice" in (destination / "README.md").read_text()
