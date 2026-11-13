"""Offline FastDialogueSidon tests; no checkpoint downloads."""

import json
import os
from dataclasses import asdict

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_audio.sts.models.dialogue_sidon.config import EncoderConfig
from mlx_audio.sts.models.fast_dialogue_sidon import Model, ModelConfig
from mlx_audio.sts.models.fast_dialogue_sidon.config import (
    GeneratorConfig,
    HeadConfig,
    VocoderConfig,
)
from mlx_audio.sts.models.fast_dialogue_sidon.model import sanitize_source_weights
from mlx_audio.sts.models.fast_dialogue_sidon.vocoder import stft_features


def small_config():
    return ModelConfig(
        latent_dim=4,
        encoder=EncoderConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            conv_depthwise_kernel_size=5,
            left_max_position_embeddings=4,
            right_max_position_embeddings=2,
        ),
        diffusion=HeadConfig(hidden_size=16, num_layers=1, num_heads=2),
        vocoder=VocoderConfig(
            latent_dim=4,
            n_fft_stft=32,
            generator_args=GeneratorConfig(
                hidden_dim=8, intermediate_dim=16, num_layers=1, kernel_size=3
            ),
        ),
    )


@pytest.fixture
def model():
    return Model(small_config())


def save_model(model, path):
    path.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(path / "model.safetensors"), dict(tree_flatten(model.parameters()))
    )
    (path / "config.json").write_text(json.dumps(asdict(model.config)))


def test_config_roundtrip_and_validation():
    config = small_config()
    assert ModelConfig.from_dict(asdict(config)) == config
    for kwargs in (
        {"latent_dim": 2},
        {"ctm": {"guidance_scale": 2}},
        {"ctm": {"alpha_t": float("nan")}},
        {"vocoder": {"latent_upsample": 3}},
        {"vocoder": {"latent_rate": 100}},
    ):
        with pytest.raises(ValueError):
            ModelConfig(**kwargs)


def test_loader_registration_and_roundtrip(model, tmp_path):
    from mlx_audio.registry import classify_model
    from mlx_audio.sts import FastDialogueSidon, load

    for kind in ("fast_dialogue_sidon", "fast-dialogue-sidon"):
        assert classify_model(kind) == "sts"
    assert classify_model("", "higurashi-ai/FastDialogueSidon") == "sts"
    assert FastDialogueSidon is Model
    save_model(model, tmp_path)
    loaded = load(tmp_path, strict=True)
    for key, value in tree_flatten(model.parameters()):
        np.testing.assert_array_equal(
            np.array(value), np.array(dict(tree_flatten(loaded.parameters()))[key])
        )


def test_original_bundle_load(model, tmp_path):
    """Exercise real source names and nested configs through the public loader."""
    from mlx_audio.sts import load

    config = model.config
    enc, head, voc, stats = {}, {}, {}, {}
    for key, value in tree_flatten(model.parameters()):
        if key.startswith("encoder."):
            key = key.removeprefix("encoder.")
            key = "model." + ("encoder." if key.startswith("layers.") else "") + key
            enc[key] = value.transpose(0, 2, 1) if value.ndim == 3 else value
        elif key.startswith(("linear1.", "linear2.")):
            enc[
                key.replace("linear1.", "speaker_1.").replace("linear2.", "speaker_2.")
            ] = value
        elif key.startswith("diffusion_head."):
            head[key.removeprefix("diffusion_head.").replace(".layers.", ".")] = value
        elif key.startswith("decoder."):
            key = key.removeprefix("decoder.")
            if value.ndim == 3:
                value = (
                    value.transpose(2, 0, 1)
                    if key == "upsampler.weight"
                    else value.transpose(0, 2, 1)
                )
            for name in ("input_norm", "final_norm", "norm"):
                if f".{name}." in key:
                    key = key.replace(f".{name}.", f".{name}.norm.")
                    break
            voc[key] = value
        else:
            stats[key] = value[None, None]
    enc["model.masked_spec_embed"] = mx.zeros((config.encoder.hidden_size,))
    encoder_config = {
        **asdict(config.encoder),
        "position_embeddings_type": "relative_key",
    }
    head_config = {
        "hidden_size": 16,
        "head_layers": 1,
        "num_heads": 2,
        "head_ffn_ratio": 4,
        "latent_size": 8,
        "cond_size": 24,
    }
    for name, weights, cfg in (
        ("encoder", enc, encoder_config),
        ("diffusion_head", head, head_config),
        ("vocoder", voc, asdict(config.vocoder)),
    ):
        directory = tmp_path / name
        directory.mkdir()
        mx.save_safetensors(str(directory / "model.safetensors"), weights)
        (directory / "config.json").write_text(json.dumps(cfg))
    mx.save_safetensors(str(tmp_path / "stats.safetensors"), stats)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "fast-dialogue-sidon",
                "latent_dim": 4,
                "ctm": asdict(config.ctm),
            }
        )
    )
    restored = load(tmp_path, strict=True)
    original = dict(tree_flatten(model.parameters()))
    for key, value in tree_flatten(restored.parameters()):
        np.testing.assert_array_equal(np.array(value), np.array(original[key]))
    # A standalone export must not depend on source bundle paths.
    from mlx_audio.sts.models.fast_dialogue_sidon.convert import convert

    output = convert(str(tmp_path), str(tmp_path / "export"), dtype="float16")
    exported_config = json.loads((output / "config.json").read_text())
    assert "model_path" not in exported_config
    assert exported_config["weight_format"] == "mlx"
    exported = load(output, strict=True)
    assert exported.dtype == mx.float16
    assert exported.latent_norm_std.dtype == mx.float32
    with pytest.raises(FileExistsError):
        convert(str(tmp_path), str(output))


def test_ctm_uses_both_times_and_exact_update(model, monkeypatch):
    calls = []

    def head(noise, t, s, conditioning):
        calls.append((np.array(t), np.array(s)))
        return mx.full(noise.shape, 0.25)

    monkeypatch.setattr(model, "diffusion_head", head)
    noise = mx.full((2, 3, 8), 0.5)
    result = model.sample_latents(mx.zeros((2, 3, 24)), initial_noise=noise)
    c = model.config.ctm
    expected = c.c_skip * 0.5 + c.c_out * (c.alpha_t * 0.5 - c.sigma_t * 0.25)
    np.testing.assert_allclose(np.array(result), expected, atol=1e-7)
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], [999, 999])
    np.testing.assert_array_equal(calls[0][1], [0, 0])
    with pytest.raises(ValueError, match="initial_noise"):
        model.sample_latents(mx.zeros((2, 3, 24)), initial_noise=mx.zeros((1, 3, 8)))


def test_raw_speaker_conditioning_and_input_padding(model, monkeypatch):
    observed = {}

    def encode(features, attention_mask=None):
        observed["frames"] = features.shape[1]
        return (
            mx.zeros((1, features.shape[1], 16)),
            mx.ones((1, features.shape[1], 4)),
            mx.full((1, features.shape[1], 4), 2),
        )

    def sample(conditioning, *args, **kwargs):
        return conditioning

    model.latent_norm_mean = mx.full((8,), 9)
    model.latent_norm_std = mx.full((8,), 3)
    monkeypatch.setattr(model, "encode", encode)
    monkeypatch.setattr(model, "sample_latents", sample)
    out = model.predict_latents(mx.zeros((1,)))
    assert observed["frames"] == 160
    np.testing.assert_array_equal(np.array(out[0, 0, :8]), [1] * 4 + [2] * 4)
    model.predict_latents(mx.zeros((51201,)))
    assert observed["frames"] == 161


@pytest.mark.parametrize("length,rate", [(1, 16000), (1761, 16000), (3601, 24000)])
def test_separate_shapes_seed_and_finite_audio(model, length, rate):
    audio = mx.zeros((length, 2))
    first = model.separate(audio, rate, seed=7)
    second = model.separate(audio, rate, seed=7)
    assert first.speakers.shape == (2, max(1, round(length * 24000 / rate)))
    assert first.sample_rate == 24000
    assert mx.all(mx.isfinite(first.speakers)).item()
    np.testing.assert_array_equal(np.array(first.speakers), np.array(second.speakers))


def test_seed_is_local_and_single_step_only(model):
    mx.random.seed(55)
    expected = mx.random.normal((3,))
    mx.eval(expected)
    mx.random.seed(55)
    model.separate(mx.zeros(200), 16000, seed=4)
    np.testing.assert_array_equal(np.array(mx.random.normal((3,))), np.array(expected))
    for steps in (0, 2, 30, 1.5):
        with pytest.raises(ValueError, match="exactly one step"):
            model.separate(mx.zeros(200), 16000, num_steps=steps)


def test_vocoder_stft_matches_torch_reflection():
    torch = pytest.importorskip("torch")
    audio = np.random.default_rng(4).standard_normal((2, 1440)).astype(np.float32)
    window = torch.hann_window(1024)
    ref = torch.stft(
        torch.from_numpy(audio),
        1024,
        hop_length=240,
        window=window,
        return_complex=True,
    )[..., :6]
    expected = torch.cat((ref.real, ref.imag[:, 1:-1]), dim=1).transpose(1, 2).numpy()
    actual = stft_features(mx.array(audio), mx.array(window.numpy()), 240, 6)
    np.testing.assert_allclose(np.array(actual), expected, atol=2e-5, rtol=2e-5)


def test_upsampler_layout_matches_torch():
    if os.environ.get("MLX_ENABLE_TF32") != "0":
        pytest.skip("Use MLX_ENABLE_TF32=0 for numerical parity")
    torch = pytest.importorskip("torch")
    config = small_config()
    model = Model(config)
    ref = torch.nn.ConvTranspose1d(4, 4, 4, stride=2, padding=1).eval()
    state = {
        "upsampler." + k: mx.array(v.detach().numpy())
        for k, v in ref.state_dict().items()
    }
    converted = sanitize_source_weights({}, {}, state, {})
    model.load_weights(list(converted.items()), strict=False)
    x = np.random.default_rng(2).standard_normal((2, 7, 4)).astype(np.float32)
    with torch.no_grad():
        expected = ref(torch.from_numpy(x).transpose(1, 2)).transpose(1, 2).numpy()
    np.testing.assert_allclose(
        np.array(model.decoder.upsampler(mx.array(x))), expected, atol=1e-6
    )


def test_invalid_latent_stats_rejected(model):
    weights = dict(tree_flatten(model.parameters()))
    weights.pop("latent_norm_mean")
    with pytest.raises(ValueError, match="latent_norm_mean"):
        model.sanitize(weights)
    weights["latent_norm_mean"] = mx.zeros((8,))
    weights["latent_norm_std"] = mx.zeros((8,))
    with pytest.raises(ValueError, match="positive"):
        model.sanitize(weights)


def test_cli_writes_two_speaker_tracks(model, tmp_path, monkeypatch):
    from mlx_audio import audio_io
    from mlx_audio.sts import generate

    monkeypatch.setattr(generate, "load_model", lambda *a, **kw: model)
    source = tmp_path / "input.wav"
    audio_io.write(str(source), np.zeros(1000, dtype=np.float32), 16000)
    generate.main(
        [
            "--model",
            "test",
            "--audio",
            str(source),
            "--output-path",
            str(tmp_path / "out.wav"),
            "--seed",
            "0",
        ]
    )
    for speaker in (1, 2):
        audio, rate = audio_io.read(str(tmp_path / f"out_speaker_{speaker}.wav"))
        assert rate == 24000 and audio.shape == (1500,)
