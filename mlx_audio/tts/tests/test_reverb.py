import json

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_audio.registry import model_type_from_config
from mlx_audio.tts.models.reverb import Model, ModelConfig
from mlx_audio.tts.models.reverb.convert import convert
from mlx_audio.tts.models.reverb.reverb import _CachedGeneration, prefix_attention_mask


def tiny_config():
    return ModelConfig(
        audio_vocab_size=9,
        audio_mask_id=8,
        num_audio_codebook=2,
        llm_config={
            "hidden_size": 16,
            "vocab_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "intermediate_size": 24,
        },
        echo={"architecture": {"embedding_rank": 8}},
    )


def inputs():
    return (
        mx.array([[[30, 30], [31, 31], [2, 3], [4, 5], [8, 8], [8, 8]]]),
        mx.array([[False, False, True, True, True, True]]),
    )


def test_config_detection_does_not_change_omnivoice():
    assert model_type_from_config({"model_type": "omnivoice"}) == "omnivoice"
    assert model_type_from_config({"model_type": "omnivoice", "echo": {}}) == "reverb"
    assert ModelConfig.from_dict({"model_type": "omnivoice"}).model_type == "reverb"


def test_prefix_mask():
    np.testing.assert_array_equal(
        np.array(prefix_attention_mask(4, 2)),
        [[True, True, False, False]] * 2 + [[True] * 4] * 2,
    )


def test_cached_target_matches_full_prefix_blocked_forward():
    model = Model(tiny_config())
    ids, mask = inputs()
    cached = _CachedGeneration(model, ids[:, :4], mask[:, :4])
    expected = model(ids, mask, prefix_length=4)[:, 4:]
    actual = cached(ids[:, 4:], mask[:, 4:])
    np.testing.assert_allclose(
        np.array(actual), np.array(expected), atol=2e-5, rtol=2e-5
    )
    # Changing a target must neither modify the prefix nor grow its KV cache.
    changed = ids.at[:, 4:, :].add(-1)
    expected = model(changed, mask, prefix_length=4)[:, 4:]
    actual = cached(changed[:, 4:], mask[:, 4:])
    np.testing.assert_allclose(
        np.array(actual), np.array(expected), atol=2e-5, rtol=2e-5
    )
    assert all(k.shape[2] == 4 for k, _ in cached.cache)
    prefix_before = model(ids, mask, prefix_length=4)[:, :4]
    prefix_after = model(changed, mask, prefix_length=4)[:, :4]
    np.testing.assert_allclose(
        np.array(prefix_before), np.array(prefix_after), atol=1e-6
    )


def test_factorized_embedding_and_audio_lookups():
    model = Model(tiny_config())
    ids, mask = inputs()
    expected_text = model.backbone.embed_tokens.project(
        model.backbone.embed_tokens.table.weight[ids[0, :2, 0]]
    )
    expected_audio = sum(
        embed.weight[ids[0, 2:, c]] for c, embed in enumerate(model.audio_embeddings)
    )
    expected = mx.concatenate([expected_text, expected_audio])[None]
    np.testing.assert_allclose(
        np.array(model._prepare_embed_inputs(ids, mask)), np.array(expected), atol=1e-6
    )


def upstream_weights(model):
    weights = dict(tree_flatten(model.parameters()))
    result = {
        key.replace("backbone.", "llm."): value
        for key, value in weights.items()
        if key.startswith("backbone.")
    }
    for name in ("audio_embeddings", "audio_heads"):
        result[f"{name}.weight"] = mx.concatenate(
            [
                weights[f"{name}.{i}.weight"]
                for i in range(model.config.num_audio_codebook)
            ]
        )
    result["codebook_layer_offsets"] = (
        mx.arange(model.config.num_audio_codebook) * model.config.audio_vocab_size
    )
    return result


def test_strict_upstream_weight_mapping_and_native_roundtrip():
    model = Model(tiny_config())
    restored = Model(tiny_config())
    sanitized = restored.sanitize(upstream_weights(model))
    restored.load_weights(list(sanitized.items()), strict=True)
    assert restored.sanitize(sanitized).keys() == sanitized.keys()
    ids, mask = inputs()
    np.testing.assert_allclose(
        np.array(restored(ids, mask)), np.array(model(ids, mask))
    )


def test_voice_codebook_major_layout():
    model = Model(tiny_config())
    model._voices = {
        "demo": {"codes": [0, 1, 2, 3, 4, 5], "frames": 3, "transcript": "Hi"}
    }
    tokens, text = model._voice_prompt("demo")
    np.testing.assert_array_equal(np.array(tokens), [[0, 3], [1, 4], [2, 5]])
    assert text == "Hi"
    with pytest.raises(ValueError, match="Unknown voice"):
        model._voice_prompt("missing")


class DummyTextTokenizer:
    def __call__(self, text, return_tensors=None, **kwargs):
        from types import SimpleNamespace

        return SimpleNamespace(input_ids=[[1, 2]] if return_tensors else [1, 2])


class DummyCodec:
    def decode(self, tokens):
        assert not bool(mx.any(tokens == 8))
        return mx.ones(tokens.shape[0] * 960)


def generation_model():
    model = Model(tiny_config())
    model.text_tokenizer = DummyTextTokenizer()
    model.audio_tokenizer = DummyCodec()
    model._voices = {
        "standard_male": {"codes": [1, 2, 3, 4], "frames": 2, "transcript": "Hello"}
    }
    return model


def test_generation_returns_real_sample_count_and_supports_batch():
    model = generation_model()
    results = model.generate_batch(
        ["Hello", "World"], duration_s=0.08, num_steps=2, position_temperature=0
    )
    assert len(results) == 2
    for i, result in enumerate(results):
        assert result.samples == result.audio.size == 1920
        assert result.token_count == 2
        assert result.sample_rate == 24000
        assert result.segment_idx == i
        assert result.processing_time_seconds > 0


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"text": ""}, "non-empty"),
        ({"speed": 0}, "speed"),
        ({"duration_s": -1}, "duration_s"),
        ({"num_steps": 0}, "num_steps"),
        ({"t_shift": 0}, "t_shift"),
        ({"stream": True}, "streaming"),
        ({"guidance_scale": 2}, "distilled"),
        ({"ref_tokens": mx.array([[1, 8]]), "ref_text": "Hello"}, "vocabulary"),
        ({"ref_tokens": mx.array([[1.0, 2.0]]), "ref_text": "Hello"}, "integer"),
        ({"ref_audio": mx.ones(100)}, "ref_text"),
    ],
)
def test_invalid_generation_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        next(generation_model().generate(**{"text": "Hello", **kwargs}))


@pytest.mark.parametrize("bits", [None, 4, 8])
def test_converter_flattens_bundle_and_retains_codec_and_license(tmp_path, bits):
    from dataclasses import asdict

    source, codec, output = (tmp_path / name for name in ("source", "codec", "output"))
    weight_dir = source / "model/safetensors/fp32"
    weight_dir.mkdir(parents=True)
    config = tiny_config()
    config.llm_config.update(hidden_size=64, intermediate_size=64, head_dim=16)
    config.echo["architecture"]["embedding_rank"] = 64
    (weight_dir / "config.json").write_text(json.dumps(asdict(config)))
    mx.save_safetensors(
        str(weight_dir / "model.safetensors"), upstream_weights(Model(config))
    )
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        (source / "model" / name).write_text("{}")
    (source / "voices").mkdir()
    (source / "voices/test.json").write_text("{}")
    (source / "README.md").write_text("Upstream attribution")
    codec.mkdir()
    (codec / "config.json").write_text("{}")
    mx.save_safetensors(str(codec / "model.safetensors"), {"codec": mx.ones(2)})
    assert convert(source, output, codec=codec, dtype="float32", bits=bits) == output
    saved_config = json.loads((output / "config.json").read_text())
    assert saved_config["model_type"] == "reverb"
    restored = Model(ModelConfig.from_dict(saved_config))
    from mlx_audio.utils import apply_quantization

    weights = mx.load(str(output / "model.safetensors"))
    apply_quantization(restored, saved_config, weights)
    restored.load_weights(list(weights.items()), strict=True)
    assert (output / "audio_tokenizer/model.safetensors").read_bytes() == (
        codec / "model.safetensors"
    ).read_bytes()
    assert (output / "voices/test.json").exists()
    assert "cc-by-nc-4.0" in (output / "README.md").read_text()
    assert (output / "README.upstream.md").read_text() == "Upstream attribution"
    with pytest.raises(ValueError, match="not empty"):
        convert(source, output, codec=codec)


def test_nested_source_cache_is_reused_and_invalidated(tmp_path, monkeypatch):
    import importlib

    module = importlib.import_module("mlx_audio.tts.models.reverb.convert")
    source = tmp_path / "source"
    checkpoint = source / "model/safetensors/fp32"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path / "hub"))
    calls = []

    def fake_convert(model, output):
        calls.append(model)
        output.mkdir()
        (output / "complete").touch()
        return output

    monkeypatch.setattr(module, "convert", fake_convert)
    first = module.prepare_model_path(source)
    assert (first / "complete").is_file()
    assert module.prepare_model_path(source) == first
    assert len(calls) == 1
    (checkpoint / "config.json").write_text('{"changed": true}')
    assert module.prepare_model_path(source) != first
    assert len(calls) == 2


def test_upstream_loader_routes_before_generic_download(tmp_path, monkeypatch):
    import mlx_audio.tts.utils as tts_utils
    import mlx_audio.utils as utils

    prepared = tmp_path / "native"
    calls = []

    def prepare(path, **kwargs):
        calls.append((path, kwargs))
        return prepared

    monkeypatch.setattr(
        "mlx_audio.tts.models.reverb.convert.prepare_model_path", prepare
    )
    monkeypatch.setattr(tts_utils, "base_load_model", lambda **kwargs: kwargs)
    result = utils.load_model("software-mansion/Reverb-TTS")
    assert result["model_path"] == prepared
    result = tts_utils.load(
        "software-mansion/Reverb-TTS", revision="abc", force_download=True
    )
    assert calls[-1][1] == {"revision": "abc", "force_download": True}
    assert result["strict"] is True
