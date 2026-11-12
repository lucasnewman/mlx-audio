import json
from dataclasses import asdict
from unittest.mock import patch

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn
from mlx.utils import tree_flatten

from mlx_audio.convert import Domain, detect_model_domain, get_model_type
from mlx_audio.registry import classify_model
from mlx_audio.stt.models.qwen2_audio.config import EncoderConfig
from mlx_audio.stt.models.sovereignai_transcribe import Model, ModelConfig
from mlx_audio.stt.models.sovereignai_transcribe.config import TextConfig
from mlx_audio.stt.models.sovereignai_transcribe.sovereignai_transcribe import (
    Float32Router,
    RoutedExperts,
    TranscriptionOutput,
    copy_supporting_files,
    write_model_card,
)
from mlx_audio.utils import get_model_class


def tiny_config():
    return ModelConfig(
        audio_config=EncoderConfig(
            d_model=16,
            encoder_layers=1,
            encoder_attention_heads=2,
            encoder_ffn_dim=32,
            num_mel_bins=4,
            max_source_positions=8,
        ),
        text_config=TextConfig(
            hidden_size=16,
            num_hidden_layers=2,
            intermediate_size=24,
            num_attention_heads=2,
            head_dim=8,
            global_head_dim=16,
            vocab_size=64,
            num_key_value_heads=1,
            num_global_key_value_heads=1,
            num_kv_shared_layers=0,
            hidden_size_per_layer_input=0,
            attention_k_eq_v=True,
            enable_moe_block=True,
            num_experts=4,
            top_k_experts=2,
            moe_intermediate_size=8,
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=8,
        ),
        audio_token_id=10,
    )


def test_config_and_registration(tmp_path):
    config = tiny_config()
    assert ModelConfig.from_dict({**asdict(config), "irrelevant": True}) == config
    assert classify_model("sovereignai_transcribe") == "stt"
    assert classify_model("sov_transcribe") is None
    assert detect_model_domain(asdict(config), tmp_path) == Domain.STT
    module, family = get_model_class("sovereignai_transcribe", [], "stt", {})
    assert family == "sovereignai_transcribe"
    assert module.Model is Model
    checkpoint = tmp_path / "SOV-Transcribe-1-4bit"
    assert detect_model_domain({}, checkpoint) == Domain.STT
    assert get_model_type({}, checkpoint, Domain.STT) == "sovereignai_transcribe"


@pytest.mark.parametrize(
    "settings",
    [
        {"model_type": "qwen2"},
        {"num_kv_shared_layers": 1},
        {"hidden_size_per_layer_input": 8},
        {"layer_types": ["full_attention"]},
        {"layer_types": ["unsupported"] * 30},
        {"top_k_experts": 129},
    ],
)
def test_invalid_decoder_config(settings):
    with pytest.raises(ValueError):
        TextConfig(**settings)


def test_sanitize_original_and_converted_weights():
    model = Model(tiny_config())
    original = {
        "model.audio_tower.conv1.weight": mx.zeros((16, 4, 3)),
        "model.audio_tower.conv2.weight": mx.zeros((16, 16, 3)),
        "model.language_model.layers.0.experts.gate_up_proj": mx.arange(
            4 * 16 * 16
        ).reshape(4, 16, 16),
        "model.language_model.layers.0.experts.down_proj": mx.zeros((4, 16, 8)),
        "model.language_model.layers.0.router.proj.weight": mx.zeros(
            (4, 16), dtype=mx.bfloat16
        ),
        "lm_head.weight": mx.zeros((64, 16)),
    }
    weights = model.sanitize(original)
    assert "lm_head.weight" not in weights
    assert weights["model.audio_tower.conv1.weight"].shape == (16, 3, 4)
    assert weights["model.audio_tower.conv2.weight"].shape == (16, 3, 16)
    prefix = "model.language_model.layers.0.experts.switch_glu."
    np.testing.assert_array_equal(
        weights[prefix + "gate_proj.weight"],
        original["model.language_model.layers.0.experts.gate_up_proj"][:, :8],
    )
    np.testing.assert_array_equal(
        weights[prefix + "up_proj.weight"],
        original["model.language_model.layers.0.experts.gate_up_proj"][:, 8:],
    )
    assert (
        weights["model.language_model.layers.0.router.proj.weight"].dtype == mx.float32
    )
    twice = model.sanitize(weights)
    assert weights.keys() == twice.keys()
    for key in weights:
        np.testing.assert_array_equal(weights[key], twice[key])


@pytest.mark.parametrize("chunk_sizes", [(5, 1, 1, 1, 1, 1, 1), (5, 5, 1)])
def test_cached_decoder_matches_full_forward_across_sliding_window(chunk_sizes):
    # Metal's FP32 GEMM and GEMV kernels use different internal precision.
    # Compare scheduling on CPU to isolate masks, offsets and cache rotation.
    with mx.stream(mx.cpu):
        _check_cached_decoder(chunk_sizes)


def _check_cached_decoder(chunk_sizes):
    mx.random.seed(7)
    model = Model(tiny_config())
    model.eval()
    ids = mx.array([[2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13]])
    expected = model(ids, logits_to_keep=0)
    cache = model.make_cache()
    chunks = []
    offset = 0
    for size in chunk_sizes:
        chunks.append(
            model(ids[:, offset : offset + size], cache=cache, logits_to_keep=0)
        )
        offset += size
    actual = mx.concatenate(chunks, axis=1)
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
    assert [c.offset for c in cache] == [11, 11]


def test_audio_encoder_and_projector():
    model = Model(tiny_config())
    output = model.get_audio_features(mx.zeros((2, 4, 16)))
    assert output.shape == (4, 16)
    assert mx.all(mx.isfinite(output)).item()
    with pytest.raises(ValueError, match="mel frames"):
        model.model.audio_tower(mx.zeros((1, 4, 15)))


def test_audio_batches_preserve_window_order():
    with mx.stream(mx.cpu):
        model = Model(tiny_config())
        model.eval()
        features = mx.random.normal((3, 4, 16))
        expected = model.get_audio_features(features)
        actual = model.get_audio_features(features, batch_size=2)
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)
    with pytest.raises(ValueError, match="audio_batch_size"):
        model.get_audio_features(features, batch_size=0)


def test_fp32_router_preserves_bf16_expert_output():
    config = tiny_config().text_config
    router = Float32Router(config)
    experts = RoutedExperts(config)
    experts.set_dtype(mx.bfloat16)
    x = mx.random.normal((1, 3, 16)).astype(mx.bfloat16)
    indices, weights = router(x)
    assert weights.dtype == mx.float32
    assert experts(x, indices, weights).dtype == mx.bfloat16


class TinyTokenizer:
    def convert_tokens_to_ids(self, token):
        return 10

    def apply_chat_template(self, messages, **kwargs):
        self.prompt = messages[0]["content"][0]["text"]
        return "<bos>prompt<|audio|><model>"

    def encode(self, text, **kwargs):
        assert "<|audio><|audio|><|audio|><audio|>" in text
        return [2, 3, 10, 10, 4]

    def decode(self, tokens, **kwargs):
        return "hello" if tokens[0] == 7 else ""


def test_projected_audio_replaces_only_audio_embeddings():
    model = Model(tiny_config())
    model._tokenizer = TinyTokenizer()
    features = mx.full((2, 16), 0.125)
    with patch.object(model, "get_audio_features", return_value=features):
        embeddings, count = model._prepare_inputs(mx.zeros((100,)), "Transcribe.")
    assert count == 5
    np.testing.assert_array_equal(embeddings[:, 2:4], features[None])
    np.testing.assert_array_equal(
        embeddings[:, :2], model._embed_text(mx.array([[2, 3]]))
    )


@pytest.mark.parametrize(
    ("limit", "reason", "count", "eos"),
    [(1, "length", 1, 106)] + [(3, "stop", 2, eos) for eos in (1, 106, 50)],
)
@pytest.mark.parametrize("stream", [False, True])
def test_generate_stopping_and_prompt_options(
    limit, reason, count, eos, stream, tmp_path
):
    from mlx_audio.stt.generate import generate_transcription

    model = Model(tiny_config())
    model._tokenizer = TinyTokenizer()
    next_tokens = iter([7, eos])

    def forward(self, *args, **kwargs):
        token = next(next_tokens)
        return mx.full((1, 1, 128), -100.0).at[:, :, token].add(200.0)

    with (
        patch.object(model, "get_audio_features", return_value=mx.ones((2, 16))),
        patch.object(model, "make_cache", return_value=[]),
        patch.object(Model, "__call__", forward),
    ):
        result = generate_transcription(
            model=model,
            audio=mx.zeros((100,)),
            output_path=str(tmp_path / "transcript"),
            max_tokens=limit,
            language="French",
            hotwords=["MLX"],
            stream=stream,
            format="json",
        )
    assert result.text == "hello"
    if not stream:
        assert result.finish_reason == reason
    assert result.generation_tokens == count
    assert result.total_tokens == result.prompt_tokens + count
    assert "French" in model._tokenizer.prompt
    assert "MLX" in model._tokenizer.prompt
    saved = json.loads((tmp_path / "transcript.json").read_text())
    assert saved["text"] == "hello"
    assert saved["segments"][0]["text"] == "hello"


@pytest.mark.parametrize("audio", [[], [[0.0]], [float("nan")], [float("inf")]])
def test_invalid_audio_is_rejected(audio):
    model = Model(tiny_config())
    model._tokenizer = TinyTokenizer()
    with pytest.raises(ValueError, match="mono waveform|non-finite"):
        model.generate(audio)


def test_stream_deltas_and_final_statistics():
    model = Model(tiny_config())
    model._tokenizer = TinyTokenizer()
    tokens = iter([7, 106])

    def forward(self, *args, **kwargs):
        token = next(tokens)
        return mx.full((1, 1, 128), -100.0).at[:, :, token].add(200.0)

    with (
        patch.object(model, "get_audio_features", return_value=mx.ones((2, 16))),
        patch.object(model, "make_cache", return_value=[]),
        patch.object(Model, "__call__", forward),
    ):
        results = list(model.generate(mx.zeros((100,)), stream=True, max_tokens=3))
    assert "".join(r.text for r in results) == "hello"
    assert results[0].is_final is False
    assert results[-1].is_final is True
    assert results[-1].finish_reason == "stop"
    assert results[-1].generation_tokens == 2
    assert results[-1].total_tokens == 7


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("output_format", ["srt", "vtt"])
def test_cli_rejects_untimed_subtitles(stream, output_format, tmp_path):
    from mlx_audio.stt.generate import generate_transcription

    model = Model(tiny_config())
    model._tokenizer = TinyTokenizer()
    with (
        patch.object(
            model,
            "_generate_stream",
            return_value=iter(
                [
                    TranscriptionOutput(
                        text="hello", generation_tokens=1, finish_reason="stop"
                    )
                ]
            ),
        ),
        pytest.raises(ValueError, match="untimed|timestamps"),
    ):
        generate_transcription(
            model=model,
            audio=mx.zeros((100,)),
            stream=stream,
            output_path=str(tmp_path / "transcript"),
            format=output_format,
        )


def test_context_limits_and_invalid_batch_sizes():
    model = Model(tiny_config())
    model._tokenizer = TinyTokenizer()
    with pytest.raises(ValueError, match="batch sizes"):
        model.generate(mx.zeros((100,)), audio_batch_size=0)
    model.args.max_position_embeddings = 5
    with (
        patch.object(model, "get_audio_features") as encode,
        pytest.raises(ValueError, match="context window"),
    ):
        model._prepare_inputs(mx.zeros((100,)), "Transcribe.")
    encode.assert_not_called()
    model.args.max_position_embeddings = 6
    with (
        patch.object(model, "get_audio_features", return_value=mx.ones((2, 16))),
        pytest.raises(ValueError, match="generation exceeds"),
    ):
        model.generate(mx.zeros((100,)), max_tokens=2)


def test_end_tokens_include_processor_and_generation_config(tmp_path):
    model = Model(tiny_config())
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": 1}))
    (tmp_path / "processor_config.json").write_text(
        json.dumps({"generation_eos_token_ids": [77]})
    )
    with patch(
        "transformers.PreTrainedTokenizerFast.from_pretrained",
        return_value=TinyTokenizer(),
    ):
        Model.post_load_hook(model, tmp_path)
    assert model._eos_token_ids == {1, 50, 77, 106}


def test_quantization_and_reload(tmp_path):
    from mlx_audio.convert import convert
    from mlx_audio.stt import load

    config = tiny_config()
    config.text_config.hidden_size = 64
    config.text_config.intermediate_size = 64
    config.text_config.moe_intermediate_size = 64
    config.text_config.head_dim = 64
    config.text_config.global_head_dim = 64
    model = Model(config)
    source, dest = tmp_path / "source", tmp_path / "mlx"
    source.mkdir()
    mx.save_safetensors(
        str(source / "model.safetensors"), dict(tree_flatten(model.parameters()))
    )
    (source / "config.json").write_text(json.dumps(asdict(config)))
    convert(str(source), str(dest), quantize=True, q_bits=4, q_group_size=64)
    nn.quantize(
        model, bits=4, group_size=64, class_predicate=model.model_quant_predicate
    )
    weights = dict(tree_flatten(model.parameters()))
    assert not any("router" in k and k.endswith("scales") for k in weights)
    assert any("switch_glu.gate_proj.scales" in k for k in weights)
    with patch.object(Model, "post_load_hook", side_effect=lambda model, _: model):
        loaded = load(dest, strict=True)
    ids = mx.array([[2, 3, 4]])
    np.testing.assert_allclose(loaded(ids), model(ids), atol=1e-6)

    # PR #1186 exports the same layers without the outer model/switch_glu names.
    alternate = tmp_path / "alternate"
    alternate.mkdir()
    alternate_weights = {
        key.removeprefix("model.").replace(".experts.switch_glu.", ".experts."): value
        for key, value in weights.items()
    }
    mx.save_safetensors(str(alternate / "model.safetensors"), alternate_weights)
    alternate_config = json.loads((dest / "config.json").read_text())
    (alternate / "config.json").write_text(json.dumps(alternate_config))
    with patch.object(Model, "post_load_hook", side_effect=lambda model, _: model):
        restored = load(alternate, strict=True)
    np.testing.assert_array_equal(restored(ids), loaded(ids))


def test_conversion_sidecars_and_card(tmp_path):
    source, dest = tmp_path / "source", tmp_path / "dest"
    source.mkdir()
    dest.mkdir()
    for name in (
        "tokenizer.json",
        "chat_template.jinja",
        "generation_config.json",
        "modeling_sovereignai_transcribe.py",
        "LICENSE",
    ):
        (source / name).write_text("test")
    copy_supporting_files(source, dest)
    assert sorted(p.name for p in dest.iterdir()) == [
        "chat_template.jinja",
        "generation_config.json",
        "tokenizer.json",
    ]
    write_model_card(
        dest, "salesforce-sovereign-intelligence/SOV-Transcribe-1", "local-model", {}
    )
    assert 'load("local-model")' in (dest / "README.md").read_text()
