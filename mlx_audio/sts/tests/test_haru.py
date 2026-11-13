"""Haru integration tests with tiny weights, without network access."""

import json
from dataclasses import asdict

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn
from mlx.utils import tree_flatten, tree_map

from mlx_audio.sts.models.haru import DuplexConfig, Model, ModelConfig, render_card
from mlx_audio.sts.models.haru.codec import CodecStream, _mixer_arrays, depthwise_mixer
from mlx_audio.sts.models.haru.prompt import build_prompt, prompt_length, reference_ids
from mlx_audio.sts.models.haru.sampling import (
    DryConfig,
    DryState,
    StreamDetokenizer,
    sample_token,
)


def tiny_config():
    return ModelConfig.from_dict(
        {
            "model_type": "haru",
            "dtype": "float32",
            "frame_samples": 8,
            "latent_dim": 4,
            "pad_token_id": 0,
            "eop_token_id": 1,
            "lm": {
                "model_type": "qwen3_5_text",
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "vocab_size": 128,
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 4,
                "linear_key_head_dim": 32,
                "linear_value_head_dim": 16,
                "linear_conv_kernel_dim": 4,
                "full_attention_interval": 4,
                "tie_word_embeddings": True,
                "eos_token_id": 2,
                "rope_parameters": {
                    "rope_type": "default",
                    "rope_theta": 10000,
                    "partial_rotary_factor": 0.5,
                },
            },
            "codec": {
                "latent_dim": 4,
                "base_channels": 4,
                "ratios": [2, 2, 2],
                "encoder_depths": [1, 1, 1, 1],
                "decoder_depths": [1, 1, 1, 1],
            },
            "head": {"hidden_size": 64, "latent_dim": 4, "num_layers": 2},
        }
    )


def tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit

    result = Tokenizer(WordLevel({f"t{i}": i for i in range(128)}, unk_token="t3"))
    result.pre_tokenizer = WhitespaceSplit()
    return result


@pytest.fixture
def model():
    mx.random.seed(7)
    result = Model(tiny_config())

    # Codec convolutions start at zero; nonzero random weights exercise streaming.
    def randomize(value):
        scale = 0.1 if value.ndim < 2 else value.shape[-1] ** -0.5
        return mx.random.normal(value.shape) * scale

    result.codec.update(tree_map(randomize, result.codec.parameters()))
    result.latent_scale = mx.array(0.2)
    result.latent_bias = mx.array(0.05)
    result.tokenizer = tokenizer()
    result.eval()
    mx.eval(result.parameters())
    return result


def test_config_roundtrip_and_registration():
    from pathlib import Path

    from mlx_audio.convert import Domain, detect_model_domain, get_model_type
    from mlx_audio.registry import classify_model
    from mlx_audio.sts.utils import infer_model_type_from_config

    config = asdict(tiny_config())
    assert asdict(ModelConfig.from_dict(config)) == config
    assert infer_model_type_from_config(config) == "haru"
    assert classify_model("haru") == "sts"
    assert detect_model_domain(config, Path("arbitrary-name")) == Domain.STS
    assert get_model_type(config, Path("arbitrary-name"), Domain.STS) == "haru"


@pytest.mark.parametrize(
    "update",
    [
        {"frame_samples": 9},
        {"latent_dim": 5},
        {"head": {"hidden_size": 99}},
        {"model_type": "other"},
    ],
)
def test_invalid_config(update):
    with pytest.raises(ValueError):
        ModelConfig.from_dict({**asdict(tiny_config()), **update})


def test_cached_backbone_matches_full_sequence(model):
    with mx.stream(mx.cpu):
        inputs = mx.random.normal((1, 9, 64))
        expected = model(inputs)
        cache = model.new_cache()
        parts = [model(inputs[:, :4], cache)]
        parts.extend(model(inputs[:, i : i + 1], cache) for i in range(4, 9))
        actual = mx.concatenate(parts, axis=1)
        np.testing.assert_allclose(
            np.asarray(actual), np.asarray(expected), atol=1e-5, rtol=1e-4
        )
    assert cache[-1].offset == 9


def test_stream_embedding_masks_and_scaling(model):
    tokens = mx.array([[3, 4, 5]])
    latents = mx.random.normal((1, 3, 4))
    actual = model.embed(
        tokens, latents, latents, mx.array([[0, 1, 1]]), mx.array([[0, 0, 1]])
    )
    text = model.embed(tokens)
    np.testing.assert_array_equal(np.asarray(actual[:, 0]), np.asarray(text[:, 0]))
    assert not np.allclose(actual[:, 1], text[:, 1])
    np.testing.assert_allclose(
        model.unscale_latents(model.scale_latents(latents)), latents, atol=1e-6
    )


def test_sanitizer_source_layout_and_idempotence(model):
    conv = mx.arange(12).reshape(3, 1, 4).astype(mx.bfloat16)
    gain = mx.array([0.001, 0.002, -0.003], dtype=mx.bfloat16)
    down = mx.arange(24).reshape(3, 2, 4)
    up = mx.arange(24).reshape(2, 3, 4)
    source = {
        "lm.layers.0.linear_attn.conv1d.weight": conv,
        "lm.norm.weight": gain,
        "lm.layers.0.input_layernorm.weight": gain,
        "lm.layers.0.linear_attn.norm.weight": gain,
        "codec.encoder.down.0.weight": down,
        "codec.decoder.up.0.weight": up,
        "head.layers.0.ada.weight": mx.ones((192, 64)),
        "head.layers.1.ada.weight": mx.full((192, 64), 2.0),
        "head.final.ada.weight": mx.full((128, 64), 3.0),
        "latent_scale": mx.array([0.2], dtype=mx.bfloat16),
    }
    result = model.sanitize(source)
    assert result["lm.norm.weight"].dtype == mx.float32
    np.testing.assert_array_equal(result["lm.norm.weight"], gain.astype(mx.float32) + 1)
    assert mx.array_equal(result["lm.layers.0.linear_attn.norm.weight"], gain).item()
    np.testing.assert_array_equal(
        result["codec.encoder.down.0.weight"], down.transpose(0, 2, 1).reshape(3, 8)
    )
    np.testing.assert_array_equal(
        result["codec.decoder.up.0.weight"], up.transpose(1, 2, 0).reshape(12, 2)
    )
    assert result["head.ada.weight"].shape == (512, 64)
    assert result["latent_scale"].shape == ()
    for name, value in model.sanitize(result).items():
        assert mx.array_equal(value, result[name]).item()
    assert "head.final.ada.weight" in source


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
def test_recurrent_decay_precision_for_source_and_converted_weights(model, dtype):
    from mlx_audio.lm.models.gated_delta import compute_g

    name = "lm.layers.0.linear_attn.dt_bias"
    bias = mx.linspace(-3, 1, 128).astype(dtype)
    # No original Conv1d tensor: this also exercises already converted weights.
    restored_bias = model.sanitize({name: bias})[name]
    assert restored_bias.dtype == mx.float32
    assert not model.cast_predicate(name)
    a = mx.linspace(-8, 3, 128).astype(dtype)
    log_rate = mx.linspace(-2, 2, 128).astype(dtype)
    expected = np.exp(
        -np.exp(np.array(log_rate.astype(mx.float32)))
        * np.logaddexp(
            0, np.array(a.astype(mx.float32)) + np.array(bias.astype(mx.float32))
        )
    )
    actual = compute_g(log_rate, a, restored_bias)
    assert actual.dtype == mx.float32
    np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=2e-6)


def test_decay_precision_over_long_recurrence(model):
    from mlx_audio.lm.models.gated_delta import gated_delta_update

    steps = 512
    a = mx.full((1, steps, 1), -4.8125, dtype=mx.bfloat16)
    name = "lm.layers.0.linear_attn.dt_bias"
    bias = model.sanitize({name: mx.array([-1.171875], dtype=mx.bfloat16)})[name]
    log_rate = mx.array([0.34375], dtype=mx.bfloat16)
    # Disable writes, leaving pure exponential decay with a closed-form result.
    q = mx.ones((1, steps, 1, 32), dtype=mx.bfloat16)
    k = mx.zeros_like(q)
    v = mx.zeros((1, steps, 1, 4), dtype=mx.bfloat16)
    state = mx.ones((1, 1, 4, 32), dtype=mx.float32)
    _, final = gated_delta_update(
        q, k, v, a, mx.full(a.shape, -mx.inf, dtype=a.dtype), log_rate, bias, state
    )
    decay = np.exp(-np.exp(0.34375) * np.logaddexp(0, -4.8125 - 1.171875))
    # Allow accumulated FP32 rounding across 512 updates on different GPUs.
    # The BF16 decay regression produces about 3% error for these inputs.
    np.testing.assert_allclose(final, decay**steps, rtol=1e-4)


def test_codec_whole_and_arbitrary_stream_chunks_agree(model):
    # CPU isolates streaming state from Metal GEMM/GEMV rounding differences.
    with mx.stream(mx.cpu):
        audio = mx.random.normal((32,)) * 0.1
        expected = model.codec.encode(audio)
        assert model.codec.encode(audio[:25]).shape == (1, 4, 4)
        stream = CodecStream(model.codec)
        for chunk in np.array_split(np.asarray(audio), [1, 5, 13, 24, 31]):
            stream.feed(mx.array(chunk))
        assert stream.frames_ready == 4
        actual = mx.concatenate([stream.pop() for _ in range(4)], axis=1)
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)
        latents = mx.random.normal((1, 4, 4))
        parts = [stream.decode(latents[:, i : i + 1]) for i in range(4)]
        np.testing.assert_allclose(
            mx.concatenate(parts, axis=1),
            model.codec.decode(latents),
            atol=1e-5,
            rtol=1e-4,
        )
        with pytest.raises(RuntimeError, match="complete frame"):
            stream.pop()


@pytest.mark.parametrize("steps", [1, 3, 9])
def test_codec_metal_mixer_matches_array_reference(steps):
    inputs = [
        mx.random.normal(shape)
        for shape in [(steps, 8), (steps, 8), (6, 8), (8, 7), (8,), (8,)]
    ]
    actual = depthwise_mixer(*inputs)
    expected = _mixer_arrays(*inputs)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, atol=2e-6, rtol=2e-6)


def test_head_sampling(model, monkeypatch):
    hidden = mx.random.normal((2, 64))
    noise = mx.random.normal((2, 4))
    result = model.head.sample(hidden, noise)
    assert result.shape == (2, 4)
    np.testing.assert_array_equal(result, model.head.sample(hidden, noise, 1.5))
    assert not np.allclose(result, model.head.sample(hidden, noise, 1.0))
    with pytest.raises(ValueError):
        model.head.sample(hidden, noise, steps=0)

    # For u(z) = z, n Euler steps over [0, 1] multiply noise by (1 + 1/n)**n.
    monkeypatch.setattr(model.head, "velocity", lambda z, c: z)
    model.head.clear_samplers()
    for steps in (1, 2, 4):
        expected = np.asarray(noise) * (1 + 1 / steps) ** steps
        np.testing.assert_allclose(
            model.head.sample(hidden, noise, steps=steps), expected, rtol=1e-6
        )


@pytest.mark.parametrize(
    "reference,speak_first",
    [
        (None, False),
        (None, True),
        (np.ones(19, dtype=np.float32) * 0.1, False),
        (np.ones(19, dtype=np.float32) * 0.1, True),
    ],
)
def test_prompt_audio_channels(model, reference, speak_first):
    prompt = build_prompt(
        model.config,
        model.codec,
        model.tokenizer,
        "t5 t6",
        reference,
        speak_first,
        mx.random.key(1),
    )
    n_ref = 0 if reference is None else len(reference)
    assert prompt.length == prompt_length(
        model.tokenizer, "t5 t6", n_ref, speak_first, 8
    )
    assert prompt.model_latents.shape == (prompt.length, 4)
    assert int(mx.sum(prompt.user_mask).item()) == (2 if speak_first else 0)
    assert reference_ids(3, 0, 1) == [0, 0, 1]
    assert "Haru" in render_card({"name": "Haru"})


def test_session_restart_reproducibility_and_limits(model):
    session = model.create_session(
        "t6 t7",
        np.ones(16, dtype=np.float32) * 0.1,
        DuplexConfig(seed=9, max_frames=3, speak_first=True),
    )
    with pytest.raises(RuntimeError, match="start"):
        session.step(np.zeros(8))
    first = session.start()
    assert first.token not in {0, 1, 2}
    with pytest.raises(ValueError, match="8 samples"):
        session.step(np.zeros(7))
    session.feed(np.zeros(3, dtype=np.float32))
    assert session.frames_ready == 0
    session.feed(np.zeros(5, dtype=np.float32))
    assert session.frames_ready == 1
    second = session.step()
    last = session.step(np.zeros(8))
    assert [first.index, second.index, last.index] == [0, 1, 2]
    assert last.done
    assert all(
        f.audio.shape == (8,) and np.isfinite(f.audio).all()
        for f in [first, second, last]
    )
    with pytest.raises(RuntimeError, match="max_frames"):
        session.step(np.zeros(8))
    restarted = session.start()
    assert restarted.text == first.text and restarted.token == first.token
    np.testing.assert_array_equal(restarted.audio, first.audio)


def test_live_warmup_preserves_session_state_and_seed(model):
    session = model.create_session(config=DuplexConfig(seed=7, max_frames=2))
    session.warmup()
    assert session.index == 0 and session.frames_ready == 0
    assert session._latent is None and session._cache is None
    assert session._decode_stream is None
    first = session.start()
    second = session.step(np.zeros(8, dtype=np.float32))
    control = model.create_session(config=DuplexConfig(seed=7, max_frames=2))
    expected = [control.start(), control.step(np.zeros(8, dtype=np.float32))]
    for actual, reference in zip([first, second], expected):
        assert actual.token == reference.token and actual.text == reference.text
        np.testing.assert_array_equal(actual.audio, reference.audio)


def test_async_decode_preserves_feedback_audio_and_restart(model):
    config = DuplexConfig(seed=9, max_frames=5)
    user = np.arange(32, dtype=np.float32).reshape(4, 8) / 100
    sync = model.create_session(config=config)
    expected = [sync.start()]
    latents = [np.array(sync._latent)]
    for chunk in user:
        expected.append(sync.step(chunk))
        latents.append(np.array(sync._latent))

    session = model.create_session(config=config)
    pending = session.start_async()
    np.testing.assert_array_equal(session._latent, latents[0])
    actual = []
    for i, chunk in enumerate(user):
        # Encode and advance the next frame before reading the previous PCM.
        session.feed(chunk[:3])
        session.feed(chunk[3:])
        next_frame = session.step_async()
        np.testing.assert_array_equal(session._latent, latents[i + 1])
        actual.append(pending.resolve())
        pending = next_frame
    # Restart with the last PCM frame still pending. Its output remains valid.
    restarted = session.start_async()
    actual.append(pending.resolve())
    for got, want in zip(actual, expected):
        assert (got.index, got.token, got.text, got.done) == (
            want.index,
            want.token,
            want.text,
            want.done,
        )
        np.testing.assert_array_equal(got.audio, want.audio)
    np.testing.assert_array_equal(restarted.resolve().audio, expected[0].audio)
    session.synchronize_decoder()


@pytest.mark.parametrize("steps", [None, 1, 2, 4])
def test_session_meanflow_steps_match_checkpoint_default(model, steps):
    # A session override must match an equivalent checkpoint configuration.
    model.head.config.num_steps = 2 if steps is None else steps
    control = model.create_session(config=DuplexConfig(seed=7, max_frames=2))
    user = np.zeros(8, dtype=np.float32)
    expected = [control.start(), control.step(user)]

    model.head.config.num_steps = 2
    session = model.create_session(
        config=DuplexConfig(seed=7, max_frames=2, meanflow_steps=steps)
    )
    session.warmup()
    first = session.start_async()
    second = session.step_async(user)
    actual = [first.resolve(), second.resolve()]
    session.synchronize_decoder()
    assert model.head.config.num_steps == 2
    for got, want in zip(actual, expected):
        assert (got.token, got.text) == (want.token, want.text)
        np.testing.assert_array_equal(got.audio, want.audio)


@pytest.mark.parametrize("steps", [0, -1, 1.5, True])
def test_session_rejects_invalid_meanflow_steps(steps):
    with pytest.raises(ValueError, match="meanflow_steps"):
        DuplexConfig(meanflow_steps=steps)


def test_resolved_and_pending_steps_share_ordered_decoder_state(model):
    config = DuplexConfig(seed=7, max_frames=4)
    control = model.create_session(config=config)
    expected = [control.start()]
    expected.extend(control.step(np.zeros(8, dtype=np.float32)) for _ in range(3))
    session = model.create_session(config=config)
    actual = [session.start()]
    pending = session.step_async(np.zeros(8, dtype=np.float32))
    following = session.step(np.zeros(8, dtype=np.float32))
    actual.extend([pending.resolve(), following])
    actual.append(session.step_async(np.zeros(8, dtype=np.float32)).resolve())
    session.synchronize_decoder()
    for got, want in zip(actual, expected):
        assert got.token == want.token
        np.testing.assert_array_equal(got.audio, want.audio)


@pytest.mark.parametrize("dry", [None, DryConfig(multiplier=6, allowed_length=0)])
def test_pending_frames_preserve_text_and_repetition_history(model, monkeypatch, dry):
    # Force repeated text so delayed token reads must preserve DRY updates.
    logits = mx.full((1, 128), -100.0)
    logits[:, 3:6] = mx.array([1.0, 0.99, 0.95])
    monkeypatch.setattr(model, "logits", lambda hidden: logits + 0)
    monkeypatch.setattr(
        model, "logits_fp32", lambda hidden, ids: mx.full((1, 2), -100.0)
    )
    config = DuplexConfig(seed=9, max_frames=9, temperature=0, dry=dry)
    user = np.arange(64, dtype=np.float32).reshape(8, 8) / 100
    sync = model.create_session(config=config)
    expected = [sync.start()]
    expected.extend(sync.step(chunk) for chunk in user)
    if dry is not None:
        assert len({frame.token for frame in expected}) > 1

    session = model.create_session(config=config)
    pending = [session.start_async()]
    # No intermediate token, latent, or PCM conversions to mask dependencies.
    pending.extend(session.step_async(chunk) for chunk in user)
    # Repeated or late reads must not consume detokenizer state again.
    last_text = pending[-1].text
    actual = [frame.resolve() for frame in pending]
    assert pending[-1].resolve().text == last_text
    for got, want in zip(actual, expected):
        assert (got.token, got.text, got.done) == (want.token, want.text, want.done)
        np.testing.assert_array_equal(got.audio, want.audio)
    session.synchronize_decoder()


def test_async_step_does_not_synchronously_evaluate_current_frame(model, monkeypatch):
    session = model.create_session(config=DuplexConfig(seed=7))
    session.start_async().resolve()

    def unexpected_eval(*args):
        raise AssertionError("async step waited for evaluation")

    with monkeypatch.context() as patch:
        patch.setattr(mx, "eval", unexpected_eval)
        pending = session.step_async(np.zeros(8, dtype=np.float32))
    # The current token is still deferred; resolving makes its text available.
    assert "token" not in pending.__dict__ and "text" not in pending.__dict__
    frame = pending.resolve()
    assert isinstance(frame.token, int) and np.isfinite(frame.audio).all()
    session.synchronize_decoder()


def test_file_generate_writes_audio_and_caps_frames(model, tmp_path):
    from mlx_audio import audio_io
    from mlx_audio.sts.generate import generate_audio

    result = generate_audio(
        model=model,
        audio=np.zeros(17, dtype=np.float32),
        tail_seconds=0,
        seed=0,
        output_path=tmp_path / "out.wav",
    )
    assert result.audio.shape == (32,)  # initial frame plus ceil(17 / 8)
    waveform, rate = audio_io.read(str(tmp_path / "out.wav"))
    assert rate == 24000 and len(waveform) == 32
    assert model.generate(
        np.zeros(17), tail_seconds=0, max_new_tokens=2
    ).audio.shape == (16,)
    model.head.config.num_steps = 4
    expected = model.generate(np.zeros(8), tail_seconds=0, seed=7)
    model.head.config.num_steps = 1
    actual = model.generate(np.zeros(8), tail_seconds=0, seed=7, meanflow_steps=4)
    np.testing.assert_array_equal(actual.audio, expected.audio)
    with pytest.raises(ValueError):
        model.generate(np.zeros((2, 17)))


def test_generic_loader_and_quantized_roundtrip(model, tmp_path):
    from mlx_audio.convert import convert
    from mlx_audio.sts.utils import load_model

    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(asdict(model.config)))
    model.tokenizer.save(str(source / "tokenizer.json"))
    mx.save_safetensors(
        str(source / "model.safetensors"), dict(tree_flatten(model.parameters()))
    )
    loaded = load_model(source, strict=True)
    x = mx.random.normal((1, 3, 64))
    np.testing.assert_allclose(loaded(x), model(x), atol=1e-5)
    converted = tmp_path / "quantized"
    convert(
        str(source),
        str(converted),
        quantize=True,
        q_bits=8,
        q_group_size=64,
        dtype="bfloat16",
    )
    quantized = load_model(converted, strict=True)
    assert isinstance(quantized.lm.embed_tokens, nn.QuantizedEmbedding)
    assert isinstance(quantized.lm.layers[0].linear_attn.in_proj_a, nn.QuantizedLinear)
    assert isinstance(quantized.lm.layers[0].linear_attn.in_proj_b, nn.QuantizedLinear)
    assert not isinstance(quantized.head.cond_proj, nn.QuantizedLinear)
    assert quantized.lm.norm.weight.dtype == mx.float32
    assert quantized.lm.layers[0].linear_attn.dt_bias.dtype == mx.float32
    assert "apache-2.0" in (converted / "README.md").read_text()
    actual = quantized(x.astype(mx.bfloat16)).astype(mx.float32)
    np.testing.assert_allclose(actual, model(x), atol=0.08, rtol=0.08)


def test_sampling_and_dry_exempt_stream_tokens():
    logits = mx.array([[1.0, 9.0, 2.0]])
    assert sample_token(logits, temperature=0).item() == 1
    assert sample_token(logits, top_k=1, key=mx.random.key(1)).item() == 1
    state = DryState(DryConfig(), {0, 1})
    for token in [3, 4, 0, 1, 3]:
        state.update(token)
    assert state.penalties[4] > 0
    assert state.history == [3, 4, 3]
    detok = StreamDetokenizer(tokenizer(), {0, 1})
    assert [detok.push(t) for t in [3, 0, 1, 0, 4]] == ["t3", " ", "", "", "t4"]
