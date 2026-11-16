"""Weight-free tests for the Stable Audio 3 pipeline."""

import json
import math
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_audio.music.models.stable_audio_3 import Model, ModelConfig, convert, t5gemma
from mlx_audio.music.models.stable_audio_3.dit import (
    HEAD_DIM,
    DiT,
    DiTConfig,
    GatedProjection,
    expo_fourier_features,
    rope,
    to_heads,
)
from mlx_audio.music.models.stable_audio_3.same import (
    DifferentialAttention,
    SAMEConfig,
    SAMEDecoder,
)
from mlx_audio.music.models.stable_audio_3.stable_audio_3 import (
    decoder_config,
    guided_velocity,
    sample_pingpong,
    sampling_schedule,
)

SAMPLE_RATE = 44_100
MEDIUM = {
    "diffusion": {
        "config": {
            "embed_dim": 1536,
            "depth": 24,
            "ff_kwargs": {"mult": 4.0},
            "attn_kwargs": {"differential": True},
        }
    },
    "pretransform": {
        "config": {
            "decoder": {
                "config": {
                    "channels": 256,
                    "c_mults": [6],
                    "transformer_depths": [12],
                    "sinusoidal_blocks": [8],
                    "sliding_window": [1, 1],
                }
            }
        }
    },
}


def values(array: mx.array) -> np.ndarray:
    return np.asarray(array.astype(mx.float32))


@pytest.fixture
def stubbed_model() -> Model:
    model = Model(ModelConfig())
    model._encode_text = lambda text: mx.ones((1, 256, 768)) * len(text)
    model.dit = lambda x, t, cross, global_cond: mx.zeros_like(x)
    model.decoder = SimpleNamespace(
        decode=lambda latents: mx.broadcast_to(
            latents[:, :1, :2], (1, latents.shape[1] * 4096, 2)
        )
    )
    return model


def test_fourier_features_use_two_pi_and_cos_before_sin() -> None:
    features = values(expo_fourier_features(mx.array([0.25])))[0]

    assert features.shape == (256,)
    assert features[0] == pytest.approx(math.cos(2 * math.pi * 0.5 * 0.25), abs=1e-5)
    assert features[128] == pytest.approx(math.sin(2 * math.pi * 0.5 * 0.25), abs=1e-5)
    assert values(expo_fourier_features(mx.array([0.0])))[0][:128] == pytest.approx(1.0)


def test_sampling_schedule_is_logsnr_with_fixed_endpoints() -> None:
    schedule = values(sampling_schedule(2))

    assert schedule[0] == 1.0
    assert schedule[-1] == 0.0
    assert schedule[1] == pytest.approx(1.0 / (1.0 + math.exp(-2.1)), abs=1e-5)


def test_pingpong_adds_noise_between_steps_only() -> None:
    sigmas = mx.array([1.0, 0.5, 0.0])
    seen = []

    def velocity(x: mx.array, t: mx.array) -> mx.array:
        seen.append(float(t[0]))
        return mx.zeros_like(x)

    start = mx.ones((1, 3, 2))
    key = mx.random.key(3)
    out = sample_pingpong(velocity, start, sigmas, key)
    _, noise_key = mx.random.split(key)
    noise = mx.random.normal(start.shape, dtype=start.dtype, key=noise_key)

    assert seen == [1.0, 0.5]
    np.testing.assert_allclose(values(out), values(0.5 * start + 0.5 * noise))


def test_guidance_keeps_only_the_orthogonal_difference() -> None:
    x = mx.array([[[1.0], [1.0]]])
    sigma = mx.array([[[0.5]]])
    cond_v = mx.array([[[0.0], [0.0]]])
    parallel_v = mx.array([[[2.0], [2.0]]])
    orthogonal_v = mx.array([[[1.0], [-1.0]]])

    np.testing.assert_allclose(
        values(guided_velocity(x, sigma, cond_v, parallel_v, 4.0)),
        values(cond_v),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        values(guided_velocity(x, sigma, cond_v, orthogonal_v, 2.0)),
        values(-orthogonal_v),
        atol=1e-5,
    )


def test_generate_uses_natural_length_and_trims(stubbed_model) -> None:
    result = next(stubbed_model.generate("rain", duration=30.0, steps=1))

    assert result.token_count == 323
    assert result.samples == 1323000
    assert result.audio.shape == (1323000, 2)
    assert result.sample_rate == SAMPLE_RATE


def test_seconds_conditioning_rounds_up_to_whole_seconds() -> None:
    model = Model(ModelConfig())

    def embed(seconds: float) -> np.ndarray:
        return values(model._seconds_embedding(seconds))

    np.testing.assert_array_equal(embed(0.4), embed(1.0))
    assert not np.array_equal(embed(1.5), embed(1.0))


def test_generate_is_seed_deterministic(stubbed_model) -> None:
    def latent_bias(seed: int) -> np.ndarray:
        return values(next(stubbed_model.generate("r", duration=1.0, seed=seed)).audio)

    np.testing.assert_array_equal(latent_bias(4), latent_bias(4))
    assert not np.array_equal(latent_bias(4), latent_bias(5))


def test_cfg_batches_conditional_and_zero_unconditional(stubbed_model) -> None:
    batches = []

    def dit(x, t, cross, global_cond):
        batches.append((x.shape[0], t.shape[0], cross, global_cond.shape[0]))
        return mx.zeros_like(x)

    stubbed_model.dit = dit
    next(stubbed_model.generate("rain", duration=1.0, steps=1, cfg_scale=3.0))

    batch, timesteps, cross, global_batch = batches[0]
    assert (batch, timesteps, global_batch) == (2, 2, 2)
    assert cross.shape == (2, 257, 768)
    assert float(mx.abs(cross[1]).max()) == 0.0
    assert float(mx.abs(cross[0, 0]).max()) > 0.0


def test_negative_prompt_replaces_zero_unconditional(stubbed_model) -> None:
    crosses = []

    def dit(x, t, cross, global_cond):
        crosses.append(cross)
        return mx.zeros_like(x)

    stubbed_model.dit = dit
    next(
        stubbed_model.generate(
            "rain",
            duration=1.0,
            steps=1,
            cfg_scale=3.0,
            negative_prompt="muffled",
        )
    )

    assert crosses[0][1, 0, 0].item() == len("muffled")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"text": " "},
        {"text": "x", "duration": 0},
        {"text": "x", "duration": 120.1},
        {"text": "x", "steps": 0},
        {"text": "x", "lyrics": "[instrumental]"},
    ],
)
def test_generate_rejects_invalid_requests(stubbed_model, kwargs) -> None:
    with pytest.raises(ValueError):
        next(stubbed_model.generate(**kwargs))


class _IndexDecoder(SAMEDecoder):
    def __init__(self, config: SAMEConfig = SAMEConfig(depth=0)):
        super().__init__(config)
        self.windows = []

    def __call__(self, latents: mx.array) -> mx.array:
        self.windows.append(latents.shape[1])
        return mx.repeat(latents[:, :, :1], 16, axis=1) * mx.ones((1, 1, 512))


@pytest.mark.parametrize("length", [3, 7, 12, 13, 323])
def test_decode_stitches_each_latent_once_with_even_windows(length) -> None:
    decoder = _IndexDecoder()
    indices = mx.arange(length, dtype=mx.float32)[None, :, None]
    audio = decoder.decode(mx.broadcast_to(indices, (1, length, 256)))

    assert audio.shape == (1, length * 4096, 2)
    assert all(window % 2 == 0 for window in decoder.windows)
    np.testing.assert_array_equal(
        values(audio)[0, ::4096, 0], np.arange(length, dtype=np.float32)
    )


@pytest.mark.parametrize(
    ("length", "windows"), [(77, {77}), (144, {144}), (400, {144})]
)
def test_sliding_window_decode_uses_config_windows_and_full_short_decode(
    length, windows
) -> None:
    decoder = _IndexDecoder(decoder_config(MEDIUM))
    indices = mx.arange(length, dtype=mx.float32)[None, :, None]
    audio = decoder.decode(mx.broadcast_to(indices, (1, length, 256)))

    assert set(decoder.windows) == windows
    np.testing.assert_array_equal(
        values(audio)[0, ::4096, 0], np.arange(length, dtype=np.float32)
    )


class _PatchDecoder(SAMEDecoder):
    def __init__(self):
        super().__init__(SAMEConfig(depth=0))

    def __call__(self, latents: mx.array) -> mx.array:
        return mx.arange(512, dtype=mx.float32)[None, None]


def test_decode_unpacks_channels_in_patch_order() -> None:
    audio = values(_PatchDecoder().decode(mx.zeros((1, 2, 256))))

    np.testing.assert_array_equal(audio[0, :256, 0], np.arange(256))
    np.testing.assert_array_equal(audio[0, :256, 1], np.arange(256, 512))


def test_sanitize_renames_drops_and_is_idempotent() -> None:
    model = Model(ModelConfig())
    weight = mx.zeros((1,))
    source = {
        "model.model.transformer.memory_tokens": weight,
        "model.model.transformer.rotary_pos_emb.inv_freq": weight,
        "conditioner.conditioners.prompt.padding_embedding": weight,
        "conditioner.conditioners.seconds_total.embedder.embedding.1.bias": weight,
        "pretransform.model.bottleneck.running_std": weight,
        "pretransform.model.bottleneck.bias": weight,
        "pretransform.model.bottleneck.scaling_factor": weight,
        "pretransform.model.encoder.layers.0.weight": weight,
        "pretransform.model.decoder.layers.1.weight": weight,
        "pretransform.model.decoder.layers.3.new_tokens": weight,
    }

    sanitized = model.sanitize(source)

    assert set(sanitized) == {
        "dit.transformer.memory_tokens",
        "padding_embedding",
        "seconds_proj.bias",
        "decoder.running_std",
        "decoder.project_in.weight",
        "decoder.new_tokens",
    }
    assert set(model.sanitize(sanitized)) == set(sanitized)


def parameter_shapes(config: ModelConfig) -> dict[str, tuple[int, ...]]:
    return {key: value.shape for key, value in tree_flatten(Model(config).parameters())}


@pytest.mark.parametrize(
    "model, expected",
    [
        (
            None,
            {
                "dit.transformer.memory_tokens": (64, 1024),
                "dit.transformer.global_cond_embedder.2.weight": (6144, 1024),
                "dit.preprocess_conv.weight": (256, 1, 256),
                "dit.to_cond_embed.0.weight": (1024, 768),
                "dit.to_timestep_embed.0.weight": (1024, 256),
                "dit.transformer.layers.19.to_scale_shift_gate": (6144,),
                "dit.transformer.layers.19.to_local_embed.0.weight": (1024, 257),
                "dit.transformer.layers.19.ff.ff.0.proj.weight": (8192, 1024),
                "dit.transformer.layers.19.self_attn.to_qkv.weight": (3072, 1024),
                "dit.transformer.layers.19.cross_attn.to_kv.weight": (2048, 1024),
                "padding_embedding": (768,),
                "seconds_proj.weight": (768, 256),
                "decoder.running_std": (1,),
                "decoder.project_in.weight": (768, 256),
                "decoder.new_tokens": (1, 1, 768),
                "decoder.mapping.weight": (512, 3, 768),
                "decoder.transformers.5.self_attn.to_qkv.weight": (3840, 768),
                "decoder.transformers.5.self_attn.q_norm.alpha": (1,),
                "decoder.transformers.5.ff.ff.0.proj.weight": (4608, 768),
            },
        ),
        (
            MEDIUM,
            {
                "dit.transformer.memory_tokens": (64, 1536),
                "dit.transformer.layers.23.self_attn.to_qkv.weight": (7680, 1536),
                "dit.transformer.layers.23.cross_attn.to_q.weight": (3072, 1536),
                "dit.transformer.layers.23.cross_attn.to_kv.weight": (4608, 1536),
                "dit.transformer.layers.23.ff.ff.0.proj.weight": (12288, 1536),
                "decoder.project_in.weight": (1536, 256),
                "decoder.new_tokens": (1, 1, 1536),
                "decoder.mapping.weight": (512, 1, 1536),
                "decoder.transformers.11.self_attn.to_qkv.weight": (7680, 1536),
                "decoder.transformers.11.ff.ff.0.proj.weight": (9216, 1536),
            },
        ),
    ],
    ids=["small", "medium"],
)
def test_parameters_match_checkpoint_layout(model, expected) -> None:
    shapes = parameter_shapes(ModelConfig(model=model))

    assert {key: shapes.get(key) for key in expected} == expected


def test_encode_text_masks_padding_and_truncates() -> None:
    model = Model(ModelConfig())
    embed = nn.Embedding(16, 8)
    encoder = t5gemma.Encoder(8, 16, 2, 1, 4)
    model.text_encoder = lambda ids, mask: encoder(embed(ids), mask)
    model.tokenizer = SimpleNamespace(
        encode=lambda text: [i % 16 for i in range(len(text))]
    )
    model.padding_embedding = mx.arange(8, dtype=mx.float32)

    hidden = values(model._encode_text("abc"))

    np.testing.assert_allclose(
        hidden[0, :3], values(encoder(embed(mx.array([[0, 1, 2]])))[0]), atol=1e-5
    )
    np.testing.assert_array_equal(hidden[0, 3:], np.tile(np.arange(8), (253, 1)))
    np.testing.assert_array_equal(
        values(model._encode_text("x" * 300)), values(model._encode_text("x" * 256))
    )


def test_last_decoder_blocks_gate_value_by_sine_of_pi_gate() -> None:
    decoder = SAMEDecoder(decoder_config(MEDIUM))
    gate = GatedProjection(2, 1, sinusoidal=True)
    gate.proj.weight = mx.eye(2)
    gate.proj.bias = mx.zeros((2,))

    gated = [block.ff.ff[0].sinusoidal for block in decoder.transformers]
    out = values(gate(mx.array([[0.5, 0.25]])))

    assert gated == [False] * 5 + [True] * 7
    assert out[0, 0] == pytest.approx(0.5 * math.sin(math.pi * 0.25), abs=1e-5)


def test_sliding_window_attention_equals_band_limited_dense_attention() -> None:
    length = 5 * 17
    attention = DifferentialAttention(128, sliding_window=True)
    x = mx.random.normal((1, length, 128), key=mx.random.key(0))
    q, k, v, q_diff, k_diff = (
        to_heads(part) for part in mx.split(attention.to_qkv(x), 5, axis=-1)
    )
    positions = mx.arange(length)
    band = mx.abs(positions[:, None] - positions[None]) <= 17

    def dense(query: mx.array, keys: mx.array) -> mx.array:
        return mx.fast.scaled_dot_product_attention(
            rope(attention.q_norm(query)),
            rope(attention.k_norm(keys)),
            v,
            scale=HEAD_DIM**-0.5,
            mask=band,
        )

    np.testing.assert_allclose(
        values(attention._attend_windowed(q, k, v, q_diff, k_diff)),
        values(dense(q, k) - dense(q_diff, k_diff)),
        atol=1e-4,
    )


def test_guidance_computes_in_fp32_and_returns_input_dtype() -> None:
    x = mx.array([[[300.0], [300.0]]], dtype=mx.float16)
    sigma = mx.array([[[0.5]]], dtype=mx.float16)
    cond_v = mx.zeros_like(x)
    parallel_v = mx.array([[[2.0], [2.0]]], dtype=mx.float16)

    out = guided_velocity(x, sigma, cond_v, parallel_v, 4.0)

    assert out.dtype == mx.float16
    np.testing.assert_allclose(values(out), values(cond_v), atol=1e-2)


def test_generate_samples_at_model_dtype_and_decodes_in_fp32(stubbed_model) -> None:
    stubbed_model.dtype = mx.float16
    dtypes = []
    decoded = []

    def dit(x, t, cross, global_cond):
        dtypes.append((x.dtype, global_cond.dtype))
        return mx.zeros_like(x)

    def decode(latents):
        decoded.append(latents.dtype)
        return mx.zeros((1, latents.shape[1] * 4096, 2))

    stubbed_model.dit = dit
    stubbed_model.decoder = SimpleNamespace(decode=decode)
    next(stubbed_model.generate("rain", duration=1.0, steps=1))

    assert dtypes == [(mx.float16, mx.float16)]
    assert decoded == [mx.float32]


@pytest.mark.parametrize(
    "dit, decoder, frames",
    [
        (DiTConfig(depth=1), SAMEConfig(depth=2), 2),
        (
            DiTConfig(depth=1, differential=True),
            SAMEConfig(
                depth=2, sinusoidal_blocks=1, conv_mapping=False, sliding_window=True
            ),
            3,
        ),
    ],
    ids=["small", "medium"],
)
def test_dit_and_decoder_forward_shapes(dit, decoder, frames) -> None:
    latents = mx.zeros((1, frames, 256))
    velocity = DiT(dit)(
        latents, mx.array([0.5]), mx.zeros((1, 257, 768)), mx.zeros((1, 768))
    )
    patches = SAMEDecoder(decoder)(latents)
    mx.eval(velocity, patches)

    assert velocity.shape == (1, frames, 256)
    assert patches.shape == (1, 16 * frames, 512)


def test_convert_writes_the_mlx_audio_layout(tmp_path) -> None:
    source = tmp_path / "optimized"
    base = tmp_path / "base"
    (source / "MLX").mkdir(parents=True)
    (base / convert.TEXT_ENCODER_DIR).mkdir(parents=True)
    mx.savez(
        str(source / "MLX" / convert.DIT_FILE),
        **{
            "cond.padding_embedding": mx.zeros((4,)),
            "cond.seconds_total_weight": mx.zeros((2, 1)),
            "to_cond_embed.0.weight": mx.zeros((3, 2)),
            "layers.0.self_attn.q_norm.weight": mx.ones((2,)),
        },
    )
    mx.savez(
        str(source / "MLX" / convert.DECODER_FILE),
        running_std=mx.ones((1,)),
        **{
            "project_in.weight": mx.zeros((3, 2)),
            "mapping.weight": mx.zeros((5, 7, 3)),
            "blocks.0.ff.glu_proj.weight": mx.zeros((6, 3)),
        },
    )
    (base / "model_config.json").write_text(json.dumps({"model_type": "other"}))
    for name in convert.TEXT_ENCODER_FILES:
        (base / convert.TEXT_ENCODER_DIR / name).write_bytes(name.encode())

    out = tmp_path / "out"
    convert.convert(out, source=str(source), base=str(base))

    shapes = {k: v.shape for k, v in mx.load(str(out / "model.safetensors")).items()}
    assert shapes == {
        "conditioner.conditioners.prompt.padding_embedding": (4,),
        "conditioner.conditioners.seconds_total.embedder.embedding.1.weight": (2, 1),
        "model.model.to_cond_embed.0.weight": (3, 2),
        "model.model.layers.0.self_attn.q_norm.gamma": (2,),
        "pretransform.model.bottleneck.running_std": (1,),
        "pretransform.model.decoder.layers.1.weight": (3, 2),
        "pretransform.model.decoder.layers.3.mapping.weight": (5, 3, 7),
        "pretransform.model.decoder.layers.3.transformers.0.ff.ff.0.proj.weight": (
            6,
            3,
        ),
    }
    for name in ("config.json", "model_config.json"):
        assert json.loads((out / name).read_text()) == {"model_type": "stable_audio_3"}
    text_dir = out / convert.TEXT_ENCODER_DIR
    assert sorted(p.name for p in text_dir.iterdir()) == sorted(
        convert.TEXT_ENCODER_FILES
    )
