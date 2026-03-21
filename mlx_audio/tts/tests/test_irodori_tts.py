import unittest

import mlx.core as mx
import numpy as np

from mlx_audio.tts.models.irodori_tts.config import (
    IrodoriDiTConfig,
    ModelConfig,
    SamplerConfig,
)
from mlx_audio.tts.models.irodori_tts.irodori_tts import Model
from mlx_audio.tts.models.irodori_tts.model import IrodoriDiT
from mlx_audio.tts.models.irodori_tts.text import encode_text, normalize_text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _small_dit_config(**overrides) -> IrodoriDiTConfig:
    """Tiny DiT config suitable for fast unit tests."""
    defaults = dict(
        latent_dim=8,
        latent_patch_size=1,
        model_dim=32,
        num_layers=2,
        num_heads=4,
        mlp_ratio=2.0,
        text_mlp_ratio=2.0,
        speaker_mlp_ratio=2.0,
        text_vocab_size=64,
        text_dim=32,
        text_layers=1,
        text_heads=4,
        speaker_dim=32,
        speaker_layers=1,
        speaker_heads=4,
        speaker_patch_size=1,
        timestep_embed_dim=16,
        adaln_rank=8,
        norm_eps=1e-5,
    )
    defaults.update(overrides)
    return IrodoriDiTConfig(**defaults)


def _small_model_config(**sampler_overrides) -> ModelConfig:
    sampler_defaults = dict(
        num_steps=1,
        cfg_scale_text=1.0,
        cfg_scale_speaker=1.0,
        sequence_length=4,
    )
    sampler_defaults.update(sampler_overrides)
    return ModelConfig(
        dit=_small_dit_config(),
        sampler=SamplerConfig(**sampler_defaults),
    )


class MockTokenizer:
    """Minimal HuggingFace-style tokenizer stub."""

    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0
    padding_side = "right"

    def encode(self, text, add_special_tokens=False):
        # Return a few dummy token IDs (independent of text content)
        return [3, 4, 5]


class FakeDACVAE:
    """DACVAE stub that matches the real API shapes."""

    def __init__(self, latent_dim: int = 8, downsample_factor: int = 1920):
        self.latent_dim = latent_dim
        self.downsample_factor = downsample_factor

    def encode(self, audio_in: mx.array) -> mx.array:
        # audio_in: (B, L, 1) → (B, latent_dim, T)
        B = audio_in.shape[0]
        T = max(1, int(audio_in.shape[1]) // self.downsample_factor)
        return mx.zeros((B, self.latent_dim, T), dtype=mx.float32)

    def decode(self, latent: mx.array) -> mx.array:
        # latent: (B, latent_dim, T) → (B, T * downsample_factor, 1)
        B, _D, T = latent.shape
        return mx.zeros((B, T * self.downsample_factor, 1), dtype=mx.float32)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TestNormalizeText(unittest.TestCase):
    def test_fullwidth_alpha_to_halfwidth(self):
        result = normalize_text("Ａｂ")
        self.assertEqual(result, "Ab")

    def test_fullwidth_digits_to_halfwidth(self):
        result = normalize_text("１２３")
        self.assertEqual(result, "123")

    def test_halfwidth_kana_to_fullwidth(self):
        # ｱ (halfwidth katakana A) → ア (fullwidth)
        result = normalize_text("ｱｲ")
        self.assertEqual(result, "アイ")

    def test_wave_dash_to_katakana_dash(self):
        result = normalize_text("ー〜ー")
        self.assertEqual(result, "ーーー")

    def test_fullwidth_exclamation_to_halfwidth(self):
        result = normalize_text("！")
        self.assertEqual(result, "!")

    def test_trailing_kuten_stripped(self):
        result = normalize_text("こんにちは。")
        self.assertFalse(result.endswith("。"))
        self.assertEqual(result, "こんにちは")

    def test_surrounding_brackets_stripped(self):
        result = normalize_text("「こんにちは」")
        self.assertEqual(result, "こんにちは")

    def test_triple_ellipsis_collapsed(self):
        result = normalize_text("……………")
        self.assertEqual(result, "……")

    def test_no_change_for_plain_text(self):
        text = "こんにちは"
        self.assertEqual(normalize_text(text), text)


class TestEncodeText(unittest.TestCase):
    def setUp(self):
        self.tok = MockTokenizer()

    def test_output_shapes(self):
        ids, mask = encode_text("hello", self.tok, max_length=10, add_bos=True)
        self.assertEqual(tuple(ids.shape), (1, 10))
        self.assertEqual(tuple(mask.shape), (1, 10))

    def test_bos_prepended(self):
        ids, mask = encode_text("hello", self.tok, max_length=10, add_bos=True)
        # First token should be bos_token_id=1
        self.assertEqual(int(ids[0, 0]), self.tok.bos_token_id)

    def test_no_bos(self):
        ids, _ = encode_text("hello", self.tok, max_length=10, add_bos=False)
        self.assertEqual(int(ids[0, 0]), 3)  # first real token

    def test_padding(self):
        ids, mask = encode_text("hello", self.tok, max_length=10, add_bos=True)
        # BOS + 3 tokens = 4 real tokens; positions 4..9 should be padded
        for i in range(4, 10):
            self.assertEqual(int(ids[0, i]), self.tok.pad_token_id)
            self.assertFalse(bool(mask[0, i]))

    def test_mask_true_for_real_tokens(self):
        ids, mask = encode_text("hello", self.tok, max_length=10, add_bos=True)
        for i in range(4):  # BOS + 3 tokens
            self.assertTrue(bool(mask[0, i]))

    def test_truncation(self):
        ids, mask = encode_text("hello", self.tok, max_length=2, add_bos=True)
        self.assertEqual(tuple(ids.shape), (1, 2))


class TestIrodoriDiTShapes(unittest.TestCase):
    def setUp(self):
        self.cfg = _small_dit_config()
        self.model = IrodoriDiT(self.cfg)

    def test_full_forward_shape(self):
        B, S = 1, 6
        x_t = mx.random.normal((B, S, self.cfg.patched_latent_dim))
        t = mx.array([0.5], dtype=mx.float32)
        text_ids = mx.zeros((B, 5), dtype=mx.int32)
        text_mask = mx.ones((B, 5), dtype=mx.bool_)
        ref_latent = mx.random.normal((B, 8, self.cfg.latent_dim))
        ref_mask = mx.ones((B, 8), dtype=mx.bool_)

        out = self.model(x_t, t, text_ids, text_mask, ref_latent, ref_mask)
        mx.eval(out)
        self.assertEqual(tuple(out.shape), (B, S, self.cfg.patched_latent_dim))

    def test_encode_conditions_shapes(self):
        B = 1
        text_ids = mx.zeros((B, 5), dtype=mx.int32)
        text_mask = mx.ones((B, 5), dtype=mx.bool_)
        ref_latent = mx.random.normal((B, 8, self.cfg.latent_dim))
        ref_mask = mx.ones((B, 8), dtype=mx.bool_)

        t_state, t_mask, s_state, s_mask = self.model.encode_conditions(
            text_ids, text_mask, ref_latent, ref_mask
        )
        mx.eval(t_state, s_state)
        self.assertEqual(tuple(t_state.shape), (B, 5, self.cfg.text_dim))
        self.assertEqual(int(s_state.shape[0]), B)
        self.assertEqual(int(s_state.shape[-1]), self.cfg.speaker_dim)

    def test_kv_cache_and_forward_with_conditions(self):
        B, S = 1, 4
        text_ids = mx.zeros((B, 5), dtype=mx.int32)
        text_mask = mx.ones((B, 5), dtype=mx.bool_)
        ref_latent = mx.zeros((B, 8, self.cfg.latent_dim))
        ref_mask = mx.ones((B, 8), dtype=mx.bool_)

        t_state, t_mask, s_state, s_mask = self.model.encode_conditions(
            text_ids, text_mask, ref_latent, ref_mask
        )
        kv_text, kv_speaker = self.model.build_kv_cache(t_state, s_state)
        self.assertEqual(len(kv_text), self.cfg.num_layers)
        self.assertEqual(len(kv_speaker), self.cfg.num_layers)

        x_t = mx.random.normal((B, S, self.cfg.patched_latent_dim))
        t = mx.array([0.3], dtype=mx.float32)
        out = self.model.forward_with_conditions(
            x_t, t, t_state, t_mask, s_state, s_mask, kv_text, kv_speaker
        )
        mx.eval(out)
        self.assertEqual(tuple(out.shape), (B, S, self.cfg.patched_latent_dim))

    def test_zero_speaker_latent(self):
        """forward should not crash with all-zero unconditional speaker."""
        B, S = 1, 4
        x_t = mx.random.normal((B, S, self.cfg.patched_latent_dim))
        t = mx.array([1.0], dtype=mx.float32)
        text_ids = mx.zeros((B, 5), dtype=mx.int32)
        text_mask = mx.ones((B, 5), dtype=mx.bool_)
        ref_latent = mx.zeros((B, 1, self.cfg.latent_dim))
        ref_mask = mx.zeros((B, 1), dtype=mx.bool_)

        out = self.model(x_t, t, text_ids, text_mask, ref_latent, ref_mask)
        mx.eval(out)
        self.assertEqual(tuple(out.shape), (B, S, self.cfg.patched_latent_dim))


class TestModelSanitize(unittest.TestCase):
    def setUp(self):
        cfg = _small_model_config()
        self.model = Model(cfg)

    def test_cond_module_key_remapped(self):
        weights = {"cond_module.0.weight": mx.zeros((1, 1), dtype=mx.float32)}
        sanitized = self.model.sanitize(weights)
        self.assertIn("model.cond_module.layers.0.weight", sanitized)
        self.assertNotIn("cond_module.0.weight", sanitized)

    def test_model_prefix_added(self):
        weights = {"blocks.0.mlp.w1.weight": mx.zeros((1, 1), dtype=mx.float32)}
        sanitized = self.model.sanitize(weights)
        self.assertIn("model.blocks.0.mlp.w1.weight", sanitized)

    def test_model_prefix_not_doubled(self):
        weights = {"model.out_proj.weight": mx.zeros((1, 1), dtype=mx.float32)}
        sanitized = self.model.sanitize(weights)
        self.assertIn("model.out_proj.weight", sanitized)
        self.assertNotIn("model.model.out_proj.weight", sanitized)

    def test_deep_cond_module_key(self):
        weights = {"cond_module.2.bias": mx.zeros((1,), dtype=mx.float32)}
        sanitized = self.model.sanitize(weights)
        self.assertIn("model.cond_module.layers.2.bias", sanitized)


class TestGenerateSmoke(unittest.TestCase):
    def _make_model(self):
        cfg = _small_model_config()
        model = Model(cfg)
        model.dacvae = FakeDACVAE(
            latent_dim=cfg.dit.latent_dim,
            downsample_factor=cfg.audio_downsample_factor,
        )
        model._tokenizer = MockTokenizer()
        return model

    def test_generate_returns_result(self):
        model = self._make_model()
        results = list(model.generate("こんにちは", rng_seed=0))
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.sample_rate, 48000)
        self.assertGreater(result.samples, 0)

    def test_generate_with_ref_audio(self):
        model = self._make_model()
        # Provide reference audio as a pre-computed mx.array (1, L)
        ref = mx.zeros((1, model.config.audio_downsample_factor * 4), dtype=mx.float32)
        results = list(model.generate("テスト", ref_audio=ref, rng_seed=1))
        self.assertEqual(len(results), 1)
        self.assertGreater(results[0].samples, 0)

    def test_generate_stream_raises(self):
        model = self._make_model()
        with self.assertRaises(NotImplementedError):
            next(model.generate("hi", stream=True))

    def test_generate_without_dacvae_raises(self):
        cfg = _small_model_config()
        model = Model(cfg)
        model._tokenizer = MockTokenizer()
        # dacvae is None by default
        with self.assertRaises(ValueError):
            next(model.generate("hi"))

    def test_result_fields(self):
        model = self._make_model()
        result = next(model.generate("テスト", rng_seed=0))
        self.assertIsNotNone(result.audio)
        self.assertIsInstance(result.token_count, int)
        self.assertGreater(result.token_count, 0)
        self.assertIsNotNone(result.audio_duration)
        self.assertGreater(result.real_time_factor, 0.0)


if __name__ == "__main__":
    unittest.main()
