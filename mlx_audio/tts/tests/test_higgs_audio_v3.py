import unittest

import mlx.core as mx
import numpy as np

from mlx_audio.tts.models.higgs_audio_v3.config import (
    HiggsAudioV3Config,
    HiggsAudioV3TextConfig,
)
from mlx_audio.tts.models.higgs_audio_v3.generation import (
    HiggsSamplerState,
    apply_delay_pattern,
    reverse_delay_pattern,
    step,
)
from mlx_audio.tts.models.higgs_audio_v3.model import Model
from mlx_audio.tts.models.higgs_audio_v3.prompt import (
    AUDIO_PLACEHOLDER_ID,
    HiggsAudioV3PromptBuilder,
    ReferenceCodes,
)
from mlx_audio.utils import get_model_name_parts
from mlx_audio.tts.utils import get_model_and_args


def _tiny_config() -> HiggsAudioV3Config:
    return HiggsAudioV3Config(
        text_config=HiggsAudioV3TextConfig(
            hidden_size=32,
            num_hidden_layers=2,
            intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            rope_theta=10000.0,
            head_dim=8,
            rms_norm_eps=1e-6,
            vocab_size=64,
            tie_word_embeddings=True,
        ),
        audio_num_codebooks=4,
        audio_codebook_size=10,
        audio_boc_token_id=8,
        audio_eoc_token_id=9,
    )


class FakeTokenizer:
    def __init__(self):
        self.vocab = {
            "<|tts|>": 1,
            "<|ref_audio|>": 2,
            "<|text|>": 3,
            "<|audio|>": 4,
            "<|ref_text|>": 5,
        }

    def get_added_vocab(self):
        return self.vocab

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [10 + (ord(ch) % 20) for ch in text]


class TestHiggsAudioV3Config(unittest.TestCase):
    def test_parses_source_config_shape(self):
        cfg = HiggsAudioV3Config.from_dict(
            {
                "model_type": "higgs_multimodal_qwen3",
                "audio_token_id": -100,
                "audio_encoder_config": {
                    "num_codebooks": 8,
                    "vocab_size": 1026,
                    "use_delay_pattern": True,
                },
                "text_config": {
                    "model_type": "qwen3",
                    "hidden_size": 2560,
                    "num_hidden_layers": 36,
                    "intermediate_size": 9728,
                    "num_attention_heads": 32,
                    "num_key_value_heads": 8,
                    "max_position_embeddings": 32768,
                    "head_dim": 128,
                    "rms_norm_eps": 1e-6,
                    "vocab_size": 151936,
                    "tie_word_embeddings": True,
                    "rope_parameters": {"rope_theta": 1000000},
                },
            }
        )
        self.assertEqual(cfg.model_type, "higgs_multimodal_qwen3")
        self.assertEqual(cfg.text_config.hidden_size, 2560)
        self.assertEqual(cfg.text_config.rope_theta, 1000000)
        self.assertEqual(cfg.audio_num_codebooks, 8)
        self.assertEqual(cfg.audio_codebook_size, 1026)
        self.assertEqual(cfg.audio_boc_token_id, 1024)
        self.assertEqual(cfg.audio_eoc_token_id, 1025)

    def test_loader_remapping(self):
        model_name = get_model_name_parts("bosonai/higgs-audio-v3-tts-4b")
        arch, model_type = get_model_and_args(
            "higgs_multimodal_qwen3",
            model_name,
        )
        self.assertEqual(model_type, "higgs_audio_v3")
        self.assertTrue(hasattr(arch, "Model"))


class TestHiggsAudioV3DelayPattern(unittest.TestCase):
    def test_delay_pattern_round_trip(self):
        codes = mx.arange(1, 13, dtype=mx.int32).reshape(3, 4)
        delayed = apply_delay_pattern(codes, boc_id=98, eoc_id=99)
        self.assertEqual(delayed.shape, (6, 4))
        np.testing.assert_array_equal(np.array(reverse_delay_pattern(delayed)), codes)

    def test_delay_pattern_padding(self):
        codes = mx.array([[1, 2, 3]], dtype=mx.int32)
        delayed = apply_delay_pattern(codes, boc_id=8, eoc_id=9)
        np.testing.assert_array_equal(
            np.array(delayed),
            np.array([[1, 8, 8], [9, 2, 8], [9, 9, 3]], dtype=np.int32),
        )


class TestHiggsAudioV3Sampler(unittest.TestCase):
    def test_ramp_in_forces_later_codebooks_to_boc(self):
        state = HiggsSamplerState(num_codebooks=4)
        logits = mx.zeros((4, 10))
        logits[:, 1] = 10.0
        out = step(
            logits,
            state,
            temperature=0.0,
            top_p=None,
            top_k=None,
            boc_id=8,
            eoc_id=9,
        )
        np.testing.assert_array_equal(np.array(out), [1, 8, 8, 8])
        self.assertEqual(state.delay_count, 1)

    def test_eoc_wind_down(self):
        state = HiggsSamplerState(num_codebooks=4, delay_count=4)
        logits = mx.zeros((4, 10))
        logits[:, 1] = 10.0
        logits[0, 9] = 20.0
        step(logits, state, temperature=0.0, top_p=None, top_k=None, boc_id=8, eoc_id=9)
        self.assertEqual(state.eoc_countdown, 2)
        self.assertFalse(state.generation_done)

        logits[0, 1] = 30.0
        step(logits, state, temperature=0.0, top_p=None, top_k=None, boc_id=8, eoc_id=9)
        self.assertEqual(state.eoc_countdown, 1)
        self.assertFalse(state.generation_done)
        step(logits, state, temperature=0.0, top_p=None, top_k=None, boc_id=8, eoc_id=9)
        self.assertTrue(state.generation_done)


class TestHiggsAudioV3Prompt(unittest.TestCase):
    def test_prompt_placeholders_match_delayed_rows(self):
        builder = HiggsAudioV3PromptBuilder(FakeTokenizer())
        delayed = mx.zeros((5, 4), dtype=mx.int32)
        prompt = builder.build_prompt(
            "target",
            references=[ReferenceCodes(codes=delayed, text="ref")],
        )
        self.assertIn(AUDIO_PLACEHOLDER_ID, prompt.token_ids)
        self.assertEqual(prompt.token_ids.count(AUDIO_PLACEHOLDER_ID), 5)
        self.assertEqual(len(prompt.audio_segments), 1)
        self.assertEqual(prompt.audio_segments[0][0], prompt.token_ids.index(-100))


class TestHiggsAudioV3Model(unittest.TestCase):
    def test_sanitize_maps_source_keys(self):
        model = Model(_tiny_config())
        weights = {
            "tied.embedding.text_embedding.weight": mx.zeros((64, 32)),
            "body.layers.0.input_layernorm.weight": mx.ones((32,)),
            "body.norm.weight": mx.ones((32,)),
            "tied.embedding.modality_embeddings.0.embedding.weight": mx.zeros((40, 32)),
            "tied.embedding.modality_embeddings.0.model.fc.weight": mx.zeros((1, 1)),
        }
        sanitized = model.sanitize(weights)
        self.assertIn("backbone.embed_tokens.weight", sanitized)
        self.assertIn("backbone.layers.0.input_layernorm.weight", sanitized)
        self.assertIn("backbone.norm.weight", sanitized)
        self.assertIn("multimodal_embedding.weight", sanitized)
        self.assertNotIn(
            "tied.embedding.modality_embeddings.0.model.fc.weight", sanitized
        )

    def test_fused_embedding_and_head_shapes(self):
        cfg = _tiny_config()
        model = Model(cfg)
        weight = mx.arange(40 * 32, dtype=mx.float32).reshape(40, 32)
        model.multimodal_embedding.weight = weight
        codes = mx.zeros((3, 4), dtype=mx.int32)
        out = model._embed_audio_codes(codes)
        self.assertEqual(out.shape, (3, 32))
        expected = weight[0] + weight[10] + weight[20] + weight[30]
        np.testing.assert_allclose(np.array(out[0]), np.array(expected))

        logits = model._audio_logits(mx.zeros((1, 32)))
        self.assertEqual(logits.shape, (1, 4, 10))

    def test_tiny_forward_shape(self):
        cfg = _tiny_config()
        model = Model(cfg)
        mx.eval(model.parameters())
        hidden = model(mx.zeros((1, 3), dtype=mx.int32))
        self.assertEqual(hidden.shape, (1, 3, cfg.text_config.hidden_size))


if __name__ == "__main__":
    unittest.main()
