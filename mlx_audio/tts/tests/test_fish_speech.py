import unittest

import mlx.core as mx

from mlx_audio.tts.models.fish_qwen3_omni.config import ModelConfig
from mlx_audio.tts.models.fish_qwen3_omni.fish_speech import Model
from mlx_audio.tts.models.fish_qwen3_omni.prompt import (
    Conversation,
    Message,
    TextPart,
    VQPart,
)
from mlx_audio.tts.utils import get_model_and_args


class FakeTokenizer:
    def __init__(self):
        self.semantic_begin_id = 1000
        self._next = 1

    def encode(self, text):
        token = self._next
        self._next += 1
        return [token]


def tiny_config() -> ModelConfig:
    return ModelConfig.from_dict(
        {
            "semantic_start_token_id": 1000,
            "semantic_end_token_id": 1007,
            "text_config": {
                "vocab_size": 32,
                "n_layer": 1,
                "n_head": 2,
                "dim": 8,
                "intermediate_size": 16,
                "n_local_heads": 1,
                "head_dim": 4,
                "norm_eps": 1e-6,
                "max_seq_len": 64,
                "attention_qk_norm": True,
            },
            "audio_decoder_config": {
                "vocab_size": 8,
                "n_layer": 1,
                "n_head": 2,
                "dim": 8,
                "intermediate_size": 16,
                "n_local_heads": 1,
                "head_dim": 4,
                "num_codebooks": 2,
                "norm_eps": 1e-6,
                "max_seq_len": 3,
            },
        }
    )


class TestFishSpeechPrompt(unittest.TestCase):
    def test_encode_for_inference_places_vq_codes_in_correct_rows(self):
        tokenizer = FakeTokenizer()
        conversation = Conversation(
            [
                Message(
                    role="user",
                    parts=[
                        TextPart("hello"),
                        VQPart(mx.array([[1, 2], [3, 4]], dtype=mx.int32)),
                        TextPart("world"),
                    ],
                    add_im_start=False,
                    add_im_end=False,
                )
            ]
        )

        values = conversation.encode_for_inference(tokenizer, num_codebooks=2)
        self.assertEqual(tuple(values.shape), (3, 4))
        self.assertEqual(values[0].tolist(), [1, 1001, 1002, 2])
        self.assertEqual(values[1].tolist(), [0, 1, 2, 0])
        self.assertEqual(values[2].tolist(), [0, 3, 4, 0])


class TestFishSpeechModel(unittest.TestCase):
    def test_model_type_remapping_uses_config_value(self):
        module, model_type = get_model_and_args("fish_qwen3_omni", ["s2", "pro"])
        self.assertEqual(model_type, "fish_qwen3_omni")
        self.assertTrue(hasattr(module, "Model"))

    def test_sanitize_remaps_upstream_keys(self):
        model = Model(tiny_config())
        weights = {
            "text_model.model.embeddings.weight": mx.zeros((4, 4)),
            "audio_decoder.embeddings.weight": mx.zeros((4, 4)),
            "audio_decoder.layers.0.attention.wqkv.weight": mx.zeros((4, 4)),
            "audio_decoder.codebook_embeddings.weight": mx.zeros((4, 4)),
        }

        sanitized = model.sanitize(weights)

        self.assertIn("model.embeddings.weight", sanitized)
        self.assertIn("model.fast_embeddings.weight", sanitized)
        self.assertIn("model.fast_layers.0.attention.wqkv.weight", sanitized)
        self.assertIn("model.codebook_embeddings.weight", sanitized)

    def test_config_from_dict_handles_upstream_nested_shape(self):
        config = tiny_config()
        self.assertEqual(config.text_config.n_layer, 1)
        self.assertEqual(config.audio_decoder_config.num_codebooks, 2)
        self.assertEqual(config.semantic_start_token_id, 1000)


if __name__ == "__main__":
    unittest.main()
