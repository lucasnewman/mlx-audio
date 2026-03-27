import unittest
from types import SimpleNamespace

import mlx.core as mx

from mlx_audio.tts.models.voxtral_tts.voxtral_tts import Model


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [101, 102]


class TestVoxtralTTSPrompt(unittest.TestCase):
    def test_encode_text_fallback_matches_mistral_common_layout(self):
        model = Model.__new__(Model)
        model.tokenizer = FakeTokenizer()
        model.config = SimpleNamespace(
            bos_token_id=1,
            begin_audio_token_id=25,
            audio_token_id=24,
        )
        model._voice_embeddings = {"casual_male": mx.zeros((147, 3072))}
        model._voice_num_audio_tokens = {"casual_male": 147}
        model._text_to_audio_token_id = 36
        model._audio_to_text_token_id = 35

        tokens = Model._encode_text(model, "Hello world.", "casual_male")

        self.assertEqual(tokens[:3], [1, 25, 24])
        self.assertEqual(tokens[1 + 1 + 147], 36)
        self.assertEqual(tokens[1 + 1 + 147 + 1 : 1 + 1 + 147 + 3], [101, 102])
        self.assertEqual(tokens[-2:], [35, 25])

    def test_encode_text_falls_back_to_voice_embedding_length(self):
        model = Model.__new__(Model)
        model.tokenizer = FakeTokenizer()
        model.config = SimpleNamespace(
            bos_token_id=1,
            begin_audio_token_id=25,
            audio_token_id=24,
        )
        model._voice_embeddings = {"casual_male": mx.zeros((3, 3072))}
        model._voice_num_audio_tokens = {}
        model._text_to_audio_token_id = 36
        model._audio_to_text_token_id = 35

        tokens = Model._encode_text(model, "Hello world.", "casual_male")

        self.assertEqual(tokens, [1, 25, 24, 24, 24, 36, 101, 102, 35, 25])


if __name__ == "__main__":
    unittest.main()
