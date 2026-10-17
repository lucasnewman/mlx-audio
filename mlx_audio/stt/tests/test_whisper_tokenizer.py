import unittest

from mlx_audio.stt.models.whisper.whisper import HFTokenizerWrapper


class TestHFTokenizerWrapper(unittest.TestCase):
    def test_non_speech_tokens_skips_empty_seed_encodings(self):
        class EmptySeedTokenizer:
            def encode(self, text, add_special_tokens=False):
                if text in {" -", " '"}:
                    return []
                return [101]

        tokenizer = HFTokenizerWrapper(EmptySeedTokenizer(), multilingual=False)

        self.assertEqual(tokenizer.non_speech_tokens, (101,))

    def test_non_speech_tokens_skips_empty_miscellaneous_encodings(self):
        miscellaneous = set("♩♪♫♬♭♮♯")

        class EmptyMiscellaneousTokenizer:
            def encode(self, text, add_special_tokens=False):
                if text.strip() in miscellaneous:
                    return []
                return [101]

        tokenizer = HFTokenizerWrapper(
            EmptyMiscellaneousTokenizer(), multilingual=False
        )

        self.assertEqual(tokenizer.non_speech_tokens, (101,))


if __name__ == "__main__":
    unittest.main()
