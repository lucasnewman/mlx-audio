import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from mlx_audio.stt.models.qwen3_asr.qwen3_asr import Qwen3ASRModel, normalize_language
from mlx_audio.stt.models.qwen3_asr.qwen3_forced_aligner import ForcedAlignerModel

SUPPORTED = ["Chinese", "English", "Cantonese", "Japanese", "French"]


def _model(outputs=()):
    model = Qwen3ASRModel.__new__(Qwen3ASRModel)
    model.config = SimpleNamespace(support_languages=SUPPORTED)
    model._tokenizer = Mock()
    model._tokenizer.encode = Mock(return_value=np.zeros((1, 1), dtype=np.int32))
    model._feature_extractor = object()
    model._generate_single_chunk = Mock(side_effect=[(text, 5, 1) for text in outputs])
    return model


def _transcribe(outputs, language):
    model = _model(outputs)
    out = model.generate(
        np.zeros(16000 * len(outputs)),
        chunk_duration=1,
        min_chunk_duration=1,
        language=language,
    )
    return model, out


class TestNormalizeLanguage(unittest.TestCase):
    def test_codes_and_names(self):
        for value, expected in [
            ("en", "English"),
            ("EN", "English"),
            ("zh", "Chinese"),
            ("zh-CN", "Chinese"),
            ("zh_TW", "Chinese"),
            ("yue", "Cantonese"),
            ("ja", "Japanese"),
            ("chinese", "Chinese"),
            ("English", "English"),
            ("Klingon", "Klingon"),
        ]:
            with self.subTest(value=value):
                self.assertEqual(normalize_language(value, SUPPORTED), expected)

    def test_empty_is_none(self):
        for value in [None, "", "  "]:
            self.assertIsNone(normalize_language(value, SUPPORTED))


class TestQwen3ASRLanguage(unittest.TestCase):
    def test_iso_code_is_prompted_as_language_name(self):
        model = _model()
        model._build_prompt(3, "en")
        prompt = model._tokenizer.encode.call_args.args[0]
        self.assertTrue(prompt.endswith("language English<asr_text>"), prompt)

    def test_forced_language_is_reported_by_name(self):
        model, out = _transcribe(["hello", "world"], "en")
        calls = model._generate_single_chunk.call_args_list
        self.assertEqual([c.kwargs["language"] for c in calls], ["English"] * 2)
        self.assertEqual([s["language"] for s in out.segments], ["English"] * 2)
        self.assertEqual(out.text, "hello world")

    def test_chinese_chunks_are_joined_without_space(self):
        _, out = _transcribe(["今天天气", "很好。"], "zh")
        self.assertEqual(out.text, "今天天气很好。")

    def test_detected_japanese_chunks_are_joined_without_space(self):
        _, out = _transcribe(
            [
                "language Japanese<asr_text>今日は",
                "language Japanese<asr_text>いい天気です。",
            ],
            None,
        )
        self.assertEqual(out.text, "今日はいい天気です。")
        self.assertEqual([s["language"] for s in out.segments], ["Japanese"] * 2)

    def test_empty_chunk_adds_no_separator(self):
        _, out = _transcribe(["hello", "", "world"], "English")
        self.assertEqual(out.text, "hello world")

    def test_untagged_output_has_no_language(self):
        model = _model()
        self.assertEqual(model.extract_language("hello"), (None, "hello"))
        self.assertEqual(model.extract_language("language None<asr_text>"), (None, ""))
        self.assertEqual(
            model.extract_language("language chinese<asr_text>你好"),
            ("Chinese", "你好"),
        )


class TestForcedAlignerLanguage(unittest.TestCase):
    def test_iso_code_selects_the_language_tokenizer(self):
        model = ForcedAlignerModel.__new__(ForcedAlignerModel)
        model._tokenizer = object()
        model._feature_extractor = object()
        model._preprocess_audio = Mock(return_value=(None, None, 1))
        model.aligner_processor = Mock()
        model.aligner_processor.encode_timestamp = Mock(side_effect=RuntimeError)

        with self.assertRaises(RuntimeError):
            model.generate(np.zeros(1600), "你好", language="zh")

        self.assertEqual(
            model.aligner_processor.encode_timestamp.call_args.args, ("你好", "Chinese")
        )


if __name__ == "__main__":
    unittest.main()
