import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import numpy as np

from mlx_audio.tts.models.confucius4.confucius4 import (
    Model,
    _cross_fade_concat,
    _segment_text,
)
from mlx_audio.tts.models.confucius4.t2s import BOS, EOS, T2SMLX


def count_words(text):
    return len(text.split())


class TestSegmentText(unittest.TestCase):
    def test_short_text_stays_whole(self):
        self.assertEqual(
            _segment_text("One sentence. And a second.", count_words, "en"),
            ["One sentence. And a second."],
        )

    def test_long_text_splits_at_sentence_ends(self):
        sentence = " ".join(["word"] * 30) + "."
        text = " ".join([sentence] * 6)
        segments = _segment_text(text, count_words, "en")

        self.assertGreater(len(segments), 1)
        self.assertEqual("".join(segments), text)
        for segment in segments:
            self.assertTrue(segment.endswith("."))
            self.assertLessEqual(count_words(segment), 90)

    def test_short_tail_joins_the_last_segment(self):
        sentence = " ".join(["word"] * 70) + "."
        tail = " ".join(["bye"] * 15) + "."
        segments = _segment_text(f"{sentence} {sentence} {tail}", count_words, "en")

        self.assertEqual(len(segments), 2)
        self.assertTrue(segments[-1].endswith(" " + tail))

    def test_chinese_counts_characters_and_drops_the_last_full_stop(self):
        segments = _segment_text("今" * 50 + "。" + "天" * 50 + "。", len, "zh")

        self.assertEqual(segments, ["今" * 50 + "。" + "天" * 50])
        segments = _segment_text("今" * 70 + "。" + "天" * 70 + "。", len, "zh")
        self.assertEqual(segments, ["今" * 70, "天" * 70])

    def test_one_long_sentence_splits_by_length(self):
        segments = _segment_text("天" * 200, len, "zh")

        self.assertEqual("".join(segments), "天" * 200)
        self.assertGreater(len(segments), 1)

    def test_punctuation_only_text_gives_no_segments(self):
        self.assertEqual(_segment_text("...", count_words, "en"), [])


class TestCrossFadeConcat(unittest.TestCase):
    def test_single_chunk_is_unchanged(self):
        chunk = np.ones(10, dtype=np.float32)

        self.assertIs(_cross_fade_concat([chunk], 100), chunk)

    def test_chunks_fade_around_a_silent_gap(self):
        first = np.ones(50, dtype=np.float32)
        second = np.ones(50, dtype=np.float32)

        merged = _cross_fade_concat([first, second], 100)

        self.assertEqual(merged.shape[0], 110)
        np.testing.assert_array_equal(merged[:40], 1)
        self.assertEqual(merged[49], 0)
        np.testing.assert_array_equal(merged[50:60], 0)
        self.assertEqual(merged[60], 0)
        np.testing.assert_array_equal(merged[70:], 1)
        np.testing.assert_array_equal(first, 1)
        np.testing.assert_array_equal(second, 1)


class TestT2SLength(unittest.TestCase):
    def test_generation_length_counts_the_prompt(self):
        t2s = T2SMLX.__new__(T2SMLX)
        hidden = mx.zeros((1, 1, 4))
        steps = []

        def transformer(x, caches=None):
            return mx.zeros((1, x.shape[1], 4)), caches

        def sample(*args):
            steps.append(1)
            return 7

        vars(t2s).update(
            _sem_token_embed=lambda token, pos: hidden,
            _prefill=lambda x: (mx.zeros((1, x.shape[1], 4)), []),
            _head=lambda h: mx.zeros((1, 1, 8)),
            _sample=sample,
            transformer=transformer,
            semantic_embed=lambda codes: mx.zeros((1, codes.shape[1], 4)),
        )

        codes, _ = t2s.generate(
            mx.zeros((1, 1, 4)), mx.zeros((1, 18, 4)), max_length=40
        )

        self.assertEqual(len(steps), 40 - 18 - 2)
        self.assertEqual(len(codes), len(steps) - 1)
        self.assertNotIn(BOS, codes)
        self.assertNotIn(EOS, codes)


class TestGenerate(unittest.TestCase):
    def make_model(self):
        model = Model.__new__(Model)
        tokenizer = MagicMock()
        tokenizer.encode.side_effect = lambda text, **kwargs: SimpleNamespace(
            ids=[1] + list(range(10, 10 + len(text.split())))
        )
        tokenizer.token_to_id.side_effect = {"</s>": 2}.get
        t2s = MagicMock()
        t2s.generate.return_value = (
            np.arange(5),
            np.zeros((1, 5, 4), dtype=np.float32),
        )
        s2a = MagicMock()
        s2a.build_mu.return_value = mx.zeros((1, 6, 80))
        s2a.solve_euler.return_value = mx.zeros((1, 80, 8))
        vars(model).update(
            sample_rate=100,
            _fbank=lambda *args: mx.zeros((1, 4, 8)),
            _mel=None,
            _win=None,
            w2v=MagicMock(),
            stats={"mean": 0.0, "std": 1.0},
            camp=MagicMock(),
            prefix=MagicMock(),
            _tok=tokenizer,
            t2s=t2s,
            s2a=s2a,
            voc=MagicMock(return_value=mx.ones((1, 1, 40))),
        )
        model.w2v.hidden17.return_value = mx.zeros((1, 4, 8))
        model.camp.inference.return_value = mx.zeros((1, 192))
        return model

    def run_generate(self, model, text):
        with (
            patch("mlx_audio.utils.load_audio", return_value=mx.zeros((1600,))),
            patch(
                "mlx_audio.tts.models.confucius4.confucius4._ref_mel",
                return_value=mx.zeros((1, 2, 80)),
            ),
        ):
            return list(model.generate(text=text, ref_audio="ref.wav", lang="en"))

    def test_text_prompt_ends_with_the_end_of_sequence_token(self):
        model = self.make_model()

        self.run_generate(model, "Two buttons.")

        ids = model.prefix.text_emb.call_args.args[0].tolist()[0]
        self.assertEqual(ids[0], 1)
        self.assertEqual(ids[-1], 2)
        self.assertEqual(ids.count(2), 1)

    def test_long_text_gives_one_result_from_every_segment(self):
        model = self.make_model()
        sentence = " ".join(["word"] * 70) + "."

        results = self.run_generate(model, f"{sentence} {sentence}")

        self.assertEqual(model.t2s.generate.call_count, 2)
        self.assertEqual(
            [call.kwargs["seed"] for call in model.t2s.generate.call_args_list],
            [0, 1],
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].token_count, 10)
        self.assertEqual(results[0].samples, 90)
        self.assertTrue(results[0].is_final_chunk)

    def test_short_text_reaches_the_model_as_written(self):
        model = self.make_model()

        self.run_generate(model, "Short text without a stop")

        self.assertEqual(model.t2s.generate.call_count, 1)
        prompt = model._tok.encode.call_args_list[-1].args[0]
        self.assertTrue(prompt.endswith(":Short text without a stop"))


if __name__ == "__main__":
    unittest.main()
