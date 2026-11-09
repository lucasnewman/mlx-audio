import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import numpy as np

from mlx_audio.tts.models.confucius4.confucius4 import Model


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


if __name__ == "__main__":
    unittest.main()
