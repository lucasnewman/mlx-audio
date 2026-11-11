from types import SimpleNamespace

import mlx.core as mx

from mlx_audio.stt.models.nemo.alignment import (
    AlignedToken,
    sentences_to_result,
    tokens_to_sentences,
)
from mlx_audio.stt.models.parakeet.parakeet import Model

SAMPLE_RATE = 16000


class ScriptedWindows(Model):
    def __init__(self, windows):
        super().__init__(SimpleNamespace(sample_rate=SAMPLE_RATE))
        self.windows = iter(windows)

    def _log_mel_spectrogram(self, audio_data):
        return audio_data

    def decode(self, mel):
        tokens = [AlignedToken(*token) for token in next(self.windows)]
        return [sentences_to_result(tokens_to_sentences(tokens))]


def test_deltas_join_to_the_final_text_when_a_merge_rewrites_the_tail():
    model = ScriptedWindows(
        [
            [
                (1, "The", 0.0, 0.5),
                (2, " bell", 3.6, 0.4),
                (3, " rang", 4.1, 0.4),
                (4, ".", 4.6, 0.3),
            ],
            [
                (3, " rang", 0.1, 0.4),
                (5, " seven", 0.7, 0.4),
                (6, " times", 1.2, 0.4),
            ],
        ]
    )
    audio = mx.ones((7 * SAMPLE_RATE,), dtype=mx.float32)

    results = list(
        model.generate(audio, chunk_duration=5.0, overlap_duration=1.0, stream=True)
    )

    assert "".join(result.text for result in results) == "The bell rang seven times"
    assert results[0].text == "The"
