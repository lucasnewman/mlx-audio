from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import numpy as np
import pytest

from mlx_audio.tts.models.llama import llama


def frame_codes(num_frames):
    return [
        offset + frame
        for frame in range(num_frames)
        for offset in (0, 4096, 8192, 12288, 16384, 20480, 24576)
    ]


def decode_to_frame_index(codes):
    frames = mx.repeat(codes[0], 2048, axis=1)
    aligned = (mx.repeat(codes[1], 1024, axis=1) == frames) & (
        mx.repeat(codes[2], 512, axis=1) == frames
    )
    return mx.where(aligned, frames, -1)[..., None].astype(mx.float32)


def test_chunked_decode_returns_each_frame_once():
    num_frames = 70
    codes = frame_codes(num_frames)
    with patch.object(llama.snac_model, "decode", decode_to_frame_index):
        audio = mx.concatenate(
            [
                llama.decode_audio_stream(codes, 0, 39),
                llama.decode_audio_stream(codes, 39, num_frames),
            ],
            axis=1,
        )

    expected = np.repeat(np.arange(num_frames), 2048)
    np.testing.assert_array_equal(np.array(audio[0]), expected)


END_OF_SPEECH = 128258


def run_generate(text, tokens, **kwargs):
    with patch("transformers.AutoTokenizer"):
        model = llama.Model(
            llama.ModelConfig(
                model_type="llama",
                hidden_size=8,
                num_hidden_layers=1,
                intermediate_size=16,
                num_attention_heads=2,
                rms_norm_eps=1e-5,
                vocab_size=16,
                head_dim=4,
                num_key_value_heads=2,
                max_position_embeddings=64,
                rope_theta=10000.0,
                tie_word_embeddings=True,
            )
        )
    segments, budgets, voices = [], [], []

    def prepare_input_ids(prompt, voice, *args, **kwargs):
        segments.append(prompt)
        voices.append(voice)
        return mx.zeros((1, len(prompt.split())), dtype=mx.int32)

    def stream_generate(*args, max_tokens, **kwargs):
        budgets.append(max_tokens)
        for token in tokens:
            yield SimpleNamespace(token=token)

    model.prepare_input_ids = prepare_input_ids
    model.tokenizer = lambda text: SimpleNamespace(input_ids=text.split())
    with (
        patch.object(llama, "stream_generate", stream_generate),
        patch.object(llama.snac_model, "decode", decode_to_frame_index),
    ):
        list(model.generate(text, **kwargs))
    return segments, budgets, voices


def test_warns_only_when_budget_ends_before_end_of_speech(capsys):
    codes = [128257, *[128266 + code for code in range(14)]]

    run_generate("one", codes, max_tokens=len(codes))
    assert "reached max_tokens" in capsys.readouterr().out

    run_generate("one", [*codes, END_OF_SPEECH], max_tokens=len(codes) + 1)
    assert "reached max_tokens" not in capsys.readouterr().out


def test_long_paragraph_is_generated_in_sentence_groups_with_their_own_budget():
    paragraph = " ".join(
        f"Sentence number {i} walks {'on and ' * (i % 5)}on down the long road home."
        for i in range(40)
    )

    segments, budgets, _ = run_generate(paragraph, [END_OF_SPEECH])

    assert len(segments) > 1
    assert all(len(s) <= llama.SEGMENT_MAX_CHARS for s in segments)
    assert all(s.endswith(".") for s in segments)
    assert " ".join(segments) == paragraph
    assert budgets == [
        max(
            llama.MIN_MAX_TOKENS,
            llama.MAX_TOKENS_PER_TEXT_TOKEN * len(s.split()),
        )
        for s in segments
    ]
    assert len(set(budgets)) > 1


def test_run_of_short_sentences_is_split_by_sentence_count():
    dialogue = " ".join(["No.", "Why not?", "Because!"] * 9)

    segments, _, _ = run_generate(dialogue, [END_OF_SPEECH])

    assert [len(llama.SENTENCE_END.findall(s + " ")) for s in segments] == [8, 8, 8, 3]
    assert " ".join(segments) == dialogue


def test_sentence_longer_than_a_segment_is_split_at_a_comma():
    sentence = ", ".join(["the road went on over the hills"] * 15) + "."

    segments, _, _ = run_generate(sentence, [END_OF_SPEECH])

    assert len(segments) > 1
    assert all(len(s) <= llama.SEGMENT_MAX_CHARS for s in segments)
    assert all(s.endswith(",") for s in segments[:-1])
    assert " ".join(segments) == sentence


@pytest.mark.parametrize(
    "text, joiner",
    [
        pytest.param(" ".join(["word"] * 150), " ", id="spaces"),
        pytest.param("a" * 700, "", id="unspaced"),
    ],
)
def test_text_without_sentence_ends_still_fits_the_segment_size(text, joiner):
    segments, _, _ = run_generate(text, [END_OF_SPEECH])

    assert len(segments) > 1
    assert all(len(s) <= llama.SEGMENT_MAX_CHARS for s in segments)
    assert joiner.join(segments) == text


def test_every_segment_without_a_voice_uses_tara():
    paragraph = " ".join(["The road went on and on down the long hill home."] * 20)

    _, _, voices = run_generate(paragraph, [END_OF_SPEECH])

    assert len(voices) > 1
    assert voices == ["tara"] * len(voices)
