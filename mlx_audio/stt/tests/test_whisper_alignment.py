from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx_audio.stt.models.whisper import timing, whisper
from mlx_audio.stt.models.whisper.decoding import DecodingResult
from mlx_audio.stt.models.whisper.whisper import AudioEncoder, Model, ModelDimensions


@pytest.mark.parametrize("dtype", [mx.float16, mx.float32])
def test_alignment_reuses_audio_features(monkeypatch, dtype):
    mx.random.seed(0)
    dims = ModelDimensions(8, 6, 16, 2, 1, 64, 40, 16, 2, 2)
    model = Model(dims, dtype=dtype)
    model.decoder.positional_embedding = mx.random.normal((40, 16)).astype(dtype)
    model.set_dtype(dtype)
    mel = mx.random.normal((2, 12, 8)).astype(dtype)
    tokens = mx.array([[1, 2, 3], [4, 5, 6]])
    expected_logits, expected_qk = model.forward_with_cross_qk(mel, tokens)
    features = model.embed_audio(mel)
    mx.eval(expected_logits, expected_qk, features)
    assert features.dtype == dtype
    assert expected_logits.dtype == dtype
    assert all(heads.dtype == dtype for heads in expected_qk)

    def unexpected_encoder(*args):
        pytest.fail("Alignment must reuse the existing encoder output")

    monkeypatch.setattr(AudioEncoder, "__call__", unexpected_encoder)
    logits, qk = model.forward_with_cross_qk(mel, tokens, audio_features=features)
    np.testing.assert_array_equal(np.array(logits), np.array(expected_logits))
    for actual, expected in zip(qk, expected_qk):
        np.testing.assert_array_equal(np.array(actual), np.array(expected))


@pytest.mark.parametrize("dtype", [mx.float16, mx.float32])
def test_generate_reuses_audio_features_for_word_timestamps(monkeypatch, dtype):
    mx.random.seed(0)
    dims = ModelDimensions(8, 6, 16, 2, 1, 64, 40, 16, 2, 2)
    model = Model(dims, dtype=dtype)
    model.decoder.positional_embedding = mx.random.normal((40, 16))
    model.set_dtype(dtype)
    window_frames = 12
    mel_frames = 2 * window_frames
    mel = mx.random.normal((mel_frames, 8)).astype(dtype)
    tokenizer = SimpleNamespace(
        sot_sequence=(0,),
        no_timestamps=19,
        eot=20,
        timestamp_begin=21,
        decode=lambda tokens: " hello" if 2 in tokens else "",
        split_to_word_tokens=lambda tokens: ([" hello", ""], [[2], [20]]),
    )
    monkeypatch.setattr(whisper, "N_FRAMES", window_frames)
    monkeypatch.setattr(model, "_prepare_audio", lambda audio: (mel, mel_frames))
    monkeypatch.setattr(model, "_detect_language", lambda mel, **kwargs: "en")
    monkeypatch.setattr(model, "get_tokenizer", lambda **kwargs: tokenizer)
    decoded_windows = []

    def decode(segment, options):
        features = model.embed_audio(segment[None, :])[0]
        decoded_windows.append((segment, features))
        return DecodingResult(
            audio_features=features,
            language="en",
            tokens=[21, 2, 26],
            text="hello",
            avg_logprob=0.0,
            no_speech_prob=0.0,
            temperature=0.0,
            compression_ratio=1.0,
        )

    monkeypatch.setattr(model, "decode", decode)
    monkeypatch.setattr(timing, "dtw", lambda matrix: np.array([[0, 1], [0, 5]]))
    encoder_calls = []
    original_encoder = AudioEncoder.__call__

    def encode(self, value):
        encoder_calls.append(value.shape)
        return original_encoder(self, value)

    monkeypatch.setattr(AudioEncoder, "__call__", encode)
    aligned_windows = []
    original_forward = model.forward_with_cross_qk

    def forward_with_cross_qk(mel, tokens, **kwargs):
        aligned_windows.append((mel[0], kwargs.get("audio_features")))
        return original_forward(mel, tokens, **kwargs)

    monkeypatch.setattr(model, "forward_with_cross_qk", forward_with_cross_qk)
    result = model.generate(
        mx.zeros(mel_frames * whisper.HOP_LENGTH), language="en", word_timestamps=True
    )

    assert len(encoder_calls) == 2
    assert len(aligned_windows) == 2
    first_features, second_features = (features for _, features in decoded_windows)
    assert not np.array_equal(np.array(first_features), np.array(second_features))
    for (decoded_mel, decoded_features), (aligned_mel, aligned_features) in zip(
        decoded_windows, aligned_windows
    ):
        np.testing.assert_array_equal(np.array(aligned_mel), np.array(decoded_mel))
        np.testing.assert_array_equal(
            np.array(aligned_features[0]), np.array(decoded_features)
        )
    assert len(result.segments) == 2
    assert [
        [word["word"] for word in segment["words"]] for segment in result.segments
    ] == [[" hello"], [" hello"]]
