"""Speaker masks, independent caches, absolute time and PCM partition invariance."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx_audio.stt.models.nemotron_asr.speaker_streaming import (
    mask_features,
    speaker_activity,
)
from mlx_audio.stt.tests.test_nemotron_asr import _build_tiny


class Diarizer:
    """Deterministic native-frame activity with a delayed, committed timeline."""

    def __init__(self, probs=None):
        self.probs = probs
        self.config = SimpleNamespace(
            processor_config=SimpleNamespace(sampling_rate=16000, hop_length=160),
            output_subsampling_factor=1,
            num_speakers=2,
        )

    def init_streaming_state(self):
        return SimpleNamespace(samples=0, frames=0)

    def feed(self, audio, state, *, final, **kwargs):
        state.samples += len(audio)
        end = state.samples // 160
        if not final:
            end = max(0, (end - 24) // 72 * 72)
        if self.probs is None:
            probs = mx.ones((end - state.frames, 2))
        else:
            probs = mx.array(self.probs[state.frames : end])
        state.frames = end
        return SimpleNamespace(speaker_probs=probs), state


def emitting_model():
    model = _build_tiny()
    model.max_symbols = 1
    model.joint = lambda enc, pred: mx.array([0.0, 0.0, 1.0] + [0.0] * 6)
    return model


def signature(deltas):
    result = {}
    for delta in deltas:
        result.setdefault(delta.speaker, []).extend(
            (token.id, token.text, token.start, token.end) for token in delta.tokens
        )
    return result


def test_mask_matches_reference_and_averages_before_threshold():
    probs = mx.array([[1.0, 0.0]] * 5 + [[0.0, 1.0]] * 3 + [[0.9, 0.9]])
    active = speaker_activity(probs, 8, 0.5)
    assert np.array(active).tolist() == [[True, False], [False, False]]
    mel = mx.array([[[-2.0, 0.0]] * 9])
    masked = np.array(mask_features(mel, active[:, 0], 8))
    np.testing.assert_allclose(masked[0, :8, 0], -2.0)
    assert masked[0, 8, 0] == 0.0
    np.testing.assert_allclose(masked[0, :, 1], -16.6355)


def test_all_active_stream_matches_unmasked_decoder_and_has_separate_caches():
    mx.random.seed(31)
    model = _build_tiny()
    mel = mx.random.normal((1, 231, 80))
    baseline = model.decode(mel)
    session = model.create_speaker_streaming_session(Diarizer())
    deltas = session._push_features(mel, mx.ones((231, 2)), final=True)
    expected = [
        (t.id, t.text, t.start, t.end) for s in baseline.sentences for t in s.tokens
    ]
    assert signature(deltas) == {"speaker_0": expected, "speaker_1": expected}
    first, second = session._encoders.values()
    assert first is not second
    assert first.attn_cache[0] is not second.attn_cache[0]
    assert first.conv_cache[0] is not second.conv_cache[0]
    assert session._decoders[0].hidden is not session._decoders[1].hidden


def test_gating_freezes_state_and_reactivation_keeps_absolute_time():
    session = emitting_model().create_speaker_streaming_session(Diarizer())
    mel = mx.full((1, 112, 80), -2.0)
    first = mx.tile(mx.array([[1.0, 0.0]]), (112, 1))
    silent = mx.zeros((112, 2))
    assert session._push_features(mel, first)[0].speaker == "speaker_0"
    # One extra chunk is decoded for delayed emissions, then the cache freezes.
    assert session._push_features(mel, silent)
    cache = session._encoders[0].attn_cache[0]
    hidden = session._decoders[0].hidden
    assert session._push_features(mel, silent) == []
    assert session._encoders[0].attn_cache[0] is cache
    assert session._decoders[0].hidden is hidden
    # A previously unseen speaker starts at the recording's time, not zero.
    second = mx.tile(mx.array([[0.0, 1.0]]), (112, 1))
    delta = session._push_features(mel, second)[0]
    assert delta.speaker == "speaker_1"
    assert delta.tokens[0].start == pytest.approx(3 * 1.12)
    resumed = session._push_features(mel, first)[0]
    assert resumed.speaker == "speaker_0"
    assert resumed.tokens[0].start == pytest.approx(4 * 1.12)
    assert session._decoders[0].hidden is not hidden


@pytest.mark.parametrize("length", [1, 159, 160, 17920, 36001])
def test_arbitrary_pcm_partitions_and_flush(length):
    mx.random.seed(7)
    model = _build_tiny()
    audio = np.random.default_rng(9).normal(0, 0.02, length).astype(np.float32)
    whole = list(model.stream_generate_speakers(audio, Diarizer()))
    session = model.create_speaker_streaming_session(Diarizer())
    split = []
    for start in range(0, length, 317):
        split.extend(session.feed(audio[start : start + 317]))
    split.extend(session.feed([], final=True))
    assert signature(whole) == signature(split)
    assert session.done
    assert session._mel.shape[1] == session._probs.shape[0] == 0
    with pytest.raises(RuntimeError):
        session.feed([])


def test_silence_empty_reset_validation_and_bounded_buffers():
    model = emitting_model()
    diar = Diarizer(np.zeros((3000, 2), np.float32))
    session = model.create_speaker_streaming_session(diar)
    for _ in range(100):
        assert session.feed(np.zeros(3200, np.float32)) == []
        assert session._frontend.buffered_samples < 1024
        assert session._mel.shape[1] < session.chunk_mel + 100
        assert session._probs.shape[0] < session.chunk_mel
    assert not session._encoders
    assert session.feed([], final=True) == []
    session.reset()
    assert not session.done
    assert session.feed([], final=True) == []
    session.reset()
    for invalid in ([float("nan")], [[0.0]]):
        with pytest.raises(ValueError, match="finite mono"):
            session.feed(invalid)
    for options in ({"threshold": 0}, {"cache_gating_buffer_size": 0}):
        with pytest.raises(ValueError):
            model.create_speaker_streaming_session(diar, **options)
    diar.config.processor_config.hop_length = 320
    with pytest.raises(ValueError, match="same sample rate"):
        model.create_speaker_streaming_session(diar)


def test_masks_are_applied_before_encoder_and_gating_can_be_disabled():
    session = emitting_model().create_speaker_streaming_session(
        Diarizer(), cache_gating=False
    )
    mel = mx.full((1, 112, 80), -3.0)
    session._push_features(mel, mx.tile(mx.array([[1.0, 0.0]]), (112, 1)))
    np.testing.assert_allclose(np.array(session._encoders[0].mel_cache), -3.0)
    np.testing.assert_allclose(np.array(session._encoders[1].mel_cache), 0.0)


def test_sessions_do_not_share_diarization_or_transcript_state():
    model = emitting_model()
    diar = Diarizer()
    first = model.create_speaker_streaming_session(diar)
    second = model.create_speaker_streaming_session(diar)
    assert first.feed(np.zeros(30000, np.float32))
    assert second._diar_state.samples == 0
    assert second._encoders == {}
    results = model.generate_speakers(np.zeros(30000, np.float32), diar)
    assert set(results) == {"speaker_0", "speaker_1"}
    assert all(result.text for result in results.values())
