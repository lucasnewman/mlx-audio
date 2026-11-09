import mlx.core as mx
import numpy as np
import pytest

from mlx_audio.stt.models.whisper.decoding import Inference
from mlx_audio.stt.models.whisper.whisper import (
    Model,
    ModelDimensions,
    SelfAttentionCache,
)

DIMS = ModelDimensions(
    n_mels=8,
    n_audio_ctx=6,
    n_audio_state=16,
    n_audio_head=2,
    n_audio_layer=1,
    n_vocab=64,
    n_text_ctx=40,
    n_text_state=16,
    n_text_head=2,
    n_text_layer=2,
)


@pytest.fixture
def model():
    mx.random.seed(0)
    m = Model(DIMS, dtype=mx.float32)
    m.decoder.positional_embedding = mx.random.normal(
        (DIMS.n_text_ctx, DIMS.n_text_state)
    )
    mx.eval(m.parameters())
    return m


@pytest.fixture
def audio_features():
    mx.random.seed(1)
    return mx.random.normal((3, DIMS.n_audio_ctx, DIMS.n_audio_state))


def _tokens(batch, length, seed=2):
    rng = np.random.default_rng(seed)
    return mx.array(rng.integers(0, DIMS.n_vocab, (batch, length)))


def _full_logits(model, tokens, xa):
    logits, _, _ = model.decoder(tokens, xa)
    return np.array(logits)


def test_incremental_decoding_matches_full_forward(model, audio_features):
    """One token at a time through the cache == one causal pass over all tokens."""
    tokens = _tokens(3, 20)
    full = _full_logits(model, tokens, audio_features)

    inf = Inference(model)
    steps = [np.array(inf.logits(tokens[:, :4], audio_features))]
    for t in range(4, tokens.shape[1]):
        steps.append(np.array(inf.logits(tokens[:, t : t + 1], audio_features)))
    incremental = np.concatenate(steps, axis=1)

    np.testing.assert_allclose(incremental, full, atol=1e-4, rtol=1e-4)


def test_multi_token_chunk_after_prefix_is_causal(model, audio_features):
    """A multi-token call at offset > 0 attends causally across the whole cache."""
    tokens = _tokens(3, 12)
    full = _full_logits(model, tokens, audio_features)

    inf = Inference(model)
    first = np.array(inf.logits(tokens[:, :5], audio_features))
    rest = np.array(inf.logits(tokens[:, 5:], audio_features))

    np.testing.assert_allclose(
        np.concatenate([first, rest], axis=1), full, atol=1e-4, rtol=1e-4
    )


def test_cache_grows_past_step_boundary(model, audio_features, monkeypatch):
    monkeypatch.setattr(SelfAttentionCache, "step", 4)
    tokens = _tokens(3, 15)
    full = _full_logits(model, tokens, audio_features)

    inf = Inference(model)
    steps = [np.array(inf.logits(tokens[:, :1], audio_features))]
    for t in range(1, tokens.shape[1]):
        steps.append(np.array(inf.logits(tokens[:, t : t + 1], audio_features)))

    cache = inf.kv_cache[0][0]
    assert cache.offset == 15
    assert cache.keys.shape[2] == 16  # four 4-token steps
    np.testing.assert_allclose(
        np.concatenate(steps, axis=1), full, atol=1e-4, rtol=1e-4
    )


def test_rearrange_kv_cache_reorders_rows(model, audio_features):
    tokens = _tokens(3, 8)
    inf = Inference(model)
    inf.logits(tokens[:, :7], audio_features)
    order = [2, 0, 0]
    inf.rearrange_kv_cache(order)
    reordered = tokens[mx.array(order)]
    got = np.array(inf.logits(reordered[:, 7:8], audio_features))

    full = _full_logits(model, reordered, audio_features[mx.array(order)])
    np.testing.assert_allclose(got[:, -1], full[:, -1], atol=1e-4, rtol=1e-4)


def test_reset_starts_a_fresh_cache(model, audio_features):
    tokens = _tokens(3, 6)
    inf = Inference(model)
    first = np.array(inf.logits(tokens, audio_features))
    inf.reset()
    again = np.array(inf.logits(tokens, audio_features))
    np.testing.assert_allclose(again, first, atol=1e-6)
