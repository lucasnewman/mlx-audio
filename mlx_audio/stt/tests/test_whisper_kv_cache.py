from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest

from mlx_audio.stt.models.whisper.decoding import Inference
from mlx_audio.stt.models.whisper.whisper import (
    Model,
    ModelDimensions,
    MultiHeadAttention,
    SelfAttentionCache,
)

# Allow float32 rounding differences between full-sequence and incremental
# attention kernels on Apple GPUs.
LOGIT_ATOL = 3e-3
LOGIT_RTOL = 1e-3

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

    np.testing.assert_allclose(incremental, full, atol=LOGIT_ATOL, rtol=LOGIT_RTOL)


def test_multi_token_chunk_after_prefix_is_causal(model, audio_features):
    """A multi-token call at offset > 0 attends causally across the whole cache."""
    tokens = _tokens(3, 12)
    full = _full_logits(model, tokens, audio_features)

    inf = Inference(model)
    first = np.array(inf.logits(tokens[:, :5], audio_features))
    rest = np.array(inf.logits(tokens[:, 5:], audio_features))

    np.testing.assert_allclose(
        np.concatenate([first, rest], axis=1), full, atol=LOGIT_ATOL, rtol=LOGIT_RTOL
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
        np.concatenate(steps, axis=1), full, atol=LOGIT_ATOL, rtol=LOGIT_RTOL
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
    np.testing.assert_allclose(
        got[:, -1], full[:, -1], atol=LOGIT_ATOL, rtol=LOGIT_RTOL
    )


def test_reset_starts_a_fresh_cache(model, audio_features):
    tokens = _tokens(3, 6)
    inf = Inference(model)
    first = np.array(inf.logits(tokens, audio_features))
    inf.reset()
    again = np.array(inf.logits(tokens, audio_features))
    np.testing.assert_allclose(again, first, atol=1e-6)


class _LegacyCrossAttention(MultiHeadAttention):
    """Original attention formula, with raw keys in the cross-attention cache."""

    def __call__(self, x, xa, mask=None, kv_cache=None):
        q = self.query(x)
        k, v = kv_cache if kv_cache is not None else (self.key(xa), self.value(xa))
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.reshape(*q.shape[:2], self.n_head, -1).transpose(0, 2, 1, 3) * scale
        kh = k.reshape(*k.shape[:2], self.n_head, -1).transpose(0, 2, 3, 1) * scale
        vh = v.reshape(*v.shape[:2], self.n_head, -1).transpose(0, 2, 1, 3)
        qk = q @ kh
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        w = mx.softmax(qk, axis=-1, precise=True)
        out = (w @ vh).transpose(0, 2, 1, 3).reshape(n_batch, n_ctx, n_state)
        return self.out(out), (k, v), qk


@pytest.mark.parametrize("dtype", [mx.float16, mx.float32])
def test_cross_attention_cache_matches_legacy_formula(dtype):
    """Prescaling preserves exact logits and timing scores through cache reuse."""
    # Whisper's real head width and audio context exercise the layouts used
    # during decoding, including after a gather for beam reordering.
    dims = replace(
        DIMS,
        n_audio_ctx=1500,
        n_audio_state=384,
        n_audio_head=6,
        n_text_state=384,
        n_text_head=6,
    )
    mx.random.seed(1196)
    model = Model(dims, dtype=dtype)
    model.decoder.positional_embedding = mx.random.normal(
        (dims.n_text_ctx, dims.n_text_state)
    )
    model.set_dtype(dtype)
    reference = Model(dims, dtype=dtype)
    for block in reference.decoder.blocks:
        block.cross_attn = _LegacyCrossAttention(dims.n_text_state, dims.n_text_head)
    reference.update(model.parameters())
    mx.eval(model.parameters(), reference.parameters())

    features = mx.random.normal((3, dims.n_audio_ctx, dims.n_audio_state)).astype(dtype)
    tokens = _tokens(3, 10)
    actual, expected = Inference(model), Inference(reference)
    scale = (dims.n_text_state // dims.n_text_head) ** -0.25

    def check_step(start, stop, stage):
        previous = (
            [cross_cache for _, cross_cache in actual.kv_cache]
            if actual.kv_cache is not None
            else None
        )
        logits, qk = actual.logits_with_cross_qk(tokens[:, start:stop], features)
        ref_logits, ref_qk = expected.logits_with_cross_qk(
            tokens[:, start:stop], features
        )
        np.testing.assert_array_equal(
            np.array(logits), np.array(ref_logits), err_msg=stage
        )
        assert len(qk) == len(ref_qk) == dims.n_text_layer
        for layer, (scores, ref_scores) in enumerate(zip(qk, ref_qk)):
            np.testing.assert_array_equal(
                np.array(scores),
                np.array(ref_scores),
                err_msg=f"{stage}, layer {layer}",
            )
            keys, values = actual.kv_cache[layer][1]
            raw_keys, raw_values = expected.kv_cache[layer][1]
            assert keys.shape == (3, dims.n_audio_ctx, dims.n_text_state)
            assert keys.dtype == values.dtype == dtype
            np.testing.assert_array_equal(np.array(keys), np.array(raw_keys * scale))
            np.testing.assert_array_equal(np.array(values), np.array(raw_values))
            if previous is not None:
                # Reusing a cache must not project or scale its arrays again.
                assert keys is previous[layer][0]
                assert values is previous[layer][1]

    check_step(0, 4, "cache creation")
    check_step(4, 5, "first cache reuse")
    check_step(5, 6, "second cache reuse")

    order = [2, 0, 0]  # Include a duplicate beam and distinct audio rows.
    actual.rearrange_kv_cache(order)
    # Reorder the raw reference cache independently of the production helper.
    expected.kv_cache = [
        (self_cache.select(order), (keys[order], values[order]))
        for self_cache, (keys, values) in expected.kv_cache
    ]
    tokens = tokens[mx.array(order)]
    features = features[mx.array(order)]
    check_step(6, 7, "cache reordering")
    check_step(7, 8, "reuse after reordering")
    check_step(8, 10, "multi-token reuse after reordering")
