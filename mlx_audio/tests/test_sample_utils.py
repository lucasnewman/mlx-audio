"""Contract tests for mlx_audio.lm.sample_utils samplers.

``make_sampler`` returns a sampler whose input must be normalized
log-probabilities: ``apply_top_p``/``apply_min_p`` run ``exp``/``log`` over the
input, so ``sum(exp(x))`` must already equal 1. Raw logits are silently wrong
here, which is why callers normalize with ``log_softmax`` or
``logits - logsumexp(logits)`` before sampling.
"""

import mlx.core as mx

from mlx_audio.lm.sample_utils import apply_top_p, make_sampler


def test_apply_top_p_masks_tail_of_normalized_logprobs():
    """Input is log-probs: the least likely token falls below 1 - top_p."""
    probs = mx.array([0.5, 0.3, 0.15, 0.05])
    logprobs = mx.log(probs)
    out = apply_top_p(logprobs, top_p=0.9)
    # Ascending cumulative probs: .05, .20, .50, 1.00. Tokens kept where
    # cumulative prob > 1 - top_p = .1, so only the .05 token is dropped.
    assert out[3].item() == -float("inf")
    assert mx.allclose(out[:3], logprobs[:3])


def test_apply_top_p_masks_per_row_on_batched_logprobs():
    """The nucleus is computed independently per row over the last axis."""
    probs = mx.array([[0.5, 0.3, 0.15, 0.05], [0.05, 0.5, 0.3, 0.15]])
    logprobs = mx.log(probs)
    out = apply_top_p(logprobs, top_p=0.9)
    assert out[0, 3].item() == -float("inf")
    assert out[1, 0].item() == -float("inf")
    assert mx.allclose(out[0, :3], logprobs[0, :3])
    assert mx.allclose(out[1, 1:], logprobs[1, 1:])


def test_apply_top_p_tight_threshold_keeps_only_top_token():
    """top_p below the runner-up's cumulative mass leaves a single survivor."""
    probs = mx.array([0.9, 0.05, 0.05])
    out = apply_top_p(mx.log(probs), top_p=0.5)
    assert out[0].item() != -float("inf")
    assert out[1].item() == -float("inf")
    assert out[2].item() == -float("inf")


def test_make_sampler_top_p_only_draws_from_nucleus():
    """End to end: a make_sampler sampler never returns a masked token.

    The sampler contract is log-probabilities in, token out; here top_p prunes
    every token but the argmax, so sampling must always return token 0.
    """
    logprobs = mx.log(mx.array([0.97, 0.01, 0.01, 0.01]))
    sampler = make_sampler(temp=1.0, top_p=0.5)
    for _ in range(32):
        assert sampler(logprobs).item() == 0
