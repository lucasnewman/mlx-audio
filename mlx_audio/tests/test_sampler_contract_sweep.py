"""Follow-up sampling contract tests: categorical takes logits, processors run
before normalization, and ``top_k >= vocab`` is a no-op filter everywhere."""

import ast
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx_audio.lm.sample_utils import apply_top_k


def test_apply_top_k_accepts_vocab_size_as_no_filter():
    """The error message documents (0, vocab]; a k that keeps every token must
    not raise, matching the no-op convention other model samplers already use."""
    logprobs = mx.log(mx.array([0.5, 0.3, 0.15, 0.05]))
    out = apply_top_k(logprobs, 4)
    assert mx.allclose(out, logprobs)


def test_apply_top_k_above_vocab_is_also_a_no_filter():
    logprobs = mx.log(mx.array([0.5, 0.3, 0.15, 0.05]))
    out = apply_top_k(logprobs, 10)
    assert mx.allclose(out, logprobs)


def test_apply_top_k_still_rejects_non_positive_values():
    logprobs = mx.log(mx.array([0.5, 0.3, 0.15, 0.05]))
    for bad in (0, -1):
        with pytest.raises(ValueError, match="top_k"):
            apply_top_k(logprobs, bad)


def test_apply_top_k_filters_below_vocab():
    logprobs = mx.log(mx.array([0.5, 0.3, 0.15, 0.05]))
    out = apply_top_k(logprobs, 2)
    assert out[2].item() == -float("inf")
    assert out[3].item() == -float("inf")
    assert mx.allclose(out[:2], logprobs[:2])


def test_moshi_sampler_logprobs_normalize_per_row():
    """mx.logsumexp without an axis reduces the whole tensor; batched CFG logits
    need per-row log-probs."""
    from mlx_audio.sts.models.moshi.utils.sampling import Sampler

    logits = mx.array([[2.0, 0.0, -1.0], [0.0, 1.0, 3.0]])
    _, logprobs = Sampler()(logits)
    sums = mx.exp(logprobs).sum(axis=-1)
    assert np.allclose(np.asarray(sums), [1.0, 1.0])


def test_mimo_sampler_categorical_tracks_the_true_distribution():
    """categorical() consumes logits; feeding it softmax probabilities flattens
    a 90/5/5 distribution toward 54/23/23, so token 0 must dominate at ~90%."""
    from mlx_audio.stt.models.mimo_v2_asr.model import MiMoSampler

    sampler = MiMoSampler(do_sample=True)
    scores = mx.log(mx.array([[0.9, 0.05, 0.05]] * 4)).astype(mx.float32)
    mx.random.seed(0)
    draws = sampler.sample(scores)
    mx.eval(draws)
    draws = np.asarray(draws)
    for _ in range(50):
        draws = np.concatenate([draws, np.asarray(sampler.sample(scores))])
    freq = float((draws == 0).mean())
    assert freq > 0.8


def test_mimo_removed_tokens_ban_applies_and_does_not_mutate():
    """process() can return the input unchanged: banning argmax must drop to
    the next token without punching -inf into the caller's array."""
    from mlx_audio.stt.models.mimo_v2_asr.model import MiMoSampler

    scores = mx.array([[0.0, 1.0, 3.0, -1.0]], dtype=mx.float32)
    before = np.asarray(scores)
    out = MiMoSampler(do_sample=False, temperature=0.0).sample(
        scores, removed_tokens=[2]
    )
    mx.eval(scores)
    assert out.tolist() == [1]
    assert np.array_equal(np.asarray(scores), before)
    assert MiMoSampler(do_sample=False, temperature=0.0).sample(scores).tolist() == [2]


def test_mimo_removed_tokens_ignores_out_of_range_ids():
    """Negative or too-large ids must not wrap around and ban a valid token."""
    from mlx_audio.stt.models.mimo_v2_asr.model import MiMoSampler

    scores = mx.array([[0.0, 1.0, 3.0, -1.0]], dtype=mx.float32)
    out = MiMoSampler(do_sample=False, temperature=0.0).sample(
        scores, removed_tokens=[-1, 99]
    )
    assert out.tolist() == [2]


def test_vibevoice_processors_run_on_logits_before_normalization():
    """Logits processors are sign-aware; running them on log-probs silently
    picks the wrong branch. They must see next_logits, then normalize."""
    source = (
        Path(__file__).parents[1]
        / "stt"
        / "models"
        / "vibevoice_asr"
        / "vibevoice_asr.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    processor_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "processor"
    ]
    assert processor_calls, "expected processor(previous, ...) calls"
    normalize_lines = {
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "logsumexp"
    }
    for call in processor_calls:
        second_arg = call.args[1] if len(call.args) > 1 else None
        assert isinstance(second_arg, ast.Name) and second_arg.id == "next_logits"
        assert any(
            call.lineno < n for n in normalize_lines
        ), "processors must run before the logsumexp normalization"


def test_moss_merge_hotwords_applies_once():
    """A duplicated import + call left hotwords injected into the prompt twice."""
    source = (
        Path(__file__).parents[1]
        / "stt"
        / "models"
        / "moss_transcribe_diarize"
        / "moss_transcribe_diarize.py"
    ).read_text(encoding="utf-8")
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "merge_hotwords"
    ]
    assert len(calls) == 1


def test_moshi_top_k_sampling_returns_a_token_at_vocab_boundary():
    """The function's contract is a sampled token; a no-op early return would
    hand callers the whole (batch, vocab) logprobs array."""
    from mlx_audio.sts.models.moshi.utils.sampling import top_k_sampling

    logits = mx.array([[2.0, 0.0, -1.0], [0.0, 1.0, 3.0]])
    for k in (3, 50):
        token = top_k_sampling(logits, k, 1.0)
        assert token.shape == (2,)
        assert token.dtype in (mx.int32, mx.int64, mx.uint32)


def test_moshi_logit_bias_does_not_mutate_caller_logits():
    from mlx_audio.sts.models.moshi.utils.sampling import Sampler

    logits = mx.array([[1.0, 0.0, -1.0]])
    before = np.asarray(logits)
    Sampler(logit_bias={0: 5.0})(logits)
    mx.eval(logits)
    assert np.array_equal(np.asarray(logits), before)


def test_lfm3_suppress_applies_and_does_not_mutate():
    """astype on same-dtype input can alias: suppressing argmax must drop to
    the next token without punching -inf into the caller's tensor."""
    from mlx_audio.sts.models.lfm3_audio.depth import sample

    logits = mx.array([[0.0, 1.0, 10.0, -1.0]], dtype=mx.float32)
    before = np.asarray(logits)
    out = sample(logits, temperature=0, suppress={2})
    mx.eval(logits, out)
    assert int(out) != 2
    assert np.array_equal(np.asarray(logits), before)


def test_fish_drama3_top_k_zero_keeps_full_candidate_set():
    """top_k=0 is the repo-wide 'disabled' sentinel; it used to build an empty
    candidate list and crash categorical. Draws must span more than argmax."""
    from mlx_audio.tts.models.fish_drama3.drama3 import sample_top_k_top_p

    logits = mx.array([[2.0, 1.9, -1.0, 1.8]])
    mx.random.seed(0)
    draws = {int(sample_top_k_top_p(logits, 0.7, 0, 0.9)[0]) for _ in range(60)}
    assert len(draws) > 1


def test_arktts_top_k_zero_does_not_collapse_to_argmax():
    """top_k=0 must disable the filter (repo convention), not secretly keep
    only the top token."""
    from mlx_audio.tts.models.arktts.arktts import ArkttsModel

    scores = mx.array([[1.0, 0.5, -1.0]])
    out = ArkttsModel._legacy_top_k_top_p(scores, 0, 1.0)
    # argmax is always kept; under the bug it was the only survivor
    assert np.isfinite(np.asarray(out)).sum() >= 2


def test_confucius4_temperature_zero_is_greedy():
    """temperature=0 used to divide by zero, producing NaN probs."""
    from mlx_audio.tts.models.confucius4.t2s import T2SMLX

    logits = np.array([0.5, 3.0, -1.0, 0.1])
    out = T2SMLX._sample(None, logits, [], 0.0, 30, 0.9, 1.0, np.random)
    assert out == 1


def test_chatterbox_turbo_temperature_zero_is_greedy():
    """temperature=0 used to fall through to categorical over raw logits."""
    from mlx_audio.tts.models.chatterbox_turbo.models.t3.t3 import T3

    logits = mx.array([[0.5, 3.0, -1.0, 2.9], [4.0, 0.0, 1.0, 3.9]])
    for _ in range(20):
        out = T3._sample_token(None, logits, 0.0, 0, 1.0, None, 1.0)
        assert out.shape == (2, 1)
        assert out.tolist() == [[1], [0]]


def test_tada_temperature_zero_is_greedy():
    """token_logits / 0 produced inf -> NaN softmax; it now argmaxes first."""
    from mlx_audio.tts.models.tada.tada import Model

    logits = mx.array([[[0.5, 3.0, -1.0, 2.9]]])
    input_ids = mx.array([[0]])
    out = Model._sample_next_token(
        None, logits, input_ids, temperature=0.0, repetition_penalty=1.0
    )
    assert out.shape == (1, 1)
    assert out.tolist() == [[1]]
