"""Reduced-precision model logits must retain a valid sampling nucleus."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import mlx.core as mx
import pytest

from mlx_audio.lm.sample_utils import apply_top_p, make_sampler
from mlx_audio.tts.models.chatterbox.t3 import t3
from mlx_audio.tts.models.dia.dia import _sample_next_token
from mlx_audio.tts.models.indextts.indextts import Model as IndexTTS
from mlx_audio.tts.models.sesame.sesame import SesameModel
from mlx_audio.tts.models.soprano import soprano


@pytest.fixture(params=[mx.float16, mx.bfloat16], ids=["float16", "bfloat16"])
def logits(request):
    # A large shared offset exposes rounding in logsumexp; the masked token
    # also ensures normalization preserves model/logits-processor exclusions.
    return mx.array(
        [[29, 32, 31, 30, -mx.inf], [31, 30, 29, 32, -mx.inf]],
        dtype=request.param,
    )


def _sample_nucleus(logprobs):
    assert mx.allclose(
        mx.exp(logprobs).sum(axis=-1), mx.ones(logprobs.shape[:-1]), atol=1e-5
    ).item()
    assert mx.isneginf(logprobs[..., -1]).all().item()
    # With this distribution, a small top_p must leave exactly the argmax.
    # Rounded normalization used to mask every token for the bfloat16 rows.
    filtered = apply_top_p(logprobs, top_p=0.05)
    assert (mx.isfinite(filtered).sum(axis=-1) == 1).all().item()
    return make_sampler(temp=0.3, top_p=0.05)(logprobs)


def test_dia_reduced_precision_nucleus(logits):
    tokens = _sample_next_token(logits, temperature=0.3, sampler=_sample_nucleus)
    assert tokens.tolist() == [1, 3]


def test_soprano_reduced_precision_nucleus(logits):
    hidden = mx.ones((1, 1, 2))
    model = SimpleNamespace(
        config=SimpleNamespace(num_hidden_layers=0),
        _forward_with_hidden_states=Mock(return_value=(logits[:1, None], hidden)),
        tokenizer=None,
        _stop_token_id=None,
    )
    with patch.object(soprano, "make_sampler", return_value=_sample_nucleus):
        generated = list(
            soprano.Model.stream_generate(
                model, mx.array([0]), max_tokens=2, temperature=0.3, top_p=0.05
            )
        )
    assert [token.item() for token, _ in generated if token is not None] == [1, 1]


def test_chatterbox_reduced_precision_nucleus(logits):
    hidden = mx.ones((1, 1, 2))
    model = SimpleNamespace(
        hp=SimpleNamespace(start_speech_token=0, stop_speech_token=1),
        _prepare_inference_context=lambda **kwargs: hidden,
        tfmr=SimpleNamespace(layers=[], model=lambda **kwargs: hidden),
        speech_head=lambda hidden: logits[:1, None],
    )
    with patch.object(t3, "make_sampler", return_value=_sample_nucleus):
        tokens = t3.T3.inference(
            model,
            t3_cond=None,
            text_tokens=mx.array([0]),
            max_new_tokens=1,
            temperature=0.3,
            top_p=0.05,
            min_p=0,
            repetition_penalty=0,
            cfg_weight=0,
        )
    assert tokens.tolist() == [[0, 1]]


class _IdentityWithCache:
    layers = []

    def __call__(self, hidden, cache=None):
        return hidden


def test_sesame_reduced_precision_nucleus(logits):
    model = SimpleNamespace(
        args=SimpleNamespace(audio_num_codebooks=2),
        caches_are_enabled=lambda: True,
        _embed_tokens=lambda tokens: mx.ones((2, 1, 1, 1), dtype=logits.dtype),
        _embed_audio=lambda codebook, tokens: mx.ones(
            (*tokens.shape, 1), dtype=logits.dtype
        ),
        backbone=_IdentityWithCache(),
        backbone_cache=[],
        codebook0_head=lambda hidden: logits,
        decoder=_IdentityWithCache(),
        projection=lambda hidden: hidden,
        audio_head=logits[:1, None],
    )
    tokens = SesameModel.generate_frame(
        model,
        tokens=mx.zeros((2, 1, 1), dtype=mx.int32),
        tokens_mask=mx.ones((2, 1, 1), dtype=mx.bool_),
        input_pos=mx.zeros((2, 1), dtype=mx.int32),
        sampler=_sample_nucleus,
    )
    assert tokens.tolist() == [[1, 1], [3, 1]]


def test_indextts_reduced_precision_nucleus(logits):
    hidden = mx.ones((1, 1, 2))
    model = SimpleNamespace(
        args=SimpleNamespace(gpt=SimpleNamespace(layers=0, stop_mel_token=1)),
        prepare_input_embedding=lambda *args: hidden,
        gpt=lambda inputs, cache: hidden,
        final_norm=lambda hidden: hidden,
        mel_head=lambda hidden: logits[:1, None],
        bigvgan=lambda latents, ref_mel: mx.zeros((1, 1, 8)),
        generate_result=lambda audio, start_time, token_count: token_count,
    )
    generated = list(
        IndexTTS.generate(
            model,
            "Hello",
            ref_audio=None,
            ref_mel=mx.zeros((1, 2, 2)),
            max_tokens=1,
            sampler=_sample_nucleus,
        )
    )
    assert generated == [1]
