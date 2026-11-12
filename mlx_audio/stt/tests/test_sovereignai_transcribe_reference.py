"""Optional numerical checks against Transformers, with tiny random weights."""

from dataclasses import asdict

import mlx.core as mx
import numpy as np
import pytest

from mlx_audio.stt.models.sovereignai_transcribe import Model
from mlx_audio.stt.tests.test_sovereignai_transcribe import tiny_config


def test_audio_encoder_matches_transformers():
    torch = pytest.importorskip("torch")
    from transformers import WhisperConfig
    from transformers.models.whisper.modeling_whisper import WhisperEncoder

    config = tiny_config()
    torch.manual_seed(4)
    reference = WhisperEncoder(WhisperConfig(**asdict(config.audio_config))).eval()
    model = Model(config)
    weights = {
        "model.audio_tower." + key: mx.array(value.detach().numpy())
        for key, value in reference.state_dict().items()
    }
    model.load_weights(list(model.sanitize(weights).items()), strict=False)
    features = np.random.default_rng(4).normal(size=(2, 4, 16)).astype(np.float32)
    with torch.no_grad():
        expected = reference(torch.from_numpy(features)).last_hidden_state.numpy()
    with mx.stream(mx.cpu):
        actual = model.model.audio_tower(mx.array(features))
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_moe_decoder_matches_transformers():
    torch = pytest.importorskip("torch")
    from transformers import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

    config = tiny_config()
    reference_config = asdict(config.text_config)
    # Legacy top-level aliases override the nested partial-RoPE parameters in
    # Transformers. The real SOV checkpoint uses only the nested parameters.
    reference_config.pop("partial_rotary_factor")
    reference_config.pop("global_partial_rotary_factor")
    torch.manual_seed(5)
    reference = Gemma4TextModel(Gemma4TextConfig(**reference_config)).eval()
    model = Model(config)
    weights = {
        "model.language_model." + key: mx.array(value.detach().numpy())
        for key, value in reference.state_dict().items()
    }
    model.load_weights(list(model.sanitize(weights).items()), strict=False)
    model.eval()
    ids = np.array([[2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13]], dtype=np.int64)
    with torch.no_grad():
        hidden = reference(torch.from_numpy(ids)).last_hidden_state
        logits = hidden @ reference.embed_tokens.weight.T
        cap = config.text_config.final_logit_softcapping
        expected = (torch.tanh(logits / cap) * cap).numpy()
    with mx.stream(mx.cpu):
        actual = model(mx.array(ids), logits_to_keep=0)
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_multiwindow_mel_matches_transformers():
    from transformers import WhisperFeatureExtractor

    config = tiny_config()
    config.audio_config.num_mel_bins = 128
    model = Model(config)
    audio = np.random.default_rng(6).normal(0, 0.1, 480321).astype(np.float32)
    audio[480000:] *= 1e-4
    extractor = WhisperFeatureExtractor(feature_size=128)
    expected = (
        extractor(
            [audio],
            sampling_rate=16000,
            padding="max_length",
            max_length=960000,
            truncation=False,
            return_tensors="np",
        )["input_features"][0]
        .reshape(128, 2, 3000)
        .transpose(1, 0, 2)
    )
    actual = model._extract_features(mx.array(audio))
    np.testing.assert_allclose(actual, expected, atol=3e-4, rtol=1e-5)
