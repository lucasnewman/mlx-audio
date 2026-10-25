from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_audio.tts.models.kokoro import pipeline as kokoro_pipeline
from mlx_audio.tts.models.kokoro.kokoro import Model


@pytest.fixture(autouse=True)
def phonemizer(monkeypatch):
    monkeypatch.setattr(
        kokoro_pipeline,
        "_get_misaki_espeak",
        lambda: SimpleNamespace(EspeakG2P=lambda **kwargs: None),
    )


def save_voice(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(path), {"voice": mx.full((2, 3), value)})


def unloaded_model():
    model = Model.__new__(Model)
    nn.Module.__init__(model)
    model.repo_id = None
    model._pipelines = {}
    return model


def test_named_voice_uses_the_loaded_snapshot_without_hub_access(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshots" / "pinned-revision"
    save_voice(snapshot / "voices" / "pf_dora.safetensors", 1.0)
    save_voice(
        tmp_path / "snapshots" / "newer-revision" / "voices" / "pf_dora.safetensors",
        2.0,
    )

    def no_download(**kwargs):
        raise AssertionError("a voice in the loaded snapshot must not access the Hub")

    monkeypatch.setattr(kokoro_pipeline, "snapshot_download", no_download)
    model = unloaded_model()
    assert Model.post_load_hook(model, snapshot) is model
    pipeline = Model._get_pipeline(model, "p")

    voice = pipeline.load_single_voice("pf_dora")

    assert bool(mx.array_equal(voice, mx.ones((2, 3))))
    assert pipeline.load_single_voice("pf_dora") is voice


def test_explicit_voice_file_takes_precedence_over_snapshot(tmp_path):
    snapshot = tmp_path / "model"
    save_voice(snapshot / "voices" / "pf_dora.safetensors", 1.0)
    explicit = tmp_path / "custom" / "pf_dora.safetensors"
    save_voice(explicit, 3.0)
    model = unloaded_model()
    Model.post_load_hook(model, snapshot)
    pipeline = Model._get_pipeline(model, "p")

    assert bool(
        mx.array_equal(pipeline.load_single_voice(str(explicit)), mx.full((2, 3), 3.0))
    )


@pytest.mark.parametrize("model_path", [None, "empty-model"])
def test_missing_local_voice_preserves_hub_fallback(tmp_path, monkeypatch, model_path):
    hub_snapshot = tmp_path / "hub"
    save_voice(hub_snapshot / "voices" / "pf_dora.safetensors", 4.0)
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(hub_snapshot)

    monkeypatch.setattr(kokoro_pipeline, "snapshot_download", download)
    pipeline = kokoro_pipeline.KokoroPipeline(
        lang_code="p",
        model=False,
        repo_id="example/kokoro",
        model_path=tmp_path / model_path if model_path else None,
    )

    assert bool(
        mx.array_equal(pipeline.load_single_voice("pf_dora"), mx.full((2, 3), 4.0))
    )
    assert calls[0]["repo_id"] == "example/kokoro"
