from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import cast

import mlx.core as mx
import pytest

from mlx_audio.stt import load
from mlx_audio.stt.models.base import STTOutput
from mlx_audio.stt.models.mega_asr.mega_asr import Model as MegaASRModel


MODEL_DIR_ENV = "MEGA_ASR_MLX_DIR"
DEFAULT_MODEL_DIR = Path(
    "/var/folders/kj/d8bkjl_n4y58ks_vx3qv9rmm0000gn/T/opencode/mega-asr-mlx"
)
FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _model_dir() -> Path:
    return Path(os.environ.get(MODEL_DIR_ENV, DEFAULT_MODEL_DIR))


def _normalize(text: str) -> str:
    text = text.casefold()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            curr.append(
                min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + (0 if ca == cb else 1),
                )
            )
        prev = curr
    return prev[-1]


def _assert_close_text(got: str, expected: str, *, max_char_distance: int) -> None:
    norm_got = _normalize(got)
    norm_expected = _normalize(expected)
    distance = _levenshtein(norm_got, norm_expected)
    assert distance <= max_char_distance, (
        f"char distance {distance} exceeded {max_char_distance}: got={got!r} expected={expected!r}"
    )


def _generate_and_measure(model: MegaASRModel, audio: Path) -> tuple[STTOutput, float]:
    mx.clear_cache()
    mx.reset_peak_memory()
    result = cast(STTOutput, model.generate(str(audio), temperature=0.0))
    mx.synchronize()
    peak_gb = float(mx.get_peak_memory() / 1e9)
    mx.clear_cache()
    return result, peak_gb


@pytest.mark.requires_weights
def test_pretrained_mega_asr_matches_reference():
    model_dir = _model_dir()
    if not model_dir.exists():
        pytest.skip(f"Mega-ASR MLX model dir missing: {model_dir}")

    ref = json.loads((FIXTURES_DIR / "reference.json").read_text())
    model = cast(MegaASRModel, cast(object, load(str(model_dir))))

    assert type(model).__module__.endswith("mega_asr.mega_asr")
    assert hasattr(model, "_router")
    assert model._deltas

    clean_audio = FIXTURES_DIR / "clean.wav"
    degraded_audio = FIXTURES_DIR / "degraded.wav"

    clean_route = model._router.route(clean_audio)
    assert clean_route["use_lora"] is False
    clean = cast(STTOutput, model.generate(str(clean_audio), temperature=0.0))
    _assert_close_text(clean.text, ref["clean"]["text"], max_char_distance=1)
    assert model._lora_active is False

    degraded_route = model._router.route(degraded_audio)
    assert degraded_route["use_lora"] is True
    degraded = cast(STTOutput, model.generate(str(degraded_audio), temperature=0.0))
    _assert_close_text(degraded.text, ref["degraded"]["text"], max_char_distance=0)
    assert model._lora_active is True

    clean_again = cast(STTOutput, model.generate(str(clean_audio), temperature=0.0))
    _assert_close_text(clean_again.text, ref["clean"]["text"], max_char_distance=1)
    assert model._lora_active is False


@pytest.mark.requires_weights
def test_pretrained_mega_asr_peak_memory_stays_below_threshold():
    model_dir = _model_dir()
    if not model_dir.exists():
        pytest.skip(f"Mega-ASR MLX model dir missing: {model_dir}")

    ref = json.loads((FIXTURES_DIR / "reference.json").read_text())
    model = cast(MegaASRModel, cast(object, load(str(model_dir))))

    clean_audio = FIXTURES_DIR / "clean.wav"
    degraded_audio = FIXTURES_DIR / "degraded.wav"

    clean, clean_peak_gb = _generate_and_measure(model, clean_audio)
    _assert_close_text(clean.text, ref["clean"]["text"], max_char_distance=1)
    assert model._lora_active is False

    degraded, degraded_peak_gb = _generate_and_measure(model, degraded_audio)
    _assert_close_text(degraded.text, ref["degraded"]["text"], max_char_distance=0)
    assert model._lora_active is True

    clean_again, clean_again_peak_gb = _generate_and_measure(model, clean_audio)
    _assert_close_text(clean_again.text, ref["clean"]["text"], max_char_distance=1)
    assert model._lora_active is False

    assert degraded_peak_gb < 8.0, {
        "clean_peak_gb": clean_peak_gb,
        "degraded_peak_gb": degraded_peak_gb,
        "clean_again_peak_gb": clean_again_peak_gb,
        "clean_text": clean.text,
        "degraded_text": degraded.text,
        "clean_again_text": clean_again.text,
    }
