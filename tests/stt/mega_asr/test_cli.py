from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


MODEL_DIR_ENV = "MEGA_ASR_MLX_DIR"
DEFAULT_MODEL_DIR = Path(
    "/var/folders/kj/d8bkjl_n4y58ks_vx3qv9rmm0000gn/T/opencode/mega-asr-mlx"
)
FIXTURES_DIR = Path(__file__).parent / "fixtures"
REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.requires_weights
def test_cli_generate_with_pretrained_model(tmp_path):
    model_dir = Path(os.environ.get(MODEL_DIR_ENV, DEFAULT_MODEL_DIR))
    if not model_dir.exists():
        pytest.skip(f"Mega-ASR MLX model dir missing: {model_dir}")

    output_path = tmp_path / "transcript"
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "mlx_audio.stt.generate",
        "--model",
        str(model_dir),
        "--audio",
        str(FIXTURES_DIR / "degraded.wav"),
        "--output-path",
        str(output_path),
        "--verbose",
    ]
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert "Transcription:" in result.stdout
    assert result.stdout.strip()
    assert output_path.with_suffix(".txt").exists()
