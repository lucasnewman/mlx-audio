from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

MODEL_DIR_ENV = "MEGA_ASR_MLX_DIR"
FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.mark.requires_weights
def test_cli_generate_with_pretrained_model(tmp_path):
    raw = os.environ.get(MODEL_DIR_ENV)
    if not raw:
        pytest.skip(
            f"set {MODEL_DIR_ENV} to a converted Mega-ASR MLX dir to run this test"
        )
    model_dir = Path(raw)
    if not model_dir.exists():
        pytest.skip(f"Mega-ASR MLX model dir missing: {model_dir}")

    output_path = tmp_path / "transcript"
    cmd = [
        sys.executable,
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
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert "Transcription:" in result.stdout
    assert result.stdout.strip()
    assert output_path.with_suffix(".txt").exists()
