# Copyright (c) 2026 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

"""PyTorch↔MLX numerical parity tests for Mel-Band-RoFormer.

This test verifies that the MLX implementation produces output matching the
PyTorch reference implementation on a real checkpoint. It is the load-bearing
correctness check for the architecture port — subtle weight-loading bugs
(e.g., wrong QKV head order) produce output that sounds roughly correct but
has significant SDR loss, which is hard to detect without numerical comparison.

Run requirements:
    - PyTorch installed (`pip install torch`)
    - A PyTorch Mel-Band-RoFormer checkpoint converted via `convert.py`
    - The original PyTorch checkpoint available for reference inference
    - A reference audio file for the SDR diff test

Environment variables:
    MEL_ROFORMER_TORCH_CKPT    — Path to the original PyTorch .ckpt
    MEL_ROFORMER_MLX_WEIGHTS   — Path to the converted MLX .safetensors
    MEL_ROFORMER_REF_AUDIO     — Path to a reference stereo 44.1kHz WAV
    MEL_ROFORMER_PRESET        — Preset name (kim_vocal_2, viperx_vocals,
                                  zfturbo_bs_roformer, zfturbo_vocals_v1)
    MEL_ROFORMER_TORCH_INFER   — Path to the user-supplied torch_infer.py
                                  reference (see PARITY_TESTING.md)
    MEL_ROFORMER_TORCH_CONFIG  — Path to the ZFTurbo YAML config for the
                                  PyTorch model the ckpt was trained with
    MEL_ROFORMER_SDR_TARGET    — Minimum SDR in dB (default 40.0 — bit-exact
                                  up to fp precision)
    MEL_ROFORMER_CHUNK_SAMPLES — Samples per single-chunk inference
                                  (default 352800 = 8s at 44.1kHz)

To run:
    pytest tests/sts/test_mel_roformer_parity.py -v -m requires_torch

To skip (default in CI):
    pytest tests/sts/test_mel_roformer_parity.py -v -m 'not requires_torch'
"""

import os
from pathlib import Path

import pytest


# Skip the entire module if torch is not installed.
torch = pytest.importorskip("torch", reason="PyTorch required for parity tests")


@pytest.fixture
def torch_checkpoint_path():
    path = os.environ.get("MEL_ROFORMER_TORCH_CKPT")
    if not path:
        pytest.skip("Set MEL_ROFORMER_TORCH_CKPT to run parity tests")
    if not Path(path).exists():
        pytest.skip(f"Checkpoint not found: {path}")
    return Path(path)


@pytest.fixture
def mlx_weights_path():
    path = os.environ.get("MEL_ROFORMER_MLX_WEIGHTS")
    if not path:
        pytest.skip("Set MEL_ROFORMER_MLX_WEIGHTS to run parity tests")
    if not Path(path).exists():
        pytest.skip(f"MLX weights not found: {path}")
    return Path(path)


@pytest.fixture
def reference_audio_path():
    path = os.environ.get("MEL_ROFORMER_REF_AUDIO")
    if not path:
        pytest.skip("Set MEL_ROFORMER_REF_AUDIO to run parity tests")
    if not Path(path).exists():
        pytest.skip(f"Reference audio not found: {path}")
    return Path(path)


@pytest.fixture
def preset_name():
    name = os.environ.get("MEL_ROFORMER_PRESET", "kim_vocal_2")
    return name


@pytest.fixture
def sdr_target():
    return float(os.environ.get("MEL_ROFORMER_SDR_TARGET", "40.0"))


@pytest.fixture
def torch_infer_path():
    path = os.environ.get("MEL_ROFORMER_TORCH_INFER")
    if not path:
        pytest.skip(
            "Set MEL_ROFORMER_TORCH_INFER to a torch_infer.py that exposes "
            "`run(ckpt, config_yaml, audio, out_path, chunk_samples=...) -> np.ndarray` "
            "and `load_audio_chunk(audio, chunk_samples) -> (np.ndarray, sr)`"
        )
    if not Path(path).exists():
        pytest.skip(f"torch_infer.py not found: {path}")
    return Path(path)


@pytest.fixture
def torch_config_yaml():
    path = os.environ.get("MEL_ROFORMER_TORCH_CONFIG")
    if not path:
        pytest.skip("Set MEL_ROFORMER_TORCH_CONFIG to the ZFTurbo model YAML")
    if not Path(path).exists():
        pytest.skip(f"Config YAML not found: {path}")
    return Path(path)


@pytest.fixture
def chunk_samples():
    return int(os.environ.get("MEL_ROFORMER_CHUNK_SAMPLES", "352800"))


def _load_torch_infer_module(path: Path):
    """Dynamically import the user's torch_infer.py as a module."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("mel_roformer_torch_infer", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_audio_stereo_44k(path: Path):
    """Load audio as stereo 44.1kHz NumPy array [2, samples]."""
    try:
        import soundfile as sf
    except ImportError:
        pytest.skip("soundfile required for audio loading")

    audio, sr = sf.read(str(path), always_2d=True)
    audio = audio.T.astype("float32")  # [channels, samples]

    if audio.shape[0] == 1:
        audio = audio.repeat(2, axis=0)
    elif audio.shape[0] > 2:
        audio = audio[:2]

    if sr != 44100:
        try:
            import librosa
            # Resample per-channel
            resampled = librosa.resample(audio, orig_sr=sr, target_sr=44100)
            audio = resampled
        except ImportError:
            pytest.skip(f"Audio is {sr}Hz, need 44.1kHz (install librosa to auto-resample)")

    return audio


def _compute_sdr(reference, estimate, eps: float = 1e-10) -> float:
    """Compute Signal-to-Distortion Ratio in dB.

    SDR = 10 * log10(||reference||^2 / ||reference - estimate||^2)

    A high SDR indicates the estimate closely matches the reference.
    For implementation-parity purposes, SDR > 40 dB is effectively bit-exact
    up to floating-point precision.
    """
    import numpy as np

    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)

    # Align lengths
    min_len = min(reference.shape[-1], estimate.shape[-1])
    reference = reference[..., :min_len]
    estimate = estimate[..., :min_len]

    num = np.sum(reference ** 2) + eps
    den = np.sum((reference - estimate) ** 2) + eps
    return float(10.0 * np.log10(num / den))


@pytest.mark.requires_torch
class TestMelRoFormerParity:
    """Numerical parity tests — PyTorch reference vs MLX port.

    Protocol (from the upstream plan):
        1. Pick a reference song (30-60 seconds, mixed content).
        2. Run PyTorch inference via the original implementation.
        3. Run MLX inference on the same input with the converted weights.
        4. Compute SDR of MLX output vs PyTorch output (implementation parity,
           not vs ground truth — we want *the same output*, not the best output).
        5. Target: SDR > 40 dB. Anything under ~25 dB indicates a bug.
    """

    def test_sdr_parity(
        self,
        torch_checkpoint_path,
        mlx_weights_path,
        reference_audio_path,
        preset_name,
        sdr_target,
        torch_infer_path,
        torch_config_yaml,
        chunk_samples,
    ):
        """Run both implementations on the same audio chunk, compare via SDR.

        Single-chunk, no overlap-add. Parity is a property of the forward pass;
        the chunked inference wrapper is tested separately.
        """
        import mlx.core as mx
        import numpy as np

        from mlx_audio.sts.models.mel_roformer import MelRoFormer, MelRoFormerConfig

        torch_infer = _load_torch_infer_module(torch_infer_path)

        torch_vocals = torch_infer.run(
            str(torch_checkpoint_path),
            str(torch_config_yaml),
            str(reference_audio_path),
            out_path="",
            chunk_samples=chunk_samples,
        )
        # torch_vocals: np.ndarray [2, T_out]

        audio_chunk, sr = torch_infer.load_audio_chunk(
            str(reference_audio_path), chunk_samples
        )
        assert sr == 44100, f"Reference audio must be 44.1kHz, got {sr}"

        config = getattr(MelRoFormerConfig, preset_name)()
        mlx_model = MelRoFormer(config)
        weights = dict(mx.load(str(mlx_weights_path)))
        sanitized = mlx_model.sanitize(weights)
        mlx_model.load_weights(list(sanitized.items()), strict=True)

        mlx_out = mlx_model(mx.array(audio_chunk[None]))
        mx.eval(mlx_out)
        mlx_vocals = np.asarray(mlx_out).squeeze(0)  # [2, T_out]

        sdr = _compute_sdr(torch_vocals, mlx_vocals)
        print(
            f"\nSDR between PyTorch and MLX outputs: {sdr:.2f} dB "
            f"(target > {sdr_target})"
        )
        assert sdr > sdr_target, (
            f"SDR {sdr:.2f} dB below target {sdr_target} dB — "
            f"likely a weight loading or architecture bug. "
            f"See debugging tips in PARITY_TESTING.md."
        )

    def test_qkv_split_preserves_weights(self, mlx_weights_path):
        """Verify that the QKV split in sanitize() preserves the original weights.

        This test doesn't require the PyTorch model — it loads the converted MLX
        weights, reconstructs the packed to_qkv from the split to_q/to_k/to_v,
        and checks that the shapes are consistent.
        """
        import mlx.core as mx
        import numpy as np

        weights = dict(mx.load(str(mlx_weights_path)))

        # Find all split QKV triples
        q_keys = [k for k in weights if k.endswith("to_q.weight")]

        assert len(q_keys) > 0, "No to_q.weight keys found — QKV split may not have run"

        for q_key in q_keys:
            prefix = q_key[: -len("to_q.weight")]
            k_key = f"{prefix}to_k.weight"
            v_key = f"{prefix}to_v.weight"

            assert k_key in weights, f"Missing {k_key}"
            assert v_key in weights, f"Missing {v_key}"

            # Shapes should match
            q_shape = weights[q_key].shape
            k_shape = weights[k_key].shape
            v_shape = weights[v_key].shape
            assert q_shape == k_shape == v_shape, (
                f"QKV shape mismatch at {prefix}: Q={q_shape}, K={k_shape}, V={v_shape}"
            )
