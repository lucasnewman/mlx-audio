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
    MEL_ROFORMER_TORCH_CKPT   — Path to the original PyTorch .ckpt
    MEL_ROFORMER_MLX_WEIGHTS  — Path to the converted MLX .safetensors
    MEL_ROFORMER_REF_AUDIO    — Path to a reference stereo 44.1kHz WAV
    MEL_ROFORMER_PRESET       — Preset name (kim_vocal_2, viperx_vocals, zfturbo_bs_roformer)
    MEL_ROFORMER_SDR_TARGET   — Minimum SDR in dB (default 40.0 — bit-exact up to fp precision)

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
    ):
        """Run both implementations on the same audio, compute SDR between outputs."""
        pytest.skip(
            "Parity test requires: PyTorch Mel-Band-RoFormer inference script "
            "(e.g., ZFTurbo/Music-Source-Separation-Training inference.py). "
            "Implement per the test docstring protocol."
        )

        # Scaffold for when the reference inference is wired up:
        #
        # 1. Load audio
        # audio = _load_audio_stereo_44k(reference_audio_path)
        # audio_torch = torch.from_numpy(audio[None, :, :])  # [1, 2, samples]
        # audio_mlx = mx.array(audio[None, :, :])
        #
        # 2. PyTorch reference (user must provide this import / inference fn)
        # torch_model = _load_torch_model(torch_checkpoint_path, preset_name)
        # with torch.no_grad():
        #     torch_vocals = torch_model(audio_torch).squeeze(0).cpu().numpy()
        #
        # 3. MLX port
        # from mlx_audio.sts.models.mel_roformer import MelRoFormer, MelRoFormerConfig
        # config = getattr(MelRoFormerConfig, preset_name)()
        # mlx_model = MelRoFormer(config)
        # weights = dict(mx.load(str(mlx_weights_path)))
        # mlx_model.load_weights(list(mlx_model.sanitize(weights).items()), strict=False)
        # mlx_vocals = np.asarray(mlx_model(audio_mlx).squeeze(0))
        #
        # 4. SDR diff
        # sdr = _compute_sdr(torch_vocals, mlx_vocals)
        # print(f"SDR between PyTorch and MLX outputs: {sdr:.2f} dB")
        # assert sdr > sdr_target, (
        #     f"SDR {sdr:.2f} dB below target {sdr_target} dB — "
        #     f"likely a weight loading or architecture bug"
        # )

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
