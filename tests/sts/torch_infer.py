"""PyTorch reference inference for Mel-Band-RoFormer parity testing.

Loaded by ``test_mel_roformer_parity.py`` via the ``MEL_ROFORMER_TORCH_INFER``
environment variable. Wraps the lucidrains ``bs_roformer.MelBandRoformer``
PyTorch model — the same implementation ZFTurbo's MSS-Training uses for
training and inference — so the test compares MLX output against the
canonical PyTorch reference, not against a custom reimplementation.

Required interface (per ``test_mel_roformer_parity.py``):

    run(checkpoint_path, config_yaml_path, audio_path, out_path="", chunk_samples=352800) -> np.ndarray [2, T_out]
        Single-chunk inference: load checkpoint, build model from YAML,
        run forward pass on the leading ``chunk_samples`` frames of the
        reference audio, return separated vocals as np.ndarray [2, T].

    load_audio_chunk(audio_path, chunk_samples) -> (np.ndarray [2, T], sr)
        Load the same chunk the MLX side will see, with the same channel
        and sample-rate normalisation.

Both functions are intentionally stateless and side-effect free (modulo
their explicit ``out_path`` arg, which the parity test passes empty).
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


# ---------- Audio loading ----------


def load_audio_chunk(
    audio_path: str | Path, chunk_samples: int
) -> Tuple[np.ndarray, int]:
    """Load a stereo 44.1kHz chunk from ``audio_path``.

    Mono inputs are duplicated to stereo. Inputs at non-44.1kHz are resampled
    via librosa. Audio is truncated (or zero-padded) to exactly
    ``chunk_samples``. Returns ``([2, T], sr)`` with ``T == chunk_samples``
    and ``sr == 44100``.
    """
    import soundfile as sf

    audio, sr = sf.read(str(audio_path), always_2d=True)
    audio = audio.T.astype(np.float32)  # [channels, samples]

    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    elif audio.shape[0] > 2:
        audio = audio[:2]

    if sr != 44100:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=44100)
        sr = 44100

    if audio.shape[1] >= chunk_samples:
        audio = audio[:, :chunk_samples]
    else:
        # Pad short clips with zeros so the chunk shape is always exact.
        pad = chunk_samples - audio.shape[1]
        audio = np.pad(audio, ((0, 0), (0, pad)))

    return audio, sr


# ---------- PyTorch reference inference ----------


def _build_torch_model(config_yaml_path: str | Path):
    """Build a ``MelBandRoformer`` from the training YAML's ``model`` block.

    Filters the YAML hyperparameters against the installed ``MelBandRoformer``
    init signature, dropping any keys the current ``bs_roformer`` release
    doesn't accept. This makes the script robust across the lucidrains
    package's evolution: Kim Vocal 2 was trained against a pre-1.0 release
    that lacked ``linear_transformer_depth``, ``mlp_expansion_factor`` etc.;
    later releases added them with defaults that don't match the checkpoint.

    Pin ``bs_roformer==0.3.10`` for Kim Vocal 2 / classic-architecture
    checkpoints. Newer releases (0.4+) reorder the ``layers`` ModuleList
    nesting and add nGPT-style normalization, breaking checkpoint
    compatibility.
    """
    import inspect

    import yaml
    import torch
    from bs_roformer.mel_band_roformer import MelBandRoformer

    with open(config_yaml_path) as f:
        cfg = yaml.unsafe_load(f)  # YAML uses !!python/tuple — needs unsafe loader

    model_cfg = dict(cfg["model"])

    # The lucidrains init defaults assume eval-time dropout 0.0 — but our
    # YAML already sets attn_dropout/ff_dropout to 0, so this is just defensive.
    model_cfg.setdefault("attn_dropout", 0)
    model_cfg.setdefault("ff_dropout", 0)

    # Disable flash-attn on platforms where it isn't available (Apple Silicon).
    # bs_roformer falls back to standard attention silently, but flash_attn=True
    # can fail at import time on systems without the CUDA flash_attn package.
    if not torch.cuda.is_available():
        model_cfg["flash_attn"] = False

    accepted = set(inspect.signature(MelBandRoformer.__init__).parameters.keys())
    filtered = {k: v for k, v in model_cfg.items() if k in accepted}
    dropped = set(model_cfg) - set(filtered)
    if dropped:
        # Surface what we ignored so a future cross-version mismatch isn't silent.
        print(
            f"  [torch_infer] Ignored YAML keys not in {MelBandRoformer.__module__}: "
            f"{sorted(dropped)}"
        )

    return MelBandRoformer(**filtered)


def _load_checkpoint(model, checkpoint_path: str | Path) -> None:
    """Load a Kim/ZFTurbo Mel-Band-RoFormer checkpoint into ``model``.

    Handles the common ``state_dict`` wrapping shapes:
      * raw state_dict
      * ``{"state_dict": state_dict}``
      * ``{"model": state_dict}``
      * Lightning-style ``{"state_dict": {"model.<...>": ...}}`` with a
        ``model.`` prefix to strip.
    """
    import torch

    raw = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

    if isinstance(raw, dict) and "state_dict" in raw:
        state_dict = raw["state_dict"]
    elif isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
        state_dict = raw["model"]
    else:
        state_dict = raw

    # Strip a leading "model." prefix if every key has it (Lightning convention).
    if all(k.startswith("model.") for k in state_dict.keys()):
        state_dict = {k[len("model.") :]: v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        # Report up to 5 to keep the message bounded.
        sample = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f" (+{len(missing) - 5} more)"
        raise RuntimeError(
            f"Missing {len(missing)} keys when loading checkpoint: {sample}{suffix}"
        )
    if unexpected:
        sample = ", ".join(unexpected[:5])
        suffix = "" if len(unexpected) <= 5 else f" (+{len(unexpected) - 5} more)"
        # Unexpected keys are usually optimizer/EMA state — log but don't fail.
        print(
            f"  [torch_infer] Ignored {len(unexpected)} unexpected keys: "
            f"{sample}{suffix}"
        )


def run(
    checkpoint_path: str | Path,
    config_yaml_path: str | Path,
    audio_path: str | Path,
    out_path: str = "",
    chunk_samples: int = 352800,
) -> np.ndarray:
    """Run PyTorch reference inference on a single audio chunk.

    Returns the separated vocals as ``np.ndarray [2, chunk_samples]``.

    ``out_path`` is accepted to match the test contract but is ignored —
    we never write to disk, the caller computes SDR in memory.
    """
    import torch

    audio, sr = load_audio_chunk(audio_path, chunk_samples)
    assert sr == 44100

    model = _build_torch_model(config_yaml_path)
    _load_checkpoint(model, checkpoint_path)
    model.eval()

    audio_t = torch.from_numpy(audio).unsqueeze(0)  # [1, 2, T]

    with torch.no_grad():
        # MelBandRoformer.forward(stems=None) returns separated vocals at
        # [B, num_stems, channels, T]. With num_stems=1 we get [B, 1, 2, T].
        vocals = model(audio_t)

    if vocals.dim() == 4:
        vocals = vocals.squeeze(1)  # [B, 2, T]

    return vocals.squeeze(0).cpu().numpy().astype(np.float32)  # [2, T]
