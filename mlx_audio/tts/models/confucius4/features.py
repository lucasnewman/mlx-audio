# Copyright (c) 2026 Hert4 (https://github.com/Hert4)
# MLX port of Confucius4-TTS (netease-youdao, Apache-2.0).
# Licensed under the Apache License, Version 2.0.
"""Torch-free SeamlessM4T-style fbank (160-d) for the w2v-bert frontend.

Reimplements transformers' SeamlessM4TFeatureExtractor in numpy: povey window,
remove-dc + preemphasis 0.97 per frame, 80 kaldi-mel, log(mel_floor), per-mel-bin
CMVN (ddof=1), stride-2 frame stacking. mel matrix + window are precomputed at
convert time (fbank_filters.npz). Validated to ~1e-6 vs transformers.
"""
import numpy as np

FRAME, HOP, NFFT, MEL_FLOOR = 400, 160, 512, 1.192092955078125e-07


def fbank_160(
    audio: np.ndarray, mel_filters: np.ndarray, window: np.ndarray
) -> np.ndarray:
    """audio (T,) float 16kHz -> (1, num_frames//2, 160)."""
    wav = audio.astype(np.float64) * (2**15)
    nfr = 1 + (len(wav) - FRAME) // HOP
    melT = mel_filters.T
    out = np.empty((nfr, 80), dtype=np.float64)
    for i in range(nfr):
        b = wav[i * HOP : i * HOP + FRAME].copy()
        b = b - b.mean()  # remove_dc_offset
        buf = np.zeros(NFFT)
        buf[1:FRAME] = b[1:] - 0.97 * b[:-1]  # preemphasis
        buf[0] = b[0] * 0.03
        buf[:FRAME] *= window
        spec = np.abs(np.fft.rfft(buf, NFFT)) ** 2
        out[i] = np.log(np.maximum(MEL_FLOOR, melT @ spec))
    out = (out - out.mean(0)) / np.sqrt(out.var(0, ddof=1) + 1e-7)  # per-bin CMVN
    n = out.shape[0] - (out.shape[0] % 2)
    return out[:n].reshape(n // 2, 160)[None].astype(np.float32)
