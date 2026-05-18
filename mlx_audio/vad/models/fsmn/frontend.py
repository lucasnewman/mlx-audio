"""
FSMN-VAD 前端特征提取: Kaldi-style Fbank + LFR + CMVN

与 FunASR WavFrontendOnline 对齐:
- Kaldi fbank (torchaudio.compliance.kaldi.fbank)
- LFR: lfr_m=5, lfr_n=1
- CMVN: Kaldi Nnet 格式 (AddShift + Rescale)
"""
import re
import numpy as np
from typing import Tuple, Optional


def load_cmvn(cmvn_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载 Kaldi Nnet 格式的 CMVN 文件 (am.mvn).

    格式:
        <AddShift> D D
        <LearnRateCoef> 0 [ shift_values ]
        <Rescale> D D
        <LearnRateCoef> 0 [ scale_values ]

    CMVN 操作: output = (input + shift) * scale
    """
    with open(cmvn_path, "r") as f:
        content = f.read()

    # 提取 AddShift 后的数值
    shift_match = re.search(r"<AddShift>.*?\[(.*?)\]", content, re.DOTALL)
    scale_match = re.search(r"<Rescale>.*?\[(.*?)\]", content, re.DOTALL)

    if not shift_match or not scale_match:
        raise ValueError(f"Cannot parse CMVN file: {cmvn_path}")

    shift = np.array([float(x) for x in shift_match.group(1).split()], dtype=np.float32)
    scale = np.array([float(x) for x in scale_match.group(1).split()], dtype=np.float32)

    return shift, scale


def compute_fbank_kaldi(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    n_mels: int = 80,
    frame_length_ms: int = 25,
    frame_shift_ms: int = 10,
    dither: float = 0.0,
) -> np.ndarray:
    """
    Kaldi-style fbank 特征提取 (通过 torchaudio).

    Returns:
        fbank: [num_frames, n_mels] float32
    """
    import torch
    import torchaudio

    waveform_tensor = torch.from_numpy(waveform).unsqueeze(0).float() * (1 << 15)

    fbank = torchaudio.compliance.kaldi.fbank(
        waveform_tensor,
        sample_frequency=sample_rate,
        num_mel_bins=n_mels,
        frame_length=frame_length_ms,
        frame_shift=frame_shift_ms,
        dither=dither,
        window_type="hamming",
    )

    return fbank.numpy()  # [T, n_mels]


def apply_lfr(
    features: np.ndarray, lfr_m: int = 5, lfr_n: int = 1
) -> np.ndarray:
    """
    Low Frame Rate: 每 lfr_n 帧取一帧, 每帧拼接 lfr_m 帧.

    FunASR 的 LFR 对前几帧做左侧填充 (用第一帧重复).

    Args:
        features: [T, D]
        lfr_m: 拼接帧数
        lfr_n: 步长

    Returns:
        [T', D * lfr_m]
    """
    T, D = features.shape
    # FunASR 左侧 padding: 重复第一帧 (lfr_m - 1) // 2 次
    left_pad = (lfr_m - 1) // 2
    if left_pad > 0:
        pad_frames = np.tile(features[0:1], (left_pad, 1))
        features = np.concatenate([pad_frames, features], axis=0)

    T_padded = features.shape[0]
    T_out = (T_padded + lfr_n - 1) // lfr_n
    out = np.zeros((T_out, D * lfr_m), dtype=np.float32)

    for i in range(T_out):
        start = i * lfr_n
        for j in range(lfr_m):
            idx = start + j
            if idx < T_padded:
                out[i, j * D : (j + 1) * D] = features[idx]
            else:
                out[i, j * D : (j + 1) * D] = features[T_padded - 1]

    return out


def apply_cmvn(
    features: np.ndarray, shift: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    """
    Kaldi CMVN: output = (input + shift) * scale
    """
    return (features + shift) * scale


def extract_features(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    n_mels: int = 80,
    frame_length_ms: int = 25,
    frame_shift_ms: int = 10,
    lfr_m: int = 5,
    lfr_n: int = 1,
    cmvn_path: Optional[str] = None,
    cmvn_shift: Optional[np.ndarray] = None,
    cmvn_scale: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Full frontend: waveform → Kaldi fbank → LFR → CMVN → [T', 400]

    CMVN can be provided via file path (cmvn_path) or directly as
    numpy arrays (cmvn_shift, cmvn_scale). Direct arrays take priority.
    """
    fbank = compute_fbank_kaldi(waveform, sample_rate, n_mels, frame_length_ms, frame_shift_ms)
    features = apply_lfr(fbank, lfr_m, lfr_n)

    if cmvn_shift is not None and cmvn_scale is not None:
        if len(cmvn_shift) == features.shape[1]:
            features = apply_cmvn(features, cmvn_shift, cmvn_scale)
    elif cmvn_path is not None:
        shift, scale = load_cmvn(cmvn_path)
        if len(shift) == features.shape[1]:
            features = apply_cmvn(features, shift, scale)

    return features
