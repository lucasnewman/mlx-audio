"""From fermion-research 0.2.11 (Copyright 2026 Fermion Research,
Apache-2.0), with explicit validation errors. See mlx_audio/licenses/NOTICE.
HF-transformers ParakeetForTDT names -> MLX Audio Parakeet-TDT names.

Target key set = mlx-community/parakeet-tdt-0.6b-v3 @ ed2b7e8c (697 tensors),
the layout mlx_audio.stt.models.parakeet loads with no sanitize().
Layout changes (MLX convs are channels-last):
  Conv2d  [O, I, H, W] -> [O, H, W, I]   (subsampling.layers.{0,2,3,5,6})
  Conv1d  [O, I, K]    -> [O, K, I]      (depthwise [1024,1,9] -> [1024,9,1];
                                          pointwise [O,I] or [O,I,1] -> [O,1,I])
  LSTM    weight_ih_lK -> lstm.K.Wx, weight_hh_lK -> lstm.K.Wh,
          bias_ih_lK + bias_hh_lK -> lstm.K.bias   (gate order i,f,g,o in both)
Dropped: BatchNorm num_batches_tracked.
"""

from __future__ import annotations

import re

import numpy as np

_SUB = {"0", "2", "3", "5", "6"}
_ATT = {
    "q_proj": "linear_q",
    "k_proj": "linear_k",
    "v_proj": "linear_v",
    "o_proj": "linear_out",
    "relative_k_proj": "linear_pos",
    "bias_u": "pos_bias_u",
    "bias_v": "pos_bias_v",
}


def hf_to_mlx(hf: dict) -> dict:
    """hf: name -> np.ndarray (any float dtype). Returns mlx-named arrays (same dtype)."""
    out, used = {}, set()

    def put(k_new, v, src):
        if k_new in out:
            raise ValueError(f"Duplicate mapped Phonon-2 tensor: {k_new}")
        out[k_new] = v
        used.update(src if isinstance(src, (list, tuple)) else [src])

    for k, v in hf.items():
        if k.endswith("num_batches_tracked"):
            used.add(k)
            continue
        m = re.fullmatch(r"encoder\.subsampling\.layers\.(\d+)\.(weight|bias)", k)
        if m:
            if m.group(1) not in _SUB:
                raise ValueError(f"Unsupported Phonon-2 subsampling tensor: {k}")
            put(
                f"encoder.pre_encode.conv.{m.group(1)}.{m.group(2)}",
                v.transpose(0, 2, 3, 1) if m.group(2) == "weight" else v,
                k,
            )
            continue
        m = re.fullmatch(r"encoder\.subsampling\.linear\.(weight|bias)", k)
        if m:
            put(f"encoder.pre_encode.out.{m.group(1)}", v, k)
            continue
        m = re.fullmatch(r"encoder\.layers\.(\d+)\.(.+)", k)
        if m:
            L, rest = m.groups()
            p = f"encoder.layers.{L}."
            a = re.fullmatch(r"self_attn\.(\w+?)(\.weight)?", rest)
            if a and a.group(1) in _ATT:
                put(p + "self_attn." + _ATT[a.group(1)] + (a.group(2) or ""), v, k)
                continue
            c = re.fullmatch(r"conv\.(pointwise_conv[12])\.weight", rest)
            if c:
                w = v if v.ndim == 3 else v[:, :, None]  # [O,I,1]
                put(p + f"conv.{c.group(1)}.weight", w.transpose(0, 2, 1), k)
                continue
            if rest == "conv.depthwise_conv.weight":
                put(p + "conv.depthwise_conv.weight", v.transpose(0, 2, 1), k)
                continue
            n = re.fullmatch(
                r"conv\.norm\.(weight|bias|running_mean|running_var)", rest
            )
            if n:
                put(p + f"conv.batch_norm.{n.group(1)}", v, k)
                continue
            if re.fullmatch(
                r"(norm_(feed_forward[12]|self_att|conv|out)|feed_forward[12]\.linear[12])\.(weight|bias)",
                rest,
            ):
                put(p + rest, v, k)
                continue
            raise KeyError(k)
        m = re.fullmatch(r"encoder_projector\.(weight|bias)", k)
        if m:
            put(f"joint.enc.{m.group(1)}", v, k)
            continue
        m = re.fullmatch(r"decoder\.decoder_projector\.(weight|bias)", k)
        if m:
            put(f"joint.pred.{m.group(1)}", v, k)
            continue
        m = re.fullmatch(r"joint\.head\.(weight|bias)", k)
        if m:
            put(f"joint.joint_net.2.{m.group(1)}", v, k)
            continue
        if k == "decoder.embedding.weight":
            put("decoder.prediction.embed.weight", v, k)
            continue
        m = re.fullmatch(r"decoder\.lstm\.weight_(ih|hh)_l(\d+)", k)
        if m:
            put(
                f"decoder.prediction.dec_rnn.lstm.{m.group(2)}.{'Wx' if m.group(1) == 'ih' else 'Wh'}",
                v,
                k,
            )
            continue
        m = re.fullmatch(r"decoder\.lstm\.bias_ih_l(\d+)", k)
        if m:
            kh = f"decoder.lstm.bias_hh_l{m.group(1)}"
            b = (v.astype(np.float32) + hf[kh].astype(np.float32)).astype(v.dtype)
            put(f"decoder.prediction.dec_rnn.lstm.{m.group(1)}.bias", b, [k, kh])
            continue
        if re.fullmatch(r"decoder\.lstm\.bias_hh_l\d+", k):
            continue
        raise KeyError(f"unmapped HF key: {k}")
    missing = set(hf) - used
    if missing:
        raise ValueError(f"Unmapped Phonon-2 tensors: {sorted(missing)[:10]}")
    return out
