# Copyright (c) 2026 Hert4 (https://github.com/Hert4)
# MLX port of Confucius4-TTS (netease-youdao, Apache-2.0).
# Licensed under the Apache License, Version 2.0.
"""Convert original Confucius4-TTS (+ deps) torch weights into an MLX model dir.
torch is used ONLY here (conversion), never at inference.

    python -m mlx_audio.tts.models.confucius4.convert --out ./confucius4-model
"""
import argparse
import glob
import re
from pathlib import Path

import numpy as np


def _fold_weight_norm(sd):
    folded, vg = {}, {}
    for k, v in sd.items():
        if k.endswith(".weight_v"):
            vg.setdefault(k[:-9], {})["v"] = v
        elif k.endswith(".weight_g"):
            vg.setdefault(k[:-9], {})["g"] = v
        else:
            folded[k] = v
    for base, d in vg.items():
        g, v = d["g"].float(), d["v"].float()
        folded[base + ".weight"] = g * v / v.flatten(1).norm(dim=1).view(-1, 1, 1)
    return folded


def main():
    import mlx.core as mx
    import safetensors.torch
    import torch
    from huggingface_hub import hf_hub_download

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./confucius4-model")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def save(name, d):
        mx.save_safetensors(
            str(out / name),
            {
                k: mx.array(v.detach().cpu().numpy().astype(np.float32))
                for k, v in d.items()
            },
        )
        print("saved", name, len(d))

    # T2S (F32 safetensors, mx-loadable as-is) -> copy
    t2s_pt = hf_hub_download(
        "netease-youdao/Confucius4-TTS", filename="t2s_model.safetensors"
    )
    save("t2s_model.safetensors", safetensors.torch.load_file(t2s_pt))

    # S2A (.pt, fold weight_norm)
    s2a_pt = hf_hub_download("netease-youdao/Confucius4-TTS", filename="s2a_model.pt")
    sd = torch.load(s2a_pt, map_location="cpu", weights_only=False)
    save(
        "s2a_mlx.safetensors",
        _fold_weight_norm(sd.state_dict() if hasattr(sd, "state_dict") else sd),
    )

    # BigVGAN (fold weight_norm)
    bv = torch.load(
        hf_hub_download(
            "nvidia/bigvgan_v2_22khz_80band_256x", filename="bigvgan_generator.pt"
        ),
        map_location="cpu",
        weights_only=False,
    )
    save("bigvgan_mlx.safetensors", _fold_weight_norm(bv.get("generator", bv)))

    # w2v-bert: feature_projection + layers 0..16 only
    from transformers import Wav2Vec2BertModel

    wsd = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0").eval().state_dict()
    keep = {
        k: v
        for k, v in wsd.items()
        if k.startswith("feature_projection.")
        or (k.startswith("encoder.layers.") and int(k.split(".")[2]) <= 16)
    }
    save("w2vbert_mlx.safetensors", keep)

    # w2v stats
    st = torch.load(
        (
            hf_hub_download("facebook/w2v-bert-2.0", filename="wav2vec2bert_stats.pt")
            if False
            else (
                glob.glob(str(Path(t2s_pt).parent / "wav2vec2bert_stats.pt"))[0]
                if glob.glob(str(Path(t2s_pt).parent / "wav2vec2bert_stats.pt"))
                else hf_hub_download(
                    "netease-youdao/Confucius4-TTS", filename="wav2vec2bert_stats.pt"
                )
            )
        ),
        map_location="cpu",
    )
    np.savez(
        str(out / "w2v_stats.npz"),
        mean=st["mean"].numpy().astype(np.float32),
        std=torch.sqrt(st["var"]).numpy().astype(np.float32),
    )
    print("saved w2v_stats.npz")

    # CAMPPlus: sanitize via mlx-audio's CAMPPlus then save (no torch at inference)
    from mlx.utils import tree_flatten, tree_unflatten

    from mlx_audio.tts.models.chatterbox.s3gen.xvector import CAMPPlus

    cm = CAMPPlus(feat_dim=80, embedding_size=192)
    csd = torch.load(
        hf_hub_download("funasr/campplus", filename="campplus_cn_common.bin"),
        map_location="cpu",
    )
    csd = csd.get("state_dict", csd)
    san = cm.sanitize(
        {k: mx.array(v.float().numpy()) for k, v in csd.items() if torch.is_tensor(v)}
    )
    mx.save_safetensors(str(out / "campplus.safetensors"), dict(san))
    print("saved campplus.safetensors", len(san))

    # precompute SeamlessM4T fbank mel matrix + povey window (torch-free at runtime)
    from transformers.audio_utils import mel_filter_bank, window_function

    mel = mel_filter_bank(
        num_frequency_bins=257,
        num_mel_filters=80,
        min_frequency=20,
        max_frequency=8000,
        sampling_rate=16000,
        norm=None,
        mel_scale="kaldi",
        triangularize_in_mel_space=True,
    )
    win = window_function(400, "povey", periodic=False)
    np.savez(
        str(out / "fbank_filters.npz"),
        mel=mel.astype(np.float32),
        window=win.astype(np.float32),
    )
    print("saved fbank_filters.npz")
    print("DONE ->", out)


if __name__ == "__main__":
    main()
