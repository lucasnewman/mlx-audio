import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file


def convert_sensevoice(src: str, dst: str):
    src_path = Path(src)
    dst_path = Path(dst)
    dst_path.mkdir(parents=True, exist_ok=True)

    pt_file = src_path / "model.pt"
    if not pt_file.exists():
        raise FileNotFoundError(f"model.pt not found in {src_path}")

    state_dict = torch.load(pt_file, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]

    save_file(state_dict, dst_path / "model.safetensors")

    config = {
        "model_type": "sensevoice",
        "vocab_size": 25055,
        "input_size": 560,
        "encoder_conf": {
            "output_size": 512,
            "attention_heads": 4,
            "linear_units": 2048,
            "num_blocks": 50,
            "tp_blocks": 20,
            "dropout_rate": 0.1,
            "attention_dropout_rate": 0.1,
            "kernel_size": 11,
            "sanm_shift": 0,
            "normalize_before": True,
        },
        "frontend_conf": {
            "fs": 16000,
            "window": "hamming",
            "n_mels": 80,
            "frame_length": 25,
            "frame_shift": 10,
            "lfr_m": 7,
            "lfr_n": 6,
        },
    }
    with open(dst_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    for fname in ["am.mvn", "chn_jpn_yue_eng_ko_spectok.bpe.model"]:
        src_file = src_path / fname
        if src_file.exists():
            shutil.copy2(src_file, dst_path / fname)

    print(f"converted {pt_file} -> {dst_path / 'model.safetensors'}")
    print(f"wrote {dst_path / 'config.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src", help="path to original SenseVoiceSmall HF directory")
    parser.add_argument("dst", help="output directory for converted model")
    args = parser.parse_args()
    convert_sensevoice(args.src, args.dst)
