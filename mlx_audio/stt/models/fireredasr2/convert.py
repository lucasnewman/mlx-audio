import argparse
import json
import math
import os
import shutil

import numpy as np
import torch
from safetensors.torch import save_file


def convert_cmvn(input_dir, output_dir):
    try:
        import kaldiio
    except ImportError:
        raise ImportError("kaldiio is required for conversion: pip install kaldiio")

    cmvn_path = os.path.join(input_dir, "cmvn.ark")
    stats = kaldiio.load_mat(cmvn_path)
    dim = stats.shape[-1] - 1
    count = stats[0, dim]

    means = stats[0, :dim] / count
    variance = stats[1, :dim] / count - means**2
    variance = np.maximum(variance, 1e-20)
    istd = 1.0 / np.sqrt(variance)

    cmvn_out = {"means": means.tolist(), "istd": istd.tolist()}
    with open(os.path.join(output_dir, "cmvn.json"), "w") as f:
        json.dump(cmvn_out, f)


def convert_weights(input_dir, output_dir):
    model_path = os.path.join(input_dir, "model.pth.tar")
    package = torch.load(model_path, map_location="cpu", weights_only=False)
    state_dict = package["model_state_dict"]
    args = package["args"]

    # remove positional encoding buffers (computed at init)
    keys_to_remove = []
    for k in state_dict:
        if "positional_encoding.pe" in k:
            keys_to_remove.append(k)
    for k in keys_to_remove:
        del state_dict[k]

    # rename: decoder.tgt_word_prj is tied to decoder.tgt_word_emb
    # keep both but they should be identical
    if (
        "decoder.tgt_word_prj.weight" in state_dict
        and "decoder.tgt_word_emb.weight" in state_dict
    ):
        # verify they're the same
        if torch.equal(
            state_dict["decoder.tgt_word_prj.weight"],
            state_dict["decoder.tgt_word_emb.weight"],
        ):
            del state_dict["decoder.tgt_word_prj.weight"]

    # remove CTC head (not used for inference via beam search)
    keys_to_remove = [k for k in state_dict if k.startswith("ctc.")]
    for k in keys_to_remove:
        del state_dict[k]

    # convert all to float32
    for k in state_dict:
        state_dict[k] = state_dict[k].float()

    save_file(state_dict, os.path.join(output_dir, "model.safetensors"))

    # generate config.json
    config = {
        "model_type": "fireredasr2",
        "idim": args.idim,
        "odim": args.odim,
        "d_model": args.d_model,
        "sos_id": args.sos_id,
        "eos_id": args.eos_id,
        "pad_id": args.pad_id,
        "blank_id": args.blank_id,
        "encoder": {
            "n_layers": args.n_layers_enc,
            "n_head": args.n_head,
            "d_model": args.d_model,
            "kernel_size": args.kernel_size,
            "pe_maxlen": args.pe_maxlen,
        },
        "decoder": {
            "n_layers": args.n_layers_dec,
            "n_head": args.n_head,
            "d_model": args.d_model,
            "pe_maxlen": args.pe_maxlen,
        },
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Convert FireRedASR2-AED to MLX format"
    )
    parser.add_argument("input_dir", help="Path to original model directory")
    parser.add_argument("output_dir", help="Path to output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Converting CMVN...")
    convert_cmvn(args.input_dir, args.output_dir)

    print("Converting weights...")
    convert_weights(args.input_dir, args.output_dir)

    # copy tokenizer files
    for fname in ["dict.txt", "train_bpe1000.model"]:
        src = os.path.join(args.input_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.output_dir, fname))
            print(f"Copied {fname}")

    print(f"Done. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
