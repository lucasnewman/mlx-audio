from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers.models.qwen3 import Qwen3Config

from mlx_audio.tts.models.moss_tts import Model, ModelConfig


def _download_reference_code() -> Path:
    local_dir = Path(tempfile.gettempdir()) / "mlx_audio_moss_tts_reference_code"
    snapshot_download(
        "OpenMOSS-Team/MOSS-TTS",
        allow_patterns=["*.py", "config.json"],
        local_dir=local_dir,
    )
    return local_dir


def _load_reference_classes():
    code_dir = _download_reference_code()
    parent = str(code_dir.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    package_name = code_dir.name
    config_mod = __import__(
        f"{package_name}.configuration_moss_tts",
        fromlist=["MossTTSDelayConfig"],
    )
    model_mod = __import__(
        f"{package_name}.modeling_moss_tts",
        fromlist=["MossTTSDelayModel"],
    )
    return config_mod.MossTTSDelayConfig, model_mod.MossTTSDelayModel


def _tiny_configs():
    moss_kwargs = dict(
        n_vq=2,
        audio_vocab_size=8,
        pad_token_id=151643,
        im_start_token_id=151644,
        im_end_token_id=151645,
        audio_user_slot_token_id=151654,
        audio_assistant_gen_slot_token_id=151656,
        audio_assistant_delay_slot_token_id=151662,
        audio_start_token_id=151652,
        audio_end_token_id=151653,
        audio_pad_code=8,
        sampling_rate=24000,
    )
    qwen_config = Qwen3Config(
        vocab_size=151669,
        hidden_size=16,
        num_hidden_layers=1,
        intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        rms_norm_eps=1e-6,
        max_position_embeddings=64,
        rope_parameters={"rope_type": "default", "rope_theta": 10000},
        tie_word_embeddings=False,
        attention_bias=False,
    )
    mlx_config = ModelConfig.from_dict(
        {
            "model_type": "moss_tts_delay",
            "language_config": qwen_config.to_dict(),
            **moss_kwargs,
        }
    )
    return qwen_config, moss_kwargs, mlx_config


def _copy_weights(torch_model, mlx_model):
    weights = {
        key: mx.array(value.detach().cpu().numpy())
        for key, value in torch_model.state_dict().items()
    }
    mlx_model.load_weights(list(weights.items()), strict=True)


def _assert_forward_parity(torch_model, mlx_model, input_ids, atol: float):
    with torch.no_grad():
        torch_outputs = torch_model(
            input_ids=torch.tensor(input_ids),
            use_cache=False,
        ).logits
    mlx_outputs = mlx_model(mx.array(input_ids, dtype=mx.int32))
    mx.eval(mlx_outputs)

    max_diff = 0.0
    for torch_logits, mlx_logits in zip(torch_outputs, mlx_outputs):
        expected = torch_logits.detach().cpu().numpy()
        actual = np.array(mlx_logits)
        finite = np.isfinite(expected)
        diff = float(np.max(np.abs(expected[finite] - actual[finite])))
        max_diff = max(max_diff, diff)
    if max_diff > atol:
        raise AssertionError(f"Forward max diff {max_diff:.6g} exceeds {atol:.6g}")
    return max_diff


def _assert_greedy_generation_parity(torch_model, mlx_model, input_ids):
    attention_mask = torch.ones(input_ids.shape[:2], dtype=torch.bool)
    with torch.no_grad():
        torch_generation = torch_model.generate(
            torch.tensor(input_ids),
            attention_mask=attention_mask,
            max_new_tokens=5,
            text_temperature=0,
            audio_temperature=0,
        )
    mlx_generation = mlx_model.generate_delay_pattern_ids(
        mx.array(input_ids, dtype=mx.int32),
        max_new_tokens=5,
        text_temperature=0,
        audio_temperature=0,
    )
    expected_start, expected_ids = torch_generation[0]
    actual_start, actual_ids = mlx_generation[0]
    expected = expected_ids.detach().cpu().numpy()
    actual = np.array(actual_ids)
    if int(expected_start) != int(actual_start) or not np.array_equal(expected, actual):
        raise AssertionError(
            "Greedy generation mismatch:\n"
            f"expected start={int(expected_start)} ids={expected.tolist()}\n"
            f"actual start={int(actual_start)} ids={actual.tolist()}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--atol", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    MossTTSDelayConfig, MossTTSDelayModel = _load_reference_classes()
    qwen_config, moss_kwargs, mlx_config = _tiny_configs()

    torch_config = MossTTSDelayConfig(language_config=qwen_config, **moss_kwargs)
    torch_model = MossTTSDelayModel(torch_config).eval()
    mlx_model = Model(mlx_config)
    _copy_weights(torch_model, mlx_model)

    forward_input = np.array(
        [[[151644, 8, 8], [100, 8, 8], [151652, 8, 8]]],
        dtype=np.int64,
    )
    max_diff = _assert_forward_parity(
        torch_model,
        mlx_model,
        forward_input,
        atol=args.atol,
    )
    _assert_greedy_generation_parity(torch_model, mlx_model, forward_input)
    print(f"MOSS-TTS Delay parity passed. max_forward_diff={max_diff:.6g}")


if __name__ == "__main__":
    main()
