from .config import ModelConfig
from .frames import DuplexConfig, Frame, PendingFrame
from .model import Model
from .prompt import DEFAULT_SYSTEM_PROMPT, render_card
from .session import DuplexSession

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "DuplexConfig",
    "DuplexSession",
    "Frame",
    "Model",
    "ModelConfig",
    "PendingFrame",
    "render_card",
]


def write_model_card(output_path, source_repo, repo_id, config):
    from pathlib import Path

    quantization = config.get("quantization", {})
    precision = (
        f"{quantization['bits']}-bit backbone"
        if "bits" in quantization
        else config.get("dtype", "bfloat16")
    )
    Path(output_path, "README.md").write_text(
        f"---\nlicense: apache-2.0\nbase_model: {source_repo}\n"
        "pipeline_tag: audio-to-audio\nlibrary_name: mlx-audio\n"
        "tags:\n- mlx\n- full-duplex\n- speech-to-speech\n---\n\n"
        f"# {repo_id}\n\nHaru-4B for MLX Audio ({precision}).\n\n"
        "```python\nfrom mlx_audio.sts.utils import load_model\n\n"
        f'model = load_model("{repo_id}")\n'
        'result = model.generate("user.wav", ref_audio="voice.wav")\n```\n\n'
        "The output is mono 24 kHz audio. Live sessions use `model.create_session()` "
        "with 3200-sample frames. Character cards and a clean reference voice clip "
        "provide persona and voice conditioning.\n\n"
        "Haru is a pretrained dialogue base model. The audio head, codec and "
        "projectors retain their original precision; 8-bit backbone quantization "
        "is recommended over 4-bit for this model.\n\n"
        "[Usage documentation](https://github.com/Blaizzy/mlx-audio/tree/main/mlx_audio/sts/models/haru). "
        "Weights and Haru reference code are Apache-2.0; "
        "[upstream license](https://github.com/higurashi-ai/Haru/blob/main/LICENSE) and "
        "[third-party notices](https://github.com/higurashi-ai/Haru/blob/main/THIRD_PARTY_NOTICES.md).\n",
        encoding="utf-8",
    )
