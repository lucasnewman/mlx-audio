from mlx_audio.stt.utils import MODEL_REMAPPING
from mlx_audio.utils import get_model_class


def test_mega_asr_in_remapping():
    assert "mega_asr" in MODEL_REMAPPING
    assert MODEL_REMAPPING["mega_asr"] == "mega_asr"


def test_get_model_class_resolves_mega_asr():
    arch, model_type = get_model_class("mega_asr", [], "stt", MODEL_REMAPPING)
    assert model_type == "mega_asr"


def test_arch_exposes_model():
    arch, _ = get_model_class("mega_asr", [], "stt", MODEL_REMAPPING)
    assert hasattr(arch, "Model")


def test_arch_exposes_modelconfig():
    arch, _ = get_model_class("mega_asr", [], "stt", MODEL_REMAPPING)
    assert hasattr(arch, "ModelConfig")


def test_modelconfig_is_megaasrconfig():
    from mlx_audio.stt.models.mega_asr import MegaASRConfig

    arch, _ = get_model_class("mega_asr", [], "stt", MODEL_REMAPPING)
    assert arch.ModelConfig is MegaASRConfig


def test_megaasrconfig_from_dict_defaults():
    from mlx_audio.stt.models.mega_asr import MegaASRConfig

    cfg = MegaASRConfig.from_dict({"model_type": "mega_asr"})
    assert cfg.model_type == "mega_asr"
    assert cfg.router_weights == "extras/router.safetensors"
    assert cfg.lora_dir == "extras/lora"


def test_megaasrconfig_from_dict_custom_paths():
    from mlx_audio.stt.models.mega_asr import MegaASRConfig

    cfg = MegaASRConfig.from_dict(
        {
            "model_type": "mega_asr",
            "router_weights": "custom/router.safetensors",
            "lora_dir": "custom_lora",
        }
    )
    assert cfg.router_weights == "custom/router.safetensors"
    assert cfg.lora_dir == "custom_lora"


def test_to_qwen3_dict_sets_correct_type():
    from mlx_audio.stt.models.mega_asr import MegaASRConfig

    cfg = MegaASRConfig.from_dict({"model_type": "mega_asr"})
    q3cfg = cfg.to_qwen3_dict()
    assert q3cfg["model_type"] == "qwen3_asr"


def test_megaasrconfig_nested_configs():
    from mlx_audio.stt.models.mega_asr import MegaASRConfig

    cfg = MegaASRConfig.from_dict(
        {
            "model_type": "mega_asr",
            "audio_config": {"num_mel_bins": 128, "d_model": 1024},
            "text_config": {"vocab_size": 151936, "num_hidden_layers": 28},
        }
    )
    assert cfg.audio_config.num_mel_bins == 128
    assert cfg.audio_config.d_model == 1024
    assert cfg.text_config.vocab_size == 151936
    assert cfg.text_config.num_hidden_layers == 28