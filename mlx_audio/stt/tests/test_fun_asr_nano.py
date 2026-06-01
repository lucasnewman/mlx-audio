import pytest

from mlx_audio.stt.models.fun_asr_nano.config import FunASRNanoConfig
from mlx_audio.stt.models.fun_asr_nano.convert import _convert_weights
from mlx_audio.stt.models.fun_asr_nano.fun_asr_nano import FunASRNano


def test_config_parses_upstream_names():
    config = FunASRNanoConfig.from_dict(
        {
            "audio_encoder_conf": {"output_size": 16, "sanm_shfit": 2},
        }
    )

    assert config.audio_encoder_conf.output_size == 16
    assert config.audio_encoder_conf.sanm_shift == 2


def test_language_mapping_uses_current_model_iso_hints():
    assert FunASRNano._map_language("zh") == "中文"
    assert FunASRNano._map_language("yue") == "中文"
    assert FunASRNano._map_language("en") == "英文"
    assert FunASRNano._map_language("ja") == "日文"
    assert FunASRNano._map_language("auto") is None
    assert FunASRNano._map_language("中文") == "中文"

    with pytest.raises(ValueError, match="Unsupported ISO language"):
        FunASRNano._map_language("ko")


def test_converter_transposes_fsmn_and_skips_tied_lm_head():
    torch = pytest.importorskip("torch")
    state = {
        "audio_encoder.encoders0.0.self_attn.fsmn_block.weight": torch.arange(
            6, dtype=torch.float32
        ).reshape(2, 1, 3),
        "llm.lm_head.weight": torch.zeros((4, 4), dtype=torch.float32),
    }

    weights = _convert_weights(state, dtype="float32")

    assert "llm.lm_head.weight" not in weights
    assert weights["audio_encoder.encoders0.0.self_attn.fsmn_block.weight"].shape == (
        2,
        3,
        1,
    )
