from dataclasses import dataclass, field, fields

from mlx_audio.lm.models.gemma4_text import ModelArgs
from mlx_audio.stt.models.qwen2_audio.config import EncoderConfig


@dataclass
class TextConfig(ModelArgs):
    hidden_size: int = 2816
    num_hidden_layers: int = 30
    intermediate_size: int = 2112
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    num_global_key_value_heads: int = 2
    num_kv_shared_layers: int = 0
    hidden_size_per_layer_input: int = 0
    sliding_window: int = 1024
    sliding_window_pattern: int = 6
    max_position_embeddings: int = 262144
    attention_k_eq_v: bool = True
    use_double_wide_mlp: bool = False
    enable_moe_block: bool = True
    num_experts: int = 128
    top_k_experts: int = 8
    moe_intermediate_size: int = 704

    def __post_init__(self):
        super().__post_init__()
        if self.model_type != "gemma4_text":
            raise ValueError("SOV-Transcribe requires a Gemma 4 text decoder")
        if self.num_kv_shared_layers or self.hidden_size_per_layer_input:
            raise ValueError(
                "SOV-Transcribe does not use shared KV or per-layer inputs"
            )
        if len(self.layer_types) != self.num_hidden_layers or not set(
            self.layer_types
        ).issubset({"sliding_attention", "full_attention"}):
            raise ValueError("Expected one supported attention type per decoder layer")
        if self.enable_moe_block and not (
            self.num_experts
            and self.top_k_experts
            and 0 < self.top_k_experts <= self.num_experts
        ):
            raise ValueError("top_k_experts must be in [1, num_experts]")


@dataclass
class ModelConfig:
    model_type: str = "sovereignai_transcribe"
    audio_config: EncoderConfig = field(default_factory=EncoderConfig)
    text_config: TextConfig = field(default_factory=TextConfig)
    audio_token_id: int = 258881
    downsample_factor: int = 4
    projector_hidden_act: str = "gelu"
    tie_word_embeddings: bool = True
    eos_token_id: int | list[int] = 1

    def __post_init__(self):
        if isinstance(self.audio_config, dict):
            self.audio_config = EncoderConfig.from_dict(self.audio_config)
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)
        if self.downsample_factor <= 0 or (
            self.audio_config.max_source_positions % self.downsample_factor
        ):
            raise ValueError(
                "Audio encoder length must be divisible by downsample_factor"
            )
        if self.projector_hidden_act != "gelu":
            raise ValueError("Only the GELU audio projector is supported")

    @classmethod
    def from_dict(cls, params):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in params.items() if k in names})
