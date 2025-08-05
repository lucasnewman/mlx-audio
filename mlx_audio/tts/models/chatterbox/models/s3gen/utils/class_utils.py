import mlx.nn as nn

from ..transformer.activation import Swish
from ..transformer.subsampling import (
    LinearNoSubsampling,
    EmbedinigNoSubsampling,
    Conv1dSubsampling2,
    Conv2dSubsampling4,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
)
from ..transformer.embedding import (
    PositionalEncoding,
    RelPositionalEncoding,
    WhisperPositionalEncoding,
    LearnablePositionalEncoding,
    NoPositionalEncoding,
    EspnetRelPositionalEncoding,
)
from ..transformer.attention import MultiHeadedAttention, RelPositionMultiHeadedAttention
from ..transformer.subsampling import LegacyLinearNoSubsampling


class Identity(nn.Module):
    """Identity module that passes input through unchanged."""

    def __call__(self, x):
        return x


COSYVOICE_ACTIVATION_CLASSES = {
    "hardtanh": nn.HardTanh,
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "selu": nn.SELU,
    "swish": Swish,
    "gelu": nn.GELU,
}

COSYVOICE_SUBSAMPLE_CLASSES = {
    "linear": LinearNoSubsampling,
    "linear_legacy": LegacyLinearNoSubsampling,
    "embed": EmbedinigNoSubsampling,
    "conv1d2": Conv1dSubsampling2,
    "conv2d": Conv2dSubsampling4,
    "conv2d6": Conv2dSubsampling6,
    "conv2d8": Conv2dSubsampling8,
    "paraformer_dummy": Identity,
}

COSYVOICE_EMB_CLASSES = {
    "embed": PositionalEncoding,
    "abs_pos": PositionalEncoding,
    "rel_pos": RelPositionalEncoding,
    "rel_pos_espnet": EspnetRelPositionalEncoding,
    "no_pos": NoPositionalEncoding,
    "abs_pos_whisper": WhisperPositionalEncoding,
    "embed_learnable_pe": LearnablePositionalEncoding,
}

COSYVOICE_ATTENTION_CLASSES = {
    "selfattn": MultiHeadedAttention,
    "rel_selfattn": RelPositionMultiHeadedAttention,
}
