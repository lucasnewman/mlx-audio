import math

import mlx.core as mx

from mlx_audio.tts.models.kokoro.modules import AlbertLayer, AlbertModelArgs


def test_albert_layer_uses_gelu_new():
    """Kokoro's PL-BERT is a transformers ALBERT with the default hidden_act="gelu_new" (tanh approximation)."""
    layer = AlbertLayer(
        AlbertModelArgs(
            num_hidden_layers=1,
            num_attention_heads=1,
            hidden_size=4,
            intermediate_size=8,
            max_position_embeddings=8,
        )
    )
    x = mx.array([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0])
    gelu_new = [
        0.5 * v * (1 + math.tanh(math.sqrt(2 / math.pi) * (v + 0.044715 * v**3)))
        for v in x.tolist()
    ]
    assert mx.allclose(layer.activation(x), mx.array(gelu_new), atol=1e-6)
