LLAMA_520M_CONFIG_DICT = dict(
    model_type="llama",
    vocab_size=8,
    hidden_size=1024,
    intermediate_size=4096,
    num_hidden_layers=30,
    num_attention_heads=16,
    rms_norm_eps=1e-05,
    # Optional fields
    head_dim=64,
    max_position_embeddings=131072,
    num_key_value_heads=16,
    attention_bias=False,
    mlp_bias=False,
    rope_theta=500000.0,
    rope_traditional=False,  # original didn’t have this; default is False
    rope_scaling=dict(
        factor=8.0,
        high_freq_factor=4.0,
        low_freq_factor=1.0,
        original_max_position_embeddings=8192,
        rope_type="llama3"
    ),
    tie_word_embeddings=False
)

LLAMA_CONFIGS = {
    "Llama_520M": LLAMA_520M_CONFIG_DICT,
}
