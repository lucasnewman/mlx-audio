import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from einops.array_api import rearrange


def sequence_mask(length: mx.array, max_length: Optional[int] = None) -> mx.array:
    """Generate sequence mask for padding."""
    if max_length is None:
        max_length = int(mx.max(length).item())
    x = mx.arange(max_length, dtype=length.dtype)
    return x[None, :] < length[:, None]


class LayerNorm(nn.Module):
    """Layer normalization with learnable parameters."""
    
    def __init__(self, channels: int, eps: float = 1e-4):
        super().__init__()
        self.channels = channels
        self.eps = eps
        
        # Initialize parameters
        self.gamma = mx.ones((channels,))
        self.beta = mx.zeros((channels,))
    
    def __call__(self, x: mx.array) -> mx.array:
        n_dims = len(x.shape)
        mean = mx.mean(x, axis=1, keepdims=True)
        variance = mx.mean((x - mean) ** 2, axis=1, keepdims=True)
        
        x = (x - mean) * mx.rsqrt(variance + self.eps)
        
        shape = [1, -1] + [1] * (n_dims - 2)
        gamma = self.gamma.reshape(shape)
        beta = self.beta.reshape(shape)
        
        x = x * gamma + beta
        return x


class ConvReluNorm(nn.Module):
    """Convolutional block with ReLU activation and layer normalization."""
    
    def __init__(
        self, 
        in_channels: int, 
        hidden_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        n_layers: int, 
        p_dropout: float
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout
        
        # Build layers
        self.conv_layers = []
        self.norm_layers = []
        
        # First layer
        self.conv_layers.append(
            nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size // 2)
        )
        self.norm_layers.append(LayerNorm(hidden_channels))
        
        # Hidden layers
        for _ in range(n_layers - 1):
            self.conv_layers.append(
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2)
            )
            self.norm_layers.append(LayerNorm(hidden_channels))
        
        # Output projection
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        # Initialize projection to zero for residual connection
        self.proj.weight = mx.zeros_like(self.proj.weight)
        self.proj.bias = mx.zeros_like(self.proj.bias)
        
        # Dropout
        self.dropout = nn.Dropout(p_dropout) if p_dropout > 0 else nn.Identity()
    
    def __call__(self, x: mx.array, x_mask: mx.array) -> mx.array:
        x_org = x
        
        for i in range(self.n_layers):
            x = self.conv_layers[i](x * x_mask)
            x = self.norm_layers[i](x)
            x = nn.relu(x)
            x = self.dropout(x)
        
        x = x_org + self.proj(x)
        return x * x_mask


class DurationPredictor(nn.Module):
    """Duration predictor module."""
    
    def __init__(
        self, 
        in_channels: int, 
        filter_channels: int, 
        kernel_size: int, 
        p_dropout: float
    ):
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.p_dropout = p_dropout
        
        self.drop = nn.Dropout(p_dropout) if p_dropout > 0 else nn.Identity()
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)
    
    def __call__(self, x: mx.array, x_mask: mx.array) -> mx.array:
        x = self.conv_1(x * x_mask)
        x = nn.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = nn.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask


class RotaryPositionalEmbeddings(nn.Module):
    """
    RoPE module - Rotary encoding transforms pairs of features by rotating in the 2D plane.
    """
    
    def __init__(self, d: int, base: int = 10_000):
        """
        Args:
            d: Number of features (must be even)
            base: Base for calculating theta
        """
        super().__init__()
        self.base = base
        self.d = int(d)
        self.cos_cached = None
        self.sin_cached = None
    
    def _build_cache(self, x: mx.array):
        """Cache cos and sin values."""
        # Return if cache is already built
        if self.cos_cached is not None and x.shape[0] <= self.cos_cached.shape[0]:
            return
        
        # Get sequence length
        seq_len = x.shape[0]
        
        # Calculate theta values
        theta = 1.0 / (self.base ** (mx.arange(0, self.d, 2, dtype=mx.float32) / self.d))
        
        # Create position indexes
        seq_idx = mx.arange(seq_len, dtype=mx.float32)
        
        # Calculate the product of position index and theta
        idx_theta = mx.outer(seq_idx, theta)
        
        # Concatenate for full dimension
        idx_theta2 = mx.concatenate([idx_theta, idx_theta], axis=1)
        
        # Cache cos and sin values with shape for broadcasting
        self.cos_cached = mx.cos(idx_theta2)[:, None, None, :]
        self.sin_cached = mx.sin(idx_theta2)[:, None, None, :]
    
    def _neg_half(self, x: mx.array) -> mx.array:
        """Rearrange features for rotation."""
        d_2 = self.d // 2
        # Return [-x[d/2:], x[:d/2]]
        return mx.concatenate([-x[..., d_2:], x[..., :d_2]], axis=-1)
    
    def __call__(self, x: mx.array) -> mx.array:
        """
        Apply rotary embeddings.
        Input shape: (batch, heads, seq_len, dim)
        """
        # Rearrange to (seq_len, batch, heads, dim) for cache building
        x = rearrange(x, "b h t d -> t b h d")
        
        # Build cache
        self._build_cache(x)
        
        # Split features - apply RoPE only to first self.d features
        x_rope, x_pass = x[..., :self.d], x[..., self.d:]
        
        # Apply rotation
        neg_half_x = self._neg_half(x_rope)
        x_rope = (x_rope * self.cos_cached[:x.shape[0]]) + (neg_half_x * self.sin_cached[:x.shape[0]])
        
        # Concatenate and rearrange back
        x = mx.concatenate([x_rope, x_pass], axis=-1)
        return rearrange(x, "t b h d -> b h t d")


class MultiHeadAttention(nn.Module):
    """Multi-head attention with rotary positional embeddings."""
    
    def __init__(
        self,
        channels: int,
        out_channels: int,
        n_heads: int,
        heads_share: bool = True,
        p_dropout: float = 0.0,
        proximal_bias: bool = False,
        proximal_init: bool = False,
    ):
        super().__init__()
        assert channels % n_heads == 0
        
        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.heads_share = heads_share
        self.proximal_bias = proximal_bias
        self.p_dropout = p_dropout
        self.attn = None
        
        self.k_channels = channels // n_heads
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        
        # Rotary positional embeddings (applied to half the head dimension)
        self.query_rotary_pe = RotaryPositionalEmbeddings(int(self.k_channels * 0.5))
        self.key_rotary_pe = RotaryPositionalEmbeddings(int(self.k_channels * 0.5))
        
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout) if p_dropout > 0 else nn.Identity()
        
        if proximal_init:
            # Copy query weights to key
            self.conv_k.weight = self.conv_q.weight
            self.conv_k.bias = self.conv_q.bias
    
    def __call__(self, x: mx.array, c: mx.array, attn_mask: Optional[mx.array] = None) -> mx.array:
        """Forward pass."""
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)
        
        x, self.attn = self.attention(q, k, v, mask=attn_mask)
        
        x = self.conv_o(x)
        return x
    
    def attention(self, query: mx.array, key: mx.array, value: mx.array, mask: Optional[mx.array] = None):
        """Compute multi-head attention."""
        b, d, t_s, t_t = (*key.shape, query.shape[2])
        
        # Reshape to separate heads
        query = rearrange(query, "b (h c) t -> b h t c", h=self.n_heads)
        key = rearrange(key, "b (h c) t -> b h t c", h=self.n_heads)
        value = rearrange(value, "b (h c) t -> b h t c", h=self.n_heads)
        
        # Apply rotary embeddings
        query = self.query_rotary_pe(query)
        key = self.key_rotary_pe(key)
        
        # Compute attention scores
        scores = mx.matmul(query, key.swapaxes(-2, -1)) / math.sqrt(self.k_channels)
        
        # Add proximal bias if enabled
        if self.proximal_bias:
            assert t_s == t_t, "Proximal bias is only available for self-attention."
            bias = self._attention_bias_proximal(t_s)
            scores = scores + bias
        
        # Apply mask
        if mask is not None:
            scores = mx.where(mask == 0, -1e4, scores)
        
        # Compute attention weights
        p_attn = mx.softmax(scores, axis=-1)
        p_attn = self.drop(p_attn)
        
        # Apply attention to values
        output = mx.matmul(p_attn, value)
        output = output.swapaxes(2, 3).reshape(b, d, t_t)
        
        return output, p_attn
    
    @staticmethod
    def _attention_bias_proximal(length: int) -> mx.array:
        """Generate proximal bias for attention."""
        r = mx.arange(length, dtype=mx.float32)
        diff = r[None, :] - r[:, None]
        return -mx.log1p(mx.abs(diff))[None, None, :, :]


class FFN(nn.Module):
    """Feed-forward network with convolutions."""
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        filter_channels: int, 
        kernel_size: int, 
        p_dropout: float = 0.0
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.drop = nn.Dropout(p_dropout) if p_dropout > 0 else nn.Identity()
    
    def __call__(self, x: mx.array, x_mask: mx.array) -> mx.array:
        x = self.conv_1(x * x_mask)
        x = nn.relu(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        return x * x_mask


class Encoder(nn.Module):
    """Transformer encoder with multiple layers."""
    
    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int = 1,
        p_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        
        self.drop = nn.Dropout(p_dropout) if p_dropout > 0 else nn.Identity()
        
        # Build layers
        self.attn_layers = []
        self.norm_layers_1 = []
        self.ffn_layers = []
        self.norm_layers_2 = []
        
        for _ in range(n_layers):
            self.attn_layers.append(
                MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout)
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))
    
    def __call__(self, x: mx.array, x_mask: mx.array) -> mx.array:
        # Create attention mask
        attn_mask = x_mask[:, None, :] * x_mask[:, :, None]
        
        for i in range(self.n_layers):
            x = x * x_mask
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)
            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        
        x = x * x_mask
        return x


class TextEncoder(nn.Module):
    """Text encoder with transformer architecture."""
    
    def __init__(
        self,
        encoder_type: str,
        encoder_params,
        duration_predictor_params,
        n_vocab: int,
        n_spks: int = 1,
        spk_emb_dim: int = 128,
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.n_vocab = n_vocab
        self.n_feats = encoder_params.n_feats
        self.n_channels = encoder_params.n_channels
        self.spk_emb_dim = spk_emb_dim
        self.n_spks = n_spks
        
        self.emb = nn.Embedding(n_vocab, self.n_channels)
        
        # Prenet
        if encoder_params.prenet:
            self.prenet = ConvReluNorm(
                self.n_channels,
                self.n_channels,
                self.n_channels,
                kernel_size=5,
                n_layers=3,
                p_dropout=0.5,
            )
        else:
            self.prenet = None
        
        # Main encoder
        encoder_dim = encoder_params.n_channels + (spk_emb_dim if n_spks > 1 else 0)
        self.encoder = Encoder(
            encoder_dim,
            encoder_params.filter_channels,
            encoder_params.n_heads,
            encoder_params.n_layers,
            encoder_params.kernel_size,
            encoder_params.p_dropout,
        )
        
        # Output projections
        self.proj_m = nn.Conv1d(encoder_dim, self.n_feats, 1)
        self.proj_w = DurationPredictor(
            encoder_dim,
            duration_predictor_params.filter_channels_dp,
            duration_predictor_params.kernel_size,
            duration_predictor_params.p_dropout,
        )
    
    def __call__(
        self, 
        x: mx.array, 
        x_lengths: mx.array, 
        spks: Optional[mx.array] = None
    ):
        """
        Run forward pass to the transformer based encoder and duration predictor.
        
        Args:
            x: text input, shape (batch_size, max_text_length)
            x_lengths: text input lengths, shape (batch_size,)
            spks: speaker ids, shape (batch_size,)
        
        Returns:
            mu: average output of the encoder, shape (batch_size, n_feats, max_text_length)
            logw: log duration predicted by the duration predictor, shape (batch_size, 1, max_text_length)
            x_mask: mask for the text input, shape (batch_size, 1, max_text_length)
        """
        # Embed tokens
        x = self.emb(x) * math.sqrt(self.n_channels)
        x = x.swapaxes(1, 2)  # (B, T, C) -> (B, C, T)
        
        # Create mask
        x_mask = sequence_mask(x_lengths, x.shape[2])
        x_mask = x_mask[:, None, :].astype(x.dtype)
        
        # Apply prenet
        if self.prenet is not None:
            x = self.prenet(x, x_mask)
        else:
            x = x
        
        # Add speaker embeddings if multi-speaker
        if self.n_spks > 1:
            spk_emb = spks[:, :, None]
            spk_emb = mx.repeat(spk_emb, x.shape[-1], axis=2)
            x = mx.concatenate([x, spk_emb], axis=1)
        
        # Encode
        x = self.encoder(x, x_mask)
        mu = self.proj_m(x) * x_mask
        
        # Predict durations (with stop gradient)
        x_dp = mx.stop_gradient(x)
        logw = self.proj_w(x_dp, x_mask)
        
        return mu, logw, x_mask
