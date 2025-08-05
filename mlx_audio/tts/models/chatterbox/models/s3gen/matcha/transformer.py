from typing import Optional, Dict, Any
import math

import mlx.core as mx
import mlx.nn as nn

class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        out_bias: bool = True,
        scale_qk: bool = True,
        only_cross_attention: bool = False,
        eps: float = 1e-5,
        rescale_output_factor: float = 1.0,
        residual_connection: bool = False,
        out_dim: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        
        self.inner_dim = dim_head * heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.is_cross_attention = cross_attention_dim is not None
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.rescale_output_factor = rescale_output_factor
        self.residual_connection = residual_connection
        self.dropout = dropout
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.scale_qk = scale_qk
        self.scale = dim_head**-0.5 if self.scale_qk else 1.0
        self.heads = heads
        self.only_cross_attention = only_cross_attention
        
        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        
        if not self.only_cross_attention:
            self.to_k = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=bias)
            self.to_v = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=bias)
        else:
            self.to_k = None
            self.to_v = None
        
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, self.out_dim, bias=out_bias),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
    
    def reshape_heads_to_batch_dim(self, tensor: mx.array) -> mx.array:
        """
        Reshape from (batch, seq, dim) to (batch * heads, seq, dim_per_head)
        """
        batch_size, seq_len, dim = tensor.shape
        head_dim = dim // self.heads
        tensor = tensor.reshape(batch_size, seq_len, self.heads, head_dim)
        tensor = tensor.transpose(0, 2, 1, 3)  # (batch, heads, seq, head_dim)
        tensor = tensor.reshape(batch_size * self.heads, seq_len, head_dim)
        return tensor
    
    def reshape_batch_dim_to_heads(self, tensor: mx.array) -> mx.array:
        """
        Reshape from (batch * heads, seq, dim_per_head) to (batch, seq, dim)
        """
        batch_heads, seq_len, head_dim = tensor.shape
        batch_size = batch_heads // self.heads
        tensor = tensor.reshape(batch_size, self.heads, seq_len, head_dim)
        tensor = tensor.transpose(0, 2, 1, 3)  # (batch, seq, heads, head_dim)
        tensor = tensor.reshape(batch_size, seq_len, self.heads * head_dim)
        return tensor
    
    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        **kwargs,
    ) -> mx.array:
        residual = hidden_states
        input_ndim = hidden_states.ndim
        
        # (batch, channel, height, width)
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.reshape(batch_size, channel, height * width)
            hidden_states = hidden_states.transpose(0, 2, 1)
        
        batch_size, sequence_length, _ = hidden_states.shape
        
        query = self.to_q(hidden_states)
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        
        if self.only_cross_attention:
            raise ValueError("only_cross_attention requires encoder_hidden_states")
        
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)
        
        dim_head = self.inner_dim // self.heads
        
        query = query.reshape(batch_size, -1, self.heads, dim_head).transpose(0, 2, 1, 3)
        key = key.reshape(batch_size, -1, self.heads, dim_head).transpose(0, 2, 1, 3)
        value = value.reshape(batch_size, -1, self.heads, dim_head).transpose(0, 2, 1, 3)
        
        scores = (query @ key.transpose(0, 1, 3, 2)) * self.scale
        
        if attention_mask is not None:
            # Expand mask for heads if needed
            if attention_mask.ndim == 3:
                # (batch, 1, seq) -> (batch, heads, 1, seq)
                attention_mask = mx.expand_dims(attention_mask, axis=1)
                attention_mask = mx.repeat(attention_mask, self.heads, axis=1)
            scores = scores + attention_mask
        
        if self.upcast_softmax:
            scores = scores.astype(mx.float32)
        
        attn_weights = mx.softmax(scores, axis=-1)
        
        if self.upcast_softmax:
            attn_weights = attn_weights.astype(query.dtype)
        
        hidden_states = attn_weights @ value
        hidden_states = hidden_states.transpose(0, 2, 1, 3)
        hidden_states = hidden_states.reshape(batch_size, -1, self.inner_dim)
        hidden_states = self.to_out(hidden_states)
        
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(0, 2, 1)
            hidden_states = hidden_states.reshape(batch_size, channel, height, width)
        
        if self.residual_connection:
            hidden_states = hidden_states + residual
        
        hidden_states = hidden_states / self.rescale_output_factor
        
        return hidden_states


class GELU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none"):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)
        self.approximate = approximate
    
    def __call__(self, x: mx.array) -> mx.array:
        x = self.proj(x)
        if self.approximate == "tanh":
            return 0.5 * x * (1.0 + mx.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))
        else:
            return nn.gelu(x)


class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)
    
    def __call__(self, x: mx.array) -> mx.array:
        x, gate = mx.split(self.proj(x), 2, axis=-1)
        return x * nn.gelu(gate)


class ApproximateGELU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)
    
    def __call__(self, x: mx.array) -> mx.array:
        x = self.proj(x)
        return 0.5 * x * (1.0 + mx.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


class Identity(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return x


class SnakeBeta(nn.Module):
    """
    A modified Snake function which uses separate parameters for the magnitude of the periodic components.
    
    Based on the paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
    https://arxiv.org/abs/2006.08195
    """
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        alpha: float = 1.0, 
        alpha_trainable: bool = True, 
        alpha_logscale: bool = True
    ):
        super().__init__()
        self.in_features = out_features if not isinstance(out_features, list) else out_features
        self.proj = nn.Linear(in_features, out_features)
        
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:
            self.alpha = mx.zeros((out_features,)) * alpha
            self.beta = mx.zeros((out_features,)) * alpha
        else:
            self.alpha = mx.ones((out_features,)) * alpha
            self.beta = mx.ones((out_features,)) * alpha
        
        self.alpha_trainable = alpha_trainable
        self.no_div_by_zero = 1e-9
    
    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass: SnakeBeta(x) = x + 1/b * sin^2(x*a)
        """
        x = self.proj(x)
        
        if self.alpha_logscale:
            alpha = mx.exp(self.alpha)
            beta = mx.exp(self.beta)
        else:
            alpha = self.alpha
            beta = self.beta
        
        x = x + (1.0 / (beta + self.no_div_by_zero)) * mx.power(mx.sin(x * alpha), 2)
        return x


class FeedForward(nn.Module):
    """
    A feed-forward layer.
    """
    
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        
        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim)
        elif activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh")
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim)
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim)
        elif activation_fn == "snakebeta":
            act_fn = SnakeBeta(dim, inner_dim)
        else:
            raise ValueError(f"Unknown activation function: {activation_fn}")
        
        self.net = nn.Sequential(
            act_fn,
            nn.Dropout(dropout) if dropout > 0 else Identity(),
            nn.Linear(inner_dim, dim_out),
            nn.Dropout(dropout) if final_dropout and dropout > 0 else Identity(),
        )
    
    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.net(hidden_states)


class BasicTransformerBlock(nn.Module):
    """
    A basic Transformer block for flow matching.
    """
    
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        final_dropout: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        
        self.norm1 = nn.LayerNorm(dim, affine=norm_elementwise_affine)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
        )
        
        if cross_attention_dim is not None or double_self_attention:
            self.norm2 = nn.LayerNorm(dim, affine=norm_elementwise_affine)
            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim if not double_self_attention else None,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
        else:
            self.norm2 = None
            self.attn2 = None
        
        self.norm3 = nn.LayerNorm(dim, affine=norm_elementwise_affine)
        self.ff = FeedForward(
            dim, 
            dropout=dropout, 
            activation_fn=activation_fn, 
            final_dropout=final_dropout
        )
        
        self._chunk_size = None
        self._chunk_dim = 0
    
    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        encoder_hidden_states: Optional[mx.array] = None,
        encoder_attention_mask: Optional[mx.array] = None,
        timestep: Optional[mx.array] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        class_labels: Optional[mx.array] = None,
    ) -> mx.array:
        norm_hidden_states = self.norm1(hidden_states)
        
        cross_attention_kwargs = cross_attention_kwargs or {}
        
        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=encoder_attention_mask if self.only_cross_attention else attention_mask,
            **cross_attention_kwargs,
        )
        
        hidden_states = attn_output + hidden_states
        
        if self.attn2 is not None:
            norm_hidden_states = self.norm2(hidden_states)
            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states
        
        norm_hidden_states = self.norm3(hidden_states)
        
        if self._chunk_size is not None:
            ff_output = self._chunked_feed_forward(norm_hidden_states)
        else:
            ff_output = self.ff(norm_hidden_states)
        
        hidden_states = ff_output + hidden_states
        
        return hidden_states
    
    def _chunked_feed_forward(self, hidden_states: mx.array) -> mx.array:
        """Apply feed-forward in chunks for memory efficiency."""
        if hidden_states.shape[self._chunk_dim] % self._chunk_size != 0:
            raise ValueError(
                f"Hidden states dimension {hidden_states.shape[self._chunk_dim]} "
                f"must be divisible by chunk size {self._chunk_size}"
            )
        
        num_chunks = hidden_states.shape[self._chunk_dim] // self._chunk_size
        chunks = mx.split(hidden_states, num_chunks, axis=self._chunk_dim)
        
        return mx.concatenate(
            [self.ff(chunk) for chunk in chunks], 
            axis=self._chunk_dim
        )
