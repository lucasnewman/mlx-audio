import mlx.core as mx
import mlx.nn as nn
from einops.array_api import rearrange
from typing import Optional


# Helper functions
def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)


# Helper classes
class Identity(nn.Module):
    """Identity module that passes input through unchanged."""

    def __call__(self, x: mx.array) -> mx.array:
        return x


class Swish(nn.Module):
    """Swish activation function."""

    def __call__(self, x: mx.array) -> mx.array:
        return x * mx.sigmoid(x)


class GLU(nn.Module):
    """Gated Linear Unit activation."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def __call__(self, x: mx.array) -> mx.array:
        # Split along the specified dimension
        out, gate = mx.split(x, 2, axis=self.dim)
        return out * mx.sigmoid(gate)


class DepthWiseConv1d(nn.Module):
    """Depth-wise 1D convolution."""

    def __init__(self, chan_in: int, chan_out: int, kernel_size: int, padding: tuple):
        super().__init__()
        self.padding = padding
        # For depth-wise conv, chan_in must equal chan_out and groups = channels
        assert chan_in == chan_out, "Depth-wise conv requires chan_in == chan_out"

        # Use groups=chan_in for depth-wise convolution
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, padding=0, groups=chan_in)

    def __call__(self, x: mx.array) -> mx.array:
        # Pad the input manually since we set padding=0 in conv
        if self.padding[0] > 0 or self.padding[1] > 0:
            x = mx.pad(x, [(0, 0), (0, 0), self.padding])

        # Apply the depth-wise convolution
        return self.conv(x)


# Attention, feedforward, and conv module
class Scale(nn.Module):
    """Scale module that multiplies output by a constant."""

    def __init__(self, scale: float, fn: nn.Module):
        super().__init__()
        self.fn = fn
        self.scale = scale

    def __call__(self, x: mx.array, **kwargs) -> mx.array:
        return self.fn(x, **kwargs) * self.scale


class PreNorm(nn.Module):
    """Pre-normalization wrapper."""

    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def __call__(self, x: mx.array, **kwargs) -> mx.array:
        x = self.norm(x)
        return self.fn(x, **kwargs)


class Attention(nn.Module):
    """Multi-head attention with relative positional embeddings."""

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0, max_pos_emb: int = 512):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.max_pos_emb = max_pos_emb
        self.rel_pos_emb = nn.Embedding(2 * max_pos_emb + 1, dim_head)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def __call__(
        self, x: mx.array, context: Optional[mx.array] = None, mask: Optional[mx.array] = None, context_mask: Optional[mx.array] = None
    ) -> mx.array:
        n = x.shape[-2]
        h = self.heads
        max_pos_emb = self.max_pos_emb
        has_context = exists(context)
        context = default(context, x)

        # Compute queries, keys, values
        q = self.to_q(x)
        kv = self.to_kv(context)
        k, v = mx.split(kv, 2, axis=-1)

        # Reshape for multi-head attention
        q = rearrange(q, "b n (h d) -> b h n d", h=h)
        k = rearrange(k, "b n (h d) -> b h n d", h=h)
        v = rearrange(v, "b n (h d) -> b h n d", h=h)

        # Compute attention scores
        # MLX doesn't have einsum, so we use matmul
        # einsum('b h i d, b h j d -> b h i j', q, k) is equivalent to:
        dots = mx.matmul(q, k.swapaxes(-2, -1)) * self.scale

        # Shaw's relative positional embedding
        seq = mx.arange(n)
        dist = seq[:, None] - seq[None, :]
        dist = mx.clip(dist, -max_pos_emb, max_pos_emb) + max_pos_emb
        rel_pos_emb = self.rel_pos_emb(dist)

        # Compute positional attention
        # einsum('b h n d, n r d -> b h n r', q, rel_pos_emb) is equivalent to:
        # q: (b, h, n, d), rel_pos_emb: (n, r, d) where r=n
        pos_attn = mx.matmul(q, rel_pos_emb.swapaxes(-2, -1)) * self.scale
        dots = dots + pos_attn

        # Apply masks if provided
        if exists(mask) or exists(context_mask):
            mask = default(mask, lambda: mx.ones(x.shape[:2]))
            context_mask = default(context_mask, mask) if not has_context else default(context_mask, lambda: mx.ones(context.shape[:2]))

            # Create attention mask
            mask = mask[:, None, :, None] * context_mask[:, None, None, :]
            mask_value = -1e9  # Large negative value for masking
            dots = mx.where(mask, dots, mask_value)

        # Apply softmax
        attn = mx.softmax(dots, axis=-1)

        # Apply attention to values
        # einsum('b h i j, b h j d -> b h i d', attn, v) is equivalent to:
        out = mx.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)

        return self.dropout(out)


class FeedForward(nn.Module):
    """Feed-forward network with Swish activation."""

    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            Swish(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.net(x)


class BatchNorm1d(nn.Module):
    """1D Batch Normalization for MLX."""

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        # Parameters
        if affine:
            self.weight = mx.ones((num_features,))
            self.bias = mx.zeros((num_features,))
        else:
            self.weight = None
            self.bias = None

        # Running statistics (not trainable)
        # In MLX, we don't have a direct way to mark these as non-trainable
        # They will be updated during forward pass but not by optimizer
        self.running_mean = mx.zeros((num_features,))
        self.running_var = mx.ones((num_features,))

        # Note: In inference mode, we'll always use running statistics
        # MLX doesn't have a built-in training/eval mode like PyTorch

    def __call__(self, x: mx.array) -> mx.array:
        # x shape: (batch, channels, length)
        # For inference, always use running statistics
        mean = self.running_mean[None, :, None]
        var = self.running_var[None, :, None]

        # Normalize
        x = (x - mean) / mx.sqrt(var + self.eps)

        # Scale and shift if affine
        if self.affine:
            x = x * self.weight[None, :, None] + self.bias[None, :, None]

        return x


class ConformerConvModule(nn.Module):
    """Conformer convolution module."""

    def __init__(self, dim: int, causal: bool = False, expansion_factor: int = 2, kernel_size: int = 31, dropout: float = 0.0):
        super().__init__()

        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        # Build the network
        self.layer_norm = nn.LayerNorm(dim)
        self.conv1 = nn.Conv1d(dim, inner_dim * 2, 1)
        self.glu = GLU(dim=1)
        self.depthwise = DepthWiseConv1d(inner_dim, inner_dim, kernel_size=kernel_size, padding=padding)
        self.batch_norm = BatchNorm1d(inner_dim) if not causal else nn.Identity()
        self.swish = Swish()
        self.conv2 = nn.Conv1d(inner_dim, dim, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def __call__(self, x: mx.array) -> mx.array:
        # Layer norm
        x = self.layer_norm(x)

        # Rearrange for convolution: (b n c) -> (b c n)
        x = rearrange(x, "b n c -> b c n")

        # First convolution
        x = self.conv1(x)

        # GLU activation
        x = self.glu(x)

        # Depth-wise convolution
        x = self.depthwise(x)

        # Batch norm (or identity if causal)
        x = self.batch_norm(x)

        # Swish activation
        x = self.swish(x)

        # Second convolution
        x = self.conv2(x)

        # Rearrange back: (b c n) -> (b n c)
        x = rearrange(x, "b c n -> b n c")

        # Dropout
        x = self.dropout(x)

        return x


# Conformer Block
class ConformerBlock(nn.Module):
    """Single Conformer block."""

    def __init__(
        self,
        *,
        dim: int,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
        conv_expansion_factor: int = 2,
        conv_kernel_size: int = 31,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        conv_dropout: float = 0.0,
        conv_causal: bool = False,
    ):
        super().__init__()

        # Components
        self.ff1 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        self.attn = Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout)
        self.conv = ConformerConvModule(
            dim=dim, causal=conv_causal, expansion_factor=conv_expansion_factor, kernel_size=conv_kernel_size, dropout=conv_dropout
        )
        self.ff2 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)

        # Wrap with prenorm and scaling
        self.attn = PreNorm(dim, self.attn)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))

        self.post_norm = nn.LayerNorm(dim)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        x = self.ff1(x) + x
        x = self.attn(x, mask=mask) + x
        x = self.conv(x) + x
        x = self.ff2(x) + x
        x = self.post_norm(x)
        return x


# Conformer
class Conformer(nn.Module):
    """Full Conformer model."""

    def __init__(
        self,
        dim: int,
        *,
        depth: int,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
        conv_expansion_factor: int = 2,
        conv_kernel_size: int = 31,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        conv_dropout: float = 0.0,
        conv_causal: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth

        # Create layers as a list first
        layers = []
        for i in range(depth):
            layers.append(
                ConformerBlock(
                    dim=dim,
                    dim_head=dim_head,
                    heads=heads,
                    ff_mult=ff_mult,
                    conv_expansion_factor=conv_expansion_factor,
                    conv_kernel_size=conv_kernel_size,
                    conv_causal=conv_causal,
                    attn_dropout=attn_dropout,
                    ff_dropout=ff_dropout,
                    conv_dropout=conv_dropout,
                )
            )

        # Store layers
        self.layers = layers

    def __call__(self, x: mx.array) -> mx.array:
        for block in self.layers:
            x = block(x)
        return x
