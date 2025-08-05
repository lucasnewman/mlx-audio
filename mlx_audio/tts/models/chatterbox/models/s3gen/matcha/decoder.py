import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from einops.array_api import pack, rearrange, repeat

from .conformer import ConformerBlock
from .transformer import BasicTransformerBlock


def get_activation(act_fn: str):
    """Get activation function by name."""
    if act_fn == "silu" or act_fn == "swish":
        return nn.SiLU()
    elif act_fn == "mish":
        return Mish()
    elif act_fn == "gelu":
        return nn.GELU()
    elif act_fn == "relu":
        return nn.ReLU()
    elif act_fn == "snake":
        assert False, "Snake activation is not implemented"
    else:
        raise ValueError(f"Unknown activation function: {act_fn}")


class Mish(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return x * mx.tanh(nn.softplus(x))


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0, "SinusoidalPosEmb requires dim to be even"

    def __call__(self, x: mx.array, scale: float = 1000) -> mx.array:
        if x.ndim < 1:
            x = mx.expand_dims(x, axis=0)

        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = mx.exp(mx.arange(half_dim, dtype=x.dtype) * -emb)
        emb = scale * x[:, None] * emb[None, :]
        emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)
        return emb


class GroupNorm(nn.Module):
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        assert num_channels % num_groups == 0, f"num_channels ({num_channels}) must be divisible by num_groups ({num_groups})"

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if affine:
            self.weight = mx.ones((num_channels,))
            self.bias = mx.zeros((num_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        # x shape: (batch, channels, length)
        batch, channels, length = x.shape

        # Reshape to separate groups
        x = x.reshape(batch, self.num_groups, channels // self.num_groups, length)

        # Compute mean and variance per group
        mean = mx.mean(x, axis=(2, 3), keepdims=True)
        var = mx.var(x, axis=(2, 3), keepdims=True)

        # Normalize
        x = (x - mean) / mx.sqrt(var + self.eps)

        # Reshape back
        x = x.reshape(batch, channels, length)

        # Apply affine transformation if enabled
        if self.affine:
            x = x * self.weight[None, :, None] + self.bias[None, :, None]

        return x


class Block1D(nn.Module):
    def __init__(self, dim: int, dim_out: int, groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(dim, dim_out, kernel_size=3, padding=1),
            GroupNorm(groups, dim_out),
            Mish(),
        )

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        output = self.block(x * mask)
        return output * mask


class ResnetBlock1D(nn.Module):
    def __init__(self, dim: int, dim_out: int, time_emb_dim: int, groups: int = 8):
        super().__init__()
        self.mlp = nn.Sequential(Mish(), nn.Linear(time_emb_dim, dim_out))

        self.block1 = Block1D(dim, dim_out, groups=groups)
        self.block2 = Block1D(dim_out, dim_out, groups=groups)

        self.res_conv = nn.Conv1d(dim, dim_out, kernel_size=1)

    def __call__(self, x: mx.array, mask: mx.array, time_emb: mx.array) -> mx.array:
        h = self.block1(x, mask)
        h = h + self.mlp(time_emb)[:, :, None]
        h = self.block2(h, mask)
        output = h + self.res_conv(x * mask)
        return output


class Downsample1D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(x)


class ConvTranspose1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        scale = 1.0 / math.sqrt(in_channels * kernel_size)
        self.weight = mx.random.uniform(-scale, scale, shape=(in_channels, out_channels, kernel_size))
        self.bias = mx.zeros((out_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        # x shape: (batch, in_channels, length)
        batch, in_channels, length = x.shape

        # Calculate output length
        output_length = (length - 1) * self.stride - 2 * self.padding + self.kernel_size

        # Upsample by inserting zeros
        if self.stride > 1:
            # Insert zeros between elements
            upsampled_length = length + (length - 1) * (self.stride - 1)
            upsampled = mx.zeros((batch, in_channels, upsampled_length))
            upsampled[:, :, :: self.stride] = x
            x = upsampled

        # Apply transposed convolution using regular convolution with flipped kernel
        weight_flipped = self.weight[:, :, ::-1]

        # Pad the input
        pad_total = self.kernel_size - 1
        pad_left = pad_total - self.padding
        pad_right = pad_total - self.padding

        if pad_left > 0 or pad_right > 0:
            x = mx.pad(x, [(0, 0), (0, 0), (pad_left, pad_right)])

        # Perform the convolution
        weight_reshaped = weight_flipped.swapaxes(0, 1)

        output = mx.conv1d(x, weight_reshaped, stride=1, padding=0)
        output = output + self.bias[None, :, None]

        # Trim to match expected output length
        if output.shape[2] != output_length:
            output = output[:, :, :output_length]

        return output


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: Optional[int] = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim: Optional[int] = None,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim)

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None

        self.act = get_activation(act_fn)

        time_embed_dim_out = out_dim if out_dim is not None else time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out)

        if post_act_fn is not None:
            self.post_act = get_activation(post_act_fn)
        else:
            self.post_act = None

    def __call__(self, sample: mx.array, condition: Optional[mx.array] = None) -> mx.array:
        if condition is not None and self.cond_proj is not None:
            sample = sample + self.cond_proj(condition)

        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)

        return sample


class Upsample1D(nn.Module):
    def __init__(self, channels: int, use_conv: bool = False, use_conv_transpose: bool = True, out_channels: Optional[int] = None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose

        if use_conv_transpose:
            self.conv = ConvTranspose1d(channels, self.out_channels, kernel_size=4, stride=2, padding=1)
        elif use_conv:
            self.conv = nn.Conv1d(channels, self.out_channels, kernel_size=3, padding=1)
        else:
            self.conv = None

    def __call__(self, x: mx.array) -> mx.array:
        assert x.shape[1] == self.channels

        if self.use_conv_transpose:
            return self.conv(x)

        batch, channels, length = x.shape
        x = mx.repeat(x, 2, axis=2)

        if self.use_conv and self.conv is not None:
            x = self.conv(x)

        return x


class ConformerWrapper(ConformerBlock):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array,
        encoder_hidden_states: Optional[mx.array] = None,
        encoder_attention_mask: Optional[mx.array] = None,
        timestep: Optional[mx.array] = None,
    ) -> mx.array:
        return super().__call__(x=hidden_states, mask=attention_mask.astype(mx.bool_))


class Decoder(nn.Module):
    """Flow matching decoder with U-Net architecture."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: tuple = (256, 256),
        dropout: float = 0.05,
        attention_head_dim: int = 64,
        n_blocks: int = 1,
        num_mid_blocks: int = 2,
        num_heads: int = 4,
        act_fn: str = "snake",
        down_block_type: str = "transformer",
        mid_block_type: str = "transformer",
        up_block_type: str = "transformer",
    ):
        super().__init__()
        channels = tuple(channels)
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Time embeddings
        self.time_embeddings = SinusoidalPosEmb(in_channels)
        time_embed_dim = channels[0] * 4
        self.time_mlp = TimestepEmbedding(
            in_channels=in_channels,
            time_embed_dim=time_embed_dim,
            act_fn="silu",
        )

        # Down blocks
        self.down_blocks = []
        output_channel = in_channels
        for i in range(len(channels)):
            input_channel = output_channel
            output_channel = channels[i]
            is_last = i == len(channels) - 1

            resnet = ResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)

            transformer_blocks = [
                self.get_block(
                    down_block_type,
                    output_channel,
                    attention_head_dim,
                    num_heads,
                    dropout,
                    act_fn,
                )
                for _ in range(n_blocks)
            ]

            downsample = (
                Downsample1D(output_channel) if not is_last else nn.Conv1d(output_channel, output_channel, kernel_size=3, padding=1)
            )

            self.down_blocks.append((resnet, transformer_blocks, downsample))

        # Mid blocks
        self.mid_blocks = []
        for i in range(num_mid_blocks):
            input_channel = channels[-1]
            output_channel = channels[-1]

            resnet = ResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)

            transformer_blocks = [
                self.get_block(
                    mid_block_type,
                    output_channel,
                    attention_head_dim,
                    num_heads,
                    dropout,
                    act_fn,
                )
                for _ in range(n_blocks)
            ]

            self.mid_blocks.append((resnet, transformer_blocks))

        # Up blocks
        self.up_blocks = []
        channels_reversed = channels[::-1] + (channels[0],)
        for i in range(len(channels_reversed) - 1):
            input_channel = channels_reversed[i]
            output_channel = channels_reversed[i + 1]
            is_last = i == len(channels_reversed) - 2

            resnet = ResnetBlock1D(
                dim=2 * input_channel,
                dim_out=output_channel,
                time_emb_dim=time_embed_dim,
            )

            transformer_blocks = [
                self.get_block(
                    up_block_type,
                    output_channel,
                    attention_head_dim,
                    num_heads,
                    dropout,
                    act_fn,
                )
                for _ in range(n_blocks)
            ]

            upsample = (
                Upsample1D(output_channel, use_conv_transpose=True)
                if not is_last
                else nn.Conv1d(output_channel, output_channel, kernel_size=3, padding=1)
            )

            self.up_blocks.append((resnet, transformer_blocks, upsample))

        # Final layers
        self.final_block = Block1D(channels_reversed[-1], channels_reversed[-1])
        self.final_proj = nn.Conv1d(channels_reversed[-1], self.out_channels, kernel_size=1)

    @staticmethod
    def get_block(block_type: str, dim: int, attention_head_dim: int, num_heads: int, dropout: float, act_fn: str):
        """Create a transformer or conformer block."""
        if block_type == "conformer":
            block = ConformerWrapper(
                dim=dim,
                dim_head=attention_head_dim,
                heads=num_heads,
                ff_mult=1,
                conv_expansion_factor=2,
                ff_dropout=dropout,
                attn_dropout=dropout,
                conv_dropout=dropout,
                conv_kernel_size=31,
            )
        elif block_type == "transformer":
            block = BasicTransformerBlock(
                dim=dim,
                num_attention_heads=num_heads,
                attention_head_dim=attention_head_dim,
                dropout=dropout,
                activation_fn=act_fn,
            )
        else:
            raise ValueError(f"Unknown block type {block_type}")

        return block

    def __call__(
        self, x: mx.array, mask: mx.array, mu: mx.array, t: mx.array, spks: Optional[mx.array] = None, cond: Optional[mx.array] = None
    ) -> mx.array:
        """
        Forward pass of the decoder.

        Args:
            x: Input tensor, shape (batch_size, in_channels, time)
            mask: Mask tensor, shape (batch_size, 1, time)
            mu: Conditioning tensor, shape (batch_size, channels, time)
            t: Time step tensor, shape (batch_size,)
            spks: Speaker embeddings, shape (batch_size, spk_dim)
            cond: Additional conditioning (placeholder)

        Returns:
            Output tensor, shape (batch_size, out_channels, time)
        """
        # Time embeddings
        t = self.time_embeddings(t)
        t = self.time_mlp(t)

        # Concatenate input with conditioning
        x = pack([x, mu], "b * t")[0]

        # Add speaker embeddings if provided
        if spks is not None:
            spks = repeat(spks, "b c -> b c t", t=x.shape[-1])
            x = pack([x, spks], "b * t")[0]

        # Down path
        hiddens = []
        masks = [mask]

        for resnet, transformer_blocks, downsample in self.down_blocks:
            mask_down = masks[-1]
            x = resnet(x, mask_down, t)

            # Reshape for transformer blocks
            x = rearrange(x, "b c t -> b t c")
            mask_down = rearrange(mask_down, "b 1 t -> b t")

            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=mask_down,
                    timestep=t,
                )

            # Reshape back
            x = rearrange(x, "b t c -> b c t")
            mask_down = rearrange(mask_down, "b t -> b 1 t")

            hiddens.append(x)  # Save for skip connections
            x = downsample(x * mask_down)

            # Downsample mask
            if isinstance(downsample, Downsample1D):
                masks.append(mask_down[:, :, ::2])
            else:
                masks.append(mask_down)

        # Remove last mask (not needed)
        masks = masks[:-1]
        mask_mid = masks[-1]

        # Middle blocks
        for resnet, transformer_blocks in self.mid_blocks:
            x = resnet(x, mask_mid, t)

            # Reshape for transformer blocks
            x = rearrange(x, "b c t -> b t c")
            mask_mid = rearrange(mask_mid, "b 1 t -> b t")

            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=mask_mid,
                    timestep=t,
                )

            # Reshape back
            x = rearrange(x, "b t c -> b c t")
            mask_mid = rearrange(mask_mid, "b t -> b 1 t")

        # Up path
        for resnet, transformer_blocks, upsample in self.up_blocks:
            mask_up = masks.pop()

            # Concatenate with skip connection
            x = pack([x, hiddens.pop()], "b * t")[0]
            x = resnet(x, mask_up, t)

            # Reshape for transformer blocks
            x = rearrange(x, "b c t -> b t c")
            mask_up = rearrange(mask_up, "b 1 t -> b t")

            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=mask_up,
                    timestep=t,
                )

            # Reshape back
            x = rearrange(x, "b t c -> b c t")
            mask_up = rearrange(mask_up, "b t -> b 1 t")

            x = upsample(x * mask_up)

        # Final processing
        x = self.final_block(x, mask)
        output = self.final_proj(x * mask)

        return output * mask
