import mlx.core as mx
import mlx.nn as nn
from einops.array_api import pack, rearrange, repeat

from .utils.mask import add_optional_chunk_mask
from .matcha.decoder import SinusoidalPosEmb, Block1D, ResnetBlock1D, Downsample1D, TimestepEmbedding, Upsample1D
from .matcha.transformer import BasicTransformerBlock


def mask_to_bias(mask: mx.array, dtype: mx.Dtype) -> mx.array:
    assert mask.dtype == mx.bool_
    mask = mask.astype(dtype)
    # attention mask bias
    mask = (1.0 - mask) * -1.0e10
    return mask


class Transpose(nn.Module):
    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def __call__(self, x: mx.array):
        x = mx.transpose(x, (self.dim0, self.dim1))
        return x


class CausalBlock1D(Block1D):
    def __init__(self, dim: int, dim_out: int):
        super(CausalBlock1D, self).__init__(dim, dim_out)
        self.block = nn.Sequential(
            CausalConv1d(dim, dim_out, 3),
            Transpose(1, 2),
            nn.LayerNorm(dim_out),
            Transpose(1, 2),
            nn.Mish(),
        )

    def __call__(self, x: mx.array, mask: mx.array):
        output = self.block(x * mask)
        return output * mask


class CausalResnetBlock1D(ResnetBlock1D):
    def __init__(self, dim: int, dim_out: int, time_emb_dim: int, groups: int = 8):
        super(CausalResnetBlock1D, self).__init__(dim, dim_out, time_emb_dim, groups)
        self.block1 = CausalBlock1D(dim, dim_out)
        self.block2 = CausalBlock1D(dim_out, dim_out)


class CausalConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super(CausalConv1d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation, groups=groups, bias=bias
        )
        assert stride == 1
        self.causal_padding = (kernel_size - 1, 0)

    def __call__(self, x: mx.array):
        # Pad only on the left side for causal convolution
        x = mx.pad(x, [(0, 0), (0, 0), self.causal_padding])
        x = super(CausalConv1d, self).__call__(x)
        return x


class ConditionalDecoder(nn.Module):
    def __init__(
        self,
        in_channels=320,
        out_channels=80,
        causal=True,
        channels=[256],
        dropout=0.0,
        attention_head_dim=64,
        n_blocks=4,
        num_mid_blocks=12,
        num_heads=8,
        act_fn="gelu",
    ):
        """
        This decoder requires an input with the same shape of the target. So, if your text content
        is shorter or longer than the outputs, please re-sampling it before feeding to the decoder.
        """
        super().__init__()
        channels = tuple(channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.causal = causal
        self.time_embeddings = SinusoidalPosEmb(in_channels)
        time_embed_dim = channels[0] * 4
        self.time_mlp = TimestepEmbedding(
            in_channels=in_channels,
            time_embed_dim=time_embed_dim,
            act_fn="silu",
        )
        self.down_blocks = []
        self.mid_blocks = []
        self.up_blocks = []

        self.static_chunk_size = 0

        output_channel = in_channels
        for i in range(len(channels)):
            input_channel = output_channel
            output_channel = channels[i]
            is_last = i == len(channels) - 1
            resnet = (
                CausalResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)
                if self.causal
                else ResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)
            )
            transformer_blocks = [
                BasicTransformerBlock(
                    dim=output_channel,
                    num_attention_heads=num_heads,
                    attention_head_dim=attention_head_dim,
                    dropout=dropout,
                    activation_fn=act_fn,
                )
                for _ in range(n_blocks)
            ]
            downsample = (
                Downsample1D(output_channel)
                if not is_last
                else CausalConv1d(output_channel, output_channel, 3)
                if self.causal
                else nn.Conv1d(output_channel, output_channel, 3, padding=1)
            )
            self.down_blocks.append([resnet, transformer_blocks, downsample])

        for _ in range(num_mid_blocks):
            input_channel = channels[-1]
            out_channels = channels[-1]
            resnet = (
                CausalResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)
                if self.causal
                else ResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)
            )

            transformer_blocks = [
                BasicTransformerBlock(
                    dim=output_channel,
                    num_attention_heads=num_heads,
                    attention_head_dim=attention_head_dim,
                    dropout=dropout,
                    activation_fn=act_fn,
                )
                for _ in range(n_blocks)
            ]

            self.mid_blocks.append([resnet, transformer_blocks])

        channels = channels[::-1] + (channels[0],)
        for i in range(len(channels) - 1):
            input_channel = channels[i] * 2
            output_channel = channels[i + 1]
            is_last = i == len(channels) - 2
            resnet = (
                CausalResnetBlock1D(
                    dim=input_channel,
                    dim_out=output_channel,
                    time_emb_dim=time_embed_dim,
                )
                if self.causal
                else ResnetBlock1D(
                    dim=input_channel,
                    dim_out=output_channel,
                    time_emb_dim=time_embed_dim,
                )
            )
            transformer_blocks = [
                BasicTransformerBlock(
                    dim=output_channel,
                    num_attention_heads=num_heads,
                    attention_head_dim=attention_head_dim,
                    dropout=dropout,
                    activation_fn=act_fn,
                )
                for _ in range(n_blocks)
            ]
            upsample = (
                Upsample1D(output_channel, use_conv_transpose=True)
                if not is_last
                else CausalConv1d(output_channel, output_channel, 3)
                if self.causal
                else nn.Conv1d(output_channel, output_channel, 3, padding=1)
            )
            self.up_blocks.append([resnet, transformer_blocks, upsample])

        self.final_block = CausalBlock1D(channels[-1], channels[-1]) if self.causal else Block1D(channels[-1], channels[-1])
        self.final_proj = nn.Conv1d(channels[-1], self.out_channels, 1)

    def __call__(self, x, mask, mu, t, spks=None, cond=None):
        """Forward pass of the UNet1DConditional model.

        Args:
            x (mx.array): shape (batch_size, in_channels, time)
            mask (mx.array): shape (batch_size, 1, time)
            t (mx.array): shape (batch_size)
            spks (mx.array, optional): shape: (batch_size, condition_channels). Defaults to None.
            cond (mx.array, optional): placeholder for future use. Defaults to None.

        Returns:
            mx.array: output tensor
        """

        t = self.time_embeddings(t).astype(t.dtype)
        t = self.time_mlp(t)

        x = pack([x, mu], "b * t")[0]

        if spks is not None:
            spks = repeat(spks, "b c -> b c t", t=x.shape[-1])
            x = pack([x, spks], "b * t")[0]
        if cond is not None:
            x = pack([x, cond], "b * t")[0]

        hiddens = []
        masks = [mask]
        for resnet, transformer_blocks, downsample in self.down_blocks:
            mask_down = masks[-1]
            x = resnet(x, mask_down, t)
            x = rearrange(x, "b c t -> b t c")

            # Create attention mask
            attn_mask = add_optional_chunk_mask(x, mask_down.astype(mx.bool_), False, False, 0, self.static_chunk_size, -1)
            attn_mask = mask_to_bias(attn_mask == 1, x.dtype)

            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=attn_mask,
                    timestep=t,
                )
            x = rearrange(x, "b t c -> b c t")
            hiddens.append(x)  # Save hidden states for skip connections
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]

        for resnet, transformer_blocks in self.mid_blocks:
            x = resnet(x, mask_mid, t)
            x = rearrange(x, "b c t -> b t c")

            # Create attention mask
            attn_mask = add_optional_chunk_mask(x, mask_mid.astype(mx.bool_), False, False, 0, self.static_chunk_size, -1)
            attn_mask = mask_to_bias(attn_mask == 1, x.dtype)

            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=attn_mask,
                    timestep=t,
                )
            x = rearrange(x, "b t c -> b c t")

        for resnet, transformer_blocks, upsample in self.up_blocks:
            mask_up = masks.pop()
            skip = hiddens.pop()
            x = pack([x[:, :, : skip.shape[-1]], skip], "b * t")[0]
            x = resnet(x, mask_up, t)
            x = rearrange(x, "b c t -> b t c")

            # Create attention mask
            attn_mask = add_optional_chunk_mask(x, mask_up.astype(mx.bool_), False, False, 0, self.static_chunk_size, -1)
            attn_mask = mask_to_bias(attn_mask == 1, x.dtype)

            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=attn_mask,
                    timestep=t,
                )
            x = rearrange(x, "b t c -> b c t")
            x = upsample(x * mask_up)

        x = self.final_block(x, mask_up)
        output = self.final_proj(x * mask_up)
        return output * mask
