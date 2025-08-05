"""ConvolutionModule definition."""

from typing import Tuple

import mlx.core as mx
import mlx.nn as nn


def glu(x: mx.array, dim: int = -1) -> mx.array:
    """Gated Linear Unit activation function.

    Args:
        x: Input tensor to be split in half along dim
        dim: Dimension along which to split

    Returns:
        Output tensor with GLU applied
    """
    # Split the tensor into two halves along the specified dimension
    size = x.shape[dim]
    assert size % 2 == 0, f"Dimension {dim} size must be even, got {size}"

    indices1 = mx.arange(size // 2)
    indices2 = mx.arange(size // 2, size)

    a = mx.take(x, indices1, axis=dim)
    b = mx.take(x, indices2, axis=dim)

    return a * mx.sigmoid(b)


class ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 15,
        activation: nn.Module = nn.ReLU(),
        norm: str = "batch_norm",
        causal: bool = False,
        bias: bool = True,
    ):
        """Construct an ConvolutionModule object.
        Args:
            channels (int): The number of channels of conv layers.
            kernel_size (int): Kernel size of conv layers.
            causal (int): Whether use causal convolution or not
        """
        super().__init__()

        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        # self.lorder is used to distinguish if it's a causal convolution,
        # if self.lorder > 0: it's a causal convolution, the input will be
        #    padded with self.lorder frames on the left in forward.
        # else: it's a symmetrical convolution
        if causal:
            padding = 0
            self.lorder = kernel_size - 1
        else:
            # kernel_size should be an odd number for none causal convolution
            assert (kernel_size - 1) % 2 == 0
            padding = (kernel_size - 1) // 2
            self.lorder = 0
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=padding,
            groups=channels,
            bias=bias,
        )

        assert norm in ["batch_norm", "layer_norm"]
        if norm == "batch_norm":
            self.use_layer_norm = False
            self.norm = nn.BatchNorm(channels, affine=True)
        else:
            self.use_layer_norm = True
            self.norm = nn.LayerNorm(channels)

        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.activation = activation

    def __call__(
        self,
        x: mx.array,
        mask_pad: mx.array = None,
        cache: mx.array = None,
    ) -> Tuple[mx.array, mx.array]:
        """Compute convolution module.
        Args:
            x (mx.array): Input tensor (#batch, time, channels).
            mask_pad (mx.array): used for batch padding (#batch, 1, time),
                (0, 0, 0) means fake mask.
            cache (mx.array): left context cache, it is only
                used in causal convolution (#batch, channels, cache_t),
                (0, 0, 0) meas fake cache.
        Returns:
            mx.array: Output tensor (#batch, time, channels).
        """
        if mask_pad is None:
            mask_pad = mx.ones((0, 0, 0), dtype=mx.bool_)
        if cache is None:
            cache = mx.zeros((0, 0, 0))

        # exchange the temporal dimension and the feature dimension
        x = mx.transpose(x, (0, 2, 1))  # (#batch, channels, time)

        # mask batch padding
        if mask_pad.shape[2] > 0:  # time > 0
            x = mx.where(mask_pad, x, 0.0)

        if self.lorder > 0:
            if cache.shape[2] == 0:  # cache_t == 0
                x = mx.pad(x, [(0, 0), (0, 0), (self.lorder, 0)])
            else:
                assert cache.shape[0] == x.shape[0]  # equal batch
                assert cache.shape[1] == x.shape[1]  # equal channel
                x = mx.concatenate((cache, x), axis=2)
            assert x.shape[2] > self.lorder
            new_cache = x[:, :, -self.lorder :]
        else:
            # It's better we just return None if no cache is required,
            # However, for JIT export, here we just fake one tensor instead of
            # None.
            new_cache = mx.zeros((0, 0, 0), dtype=x.dtype)

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        x = glu(x, dim=1)  # (batch, channel, dim)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)
        if self.use_layer_norm:
            x = mx.transpose(x, (0, 2, 1))
        x = self.activation(self.norm(x))
        if self.use_layer_norm:
            x = mx.transpose(x, (0, 2, 1))
        x = self.pointwise_conv2(x)
        # mask batch padding
        if mask_pad.shape[2] > 0:  # time > 0
            x = mx.where(mask_pad, x, 0.0)

        return mx.transpose(x, (0, 2, 1)), new_cache
