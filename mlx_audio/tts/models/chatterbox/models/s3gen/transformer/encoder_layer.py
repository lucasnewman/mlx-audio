"""Encoder self-attention layer definition."""

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


class TransformerEncoderLayer(nn.Module):
    """Encoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention`
            instance can be used as the argument.
        feed_forward (nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool):
            True: use layer_norm before each sub-block.
            False: to use layer_norm after each sub-block.
    """

    def __init__(
        self,
        size: int,
        self_attn: nn.Module,
        feed_forward: nn.Module,
        dropout_rate: float,
        normalize_before: bool = True,
    ):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(size, eps=1e-12)
        self.norm2 = nn.LayerNorm(size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before

    def __call__(
        self,
        x: mx.array,
        mask: mx.array,
        pos_emb: mx.array,
        mask_pad: mx.array = None,
        att_cache: mx.array = None,
        cnn_cache: mx.array = None,
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
        """Compute encoded features.

        Args:
            x (mx.array): (#batch, time, size)
            mask (mx.array): Mask tensor for the input (#batch, time，time),
                (0, 0, 0) means fake mask.
            pos_emb (mx.array): just for interface compatibility
                to ConformerEncoderLayer
            mask_pad (mx.array): does not used in transformer layer,
                just for unified api with conformer.
            att_cache (mx.array): Cache tensor of the KEY & VALUE
                (#batch=1, head, cache_t1, d_k * 2), head * d_k == size.
            cnn_cache (mx.array): Convolution cache in conformer layer
                (#batch=1, size, cache_t2), not used here, it's for interface
                compatibility to ConformerEncoderLayer.
        Returns:
            mx.array: Output tensor (#batch, time, size).
            mx.array: Mask tensor (#batch, time, time).
            mx.array: att_cache tensor,
                (#batch=1, head, cache_t1 + time, d_k * 2).
            mx.array: cnn_cahce tensor (#batch=1, size, cache_t2).

        """
        if mask_pad is None:
            mask_pad = mx.ones((0, 0, 0), dtype=mx.bool_)
        if att_cache is None:
            att_cache = mx.zeros((0, 0, 0, 0))
        if cnn_cache is None:
            cnn_cache = mx.zeros((0, 0, 0, 0))

        residual = x
        if self.normalize_before:
            x = self.norm1(x)
        x_att, new_att_cache = self.self_attn(x, x, x, mask, pos_emb=pos_emb, cache=att_cache)
        x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        fake_cnn_cache = mx.zeros((0, 0, 0), dtype=x.dtype)
        return x, mask, new_att_cache, fake_cnn_cache


class ConformerEncoderLayer(nn.Module):
    """Encoder layer module.
    Args:
        size (int): Input dimension.
        self_attn (nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention`
            instance can be used as the argument.
        feed_forward (nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        feed_forward_macaron (nn.Module): Additional feed-forward module
             instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        conv_module (nn.Module): Convolution module instance.
            `ConvlutionModule` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool):
            True: use layer_norm before each sub-block.
            False: use layer_norm after each sub-block.
    """

    def __init__(
        self,
        size: int,
        self_attn: nn.Module,
        feed_forward: Optional[nn.Module] = None,
        feed_forward_macaron: Optional[nn.Module] = None,
        conv_module: Optional[nn.Module] = None,
        dropout_rate: float = 0.1,
        normalize_before: bool = True,
    ):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.norm_ff = nn.LayerNorm(size, eps=1e-12)  # for the FNN module
        self.norm_mha = nn.LayerNorm(size, eps=1e-12)  # for the MHA module
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = nn.LayerNorm(size, eps=1e-12)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        if self.conv_module is not None:
            self.norm_conv = nn.LayerNorm(size, eps=1e-12)  # for the CNN module
            self.norm_final = nn.LayerNorm(size, eps=1e-12)  # for the final output of the block
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before

    def __call__(
        self,
        x: mx.array,
        mask: mx.array,
        pos_emb: mx.array,
        mask_pad: mx.array = None,
        att_cache: mx.array = None,
        cnn_cache: mx.array = None,
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
        """Compute encoded features.

        Args:
            x (mx.array): (#batch, time, size)
            mask (mx.array): Mask tensor for the input (#batch, time，time),
                (0, 0, 0) means fake mask.
            pos_emb (mx.array): positional encoding, must not be None
                for ConformerEncoderLayer.
            mask_pad (mx.array): batch padding mask used for conv module.
                (#batch, 1，time), (0, 0, 0) means fake mask.
            att_cache (mx.array): Cache tensor of the KEY & VALUE
                (#batch=1, head, cache_t1, d_k * 2), head * d_k == size.
            cnn_cache (mx.array): Convolution cache in conformer layer
                (#batch=1, size, cache_t2)
        Returns:
            mx.array: Output tensor (#batch, time, size).
            mx.array: Mask tensor (#batch, time, time).
            mx.array: att_cache tensor,
                (#batch=1, head, cache_t1 + time, d_k * 2).
            mx.array: cnn_cahce tensor (#batch, size, cache_t2).
        """
        if mask_pad is None:
            mask_pad = mx.ones((0, 0, 0), dtype=mx.bool_)
        if att_cache is None:
            att_cache = mx.zeros((0, 0, 0, 0))
        if cnn_cache is None:
            cnn_cache = mx.zeros((0, 0, 0, 0))

        # whether to use macaron style
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(self.feed_forward_macaron(x))
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        # multi-headed self-attention module
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)
        x_att, new_att_cache = self.self_attn(x, x, x, mask, pos_emb, att_cache)
        x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm_mha(x)

        # convolution module
        # Fake new cnn cache here, and then change it in conv_module
        new_cnn_cache = mx.zeros((0, 0, 0), dtype=x.dtype)
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
            x = residual + self.dropout(x)

            if not self.normalize_before:
                x = self.norm_conv(x)

        # feed forward module
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)

        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.conv_module is not None:
            x = self.norm_final(x)

        return x, mask, new_att_cache, new_cnn_cache
