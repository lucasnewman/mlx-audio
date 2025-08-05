"""Subsampling layer definition."""

from typing import Tuple, Union

import mlx.core as mx
import mlx.nn as nn


class BaseSubsampling(nn.Module):
    def __init__(self):
        super().__init__()
        self.right_context = 0
        self.subsampling_rate = 1

    def position_encoding(self, offset: Union[int, mx.array], size: int) -> mx.array:
        return self.pos_enc.position_encoding(offset, size)


class EmbedinigNoSubsampling(BaseSubsampling):
    """Embedding input without subsampling"""

    def __init__(self, idim: int, odim: int, dropout_rate: float, pos_enc_class: nn.Module):
        super().__init__()
        self.embed = nn.Embedding(idim, odim)
        self.pos_enc = pos_enc_class

    def __call__(self, x: mx.array, x_mask: mx.array, offset: Union[int, mx.array] = 0) -> Tuple[mx.array, mx.array, mx.array]:
        """Input x.

        Args:
            x (mx.array): Input tensor (#batch, time, idim).
            x_mask (mx.array): Input mask (#batch, 1, time).

        Returns:
            mx.array: linear input tensor (#batch, time', odim),
                where time' = time .
            mx.array: linear input mask (#batch, 1, time'),
                where time' = time .

        """
        x = self.embed(x)
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask


class LinearNoSubsampling(BaseSubsampling):
    """Linear transform the input without subsampling

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim: int, odim: int, dropout_rate: float, pos_enc_class: nn.Module):
        """Construct an linear object."""
        super().__init__()
        self.out = nn.Sequential(
            nn.Linear(idim, odim),
            nn.LayerNorm(odim, eps=1e-5),
            nn.Dropout(dropout_rate),
        )
        self.pos_enc = pos_enc_class
        self.right_context = 0
        self.subsampling_rate = 1

    def __call__(self, x: mx.array, x_mask: mx.array, offset: Union[int, mx.array] = 0) -> Tuple[mx.array, mx.array, mx.array]:
        """Input x.

        Args:
            x (mx.array): Input tensor (#batch, time, idim).
            x_mask (mx.array): Input mask (#batch, 1, time).

        Returns:
            mx.array: linear input tensor (#batch, time', odim),
                where time' = time .
            mx.array: linear input mask (#batch, 1, time'),
                where time' = time .

        """
        x = self.out(x)
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask


class Conv1dSubsampling2(BaseSubsampling):
    """Convolutional 1D subsampling (to 1/2 length).
       It is designed for Whisper, ref:
       https://github.com/openai/whisper/blob/main/whisper/model.py

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim: int, odim: int, dropout_rate: float, pos_enc_class: nn.Module):
        """Construct an Conv1dSubsampling2 object."""
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(idim, odim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(odim, odim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.pos_enc = pos_enc_class
        # The right context for every conv layer is computed by:
        # (kernel_size - 1) * frame_rate_of_this_layer
        self.subsampling_rate = 2
        # 4 = (3 - 1) * 1 + (3 - 1) * 1
        self.right_context = 4

    def __call__(self, x: mx.array, x_mask: mx.array, offset: Union[int, mx.array] = 0) -> Tuple[mx.array, mx.array, mx.array]:
        """Subsample x.

        Args:
            x (mx.array): Input tensor (#batch, time, idim).
            x_mask (mx.array): Input mask (#batch, 1, time).

        Returns:
            mx.array: Subsampled tensor (#batch, time', odim),
                where time' = time // 2.
            mx.array: Subsampled mask (#batch, 1, time'),
                where time' = time // 2.
            mx.array: positional encoding

        """
        time = x.shape[1]
        x = mx.transpose(x, (0, 2, 1))  # (b, f, t)
        x = self.conv(x)
        x = mx.transpose(x, (0, 2, 1))  # (b, t, f)
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, (time + 1) % 2 :: 2]


class Conv2dSubsampling4(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim: int, odim: int, dropout_rate: float, pos_enc_class: nn.Module):
        """Construct an Conv2dSubsampling4 object."""
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, odim, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(odim, odim, 3, stride=2),
            nn.ReLU(),
        )
        self.out = nn.Sequential(nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim))
        self.pos_enc = pos_enc_class
        # The right context for every conv layer is computed by:
        # (kernel_size - 1) * frame_rate_of_this_layer
        self.subsampling_rate = 4
        # 6 = (3 - 1) * 1 + (3 - 1) * 2
        self.right_context = 6

    def __call__(self, x: mx.array, x_mask: mx.array, offset: Union[int, mx.array] = 0) -> Tuple[mx.array, mx.array, mx.array]:
        """Subsample x.

        Args:
            x (mx.array): Input tensor (#batch, time, idim).
            x_mask (mx.array): Input mask (#batch, 1, time).

        Returns:
            mx.array: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            mx.array: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.
            mx.array: positional encoding

        """
        x = mx.expand_dims(x, axis=1)  # (b, c=1, t, f)
        x = self.conv(x)
        b, c, t, f = x.shape
        x = mx.transpose(x, (0, 2, 1, 3)).reshape(b, t, c * f)
        x = self.out(x)
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, 2::2][:, :, 2::2]


class Conv2dSubsampling6(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/6 length).
    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (nn.Module): Custom position encoding layer.
    """

    def __init__(self, idim: int, odim: int, dropout_rate: float, pos_enc_class: nn.Module):
        """Construct an Conv2dSubsampling6 object."""
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, odim, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(odim, odim, 5, stride=3),
            nn.ReLU(),
        )
        self.linear = nn.Linear(odim * (((idim - 1) // 2 - 2) // 3), odim)
        self.pos_enc = pos_enc_class
        # 10 = (3 - 1) * 1 + (5 - 1) * 2
        self.subsampling_rate = 6
        self.right_context = 10

    def __call__(self, x: mx.array, x_mask: mx.array, offset: Union[int, mx.array] = 0) -> Tuple[mx.array, mx.array, mx.array]:
        """Subsample x.
        Args:
            x (mx.array): Input tensor (#batch, time, idim).
            x_mask (mx.array): Input mask (#batch, 1, time).

        Returns:
            mx.array: Subsampled tensor (#batch, time', odim),
                where time' = time // 6.
            mx.array: Subsampled mask (#batch, 1, time'),
                where time' = time // 6.
            mx.array: positional encoding
        """
        x = mx.expand_dims(x, axis=1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.shape
        x = mx.transpose(x, (0, 2, 1, 3)).reshape(b, t, c * f)
        x = self.linear(x)
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, 2::2][:, :, 4::3]


class Conv2dSubsampling8(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/8 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim: int, odim: int, dropout_rate: float, pos_enc_class: nn.Module):
        """Construct an Conv2dSubsampling8 object."""
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, odim, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(odim, odim, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(odim, odim, 3, stride=2),
            nn.ReLU(),
        )
        self.linear = nn.Linear(odim * ((((idim - 1) // 2 - 1) // 2 - 1) // 2), odim)
        self.pos_enc = pos_enc_class
        self.subsampling_rate = 8
        # 14 = (3 - 1) * 1 + (3 - 1) * 2 + (3 - 1) * 4
        self.right_context = 14

    def __call__(self, x: mx.array, x_mask: mx.array, offset: Union[int, mx.array] = 0) -> Tuple[mx.array, mx.array, mx.array]:
        """Subsample x.

        Args:
            x (mx.array): Input tensor (#batch, time, idim).
            x_mask (mx.array): Input mask (#batch, 1, time).

        Returns:
            mx.array: Subsampled tensor (#batch, time', odim),
                where time' = time // 8.
            mx.array: Subsampled mask (#batch, 1, time'),
                where time' = time // 8.
            mx.array: positional encoding
        """
        x = mx.expand_dims(x, axis=1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.shape
        x = mx.transpose(x, (0, 2, 1, 3)).reshape(b, t, c * f)
        x = self.linear(x)
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, 2::2][:, :, 2::2][:, :, 2::2]


class LegacyLinearNoSubsampling(BaseSubsampling):
    """Linear transform the input without subsampling

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim: int, odim: int, dropout_rate: float, pos_enc_class: nn.Module):
        """Construct an linear object."""
        super().__init__()
        self.out = nn.Sequential(
            nn.Linear(idim, odim),
            nn.LayerNorm(odim, eps=1e-5),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
        )
        self.pos_enc = pos_enc_class
        self.right_context = 0
        self.subsampling_rate = 1

    def __call__(self, x: mx.array, x_mask: mx.array, offset: Union[int, mx.array] = 0) -> Tuple[mx.array, mx.array, mx.array]:
        """Input x.

        Args:
            x (mx.array): Input tensor (#batch, time, idim).
            x_mask (mx.array): Input mask (#batch, 1, time).

        Returns:
            mx.array: linear input tensor (#batch, time', odim),
                where time' = time .
            mx.array: linear input mask (#batch, 1, time'),
                where time' = time .

        """
        x = self.out(x)
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask
