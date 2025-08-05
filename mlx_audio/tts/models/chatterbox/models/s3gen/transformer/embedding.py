"""Positonal Encoding Module."""

import math
from typing import Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    """Positional encoding.

    :param int d_model: embedding dim
    :param float dropout_rate: dropout rate
    :param int max_len: maximum input length

    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))
    """

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000, reverse: bool = False):
        """Construct an PositionalEncoding object."""
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.max_len = max_len

        pe = mx.zeros((max_len, d_model))
        position = mx.arange(0, max_len, dtype=mx.float32).reshape(-1, 1)
        div_term = mx.exp(mx.arange(0, d_model, 2, dtype=mx.float32) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = mx.sin(position * div_term)
        pe[:, 1::2] = mx.cos(position * div_term)
        self.pe = mx.expand_dims(pe, axis=0)

    def __call__(self, x: mx.array, offset: Union[int, mx.array] = 0) -> Tuple[mx.array, mx.array]:
        """Add positional encoding.

        Args:
            x (mx.array): Input. Its shape is (batch, time, ...)
            offset (int, mx.array): position offset

        Returns:
            mx.array: Encoded tensor. Its shape is (batch, time, ...)
            mx.array: for compatibility to RelPositionalEncoding
        """
        pos_emb = self.position_encoding(offset, x.shape[1], False)
        x = x * self.xscale + pos_emb
        return self.dropout(x), self.dropout(pos_emb)

    def position_encoding(self, offset: Union[int, mx.array], size: int, apply_dropout: bool = True) -> mx.array:
        """For getting encoding in a streaming fashion

        Attention!!!!!
        we apply dropout only once at the whole utterance level in a none
        streaming way, but will call this function several times with
        increasing input size in a streaming scenario, so the dropout will
        be applied several times.

        Args:
            offset (int or mx.array): start offset
            size (int): required size of position encoding

        Returns:
            mx.array: Corresponding encoding
        """
        if isinstance(offset, int):
            assert offset + size <= self.max_len
            pos_emb = self.pe[:, offset : offset + size]
        elif isinstance(offset, mx.array) and offset.ndim == 0:  # scalar
            assert offset + size <= self.max_len
            pos_emb = self.pe[:, offset : offset + size]
        else:  # for batched streaming decoding on GPU
            assert mx.max(offset) + size <= self.max_len
            index = mx.expand_dims(offset, axis=1) + mx.arange(0, size)  # B X T
            flag = index > 0
            # remove negative offset
            index = index * flag
            # Use gather to emulate embedding lookup
            pos_emb = mx.take(self.pe[0], index, axis=0)  # B X T X d_model

        if apply_dropout:
            pos_emb = self.dropout(pos_emb)
        return pos_emb


class RelPositionalEncoding(PositionalEncoding):
    """Relative positional encoding module.
    See : Appendix B in https://arxiv.org/abs/1901.02860
    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
    """

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000):
        """Initialize class."""
        super().__init__(d_model, dropout_rate, max_len, reverse=True)

    def __call__(self, x: mx.array, offset: Union[int, mx.array] = 0) -> Tuple[mx.array, mx.array]:
        """Compute positional encoding.
        Args:
            x (mx.array): Input tensor (batch, time, `*`).
        Returns:
            mx.array: Encoded tensor (batch, time, `*`).
            mx.array: Positional embedding tensor (1, time, `*`).
        """
        x = x * self.xscale
        pos_emb = self.position_encoding(offset, x.shape[1], False)
        return self.dropout(x), self.dropout(pos_emb)


class WhisperPositionalEncoding(PositionalEncoding):
    """Sinusoids position encoding used in openai-whisper.encoder"""

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 1500):
        super().__init__(d_model, dropout_rate, max_len)
        self.xscale = 1.0
        log_timescale_increment = np.log(10000) / (d_model // 2 - 1)
        inv_timescales = mx.exp(-log_timescale_increment * mx.arange(d_model // 2))
        scaled_time = mx.arange(max_len).reshape(-1, 1) * inv_timescales.reshape(1, -1)
        pe = mx.concatenate([mx.sin(scaled_time), mx.cos(scaled_time)], axis=1)
        self.pe = mx.expand_dims(pe, axis=0)


class LearnablePositionalEncoding(PositionalEncoding):
    """Learnable position encoding used in openai-whisper.decoder"""

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 448):
        super().__init__(d_model, dropout_rate, max_len)
        # NOTE(xcsong): overwrite self.pe & self.xscale
        self.pe = mx.random.normal((1, max_len, d_model))
        self.xscale = 1.0


class NoPositionalEncoding(nn.Module):
    """No position encoding"""

    def __init__(self, d_model: int, dropout_rate: float):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout_rate)

    def __call__(self, x: mx.array, offset: Union[int, mx.array] = 0) -> Tuple[mx.array, mx.array]:
        """Just return zero vector for interface compatibility"""
        pos_emb = mx.zeros((1, x.shape[1], self.d_model), dtype=x.dtype)
        return self.dropout(x), pos_emb

    def position_encoding(self, offset: Union[int, mx.array], size: int) -> mx.array:
        return mx.zeros((1, size, self.d_model))


class EspnetRelPositionalEncoding(nn.Module):
    """Relative positional encoding module (new implementation).

    Details can be found in https://github.com/espnet/espnet/pull/2816.

    See : Appendix B in https://arxiv.org/abs/1901.02860

    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.

    """

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000):
        """Construct an PositionalEncoding object."""
        super(EspnetRelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = nn.Dropout(p=dropout_rate)
        self._pe = None
        self.extend_pe(mx.zeros((1, max_len)))

    def extend_pe(self, x: mx.array):
        """Reset the positional encodings."""
        if self._pe is not None:
            # self.pe contains both positive and negative parts
            # the length of self.pe is 2 * input_len - 1
            if self._pe.shape[1] >= x.shape[1] * 2 - 1:
                if self._pe.dtype != x.dtype:
                    self._pe = self._pe.astype(x.dtype)
                return
        # Suppose `i` means to the position of query vecotr and `j` means the
        # position of key vector. We use position relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).
        pe_positive = mx.zeros((x.shape[1], self.d_model))
        pe_negative = mx.zeros((x.shape[1], self.d_model))
        position = mx.arange(0, x.shape[1], dtype=mx.float32).reshape(-1, 1)
        div_term = mx.exp(mx.arange(0, self.d_model, 2, dtype=mx.float32) * -(math.log(10000.0) / self.d_model))
        pe_positive[:, 0::2] = mx.sin(position * div_term)
        pe_positive[:, 1::2] = mx.cos(position * div_term)
        pe_negative[:, 0::2] = mx.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = mx.cos(-1 * position * div_term)

        # Reserve the order of positive indices and concat both positive and
        # negative indices. This is used to support the shifting trick
        # as in https://arxiv.org/abs/1901.02860
        pe_positive_reversed = pe_positive[::-1, :]
        pe_positive = mx.expand_dims(pe_positive_reversed, axis=0)
        pe_negative = mx.expand_dims(pe_negative[1:], axis=0)
        pe = mx.concatenate([pe_positive, pe_negative], axis=1)
        self._pe = pe.astype(x.dtype)

    def __call__(self, x: mx.array, offset: Union[int, mx.array] = 0) -> Tuple[mx.array, mx.array]:
        """Add positional encoding.

        Args:
            x (mx.array): Input tensor (batch, time, `*`).

        Returns:
            mx.array: Encoded tensor (batch, time, `*`).

        """
        self.extend_pe(x)
        x = x * self.xscale
        pos_emb = self.position_encoding(size=x.shape[1], offset=offset)
        return self.dropout(x), self.dropout(pos_emb)

    def position_encoding(self, offset: Union[int, mx.array], size: int) -> mx.array:
        """For getting encoding in a streaming fashion

        Attention!!!!!
        we apply dropout only once at the whole utterance level in a none
        streaming way, but will call this function several times with
        increasing input size in a streaming scenario, so the dropout will
        be applied several times.

        Args:
            offset (int or mx.array): start offset
            size (int): required size of position encoding

        Returns:
            mx.array: Corresponding encoding
        """
        pos_emb = self._pe[
            :,
            self._pe.shape[1] // 2 - size + 1 : self._pe.shape[1] // 2 + size,
        ]
        return pos_emb
