import math
from typing import Tuple

import mlx.core as mx
import mlx.nn as nn


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, n_head: int, n_feat: int, dropout_rate: float, key_bias: bool = True):
        """Construct an MultiHeadedAttention object."""
        super().__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat, bias=key_bias)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(self, query: mx.array, key: mx.array, value: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        """Transform query, key and value.

        Args:
            query (mx.array): Query tensor (#batch, time1, size).
            key (mx.array): Key tensor (#batch, time2, size).
            value (mx.array): Value tensor (#batch, time2, size).

        Returns:
            mx.array: Transformed query tensor, size
                (#batch, n_head, time1, d_k).
            mx.array: Transformed key tensor, size
                (#batch, n_head, time2, d_k).
            mx.array: Transformed value tensor, size
                (#batch, n_head, time2, d_k).

        """
        n_batch = query.shape[0]
        q = self.linear_q(query).reshape(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).reshape(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).reshape(n_batch, -1, self.h, self.d_k)
        q = mx.transpose(q, (0, 2, 1, 3))  # (batch, head, time1, d_k)
        k = mx.transpose(k, (0, 2, 1, 3))  # (batch, head, time2, d_k)
        v = mx.transpose(v, (0, 2, 1, 3))  # (batch, head, time2, d_k)

        return q, k, v

    def forward_attention(self, value: mx.array, scores: mx.array, mask: mx.array = None) -> mx.array:
        """Compute attention context vector.

        Args:
            value (mx.array): Transformed value, size
                (#batch, n_head, time2, d_k).
            scores (mx.array): Attention score, size
                (#batch, n_head, time1, time2).
            mask (mx.array): Mask, size (#batch, 1, time2) or
                (#batch, time1, time2), (0, 0, 0) means fake mask.

        Returns:
            mx.array: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.shape[0]

        if mask is not None and mask.shape[2] > 0:  # time2 > 0
            mask = mx.expand_dims(mask, axis=1) == 0  # (batch, 1, *, time2)
            # For last chunk, time2 might be larger than scores.shape[-1]
            if mask.shape[-1] > scores.shape[-1]:
                mask = mask[:, :, :, : scores.shape[-1]]  # (batch, 1, *, time2)
            scores = mx.where(mask, -float("inf"), scores)
            attn = mx.where(mask, 0.0, mx.softmax(scores, axis=-1))  # (batch, head, time1, time2)
        else:
            attn = mx.softmax(scores, axis=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(attn)
        x = mx.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = mx.transpose(x, (0, 2, 1, 3)).reshape(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def __call__(
        self, query: mx.array, key: mx.array, value: mx.array, mask: mx.array = None, pos_emb: mx.array = None, cache: mx.array = None
    ) -> Tuple[mx.array, mx.array]:
        """Compute scaled dot product attention.

        Args:
            query (mx.array): Query tensor (#batch, time1, size).
            key (mx.array): Key tensor (#batch, time2, size).
            value (mx.array): Value tensor (#batch, time2, size).
            mask (mx.array): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
            cache (mx.array): Cache tensor (1, head, cache_t, d_k * 2),
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`

        Returns:
            mx.array: Output tensor (#batch, time1, d_model).
            mx.array: Cache tensor (1, head, cache_t + time1, d_k * 2)
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`

        """
        if mask is None:
            mask = mx.ones((0, 0, 0), dtype=mx.bool_)
        if cache is None:
            cache = mx.zeros((0, 0, 0, 0))

        q, k, v = self.forward_qkv(query, key, value)

        if cache.shape[0] > 0:
            key_cache, value_cache = mx.split(cache, 2, axis=-1)
            k = mx.concatenate([key_cache, k], axis=2)
            v = mx.concatenate([value_cache, v], axis=2)

        new_cache = mx.concatenate((k, v), axis=-1)

        scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask), new_cache


class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self, n_head: int, n_feat: int, dropout_rate: float, key_bias: bool = True):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate, key_bias)
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = mx.random.uniform(shape=(self.h, self.d_k), low=-0.1, high=0.1)
        self.pos_bias_v = mx.random.uniform(shape=(self.h, self.d_k), low=-0.1, high=0.1)

    def rel_shift(self, x: mx.array) -> mx.array:
        """Compute relative positional encoding.

        Args:
            x (mx.array): Input tensor (batch, head, time1, 2*time1-1).
            time1 means the length of query vector.

        Returns:
            mx.array: Output tensor.

        """
        zero_pad = mx.zeros((x.shape[0], x.shape[1], x.shape[2], 1), dtype=x.dtype)
        x_padded = mx.concatenate([zero_pad, x], axis=-1)

        x_padded = x_padded.reshape(x.shape[0], x.shape[1], x.shape[3] + 1, x.shape[2])
        x = x_padded[:, :, 1:].reshape(x.shape)[:, :, :, : x.shape[-1] // 2 + 1]
        return x

    def __call__(
        self, query: mx.array, key: mx.array, value: mx.array, mask: mx.array = None, pos_emb: mx.array = None, cache: mx.array = None
    ) -> Tuple[mx.array, mx.array]:
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        Args:
            query (mx.array): Query tensor (#batch, time1, size).
            key (mx.array): Key tensor (#batch, time2, size).
            value (mx.array): Value tensor (#batch, time2, size).
            mask (mx.array): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2), (0, 0, 0) means fake mask.
            pos_emb (mx.array): Positional embedding tensor
                (#batch, time2, size).
            cache (mx.array): Cache tensor (1, head, cache_t, d_k * 2),
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`
        Returns:
            mx.array: Output tensor (#batch, time1, d_model).
            mx.array: Cache tensor (1, head, cache_t + time1, d_k * 2)
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`
        """
        if mask is None:
            mask = mx.ones((0, 0, 0), dtype=mx.bool_)
        if pos_emb is None:
            pos_emb = mx.zeros((0,))
        if cache is None:
            cache = mx.zeros((0, 0, 0, 0))

        q, k, v = self.forward_qkv(query, key, value)
        q = mx.transpose(q, (0, 2, 1, 3))  # (batch, time1, head, d_k)

        if cache.shape[0] > 0:
            key_cache, value_cache = mx.split(cache, 2, axis=-1)
            k = mx.concatenate([key_cache, k], axis=2)
            v = mx.concatenate([value_cache, v], axis=2)

        new_cache = mx.concatenate((k, v), axis=-1)

        n_batch_pos = pos_emb.shape[0]
        p = self.linear_pos(pos_emb).reshape(n_batch_pos, -1, self.h, self.d_k)
        p = mx.transpose(p, (0, 2, 1, 3))  # (batch, head, time1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = mx.transpose(q + self.pos_bias_u, (0, 2, 1, 3))
        # (batch, head, time1, d_k)
        q_with_bias_v = mx.transpose(q + self.pos_bias_v, (0, 2, 1, 3))

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = mx.matmul(q_with_bias_u, mx.transpose(k, (0, 1, 3, 2)))

        # compute matrix b and matrix d
        # (batch, head, time1, time2)
        matrix_bd = mx.matmul(q_with_bias_v, mx.transpose(p, (0, 1, 3, 2)))
        # NOTE(Xiang Lyu): Keep rel_shift since espnet rel_pos_emb is used
        if matrix_ac.shape != matrix_bd.shape:
            matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)  # (batch, head, time1, time2)

        return self.forward_attention(v, scores, mask), new_cache
