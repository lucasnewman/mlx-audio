import mlx.core as mx
import logging

'''
def subsequent_mask(
        size: int,
) -> mx.array:
    """Create mask for subsequent steps (size, size).

    This mask is used only in decoder which works in an auto-regressive mode.
    This means the current step could only do attention with its left steps.

    In encoder, fully attention is used when streaming is not necessary and
    the sequence is not long. In this case, no attention mask is needed.

    When streaming is needed, chunk-based attention is used in encoder. See
    subsequent_chunk_mask for the chunk-based attention mask.

    Args:
        size (int): size of mask

    Returns:
        mx.array: mask

    Examples:
        >>> subsequent_mask(3)
        [[1, 0, 0],
         [1, 1, 0],
         [1, 1, 1]]
    """
    ret = mx.ones((size, size), dtype=mx.bool_)
    return mx.tril(ret)
'''


def subsequent_chunk_mask(
        size: int,
        chunk_size: int,
        num_left_chunks: int = -1,
) -> mx.array:
    """Create mask for subsequent steps (size, size) with chunk size,
       this is for streaming encoder

    Args:
        size (int): size of mask
        chunk_size (int): size of chunk
        num_left_chunks (int): number of left chunks
            <0: use full chunk
            >=0: use num_left_chunks

    Returns:
        mx.array: mask

    Examples:
        >>> subsequent_chunk_mask(4, 2)
        [[1, 1, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 1],
         [1, 1, 1, 1]]
    """
    pos_idx = mx.arange(size)
    block_value = (pos_idx // chunk_size + 1) * chunk_size
    ret = pos_idx[:, None] < block_value[None, :]
    return ret


def add_optional_chunk_mask(xs: mx.array,
                            masks: mx.array,
                            use_dynamic_chunk: bool,
                            use_dynamic_left_chunk: bool,
                            decoding_chunk_size: int,
                            static_chunk_size: int,
                            num_decoding_left_chunks: int,
                            enable_full_context: bool = True):
    """ Apply optional mask for encoder.

    Args:
        xs (mx.array): padded input, (B, L, D), L for max length
        mask (mx.array): mask for xs, (B, 1, L)
        use_dynamic_chunk (bool): whether to use dynamic chunk or not
        use_dynamic_left_chunk (bool): whether to use dynamic left chunk for
            training.
        decoding_chunk_size (int): decoding chunk size for dynamic chunk, it's
            0: default for training, use random dynamic chunk.
            <0: for decoding, use full chunk.
            >0: for decoding, use fixed chunk size as set.
        static_chunk_size (int): chunk size for static chunk training/decoding
            if it's greater than 0, if use_dynamic_chunk is true,
            this parameter will be ignored
        num_decoding_left_chunks: number of left chunks, this is for decoding,
            the chunk size is decoding_chunk_size.
            >=0: use num_decoding_left_chunks
            <0: use all left chunks
        enable_full_context (bool):
            True: chunk size is either [1, 25] or full context(max_len)
            False: chunk size ~ U[1, 25]

    Returns:
        mx.array: chunk mask of the input xs.
    """
    # Whether to use chunk mask or not
    if use_dynamic_chunk:
        max_len = xs.shape[1]
        if decoding_chunk_size < 0:
            chunk_size = max_len
            num_left_chunks = -1
        elif decoding_chunk_size > 0:
            chunk_size = decoding_chunk_size
            num_left_chunks = num_decoding_left_chunks
        else:
            # chunk size is either [1, 25] or full context(max_len).
            # Since we use 4 times subsampling and allow up to 1s(100 frames)
            # delay, the maximum frame is 100 / 4 = 25.
            key = mx.random.key(0)
            chunk_size = int(mx.random.randint(key, shape=(), low=1, high=max_len).item())
            num_left_chunks = -1
            if chunk_size > max_len // 2 and enable_full_context:
                chunk_size = max_len
            else:
                chunk_size = chunk_size % 25 + 1
                if use_dynamic_left_chunk:
                    max_left_chunks = (max_len - 1) // chunk_size
                    key = mx.random.key(1)
                    num_left_chunks = int(mx.random.randint(
                        key, shape=(), low=0, high=max_left_chunks
                    ).item())
        
        chunk_masks = subsequent_chunk_mask(xs.shape[1], chunk_size,
                                            num_left_chunks)  # (L, L)
        chunk_masks = mx.expand_dims(chunk_masks, axis=0)  # (1, L, L)
        chunk_masks = masks & chunk_masks  # (B, L, L)
    elif static_chunk_size > 0:
        num_left_chunks = num_decoding_left_chunks
        chunk_masks = subsequent_chunk_mask(xs.shape[1], static_chunk_size,
                                            num_left_chunks)  # (L, L)
        chunk_masks = mx.expand_dims(chunk_masks, axis=0)  # (1, L, L)
        chunk_masks = masks & chunk_masks  # (B, L, L)
    else:
        chunk_masks = masks
    
    assert chunk_masks.dtype == mx.bool_
    
    # Check for timesteps where all values are False
    chunk_sum = mx.sum(chunk_masks, axis=-1)
    if mx.sum(chunk_sum == 0).item() != 0:
        logging.warning('get chunk_masks all false at some timestep, force set to true, '
                       'make sure they are masked in future computation!')
        # Find indices where sum is 0 and set them to True
        zero_mask = chunk_sum == 0
        chunk_masks = mx.where(
            mx.expand_dims(zero_mask, axis=-1),
            mx.ones_like(chunk_masks),
            chunk_masks
        )
    
    return chunk_masks


def make_pad_mask(lengths: mx.array, max_len: int = 0) -> mx.array:
    """Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (mx.array): Batch of lengths (B,).
        max_len (int): Maximum length. If 0, use max of lengths.
        
    Returns:
        mx.array: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = mx.array([5, 3, 2])
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0, 0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = lengths.shape[0]
    max_len = max_len if max_len > 0 else int(mx.max(lengths).item())
    
    seq_range = mx.arange(0, max_len, dtype=mx.int32)
    # Expand for broadcasting
    seq_range_expand = mx.expand_dims(seq_range, axis=0)
    seq_range_expand = mx.broadcast_to(seq_range_expand, (batch_size, max_len))
    
    seq_length_expand = mx.expand_dims(lengths, axis=-1)
    mask = seq_range_expand >= seq_length_expand
    
    return mask


def make_non_pad_mask(lengths: mx.array, max_len: int = 0) -> mx.array:
    """Make mask tensor containing indices of non-padded part.
    
    This is the inverse of make_pad_mask.
    
    Args:
        lengths (mx.array): Batch of lengths (B,).
        max_len (int): Maximum length. If 0, use max of lengths.
        
    Returns:
        mx.array: Mask tensor containing indices of non-padded part.
        
    Examples:
        >>> lengths = mx.array([5, 3, 2])
        >>> make_non_pad_mask(lengths)
        masks = [[1, 1, 1, 1, 1],
                 [1, 1, 1, 0, 0],
                 [1, 1, 0, 0, 0]]
    """
    return ~make_pad_mask(lengths, max_len)
