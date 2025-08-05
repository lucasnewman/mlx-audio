from typing import Union
import mlx.core as mx
import mlx.nn as nn


class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, seq_len, model_dim, init=.02):
        super().__init__()
        self.emb = nn.Embedding(seq_len, model_dim)
        
        # Initialize embeddings with normal distribution
        # MLX doesn't have in-place operations like PyTorch's .normal_()
        # We need to set the weight directly
        self.emb.weight = mx.random.normal(
            shape=(seq_len, model_dim),
            loc=0.0,
            scale=init
        )
    
    def __call__(self, x):
        """
        Returns positional embeddings for index 0 up to the length of x
        """
        sl = x.shape[1]
        # MLX doesn't have device management like PyTorch
        # Arrays are automatically on the appropriate device
        return self.emb(mx.arange(0, sl))
    
    def get_fixed_embedding(self, idx: Union[int, mx.array]):
        """
        Args:
            idx: scalar int or an integer array of shape (T,) or (B, T)
            
        Returns:
            positional embeddings for given indices, shape (B, T, dim), ie (1, 1, dim) for int input
        """
        # Convert to MLX array if needed
        if isinstance(idx, int):
            idx = mx.array(idx)
        elif not isinstance(idx, mx.array):
            idx = mx.array(idx)
        
        # Ensure idx is at least 2D
        if idx.ndim == 0:
            idx = idx.reshape(1, 1)
        elif idx.ndim == 1:
            idx = idx.reshape(1, -1)
        
        assert idx.ndim == 2, f"Expected 2D array but got {idx.ndim}D"
        
        return self.emb(idx)  # (B, T, dim)
