"""Positionwise feed forward layer definition."""

import mlx.core as mx
import mlx.nn as nn


class PositionwiseFeedForward(nn.Module):
    """Positionwise feed forward layer.

    FeedForward are appied on each position of the sequence.
    The output dim is same with the input dim.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.
        activation (nn.Module): Activation function
    """

    def __init__(
        self,
        idim: int,
        hidden_units: int,
        dropout_rate: float,
        activation: nn.Module = nn.ReLU(),
    ):
        """Construct a PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(idim, hidden_units)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)
        self.w_2 = nn.Linear(hidden_units, idim)

    def __call__(self, xs: mx.array) -> mx.array:
        """Forward function.

        Args:
            xs: input tensor (B, L, D)
        Returns:
            output tensor, (B, L, D)
        """
        return self.w_2(self.dropout(self.activation(self.w_1(xs))))


def topk(x: mx.array, k: int, axis: int = -1):
    """Get top k values and indices along an axis.

    Args:
        x: Input array
        k: Number of top elements to return
        axis: Axis along which to find top k

    Returns:
        values: Top k values
        indices: Indices of top k values
    """
    # Get indices that would sort the array in descending order
    sorted_indices = mx.argsort(x, axis=axis)
    # Reverse to get descending order
    sorted_indices = mx.take(sorted_indices, mx.arange(x.shape[axis] - 1, -1, -1), axis=axis)
    # Take top k indices
    top_indices = mx.take(sorted_indices, mx.arange(k), axis=axis)
    # Gather the corresponding values
    top_values = mx.take_along_axis(x, top_indices, axis=axis)

    return top_values, top_indices


class MoEFFNLayer(nn.Module):
    """
    Mixture of expert with Positionwise feed forward layer
    See also figure 1 in https://arxiv.org/pdf/2305.15663.pdf
    The output dim is same with the input dim.

    Modified from https://github.com/Lightning-AI/lit-gpt/pull/823
                  https://github.com/mistralai/mistral-src/blob/b46d6/moe_one_file_ref.py#L203-L219
    Args:
        n_expert: number of expert.
        n_expert_per_token: The actual number of experts used for each frame
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.
        activation (nn.Module): Activation function
    """

    def __init__(
        self,
        n_expert: int,
        n_expert_per_token: int,
        idim: int,
        hidden_units: int,
        dropout_rate: float,
        activation: nn.Module = nn.ReLU(),
    ):
        super(MoEFFNLayer, self).__init__()
        self.gate = nn.Linear(idim, n_expert, bias=False)
        self.experts = [PositionwiseFeedForward(idim, hidden_units, dropout_rate, activation) for _ in range(n_expert)]
        self.n_expert_per_token = n_expert_per_token

    def __call__(self, xs: mx.array) -> mx.array:
        """Foward function.
        Args:
            xs: input tensor (B, L, D)
        Returns:
            output tensor, (B, L, D)

        """
        B, L, D = xs.shape  # batch size, sequence length, embedding dimension (idim)
        xs_flat = xs.reshape(-1, D)  # (B*L, D)
        router = self.gate(xs_flat)  # (B*L, n_expert)
        logits, indices = topk(router, self.n_expert_per_token)  # (B*L, n_expert_per_token)
        weights = mx.softmax(logits, axis=1).astype(xs.dtype)  # (B*L, n_expert_per_token)

        output = mx.zeros_like(xs_flat)  # (B*L, D)

        for i, expert in enumerate(self.experts):
            # Find which tokens are assigned to this expert
            mask = indices == i
            # Get batch indices where this expert is selected
            batch_indices = mx.where(mask)

            if len(batch_indices) > 0 and len(batch_indices[0]) > 0:
                # Extract the relevant batch indices and expert indices
                batch_idx = batch_indices[0]
                ith_expert = batch_indices[1]

                # Get the inputs for this expert
                expert_input = xs_flat[batch_idx]
                # Get the weights for this expert
                expert_weights = weights[batch_idx, ith_expert]
                # Apply expert and accumulate weighted output
                expert_output = expert(expert_input)
                output = output.at[batch_idx].add(expert_weights[:, None] * expert_output)

        return output.reshape(B, L, D)
