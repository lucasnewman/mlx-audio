"""Swish() activation function for Conformer."""

import mlx.core as mx
import mlx.nn as nn


class Swish(nn.Module):
    """Construct an Swish object."""

    def __call__(self, x: mx.array) -> mx.array:
        """Return Swish activation function."""
        return x * mx.sigmoid(x)


# Implementation adapted from https://github.com/EdwardDixon/snake under the MIT license.
# LICENSE is in incl_licenses directory.
class Snake(nn.Module):
    """
    Implementation of a sine-based periodic activation function
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter
    References:
        - This activation function is from this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snake(256)
        >>> x = mx.random.normal((1, 256, 100))
        >>> x = a1(x)
    """

    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - alpha: trainable parameter
                alpha is initialized to 1 by default, higher values = higher-frequency.
                alpha will be trained along with the rest of your model.
        """
        super(Snake, self).__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # log scale alphas initialized to zeros
            self.alpha = mx.zeros(in_features) * alpha
        else:  # linear scale alphas initialized to ones
            self.alpha = mx.ones(in_features) * alpha

        # In MLX, we control trainability by freezing parameters if needed
        if not alpha_trainable:
            self.freeze()

        self.no_div_by_zero = 0.000000001

    def __call__(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        Snake ∶= x + 1/a * sin^2 (xa)
        """
        alpha = mx.expand_dims(mx.expand_dims(self.alpha, 0), -1)  # line up with x to [B, C, T]
        if self.alpha_logscale:
            alpha = mx.exp(alpha)
        x = x + (1.0 / (alpha + self.no_div_by_zero)) * mx.power(mx.sin(x * alpha), 2)
        return x
