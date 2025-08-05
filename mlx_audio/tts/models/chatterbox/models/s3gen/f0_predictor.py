import mlx.core as mx
import mlx.nn as nn

from .conv import WNConv1d


class ConvRNNF0Predictor(nn.Module):
    def __init__(self, num_class: int = 1, in_channels: int = 80, cond_channels: int = 512):
        super().__init__()

        self.num_class = num_class
        self.condnet = nn.Sequential(
            WNConv1d(in_channels, cond_channels, kernel_size=3, padding=1),
            nn.ELU(),
            WNConv1d(cond_channels, cond_channels, kernel_size=3, padding=1),
            nn.ELU(),
            WNConv1d(cond_channels, cond_channels, kernel_size=3, padding=1),
            nn.ELU(),
            WNConv1d(cond_channels, cond_channels, kernel_size=3, padding=1),
            nn.ELU(),
            WNConv1d(cond_channels, cond_channels, kernel_size=3, padding=1),
            nn.ELU(),
        )
        self.classifier = nn.Linear(cond_channels, self.num_class)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.condnet(x)
        x = mx.transpose(x, (0, 2, 1))
        return mx.abs(self.classifier(x).squeeze(-1))
