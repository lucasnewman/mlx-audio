from typing import List

import mlx.core as mx
import mlx.nn as nn

from .utils.kaldi import fbank


def pad_list(xs: List[mx.array], pad_value: float = 0.0) -> mx.array:
    """Perform padding for the list of arrays.

    Args:
        xs: List of arrays [(T_1, *), (T_2, *), ..., (T_B, *)]
        pad_value: Value for padding

    Returns:
        Padded array (B, Tmax, *)
    """
    n_batch = len(xs)
    max_len = max(x.shape[0] for x in xs)

    first_shape = xs[0].shape
    pad_shape = (n_batch, max_len) + first_shape[1:]
    pad = mx.full(pad_shape, pad_value, dtype=xs[0].dtype)

    for i in range(n_batch):
        pad[i, : xs[i].shape[0]] = xs[i]

    return pad


def extract_feature(audio: List[mx.array]) -> tuple[mx.array, List[int], List[int]]:
    """Extract filterbank features from audio.

    Args:
        audio: List of audio waveforms

    Returns:
        Tuple of (padded_features, feature_lengths, audio_lengths)
    """
    features = []
    feature_times = []
    feature_lengths = []

    for au in audio:
        feature = fbank(
            au[None, :],
            num_mel_bins=80,
            use_energy=False,
            use_log_fbank=True,
        )
        feature = feature - mx.mean(feature, axis=0, keepdims=True)
        features.append(feature)
        feature_times.append(au.shape[0])
        feature_lengths.append(feature.shape[0])

    features_padded = pad_list(features, pad_value=0.0)

    return features_padded, feature_lengths, feature_times


class BasicResBlock(nn.Module):
    """Basic residual block for FCM."""

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=(stride, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm(planes)

        self.shortcut = None
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=(stride, 1),
                    bias=False,
                ),
                nn.BatchNorm(self.expansion * planes, affine=True, track_running_stats=True),
            )

    def __call__(self, x: mx.array) -> mx.array:
        out = nn.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.shortcut is not None:
            out = out + self.shortcut(x)
        else:
            out = out + x

        out = nn.relu(out)
        return out


class FCM(nn.Module):
    """Feature Channel Module."""

    def __init__(self, block=BasicResBlock, num_blocks: List[int] = [2, 2], m_channels: int = 32, feat_dim: int = 80):
        super().__init__()
        self.in_planes = m_channels
        self.conv1 = nn.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm(m_channels, affine=True, track_running_stats=True)

        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, m_channels, num_blocks[0], stride=2)

        self.conv2 = nn.Conv2d(m_channels, m_channels, kernel_size=3, stride=(2, 1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm(m_channels)
        self.out_channels = m_channels * (feat_dim // 8)

    def _make_layer(self, block, planes: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def __call__(self, x: mx.array) -> mx.array:
        # Add channel dimension
        x = mx.expand_dims(x, axis=1)

        out = nn.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = nn.relu(self.bn2(self.conv2(out)))

        # Reshape: (B, C, F, T) -> (B, C*F, T)
        shape = out.shape
        out = out.reshape(shape[0], shape[1] * shape[2], shape[3])
        return out


def get_nonlinear(config_str: str, channels: int) -> nn.Sequential:
    """Build nonlinear activation sequence from config string."""
    layers = []
    for name in config_str.split("-"):
        if name == "relu":
            layers.append(nn.ReLU())
        elif name == "prelu":
            layers.append(nn.PReLU(num_parameters=channels))
        elif name == "batchnorm":
            layers.append(nn.BatchNorm(channels))
        elif name == "batchnorm_":
            layers.append(nn.BatchNorm(channels, affine=False))
        else:
            raise ValueError(f"Unexpected module ({name}).")
    return nn.Sequential(*layers)


def statistics_pooling(x: mx.array, axis: int = -1, keepdims: bool = False) -> mx.array:
    """Compute mean and std statistics pooling."""
    mean = mx.mean(x, axis=axis, keepdims=keepdims)
    std = mx.sqrt(mx.var(x, axis=axis, keepdims=keepdims))

    if keepdims:
        stats = mx.concatenate([mean, std], axis=-1)
    else:
        stats = mx.concatenate([mean, std], axis=-1)

    return stats


class StatsPool(nn.Module):
    """Statistics pooling layer."""

    def __call__(self, x: mx.array) -> mx.array:
        return statistics_pooling(x)


class TDNNLayer(nn.Module):
    """Time Delay Neural Network layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
        config_str: str = "batchnorm-relu",
    ):
        super().__init__()
        if padding < 0:
            assert kernel_size % 2 == 1, f"Expect odd kernel size, got {kernel_size}"
            padding = (kernel_size - 1) // 2 * dilation

        self.linear = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.linear(x)
        x = self.nonlinear(x)
        return x


class CAMLayer(nn.Module):
    """Context Aware Masking layer."""

    def __init__(
        self,
        bn_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        bias: bool,
        reduction: int = 2,
    ):
        super().__init__()
        self.linear_local = nn.Conv1d(
            bn_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.linear1 = nn.Conv1d(bn_channels, bn_channels // reduction, 1)
        self.linear2 = nn.Conv1d(bn_channels // reduction, out_channels, 1)

    def __call__(self, x: mx.array) -> mx.array:
        y = self.linear_local(x)

        # Global context
        context = mx.mean(x, axis=-1, keepdims=True) + self.seg_pooling(x)
        context = nn.relu(self.linear1(context))
        m = nn.sigmoid(self.linear2(context))

        return y * m

    def seg_pooling(self, x: mx.array, seg_len: int = 100, stype: str = "avg") -> mx.array:
        """Segment pooling operation."""
        # Get dimensions
        B, C, T = x.shape

        # Calculate number of segments
        num_segs = (T + seg_len - 1) // seg_len

        # Pad if necessary to make divisible by seg_len
        pad_len = num_segs * seg_len - T
        if pad_len > 0:
            x_padded = mx.pad(x, [(0, 0), (0, 0), (0, pad_len)])
        else:
            x_padded = x

        # Reshape for pooling
        x_reshaped = x_padded.reshape(B, C, num_segs, seg_len)

        # Pool over segments
        if stype == "avg":
            seg = mx.mean(x_reshaped, axis=3)
        elif stype == "max":
            seg = mx.max(x_reshaped, axis=3)
        else:
            raise ValueError("Wrong segment pooling type.")

        # Expand back to original time dimension
        seg = mx.expand_dims(seg, axis=-1)
        seg = mx.repeat(seg, seg_len, axis=-1)
        seg = seg.reshape(B, C, -1)

        # Trim to original length
        seg = seg[:, :, :T]

        return seg


class CAMDenseTDNNLayer(nn.Module):
    """CAM Dense TDNN layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bn_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = False,
        config_str: str = "batchnorm-relu",
        memory_efficient: bool = False,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, f"Expect odd kernel size, got {kernel_size}"
        padding = (kernel_size - 1) // 2 * dilation

        self.nonlinear1 = get_nonlinear(config_str, in_channels)
        self.linear1 = nn.Conv1d(in_channels, bn_channels, 1, bias=False)
        self.nonlinear2 = get_nonlinear(config_str, bn_channels)
        self.cam_layer = CAMLayer(
            bn_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.linear1(self.nonlinear1(x))
        x = self.cam_layer(self.nonlinear2(x))
        return x


class CAMDenseTDNNBlock(nn.Module):
    """CAM Dense TDNN block with dense connections."""

    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        out_channels: int,
        bn_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = False,
        config_str: str = "batchnorm-relu",
        memory_efficient: bool = False,
    ):
        super().__init__()
        self.layers = []

        for i in range(num_layers):
            layer = CAMDenseTDNNLayer(
                in_channels=in_channels + i * out_channels,
                out_channels=out_channels,
                bn_channels=bn_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                bias=bias,
                config_str=config_str,
                memory_efficient=memory_efficient,
            )
            self.layers.append(layer)

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = mx.concatenate([x, layer(x)], axis=1)
        return x


class TransitLayer(nn.Module):
    """Transition layer to reduce channels."""

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True, config_str: str = "batchnorm-relu"):
        super().__init__()
        self.nonlinear = get_nonlinear(config_str, in_channels)
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.nonlinear(x)
        x = self.linear(x)
        return x


class DenseLayer(nn.Module):
    """Dense layer with nonlinearity."""

    def __init__(self, in_channels: int, out_channels: int, bias: bool = False, config_str: str = "batchnorm-relu"):
        super().__init__()
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim == 2:
            x = mx.expand_dims(x, axis=-1)
            x = self.linear(x)
            x = mx.squeeze(x, axis=-1)
        else:
            x = self.linear(x)
        x = self.nonlinear(x)
        return x


class CAMPPlus(nn.Module):
    """CAM++ model for speaker verification."""

    def __init__(
        self,
        feat_dim: int = 80,
        embedding_size: int = 192,
        growth_rate: int = 32,
        bn_size: int = 4,
        init_channels: int = 128,
        config_str: str = "batchnorm-relu",
        memory_efficient: bool = True,  # Ignored for inference
        output_level: str = "segment",
        **kwargs,
    ):
        super().__init__()

        self.head = FCM(feat_dim=feat_dim)
        channels = self.head.out_channels
        self.output_level = output_level

        # Build xvector network
        layers: list[tuple[str, nn.Module]] = [
            (
                "tdnn",
                TDNNLayer(
                    channels,
                    init_channels,
                    5,
                    stride=2,
                    dilation=1,
                    padding=-1,
                    config_str=config_str,
                ),
            )
        ]

        channels = init_channels
        for i, (num_layers, kernel_size, dilation) in enumerate(zip((12, 24, 16), (3, 3, 3), (1, 2, 2))):
            block = CAMDenseTDNNBlock(
                num_layers=num_layers,
                in_channels=channels,
                out_channels=growth_rate,
                bn_channels=bn_size * growth_rate,
                kernel_size=kernel_size,
                dilation=dilation,
                config_str=config_str,
                memory_efficient=memory_efficient,
            )
            layers.append((f"tdnnd{i + 1}", block))

            channels = channels + num_layers * growth_rate

            transit = TransitLayer(channels, channels // 2, bias=False, config_str=config_str)
            layers.append((f"transit{i + 1}", transit))

            channels //= 2

        layers.append(("out_nonlinear", get_nonlinear(config_str, channels)))

        if self.output_level == "segment":
            layers.append(("stats", StatsPool()))
            layers.append(("dense", DenseLayer(channels * 2, embedding_size, config_str="batchnorm_")))
        else:
            assert self.output_level == "frame", "`output_level` should be set to 'segment' or 'frame'."

        self.xvector = nn.Sequential(*[layer for _, layer in layers])

    def __call__(self, x: mx.array) -> mx.array:
        # Input shape: (B, T, F)
        x = x.transpose(0, 2, 1)  # (B, T, F) => (B, F, T)
        x = self.head(x)
        x = self.xvector(x)

        if self.output_level == "frame":
            x = x.transpose(0, 2, 1)  # (B, C, T) => (B, T, C)

        return x

    def inference(self, audio_list: List[mx.array]) -> mx.array:
        """Run inference on a list of audio waveforms.

        Args:
            audio_list: List of audio waveforms (each is 1D array)

        Returns:
            Speaker embeddings or frame-level features
        """
        speech, speech_lengths, speech_times = extract_feature(audio_list)
        results = self(speech.astype(mx.float32))
        return results
