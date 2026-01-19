import inspect
from dataclasses import dataclass

import mlx.core as mx
import numpy as np


@dataclass
class BaseModelArgs:
    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


def check_array_shape(arr):
    """
    Check if a 3D array is already in MLX Conv1d format.

    MLX Conv1d weight_v format: (out_channels, kernel_size, in_channels)
    PyTorch format: (out_channels, in_channels, kernel_size)

    Typical kernel_sizes are 1, 3, 5, 7. We use this to determine which format
    the weights are in by checking which dimension is likely the kernel_size.

    Returns:
        True if already in MLX format (no transpose needed)
        False if in PyTorch format (transpose needed)
    """
    shape = arr.shape

    if len(shape) != 3:
        return False

    dim0, dim1, dim2 = shape
    TYPICAL_KERNEL_SIZES = {1, 3, 5, 7}

    dim1_is_kernel = dim1 in TYPICAL_KERNEL_SIZES
    dim2_is_kernel = dim2 in TYPICAL_KERNEL_SIZES

    if dim1_is_kernel and not dim2_is_kernel:
        # Only middle dimension is a typical kernel size -> MLX format
        return True
    elif dim2_is_kernel and not dim1_is_kernel:
        # Only last dimension is a typical kernel size -> PyTorch format
        return False
    elif dim1_is_kernel and dim2_is_kernel:
        # Both are typical kernel sizes (e.g., 1 and 3)
        # The larger value is more likely the actual kernel_size
        # (1x1 convs are common, so 1 is often in_channels not kernel)
        if dim1 >= dim2:
            return True  # MLX format (dim1 is kernel)
        else:
            return False  # PyTorch format (dim2 is kernel)
    else:
        # Neither is a typical kernel size, use size comparison
        return dim1 <= dim2


def adjust_speed(audio_array, speed_factor):
    """
    Adjust the speed of the audio by resampling
    speed_factor > 1: faster
    speed_factor < 1: slower
    """
    # Ensure we're working with MLX arrays
    if not isinstance(audio_array, mx.array):
        audio_array = mx.array(audio_array)

    # Calculate new length
    old_length = audio_array.shape[0]
    new_length = int(old_length / speed_factor)

    # Create new time points
    old_indices = mx.arange(old_length)
    new_indices = mx.linspace(0, old_length - 1, new_length)

    # Resample using linear interpolation
    # Since mx doesn't have interp, we'll implement it directly
    indices_floor = mx.floor(new_indices).astype(mx.int32)
    indices_ceil = mx.minimum(indices_floor + 1, old_length - 1)
    weights_ceil = new_indices - indices_floor
    weights_floor = 1.0 - weights_ceil

    # Perform the interpolation
    result = (
        weights_floor.reshape(-1, 1) * audio_array[indices_floor]
        + weights_ceil.reshape(-1, 1) * audio_array[indices_ceil]
    )

    return result


@dataclass
class GenerationResult:
    audio: mx.array
    samples: int
    sample_rate: int
    segment_idx: int
    token_count: int
    audio_samples: int
    audio_duration: str
    real_time_factor: float
    prompt: dict
    audio_samples: dict
    processing_time_seconds: float
    peak_memory_usage: float
