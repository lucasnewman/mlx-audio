import mlx.core as mx
import numpy as np


def load_voice_tensor(path: str) -> mx.array:
    """
    Load a voice pack file into an MLX array.

    Supports both .safetensors (dict with "voice" key) and .npy (direct array) formats.

    Args:
        path: Path to the voice file (.safetensors or .npy)

    Returns:
        mx.array: The voice tensor
    """
    if path.endswith(".npy"):
        # NPY files contain the voice tensor directly
        data = np.load(path)
        return mx.array(data)
    else:
        # Safetensors files have the tensor under "voice" key
        weights = mx.load(path)
        return weights["voice"]
