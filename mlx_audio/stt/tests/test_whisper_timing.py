import mlx.core as mx
import numpy as np
import pytest

from mlx_audio.stt.models.whisper.timing import median_filter


@pytest.mark.parametrize("width", [1, 3, 7, 13])
@pytest.mark.parametrize("shape", [(19,), (2, 3, 19)])
@pytest.mark.parametrize("dtype", [mx.float16, mx.float32, mx.int32])
def test_median_filter_matches_framewise_numpy(width, shape, dtype):
    rng = np.random.default_rng(42)
    values = rng.integers(-4, 5, size=shape).astype(np.float32)
    if dtype != mx.int32:
        values.flat[:6] = [np.nan, np.inf, -np.inf, 0.0, -0.0, -0.0]
    x = mx.array(values, dtype=dtype)
    values = np.array(x)
    padding = [(0, 0)] * values.ndim
    padding[-1] = (width // 2, width // 2)
    padded = np.pad(values, padding, mode="reflect")
    with np.errstate(invalid="ignore"):
        expected = np.stack(
            [
                np.median(padded[..., i : i + width].astype(np.float32), axis=-1)
                for i in range(values.shape[-1])
            ],
            axis=-1,
        )
        actual = np.array(median_filter(x, width))
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def test_median_filter_short_input_is_unchanged():
    x = mx.array([-0.0, np.nan, np.inf])
    assert median_filter(x, 7) is x
