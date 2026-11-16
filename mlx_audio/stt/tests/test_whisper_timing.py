import mlx.core as mx
import numpy as np
import pytest

from mlx_audio.stt.models.whisper.timing import _backtrace, dtw, median_filter


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


def _scalar_dtw(x):
    N, M = x.shape
    cost = np.full((N + 1, M + 1), np.inf, dtype=np.float32)
    trace = np.full((N + 1, M + 1), -1.0, dtype=np.float32)
    cost[0, 0] = 0
    for j in range(1, M + 1):
        for i in range(1, N + 1):
            c0, c1, c2 = cost[i - 1, j - 1], cost[i - 1, j], cost[i, j - 1]
            if c0 < c1 and c0 < c2:
                c, t = c0, 0
            elif c1 < c0 and c1 < c2:
                c, t = c1, 1
            else:
                c, t = c2, 2
            cost[i, j] = x[i - 1, j - 1] + c
            trace[i, j] = t
    return _backtrace(trace)


@pytest.mark.parametrize("shape", [(1, 1), (1, 9), (9, 1), (6, 23), (23, 6)])
@pytest.mark.parametrize(
    "kind", ["normal", "ties", "constant", "non_finite", "float64"]
)
def test_dtw_matches_scalar_reference(shape, kind):
    rng = np.random.default_rng(0)
    if kind == "normal":
        x = rng.normal(size=shape).astype(np.float32)
    elif kind == "ties":
        x = rng.integers(0, 3, size=shape).astype(np.float32)
    elif kind == "constant":
        x = np.zeros(shape, dtype=np.float32)
    elif kind == "non_finite":
        x = rng.normal(size=shape).astype(np.float32)
        x.flat[0::4] = np.inf
        x.flat[1::4] = -np.inf
        x.flat[2::4] = np.nan
    else:
        x = rng.normal(size=shape)
    np.testing.assert_array_equal(dtw(x), _scalar_dtw(x))
