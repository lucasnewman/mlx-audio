"""Read the official fermion-five-value-parakeet-v1 container without native code.

Adapted from fermion-research 0.2.11, Copyright 2026 Fermion Research,
Apache-2.0. Streaming records and explicit format checks added. See LICENSE/NOTICE.
"""

import json
import math

import numpy as np

FORMAT = "fermion-five-value-parakeet-v1"


def five_value(blob, shape):
    if len(shape) != 2:
        raise ValueError("Five-value records must be matrices")
    rows, cols = shape
    n = rows * ((cols + 4) // 5)
    if len(blob) < n + rows * 4:
        raise ValueError("Truncated five-value record")
    encoded = np.frombuffer(blob, np.uint8, count=n).reshape(rows, -1)
    if np.any(encoded > 242):
        raise ValueError("Invalid base-3 byte")
    codes = (encoded[..., None].astype(np.uint16) // np.array([1, 3, 9, 27, 81])) % 3
    sign = codes.reshape(rows, -1)[:, :cols].astype(np.int8) - 1
    mask = sign != 0
    count = int(mask.sum())
    nbits = (count + 7) // 8
    if len(blob) != n + nbits + 4 * rows:
        raise ValueError("Five-value record length disagrees with its shape")
    hi_mask = np.zeros(shape, dtype=bool)
    hi_mask[mask] = np.unpackbits(
        np.frombuffer(blob, np.uint8, count=nbits, offset=n), bitorder="little"
    )[:count]
    lo = np.frombuffer(blob, "<f2", count=rows, offset=n + nbits).copy()
    hi = np.frombuffer(blob, "<f2", count=rows, offset=n + nbits + rows * 2).copy()
    if (
        not np.all(np.isfinite(lo))
        or not np.all(np.isfinite(hi))
        or np.any(lo < 0)
        or np.any(hi < lo)
    ):
        raise ValueError("Invalid five-value palettes")
    return sign, hi_mask, lo, hi


def dense_record(blob, shape, kind):
    total = math.prod(shape)
    if kind == "fp16":
        if len(blob) != 2 * total:
            raise ValueError("Invalid fp16 record length")
        return np.frombuffer(blob, "<f2").reshape(shape).copy()
    rows = shape[0]
    if kind not in ("int6", "int8"):
        raise ValueError(f"Unsupported Phonon-2 record kind: {kind}")
    body = (total + 3) // 4 * 3 if kind == "int6" else total
    if len(blob) != body + 2 * rows:
        raise ValueError("Invalid integer record length")
    scales = np.frombuffer(blob, "<f2", count=rows, offset=body).astype(np.float32)
    if kind == "int8":
        q = np.frombuffer(blob, np.int8, count=total).astype(np.float32)
    else:
        b = np.frombuffer(blob, np.uint8, count=body).reshape(-1, 3).astype(np.uint32)
        words = b[:, 0] | b[:, 1] << 8 | b[:, 2] << 16
        q = ((words[:, None] >> np.array([0, 6, 12, 18], np.uint32)) & 63).reshape(-1)[
            :total
        ].astype(np.float32) - 32
    return (q.reshape(rows, -1) * scales[:, None]).reshape(shape)


def records(path):
    """Yield validated (entry, payload) records; never expand the entire encoder."""
    with open(path, "rb") as f:
        size = f.seek(0, 2)
        f.seek(0)
        raw = f.read(8)
        length = int.from_bytes(raw, "little")
        if len(raw) != 8 or not 0 < length <= min(size - 8, 16 << 20):
            raise ValueError("Invalid Phonon-2 container header length")
        header = json.loads(f.read(length))
        if (
            not isinstance(header, dict)
            or header.get("format") != FORMAT
            or not isinstance(header.get("index"), list)
        ):
            raise ValueError("Unsupported Phonon-2 container format")
        names = set()
        for entry in header["index"]:
            if not isinstance(entry, dict):
                raise ValueError("Invalid Phonon-2 container record")
            name, kind, shape, count = (entry.get(k) for k in ("n", "k", "shape", "b"))
            if (
                not isinstance(name, str)
                or not name
                or name in names
                or not isinstance(shape, list)
                or (not shape and kind != "fp16")
                or any(type(d) is not int or d <= 0 for d in shape)
                or type(count) is not int
                or count <= 0
                or count > size - f.tell()
            ):
                raise ValueError(
                    f"Invalid or duplicate Phonon-2 container record: {name!r}"
                )
            names.add(name)
            blob = f.read(count)
            yield entry, (
                five_value(blob, shape)
                if kind == "five_value"
                else dense_record(blob, shape, kind)
            )
        if f.read(1):
            raise ValueError("Trailing bytes in Phonon-2 container")
