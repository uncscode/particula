"""Storage validation for in-place CPU array updates."""

import numpy as np


def validate_writable_arrays(fields, writers):
    """Require each destination to have independent, writable elements."""
    for name in writers:
        target = fields[name]
        if not target.flags.writeable:
            raise ValueError(f"{name} must be writable")
        if any(
            np.shares_memory(target, other)
            for key, other in fields.items()
            if key != name
        ):
            raise ValueError("CPU writer fields must not alias state")
        if target.size == 0:
            continue
        # A stride-span proof handles contiguous, transposed, reversed and
        # ordinary sliced arrays without a quadratic scan over box pairs.
        span = target.itemsize
        for stride, length in sorted(
            (abs(stride), length)
            for stride, length in zip(target.strides, target.shape, strict=True)
            if length > 1
        ):
            if stride < span:
                # Interleaved views can fail the simple span proof without
                # overlapping. Resolve only these unusual layouts exactly.
                offsets = np.array([0], dtype=np.int64)
                for size, step in zip(
                    target.shape, target.strides, strict=True
                ):
                    offsets = (
                        offsets[:, None] + np.arange(size) * step
                    ).ravel()
                offsets.sort()
                if np.any(np.diff(offsets) < target.itemsize):
                    raise ValueError("CPU writer elements must not overlap")
                break
            span += stride * (length - 1)
