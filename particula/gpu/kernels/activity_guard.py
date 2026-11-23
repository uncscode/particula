"""Propagate neutral activity intervals through the configured box rate law."""

from typing import TYPE_CHECKING, Any

import warp as wp

if TYPE_CHECKING:
    FloatTensor = Any
    IntMatrix = Any
else:
    FloatTensor = wp.array3d(dtype=wp.float64)
    IntMatrix = wp.array2d(dtype=int)


@wp.kernel(enable_backward=False)
def _bound(
    activity: FloatTensor,
    error: FloatTensor,
    sign: wp.float64,
    out: FloatTensor,
):
    b, s, k = wp.tid()  # type: ignore[misc]
    out[b, s, k] = activity[b, s, k] * wp.exp(sign * error[b, s, k])


@wp.kernel(enable_backward=False)
def _crosses(
    low: FloatTensor,
    high: FloatTensor,
    error: FloatTensor,
    mask: IntMatrix,
):
    b, s = wp.tid()  # type: ignore[misc]
    uncertain = int(0)
    for k in range(low.shape[2]):
        if error[b, s, k] > wp.float64(0.0):
            a = low[b, s, k]
            z = high[b, s, k]
            if (
                not wp.isfinite(a)
                or not wp.isfinite(z)
                or (a <= wp.float64(0.0) and z >= wp.float64(0.0))
                or (a >= wp.float64(0.0) and z <= wp.float64(0.0))
            ):
                uncertain = 1
    mask[b, s] = uncertain


class BoxFluxGuard:
    """Use the configured box rate law at both activity endpoints."""

    def __init__(self, binding):
        """Allocate interval and proposal buffers before capture."""
        self.binding = binding
        self.bound = wp.empty_like(binding.values)
        self.flat = self.bound.flatten()
        self.low = wp.empty_like(self.bound)
        self.high = wp.empty_like(self.bound)

    def refine(self, rate, masses, counts, temperature, invalid, stage):
        """Refresh all targets for each row whose rate interval spans zero."""
        b = self.binding
        for sign, output in ((-1.0, self.low), (1.0, self.high)):
            wp.launch(
                _bound,
                self.bound.shape,
                inputs=[b.values, b.errors, wp.float64(sign), self.bound],
                device=b.device,
            )
            rate(self.flat, output)
        wp.launch(
            _crosses,
            b.uncertain.shape,
            inputs=[self.low, self.high, b.errors, b.uncertain],
            device=b.device,
        )
        b.evaluate(masses, counts, temperature, invalid, stage, refine=True)
