"""Differentiable log-diameter response for the SiMBA inverse workflow."""

import warp as wp

from particula.execution._condensation_ode_kernels import (
    FloatMatrix,
    FloatVector,
)


@wp.func
def _below(edge: wp.float64, diameter: wp.float64, width: wp.float64):
    """Probability of a reported diameter below a finite detector edge."""
    z = wp.log(edge / diameter) / (wp.sqrt(wp.float64(2)) * width)
    return wp.float64(0.5) * wp.erfc(-z)


@wp.kernel
def observe_number_spectrum(
    diameter: FloatMatrix,
    number: FloatMatrix,
    edges: FloatVector,
    log_width: wp.float64,
    output: FloatMatrix,
):
    """Integrate counts per m³ by diameter, including underflow and overflow.

    Diameter and strictly ascending edges are in metres. Each cohort has a
    Gaussian response in natural-log diameter with positive log_width. The
    output columns are number concentrations per bin, including both tails.
    """
    box, channel = wp.tid()  # type: ignore[misc]
    total = wp.float64(0)
    for j in range(diameter.shape[1]):
        left = wp.float64(0)
        right = wp.float64(1)
        if channel > 0:
            left = _below(edges[channel - 1], diameter[box, j], log_width)
        if channel < edges.shape[0]:
            right = _below(edges[channel], diameter[box, j], log_width)
        total += (right - left) * number[box, j]
    output[box, channel] = total
