"""Compute straight-line particle trajectories and face arrival times."""

import warp as wp

from particula.execution._condensation_ode_kernels import FloatMatrix as Matrix

F = wp.float64


@wp.kernel
def arrivals(
    position: Matrix,
    velocity: Matrix,
    lower: Matrix,
    upper: Matrix,
    until: F,
    times: Matrix,
):
    """Compute time to the first outward face on each axis, in seconds."""
    p, axis = wp.tid()  # type: ignore[misc]  # 2-D launch
    v = velocity[p, axis]
    t = until
    if v > F(0.0):
        t = wp.max(F(0.0), (upper[p, axis] - position[p, axis]) / v)
    elif v < F(0.0):
        t = wp.max(F(0.0), (lower[p, axis] - position[p, axis]) / v)
    times[p, axis] = t


@wp.kernel
def displace(position: Matrix, velocity: Matrix, dt: F, output: Matrix):
    """Advance all positions by the accepted duration."""
    p, axis = wp.tid()  # type: ignore[misc]  # 2-D launch
    output[p, axis] = position[p, axis] + velocity[p, axis] * dt
