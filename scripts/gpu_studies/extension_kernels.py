"""Downloadable Warp kernel for the public custom-process tutorial."""

from typing import no_type_check

import warp as wp


@no_type_check
@wp.kernel
def prescribed_warming(
    temperature: wp.array(dtype=wp.float64),
    warming: wp.array(dtype=wp.float64),
    clock: wp.array(dtype=wp.float64),
    failure: wp.array(dtype=wp.int32),
    rate: wp.float64,
    dt: wp.float64,
):
    """Integrate dT/dt = rate * time; rate is K/s² and dt is seconds."""
    box = wp.tid()
    if failure[0] == 0:
        increment = rate * (clock[0] * dt + wp.float64(0.5) * dt * dt)
        temperature[box] += increment
        warming[box] += increment
