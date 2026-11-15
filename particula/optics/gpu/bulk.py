"""Prepared bounded-memory optical reductions across particle populations."""

from dataclasses import dataclass

import numpy as np
import warp as wp

from particula.optics.gpu._bulk import (
    accumulate_angular,
    accumulate_scalar,
    clear_bulk,
    finalize_bulk,
    gather_cases,
)
from particula.optics.gpu.workspace import _array, _capacities, prepare_mie


@dataclass(frozen=True)
class PopulationWorkspace:
    """Retained streaming population outputs, commands and scratch.

    Attributes:
        optics: (groups,9): extinction, scattering, absorption, backscatter,
            radiation pressure [m⁻¹], SSA, g, extinction/scattering validity.
        mueller: (groups,angles,4,4) differential matrices [m⁻¹/sr]; zero
            length angle axis when angular output was not requested.
        status: Per-group status; 0 success, 2 input/upstream error, 3 overflow.
        particle_tile: Fixed maximum live particle scratch count.

    Groups flatten box/wavelength axes. Outputs require enqueue and successful
    observation; inputs may be updated in place on the prepared stream.
    """

    optics: wp.array
    mueller: wp.array
    status: wp.array
    particle_tile: int
    _commands: tuple
    _resources: tuple
    _device: str
    _stream: wp.Stream | None

    def enqueue(self) -> None:
        """Enqueue the retained streaming calculation without host transfers."""
        for command in self._commands:
            command.launch()

    def check(self) -> None:
        """Explicitly synchronize and reject failed population results."""
        if self._stream is not None:
            wp.synchronize_stream(self._stream)
        else:
            wp.synchronize_device(self._device)
        values = self.status.numpy()
        if np.any(values):
            raise RuntimeError(f"Population optics failed: {np.unique(values)}")


def prepare_population_mie(
    relative_index: wp.array,
    size_parameter: wp.array,
    radius: wp.array,
    number_density_weights: wp.array,
    *,
    angles: wp.array | None = None,
    particle_tile: int = 256,
    angle_tile: int = 128,
    coefficient_capacity: int = 10090,
    stream=None,
) -> PopulationWorkspace:
    """Prepare streamed bulk optics without retaining particle optical output.

    Args:
        relative_index: Contiguous (groups,particles) vec2d relative n+ik.
        size_parameter: Same shape float64 x=2*pi*r*n_medium/lambda_vacuum.
        radius: Same shape float64 SI radius [m], consistent with x.
        number_density_weights: Same shape float64 density weights [m⁻³].
            Callers supply discrete, resolved-count/volume or PDF weights.
        angles: Optional shared 1D float64 angles [rad], on the same device.
        particle_tile: Maximum particles in scratch, default 256.
        angle_tile: Angular launch width, default 128.
        coefficient_capacity: Maximum retained series order, default 10090.
        stream: Optional matching CUDA stream; used for all work and checks.

    Returns:
        Population workspace. Scratch is bounded by particle_tile, series
        capacity and angle count, independently of total particle count.
        No CPU transfers, fallback or scientific launches during preparation.

    Raises:
        ValueError: If shapes/devices/capacities conflict. Physical-value errors
            are reported by check after enqueue, without host input readback.
    """
    shape: tuple[int, ...] = getattr(relative_index, "shape", ())
    device = getattr(relative_index, "device", None)
    if device is None or len(shape) != 2 or shape[0] < 1 or shape[1] < 1:
        raise ValueError("Require nonempty (groups,particles) device inputs")
    _array(relative_index, wp.vec2d, shape, device, "relative_index")
    for name, value in (
        ("size_parameter", size_parameter),
        ("radius", radius),
        ("number_density_weights", number_density_weights),
    ):
        _array(value, wp.float64, shape, device, name)
    _capacities(coefficient_capacity, particle_tile, angle_tile)
    groups, particles = shape
    tile = min(particle_tile, particles)

    def empty(dtype):
        return wp.empty(tile, dtype=dtype, device=device)

    index, size = empty(wp.vec2d), empty(wp.float64)
    r, weight, scale = empty(wp.float64), empty(wp.float64), empty(wp.float64)
    gather_status = empty(wp.int32)
    mie = prepare_mie(
        index,
        size,
        angles=angles,
        matrix_scale=scale if angles is not None else None,
        coefficient_capacity=coefficient_capacity,
        case_tile=tile,
        angle_tile=angle_tile,
        stream=stream,
    )
    stream = mie._stream
    angle_count = 0 if angles is None else angles.shape[0]
    totals = wp.empty((groups, 9), dtype=wp.float64, device=device)
    matrices = wp.empty(
        (groups, angle_count, 16), dtype=wp.float64, device=device
    )
    status = wp.zeros(groups, dtype=wp.int32, device=device)
    commands = []

    def record(kernel, dim, inputs):
        commands.append(
            wp.launch(
                kernel,
                dim=dim,
                inputs=inputs,
                device=device,
                stream=stream,
                record_cmd=True,
            )
        )

    record(clear_bulk, groups, [totals, status, matrices])
    for group in range(groups):
        for start in range(0, particles, tile):
            record(
                gather_cases,
                tile,
                [
                    relative_index,
                    size_parameter,
                    radius,
                    number_density_weights,
                    group,
                    start,
                    min(tile, particles - start),
                    index,
                    size,
                    r,
                    weight,
                    scale,
                    gather_status,
                ],
            )
            commands.extend(mie._commands)
            record(
                accumulate_scalar,
                1,
                [
                    mie.efficiencies,
                    r,
                    weight,
                    gather_status,
                    mie.status,
                    group,
                    totals,
                    status,
                ],
            )
            if angles is not None:
                record(
                    accumulate_angular,
                    (angle_count, 16),
                    [
                        mie.mueller,
                        weight,
                        mie.mueller_status,
                        group,
                        matrices,
                        status,
                    ],
                )
    record(finalize_bulk, groups, [totals, status])
    resources = (
        relative_index,
        size_parameter,
        radius,
        number_density_weights,
        angles,
        mie,
        index,
        size,
        r,
        weight,
        scale,
        gather_status,
        matrices,
    )
    return PopulationWorkspace(
        totals,
        matrices.reshape((groups, angle_count, 4, 4)),
        status,
        tile,
        tuple(commands),
        resources,
        device.alias,
        stream,
    )
