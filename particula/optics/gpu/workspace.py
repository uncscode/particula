"""Explicit preparation and observation for tiled resident optical kernels.

Preparation validates metadata, allocates storage and compiles retained launch
commands without advancing the calculation. Enqueue reuses those resources;
check is a deliberate synchronization/readback boundary. Input values may be
updated in place on the selected stream between calls; layouts stay fixed.
"""

from dataclasses import dataclass

import numpy as np
import warp as wp

from particula.optics.gpu.angular import (
    mueller_kernel,
    scattering_amplitudes_kernel,
)
from particula.optics.gpu.mie import (
    mie_coefficients_kernel,
    mie_efficiencies_kernel,
)


def _array(value, dtype, shape, device, name):
    """Validate device array metadata without inspecting its device values."""
    if (
        getattr(value, "dtype", None) != dtype
        or getattr(value, "shape", None) != shape
        or getattr(value, "device", None) != device
        or not getattr(value, "is_contiguous", False)
    ):
        raise ValueError(
            f"{name} requires contiguous {dtype} {shape} on {device}"
        )


@dataclass(frozen=True)
class MieWorkspace:
    """Prepared numerical outputs and retained device launch commands.

    Attributes:
        efficiencies: (cases,7): Qext,Qsca,Qabs,g,Qback,Qpr,scattering-valid.
        status: (cases,) coefficient status; zero means success.
        perpendicular: Optional (cases,angles) complex S1 as vec2d.
        parallel: Optional (cases,angles) complex S2 as vec2d.
        mueller: Optional (cases,angles,4,4) scaled Mueller matrices.
        angular_status: Optional per-angle amplitude status.
        mueller_status: Optional per-angle matrix status.
        coefficient_capacity: Maximum orders retained in each tiled case.
        case_tile: Maximum cases sharing reusable coefficient scratch.
        angle_tile: Maximum angles in each angular launch.

    Outputs are undefined until enqueue and successful check. Direct device
    consumers may instead inspect status on-device. This object is not safe
    for concurrent execution on multiple streams.
    """

    efficiencies: wp.array
    status: wp.array
    perpendicular: wp.array | None
    parallel: wp.array | None
    mueller: wp.array | None
    angular_status: wp.array | None
    mueller_status: wp.array | None
    coefficient_capacity: int
    case_tile: int
    angle_tile: int
    _commands: tuple
    _resources: tuple
    _device: str
    _stream: wp.Stream | None

    def enqueue(self) -> None:
        """Enqueue retained commands, without allocation or host readback.

        Callers order their input writes and consumers on the prepared stream.
        Numeric failures remain in status buffers until explicit observation.
        """
        for command in self._commands:
            command.launch()

    def check(self) -> None:
        """Synchronize and inspect status arrays; raise on numeric failure.

        Raises:
            RuntimeError: If any coefficient, angle or matrix case failed.
        """
        if self._stream is not None:
            wp.synchronize_stream(self._stream)
        else:
            wp.synchronize_device(self._device)
        for name, array in (
            ("coefficients", self.status),
            ("angles", self.angular_status),
            ("matrices", self.mueller_status),
        ):
            if array is not None:
                values = array.numpy()
                if np.any(values):
                    codes = np.unique(values[values != 0]).tolist()
                    raise RuntimeError(f"Mie {name} failed with status {codes}")


def _capacities(coefficient_capacity, case_tile, angle_tile):
    """Admit positive integer workspace capacities."""
    for name, value in (
        ("coefficient_capacity", coefficient_capacity),
        ("case_tile", case_tile),
        ("angle_tile", angle_tile),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if coefficient_capacity > 10090:
        raise ValueError("Coefficient capacity exceeds qualified x=10000 bound")


def _admit(
    relative_index,
    size_parameter,
    angles,
    matrix_scale,
    coefficient_capacity,
    case_tile,
    angle_tile,
    stream,
):
    """Admit layouts and stream before allocating or compiling commands."""
    device = getattr(relative_index, "device", None)
    shape: tuple[int, ...] = getattr(relative_index, "shape", ())
    if device is None or len(shape) != 1 or shape[0] < 1:
        raise ValueError("Require at least one device-resident optical case")
    count = shape[0]
    _array(relative_index, wp.vec2d, (count,), device, "relative_index")
    _array(size_parameter, wp.float64, (count,), device, "size_parameter")
    _capacities(coefficient_capacity, case_tile, angle_tile)
    if stream is not None and (not device.is_cuda or stream.device != device):
        raise ValueError("Stream must belong to the selected CUDA device")
    if stream is None and device.is_cuda:
        stream = wp.get_stream(device)
    angle_count = 0
    if angles is not None:
        angle_shape: tuple[int, ...] = getattr(angles, "shape", ())
        if len(angle_shape) != 1 or angle_shape[0] < 1:
            raise ValueError("Require at least one angular coordinate")
        angle_count = angle_shape[0]
        _array(angles, wp.float64, (angle_count,), device, "angles")
    elif matrix_scale is not None:
        raise ValueError("Matrix scale requires angular outputs")
    if matrix_scale is not None:
        _array(matrix_scale, wp.float64, (count,), device, "matrix_scale")
    return device, count, angle_count, stream


def prepare_mie(
    relative_index: wp.array,
    size_parameter: wp.array,
    *,
    angles: wp.array | None = None,
    matrix_scale: wp.array | None = None,
    coefficient_capacity: int = 10090,
    case_tile: int = 256,
    angle_tile: int = 128,
    stream=None,
) -> MieWorkspace:
    """Prepare tiled device optics for already-uploaded numeric inputs.

    Args:
        relative_index: Contiguous (cases,) vec2d relative n+ik on CPU/CUDA.
        size_parameter: Contiguous (cases,) float64 x, same device.
        angles: Optional contiguous (angles,) float64 radians, same device.
        matrix_scale: Optional (cases,) float64 multiplier for Mueller values;
            default one. Only accepted when angles are requested.
        coefficient_capacity: Positive scratch order bound, at most 10090.
            Insufficient capacity reports per-case status rather than truncates.
        case_tile: Positive maximum cases in coefficient scratch (default 256).
        angle_tile: Positive angular launch width (default 128).
        stream: Optional CUDA stream on the same device. Input writes and
            downstream consumers must use this stream or explicit ordering.

    Returns:
        Workspace retaining inputs, output arrays, scratch and launch commands.
        Scratch scales with case_tile times coefficient_capacity; requested
        per-particle angular outputs still scale with cases times angles.
        No host values are copied, and preparation launches no scientific work.

    Raises:
        ValueError: If input metadata, device, stream or capacities conflict.
        RuntimeError: If the explicitly requested device cannot be used.
    """
    device, count, angle_count, stream = _admit(
        relative_index,
        size_parameter,
        angles,
        matrix_scale,
        coefficient_capacity,
        case_tile,
        angle_tile,
        stream,
    )
    tile = min(case_tile, count)

    def empty(shape, dtype):
        return wp.empty(shape, dtype=dtype, device=device)

    derivative = empty((tile, coefficient_capacity + 1), wp.vec2d)
    a = empty((tile, coefficient_capacity), wp.vec2d)
    b = empty((tile, coefficient_capacity), wp.vec2d)
    terms = empty((count,), wp.int32)
    status = wp.zeros(count, dtype=wp.int32, device=device)
    efficiency = empty((count, 7), wp.float64)
    s1 = s2 = matrix = angle_status = matrix_status = None
    if angles is not None:
        s1 = empty((count, angle_count), wp.vec2d)
        s2 = empty((count, angle_count), wp.vec2d)
        matrix = empty((count, angle_count, 4, 4), wp.float64)
        angle_status = wp.zeros(
            (count, angle_count), dtype=wp.int32, device=device
        )
        matrix_status = wp.zeros_like(angle_status)
        if matrix_scale is None:
            matrix_scale = wp.ones(count, dtype=wp.float64, device=device)
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

    for start in range(0, count, tile):
        stop = min(start + tile, count)
        length = stop - start
        indices, sizes = relative_index[start:stop], size_parameter[start:stop]
        counts, codes = terms[start:stop], status[start:stop]
        record(
            mie_coefficients_kernel,
            length,
            [indices, sizes, derivative, a, b, counts, codes],
        )
        record(
            mie_efficiencies_kernel,
            length,
            [indices, sizes, a, b, counts, codes, efficiency[start:stop]],
        )
        if angles is not None:
            assert s1 is not None and s2 is not None  # noqa: S101
            assert matrix is not None and matrix_scale is not None  # noqa: S101
            assert angle_status is not None and matrix_status is not None  # noqa: S101
            for lower in range(0, angle_count, angle_tile):
                upper = min(lower + angle_tile, angle_count)
                per = s1[start:stop, lower:upper]
                par = s2[start:stop, lower:upper]
                angular_codes = angle_status[start:stop, lower:upper]
                record(
                    scattering_amplitudes_kernel,
                    (length, upper - lower),
                    [
                        a,
                        b,
                        counts,
                        codes,
                        angles[lower:upper],
                        per,
                        par,
                        angular_codes,
                    ],
                )
                record(
                    mueller_kernel,
                    (length, upper - lower),
                    [
                        per,
                        par,
                        angular_codes,
                        matrix_scale[start:stop],
                        matrix[start:stop, lower:upper],
                        matrix_status[start:stop, lower:upper],
                    ],
                )
    resources = (
        relative_index,
        size_parameter,
        angles,
        matrix_scale,
        derivative,
        a,
        b,
        terms,
    )
    return MieWorkspace(
        efficiency,
        status,
        s1,
        s2,
        matrix,
        angle_status,
        matrix_status,
        coefficient_capacity,
        tile,
        angle_tile,
        tuple(commands),
        resources,
        device.alias,
        stream,
    )
