"""Read-only direct-Warp nucleation configuration and preflight.

This concrete-only P1 boundary owns fixed-capacity schema, ownership, and
scientific-domain validation for the deferred direct GPU nucleation path.  It
does not calculate rates, allocate fallback work storage, transfer state to the
host, activate slots, or mutate caller-owned data.  P2--P7 own those actions.
"""

# mypy: disable-error-code="valid-type, misc"

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np

try:
    import warp as wp
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "Warp is required for GPU nucleation helpers. "
        "Install with: pip install warp-lang"
    ) from exc

from particula.gpu.kernels.environment import _is_warp_array_like
from particula.util.constants import AVOGADRO_NUMBER


def _real(value: object, name: str, *, positive: bool = False) -> float:
    """Normalize a finite non-Boolean real scalar."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real scalar.")
    result = float(value)
    if (
        not np.isfinite(result)
        or (positive and result <= 0.0)
        or (not positive and result < 0.0)
    ):
        qualifier = "positive" if positive else "finite and nonnegative"
        raise ValueError(f"{name} must be {qualifier}.")
    return result


def _floating_scalar(value: object, name: str) -> float:
    """Normalize one finite positive Python or NumPy floating scalar."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (float, np.floating)
    ):
        raise TypeError(f"{name} must be a floating scalar.")
    return _real(value, name, positive=True)


@dataclass(frozen=True)
class NucleationConfig:
    """Immutable potential-nucleation configuration.

    Coefficient units are ``m^-3 s^-1`` for ``activation`` and the
    corresponding kinetic coefficient units for ``kinetic``.  Formation
    diameter is in m; temperature is in K; number concentration is in
    ``#/m^3``.  The optional saturation bounds are dimensionless.
    """

    rate_law: str
    coefficient: float
    survival_factor: float
    precursor_index: int
    molecule_counts: tuple[int, ...]
    formation_diameter: float
    precursor_number_concentration_lower: float
    precursor_number_concentration_upper: float
    temperature_lower: float
    temperature_upper: float
    saturation_lower: float | None = None
    saturation_upper: float | None = None

    def __post_init__(self) -> None:
        """Validate scalar-only configuration consistency."""
        if self.rate_law not in ("activation", "kinetic"):
            raise ValueError("rate_law must be activation or kinetic.")
        _real(self.coefficient, "coefficient")
        _real(self.survival_factor, "survival_factor")
        if isinstance(self.precursor_index, (bool, np.bool_)) or not isinstance(
            self.precursor_index, (int, np.integer)
        ):
            raise TypeError("precursor_index must be a nonnegative integer.")
        if self.precursor_index < 0:
            raise ValueError("precursor_index must be nonnegative.")
        _real(self.formation_diameter, "formation_diameter", positive=True)
        if (
            not isinstance(self.molecule_counts, tuple)
            or not self.molecule_counts
        ):
            raise ValueError("molecule_counts must be a nonempty tuple.")
        if any(
            isinstance(count, (bool, np.bool_))
            or not isinstance(count, (int, np.integer))
            for count in self.molecule_counts
        ):
            raise TypeError("molecule_counts must contain integers.")
        if any(count < 0 for count in self.molecule_counts) or not any(
            count > 0 for count in self.molecule_counts
        ):
            raise ValueError(
                "molecule_counts must be nonnegative with a positive count."
            )
        self._interval(
            self.precursor_number_concentration_lower,
            self.precursor_number_concentration_upper,
            "precursor_number_concentration",
            positive=False,
        )
        self._interval(
            self.temperature_lower,
            self.temperature_upper,
            "temperature",
            positive=True,
        )
        if (self.saturation_lower is None) != (self.saturation_upper is None):
            raise ValueError("saturation bounds must be supplied as a pair.")
        if self.saturation_lower is not None:
            self._interval(
                self.saturation_lower,
                self.saturation_upper,
                "saturation",
                positive=False,
            )

    @staticmethod
    def _interval(
        lower: object, upper: object, name: str, *, positive: bool
    ) -> None:
        if _real(lower, f"{name}_lower", positive=positive) > _real(
            upper, f"{name}_upper", positive=positive
        ):
            raise ValueError(f"{name}_lower must not exceed {name}_upper.")


@dataclass(frozen=True)
class NucleationScratchBuffers:
    """Caller-owned future ``float64 (B,)`` planning sidecars.

    ``precursor_number_concentration`` [#/m^3], ``potential_rate`` [#/m^3/s],
    and ``potential_demand`` are stable same-device contiguous buffers.
    """

    precursor_number_concentration: Any
    potential_rate: Any
    potential_demand: Any


@dataclass(frozen=True)
class NucleationFinalizedDemandBuffers:
    """Caller-owned future finalized count, demand, and mass-change buffers."""

    accepted_counts: Any
    accepted_demand: Any
    precursor_mass_change: Any


@dataclass(frozen=True)
class NucleationDiagnosticBuffers:
    """Caller-owned future diagnostic sidecars.

    ``selected_slot_indices`` has shape ``(B, N)``.  Its future unused tail
    convention is ``-1``; P1 deliberately does not inspect or initialize it.
    """

    gate_codes: Any
    selected_slot_indices: Any


@dataclass(frozen=True)
class _NucleationPreflight:
    """Private normalized P1 metadata; not a supported public API."""

    particles: Any
    gas: Any
    config: NucleationConfig
    n_boxes: int
    n_particles: int
    n_species: int
    device: Any
    temperature: Any
    saturation: Any | None
    has_eligible_boxes: bool
    has_gated_boxes: bool
    gate_reason: str | None
    potential_rate: float = 0.0
    potential_demand: float = 0.0
    accepted_count: int = 0
    accepted_demand: float = 0.0
    precursor_mass_change: float = 0.0


def _field(container: Any, name: str) -> Any:
    """Get a required field without accepting host substitutes."""
    try:
        return getattr(container, name)
    except AttributeError as exc:
        raise ValueError(f"{name} must be a Warp array.") from exc


def _array(
    container: Any, name: str, dtype: Any, shape: tuple[int, ...], device: Any
) -> Any:
    """Validate one same-device contiguous Warp array schema."""
    value = _field(container, name)
    if not _is_warp_array_like(value):
        raise ValueError(f"{name} must be a Warp array.")
    if value.dtype != dtype:
        raise ValueError(f"{name} must use the required dtype.")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} shape must match particle masses.")
    if str(value.device) != str(device):
        raise ValueError(f"{name} device must match particle device.")
    _memory_range(value)  # Also rejects non-contiguous views.
    return value


def _memory_range(array: Any) -> tuple[int, int] | None:
    """Return byte bounds after rejecting unsafe noncontiguous views."""
    sizes = {wp.float64: 8, wp.int32: 4}
    size = sizes.get(array.dtype)
    if size is None:
        raise ValueError("overlap-checked arrays use an unsupported dtype.")
    expected: list[int] = []
    stride = size
    for dimension in reversed(array.shape):
        expected.insert(0, stride)
        stride *= dimension
    if getattr(array, "strides", None) is not None and tuple(
        array.strides
    ) != tuple(expected):
        raise ValueError("overlap-checked Warp arrays must be contiguous.")
    count = int(np.prod(array.shape, dtype=np.int64))
    return (
        None if count == 0 else (int(array.ptr), int(array.ptr) + count * size)
    )


def _no_overlap(arrays: tuple[Any, ...]) -> None:
    """Reject any identity or byte-range alias among future mutable state."""
    for index, first in enumerate(arrays):
        first_range = _memory_range(first)
        for second in arrays[index + 1 :]:
            second_range = _memory_range(second)
            if first is second or (
                first_range is not None
                and second_range is not None
                and first_range[0] < second_range[1]
                and second_range[0] < first_range[1]
            ):
                raise ValueError(
                    "nucleation state and sidecars must not overlap."
                )


@wp.kernel
def _invalid_float(
    values: wp.array(dtype=wp.float64),
    positive: wp.int32,
    invalid: wp.array(dtype=wp.int32),
) -> None:
    index = wp.tid()
    if (
        not wp.isfinite(values[index])
        or (positive != 0 and values[index] <= 0.0)
        or (positive == 0 and values[index] < 0.0)
    ):
        wp.atomic_add(invalid, 0, 1)


@wp.kernel
def _invalid_float_2d(
    values: wp.array2d(dtype=wp.float64),
    positive: wp.int32,
    invalid: wp.array(dtype=wp.int32),
) -> None:
    """Record invalid two-dimensional float values without modifying them."""
    row, column = wp.tid()
    value = values[row, column]
    if (
        not wp.isfinite(value)
        or (positive != 0 and value <= 0.0)
        or (positive == 0 and value < 0.0)
    ):
        wp.atomic_add(invalid, 0, 1)


@wp.kernel
def _invalid_float_3d(
    values: wp.array3d(dtype=wp.float64),
    positive: wp.int32,
    invalid: wp.array(dtype=wp.int32),
) -> None:
    """Record invalid three-dimensional float values without modifying them."""
    row, column, lane = wp.tid()
    value = values[row, column, lane]
    if (
        not wp.isfinite(value)
        or (positive != 0 and value <= 0.0)
        or (positive == 0 and value < 0.0)
    ):
        wp.atomic_add(invalid, 0, 1)


@wp.kernel
def _invalid_interval_1d(
    values: wp.array(dtype=wp.float64),
    lower: wp.float64,
    upper: wp.float64,
    invalid: wp.array(dtype=wp.int32),
) -> None:
    """Record one-dimensional values outside their configured interval."""
    index = wp.tid()
    if values[index] < lower or values[index] > upper:
        wp.atomic_add(invalid, 0, 1)


@wp.kernel
def _invalid_interval_2d(
    values: wp.array2d(dtype=wp.float64),
    lower: wp.float64,
    upper: wp.float64,
    invalid: wp.array(dtype=wp.int32),
) -> None:
    """Record two-dimensional values outside their configured interval."""
    row, column = wp.tid()
    if values[row, column] < lower or values[row, column] > upper:
        wp.atomic_add(invalid, 0, 1)


@wp.kernel
def _invalid_binary(
    values: wp.array2d(dtype=wp.int32),
    invalid: wp.array(dtype=wp.int32),
) -> None:
    """Record partitioning values outside the binary representation."""
    row, column = wp.tid()
    value = values[row, column]
    if value != 0 and value != 1:
        wp.atomic_add(invalid, 0, 1)


@wp.kernel
def _disabled_precursor(
    values: wp.array2d(dtype=wp.int32),
    index: wp.int32,
    invalid: wp.array(dtype=wp.int32),
) -> None:
    """Record disabled precursor species by box."""
    box = wp.tid()
    if values[box, index] != 1:
        wp.atomic_add(invalid, 0, 1)


@wp.kernel
def _precursor_status(
    concentration: wp.array2d(dtype=wp.float64),
    molar_mass: wp.array(dtype=wp.float64),
    precursor_index: wp.int32,
    lower: wp.float64,
    upper: wp.float64,
    status: wp.array(dtype=wp.int32),
) -> None:
    """Record zero and out-of-domain precursor concentration by box."""
    box = wp.tid()
    number_concentration = (
        concentration[box, precursor_index]
        * wp.float64(AVOGADRO_NUMBER)
        / molar_mass[precursor_index]
    )
    if number_concentration == 0.0:
        wp.atomic_add(status, 1, 1)
    elif number_concentration < lower or number_concentration > upper:
        wp.atomic_add(status, 0, 1)


@wp.kernel
def _saturation_status(
    saturation: wp.array2d(dtype=wp.float64),
    precursor_index: wp.int32,
    lower: wp.float64,
    upper: wp.float64,
    status: wp.array(dtype=wp.int32),
) -> None:
    """Record lower gates and upper saturation-domain failures by box."""
    box = wp.tid()
    value = saturation[box, precursor_index]
    if value < lower:
        wp.atomic_add(status, 1, 1)
    elif value > upper:
        wp.atomic_add(status, 0, 1)


def _scan(values: Any, name: str, *, positive: bool = False) -> None:
    """Perform a read-only finite-domain scan with scalar status readback."""
    invalid = wp.zeros(1, dtype=wp.int32, device=values.device)
    kernels = {1: _invalid_float, 2: _invalid_float_2d, 3: _invalid_float_3d}
    wp.launch(
        kernels[values.ndim],
        dim=values.shape,
        inputs=[values, wp.int32(positive), invalid],
        device=values.device,
    )
    if int(invalid.numpy()[0]):
        domain = "positive and finite" if positive else "finite and nonnegative"
        raise ValueError(f"{name} must be {domain}.")


@wp.kernel
def _invalid_nonfinite_2d(
    values: wp.array2d(dtype=wp.float64),
    invalid: wp.array(dtype=wp.int32),
) -> None:
    """Record nonfinite two-dimensional values without modifying them."""
    row, column = wp.tid()
    if not wp.isfinite(values[row, column]):
        wp.atomic_add(invalid, 0, 1)


def _scan_finite_2d(values: Any, name: str) -> None:
    """Perform a read-only scan that permits finite signed values."""
    invalid = wp.zeros(1, dtype=wp.int32, device=values.device)
    wp.launch(
        _invalid_nonfinite_2d,
        dim=values.shape,
        inputs=[values, invalid],
        device=values.device,
    )
    if int(invalid.numpy()[0]):
        raise ValueError(f"{name} must be finite.")


def _scan_interval(values: Any, name: str, lower: float, upper: float) -> None:
    """Perform a read-only configured-interval scan with status readback."""
    invalid = wp.zeros(1, dtype=wp.int32, device=values.device)
    kernels = {1: _invalid_interval_1d, 2: _invalid_interval_2d}
    wp.launch(
        kernels[values.ndim],
        dim=values.shape,
        inputs=[values, lower, upper, invalid],
        device=values.device,
    )
    if int(invalid.numpy()[0]):
        raise ValueError(f"{name} is outside configured bounds.")


def _scan_binary(values: Any) -> None:
    """Validate binary partitioning with a device-resident read-only scan."""
    invalid = wp.zeros(1, dtype=wp.int32, device=values.device)
    wp.launch(
        _invalid_binary,
        dim=values.shape,
        inputs=[values, invalid],
        device=values.device,
    )
    if int(invalid.numpy()[0]):
        raise ValueError("gas.partitioning must be binary.")


def _validate_sidecars(
    scratch: NucleationScratchBuffers | None,
    finalized: NucleationFinalizedDemandBuffers | None,
    diagnostics: NucleationDiagnosticBuffers | None,
    n_boxes: int,
    n_particles: int,
    n_species: int,
    device: Any,
) -> tuple[Any, ...]:
    """Validate supplied sidecar schemas only; stale contents are untouched."""
    arrays: list[Any] = []
    if scratch is not None:
        if not isinstance(scratch, NucleationScratchBuffers):
            raise ValueError("scratch must be NucleationScratchBuffers.")
        for name in (
            "precursor_number_concentration",
            "potential_rate",
            "potential_demand",
        ):
            arrays.append(_array(scratch, name, wp.float64, (n_boxes,), device))
    if finalized is not None:
        if not isinstance(finalized, NucleationFinalizedDemandBuffers):
            raise ValueError(
                "finalized_demand must be NucleationFinalizedDemandBuffers."
            )
        arrays.extend(
            (
                _array(
                    finalized, "accepted_counts", wp.int32, (n_boxes,), device
                ),
                _array(
                    finalized, "accepted_demand", wp.float64, (n_boxes,), device
                ),
                _array(
                    finalized,
                    "precursor_mass_change",
                    wp.float64,
                    (n_boxes, n_species),
                    device,
                ),
            )
        )
    if diagnostics is not None:
        if not isinstance(diagnostics, NucleationDiagnosticBuffers):
            raise ValueError("diagnostics must be NucleationDiagnosticBuffers.")
        arrays.extend(
            (
                _array(diagnostics, "gate_codes", wp.int32, (n_boxes,), device),
                _array(
                    diagnostics,
                    "selected_slot_indices",
                    wp.int32,
                    (n_boxes, n_particles),
                    device,
                ),
            )
        )
    return tuple(arrays)


def _preflight_nucleation(  # noqa: C901
    particles: Any,
    gas: Any,
    config: NucleationConfig,
    time_step: Any,
    temperature: Any | None = None,
    saturation: Any | None = None,
    environment: Any | None = None,
    scratch: NucleationScratchBuffers | None = None,
    finalized_demand: NucleationFinalizedDemandBuffers | None = None,
    diagnostics: NucleationDiagnosticBuffers | None = None,
) -> _NucleationPreflight:
    """Validate the deferred P1 boundary without writing caller-owned arrays."""
    masses = _field(particles, "masses")
    if (
        not _is_warp_array_like(masses)
        or masses.dtype != wp.float64
        or masses.ndim != 3
    ):
        raise ValueError(
            "particles.masses must be a rank-3 float64 Warp array."
        )
    _memory_range(masses)
    n_boxes, n_particles, n_species = tuple(masses.shape)
    device = masses.device
    particle_arrays = (
        masses,
        _array(
            particles,
            "concentration",
            wp.float64,
            (n_boxes, n_particles),
            device,
        ),
        _array(particles, "charge", wp.float64, (n_boxes, n_particles), device),
        _array(particles, "density", wp.float64, (n_species,), device),
        _array(particles, "volume", wp.float64, (n_boxes,), device),
    )
    gas_arrays = (
        _array(gas, "molar_mass", wp.float64, (n_species,), device),
        _array(gas, "concentration", wp.float64, (n_boxes, n_species), device),
        _array(gas, "partitioning", wp.int32, (n_boxes, n_species), device),
    )
    temperature_value: Any
    saturation_value: Any | None = None
    external: list[Any] = []
    if environment is not None:
        if temperature is not None or saturation is not None:
            raise ValueError(
                "direct temperature and saturation require no environment."
            )
        temperature_value = _array(
            environment, "temperature", wp.float64, (n_boxes,), device
        )
        external.append(temperature_value)
        if config.saturation_lower is not None:
            saturation_value = _array(
                environment,
                "saturation_ratio",
                wp.float64,
                (n_boxes, n_species),
                device,
            )
            external.append(saturation_value)
    else:
        if _is_warp_array_like(temperature):
            temperature_value = _array(
                type("Direct", (), {"temperature": temperature})(),
                "temperature",
                wp.float64,
                (n_boxes,),
                device,
            )
            external.append(temperature_value)
        else:
            temperature_value = _floating_scalar(temperature, "temperature")
        if config.saturation_lower is not None:
            if not _is_warp_array_like(saturation):
                raise ValueError(
                    "saturation must be a Warp array when configured."
                )
            saturation_value = _array(
                type("Direct", (), {"saturation": saturation})(),
                "saturation",
                wp.float64,
                (n_boxes, n_species),
                device,
            )
            external.append(saturation_value)
        elif saturation is not None:
            raise ValueError("saturation is not configured.")
    sidecars = _validate_sidecars(
        scratch,
        finalized_demand,
        diagnostics,
        n_boxes,
        n_particles,
        n_species,
        device,
    )
    _no_overlap(particle_arrays + gas_arrays + tuple(external) + sidecars)
    normalized_time_step = _real(time_step, "time_step")
    for value, name, positive in zip(
        (particle_arrays[0], particle_arrays[1], *particle_arrays[3:]),
        (
            "particles.masses",
            "particles.concentration",
            "particles.density",
            "particles.volume",
        ),
        (False, False, True, True),
        strict=True,
    ):
        _scan(value, name, positive=positive)
    _scan_finite_2d(particle_arrays[2], "particles.charge")
    for value, name, positive in zip(
        gas_arrays[:2],
        ("gas.molar_mass", "gas.concentration"),
        (True, False),
        strict=True,
    ):
        _scan(value, name, positive=positive)
    if isinstance(temperature_value, float):
        if (
            not config.temperature_lower
            <= temperature_value
            <= config.temperature_upper
        ):
            raise ValueError("temperature is outside configured bounds.")
    else:
        _scan(temperature_value, "temperature", positive=True)
        _scan_interval(
            temperature_value,
            "temperature",
            config.temperature_lower,
            config.temperature_upper,
        )
    if saturation_value is not None:
        _scan(saturation_value, "saturation")
    if (
        config.precursor_index >= n_species
        or len(config.molecule_counts) != n_species
    ):
        raise ValueError(
            "precursor index and molecule_counts must match species."
        )
    if any(count > np.iinfo(np.int32).max for count in config.molecule_counts):
        raise ValueError("molecule_counts must fit int32.")
    partitioning = gas_arrays[2]
    _scan_binary(partitioning)
    precursor_enabled = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(
        _disabled_precursor,
        dim=n_boxes,
        inputs=[partitioning, config.precursor_index, precursor_enabled],
        device=device,
    )
    if int(precursor_enabled.numpy()[0]):
        raise ValueError("precursor partitioning must be enabled.")
    if n_boxes == 0 or n_particles == 0:
        return _NucleationPreflight(
            particles,
            gas,
            config,
            n_boxes,
            n_particles,
            n_species,
            device,
            temperature_value,
            saturation_value,
            False,
            False,
            None,
        )
    # P1 has no rate work.  These conservative aggregate gates preserve the
    # required all-box zero diagnostics while later phases own mixed-box work.
    reason = None
    if normalized_time_step == 0.0:
        reason = "zero_time"
    elif float(config.coefficient) == 0.0:
        reason = "zero_coefficient"
    elif float(config.survival_factor) == 0.0:
        reason = "zero_survival"
    if reason is not None:
        return _NucleationPreflight(
            particles,
            gas,
            config,
            n_boxes,
            n_particles,
            n_species,
            device,
            temperature_value,
            saturation_value,
            False,
            True,
            reason,
        )
    precursor_status = wp.zeros(2, dtype=wp.int32, device=device)
    wp.launch(
        _precursor_status,
        dim=n_boxes,
        inputs=[
            gas_arrays[1],
            gas_arrays[0],
            config.precursor_index,
            config.precursor_number_concentration_lower,
            config.precursor_number_concentration_upper,
            precursor_status,
        ],
        device=device,
    )
    precursor_values = precursor_status.numpy()
    if int(precursor_values[0]):
        raise ValueError(
            "precursor_number_concentration is outside configured bounds."
        )
    zero_precursor_boxes = int(precursor_values[1])
    low_saturation_boxes = 0
    if saturation_value is not None:
        saturation_status = wp.zeros(2, dtype=wp.int32, device=device)
        wp.launch(
            _saturation_status,
            dim=n_boxes,
            inputs=[
                saturation_value,
                config.precursor_index,
                config.saturation_lower,
                config.saturation_upper,
                saturation_status,
            ],
            device=device,
        )
        saturation_values = saturation_status.numpy()
        if int(saturation_values[0]):
            raise ValueError("saturation is outside configured bounds.")
        low_saturation_boxes = int(saturation_values[1])
    gated_boxes = max(zero_precursor_boxes, low_saturation_boxes)
    if gated_boxes == n_boxes:
        gate_reason = (
            "zero_precursor" if zero_precursor_boxes else "low_saturation"
        )
        return _NucleationPreflight(
            particles,
            gas,
            config,
            n_boxes,
            n_particles,
            n_species,
            device,
            temperature_value,
            saturation_value,
            False,
            True,
            gate_reason,
        )
    return _NucleationPreflight(
        particles,
        gas,
        config,
        n_boxes,
        n_particles,
        n_species,
        device,
        temperature_value,
        saturation_value,
        True,
        bool(gated_boxes),
        None,
    )
