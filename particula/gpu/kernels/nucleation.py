"""Define concrete-only P1 validation and P2 demand planning for GPU nucleation.

This unexported module validates fixed-capacity particle, gas, environment,
and caller-owned sidecar schemas for a GPU nucleation path. P1 is read-only;
private P2 calculates and commits demand sidecars only.
It also performs no hidden host transfer, CPU fallback, fallback work-buffer
allocation, slot activation, or exhaustion handling. Later P2--P7 phases own
particle mutation, public API, and user-facing documentation.

The preflight accepts only same-device, contiguous Warp arrays with fixed
shapes. Frozen records prevent rebinding their fields but do not copy, freeze,
or otherwise transfer their caller-owned arrays.
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

_P2_GATE_ELIGIBLE = 0
_P2_GATE_ZERO_TIME = 1
_P2_GATE_ZERO_COEFFICIENT = 2
_P2_GATE_ZERO_SURVIVAL = 3
_P2_GATE_ZERO_PRECURSOR = 4
_P2_GATE_LOW_SATURATION = 5
_P2_GATE_ZERO_INVENTORY = 6
_P2_GATE_GAS_LIMITED_OFFSET = 7


def _real(value: object, name: str, *, positive: bool = False) -> float:
    """Normalize a finite, non-Boolean real scalar.

    Args:
        value: Scalar to validate.
        name: Value name used in validation errors.
        positive: Require a strictly positive rather than nonnegative value.

    Returns:
        Validated Python floating-point scalar.

    Raises:
        TypeError: If ``value`` is not a supported real scalar.
        ValueError: If ``value`` is nonfinite or outside the required domain.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real scalar.")
    try:
        result = float(value)
    except OverflowError as exc:
        qualifier = "positive" if positive else "finite and nonnegative"
        raise ValueError(f"{name} must be {qualifier}.") from exc
    if (
        not np.isfinite(result)
        or (positive and result <= 0.0)
        or (not positive and result < 0.0)
    ):
        qualifier = "positive" if positive else "finite and nonnegative"
        raise ValueError(f"{name} must be {qualifier}.")
    return result


def _floating_scalar(value: object, name: str) -> float:
    """Normalize a finite positive Python or NumPy floating scalar.

    Args:
        value: Scalar to validate.
        name: Value name used in validation errors.

    Returns:
        Validated positive scalar.

    Raises:
        TypeError: If ``value`` is not a Python or NumPy floating scalar.
        ValueError: If ``value`` is nonfinite or not positive.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (float, np.floating)
    ):
        raise TypeError(f"{name} must be a floating scalar.")
    return _real(value, name, positive=True)


@dataclass(frozen=True)
class NucleationConfig:
    """Store immutable scalar controls for a potential nucleation calculation.

    The activation law is ``J = survival_factor * coefficient * C`` and the
    kinetic law is ``J = survival_factor * coefficient * C²``, where ``J`` is
    formation rate [#/m³/s] and ``C`` is precursor number concentration
    [#/m³]. Therefore, activation ``coefficient`` has units [1/s] and kinetic
    ``coefficient`` has units [m³/s]. This record performs scalar consistency
    checks only; array-dependent species and physical-state validation belongs
    to :func:`_preflight_nucleation`.

    Attributes:
        rate_law: Either ``"activation"`` or ``"kinetic"``.
        coefficient: Nonnegative rate-law coefficient, in [1/s] or [m³/s].
        survival_factor: Nonnegative, dimensionless formation survival factor.
        precursor_index: Zero-based precursor species index.
        molecule_counts: Nonnegative, dimensionless molecule counts per
            species, with at least one positive count.
        formation_diameter: Positive formation diameter [m].
        precursor_number_concentration_lower: Inclusive lower bound for ``C``
            [#/m³].
        precursor_number_concentration_upper: Inclusive upper bound for ``C``
            [#/m³].
        temperature_lower: Inclusive temperature lower bound [K].
        temperature_upper: Inclusive temperature upper bound [K].
        saturation_lower: Optional inclusive, dimensionless saturation lower
            bound. Must be paired with ``saturation_upper``.
        saturation_upper: Optional inclusive, dimensionless saturation upper
            bound. Must be paired with ``saturation_lower``.
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
        """Validate scalar-only configuration consistency.

        Raises:
            TypeError: If a scalar or molecule count has an unsupported type.
            ValueError: If a value is nonfinite, out of range, or inconsistent.
        """
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
        """Validate an ordered scalar interval.

        Args:
            lower: Proposed inclusive lower endpoint.
            upper: Proposed inclusive upper endpoint.
            name: Name used in validation errors.
            positive: Whether both endpoints must be strictly positive.

        Raises:
            TypeError: If either endpoint is not a real scalar.
            ValueError: If an endpoint is invalid or the interval is reversed.
        """
        if _real(lower, f"{name}_lower", positive=positive) > _real(
            upper, f"{name}_upper", positive=positive
        ):
            raise ValueError(f"{name}_lower must not exceed {name}_upper.")


@dataclass(frozen=True)
class NucleationScratchBuffers:
    """Reference caller-owned P2 planning sidecars without taking ownership.

    Attributes:
        precursor_number_concentration: Same-device contiguous ``wp.float64``
            array shaped ``(B,)`` for precursor number concentration [#/m³].
        potential_rate: Same-device contiguous ``wp.float64`` array shaped
            ``(B,)`` for potential formation rate [#/m³/s].
        potential_demand: Same-device contiguous ``wp.float64`` array shaped
            ``(B,)`` for P2 potential source demand [#/m³]. P3 does not write
            this field.
    """

    precursor_number_concentration: Any
    potential_rate: Any
    potential_demand: Any


@dataclass(frozen=True)
class NucleationFinalizedDemandBuffers:
    """Reference caller-owned P2/P3 finalized demand and transfer sidecars.

    Attributes:
        accepted_counts: Same-device contiguous ``wp.int32`` array shaped
            ``(B,)`` for P3 finalized accepted source counts. P2 preserves it.
        accepted_demand: Same-device contiguous ``wp.float64`` array shaped
            ``(B,)`` for P2 inventory-admitted source demand [#/m³].
        precursor_mass_change: Same-device contiguous ``wp.float64`` array
            shaped ``(B, S)`` for P2 planned positive species removal [kg/m³].
    """

    accepted_counts: Any
    accepted_demand: Any
    precursor_mass_change: Any


@dataclass(frozen=True)
class NucleationDiagnosticBuffers:
    """Reference caller-owned P2/P3 diagnostic sidecars.

    Attributes:
        gate_codes: Same-device contiguous ``wp.int32`` array shaped ``(B,)``
            for P2 gate and limiting-species diagnostics.
        selected_slot_indices: Same-device contiguous ``wp.int32`` array
            shaped ``(B, N)`` for P3 selected slots. That phase
            reserves ``-1`` for unused tails; P1 does not inspect, initialize,
            or otherwise alter stale output values.
    """

    gate_codes: Any
    selected_slot_indices: Any


@dataclass(frozen=True)
class _NucleationPreflight:
    """Retain normalized read-only P1 metadata for private phase handoff.

    This private test seam is not a supported API. Its references preserve
    caller identity and never authorize mutation. Zero-valued diagnostic
    fields describe all-box gates without clearing supplied stale sidecars.

    Attributes:
        particles: Validated caller-owned Warp particle container.
        gas: Validated caller-owned Warp gas container.
        config: Validated nucleation configuration.
        n_boxes: Fixed number of boxes ``B``.
        n_particles: Fixed particle capacity ``N`` per box.
        n_species: Fixed species count ``S``.
        device: Shared Warp device for all retained arrays.
        temperature: Validated scalar or same-device ``(B,)`` temperature [K].
        saturation: Validated same-device ``(B, S)`` saturation, if configured.
        has_eligible_boxes: Whether at least one box can proceed past P1.
        has_gated_boxes: Whether at least one valid box is currently gated.
        gate_reason: All-box gate reason, or ``None`` for mixed/no gates.
        potential_rate: All-box gate diagnostic potential rate [#/m³/s].
        potential_demand: All-box gate diagnostic potential demand.
        accepted_count: All-box gate diagnostic accepted count.
        accepted_demand: All-box gate diagnostic accepted demand.
        precursor_mass_change: All-box gate diagnostic precursor change [kg/m³].
    """

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
    normalized_time_step: float
    potential_rate: float = 0.0
    potential_demand: float = 0.0
    accepted_count: int = 0
    accepted_demand: float = 0.0
    precursor_mass_change: float = 0.0


def _field(container: Any, name: str) -> Any:
    """Get a required container field without accepting host substitutes.

    Args:
        container: Object expected to carry a Warp-backed field.
        name: Required field name.

    Returns:
        Field value, before its Warp-array schema is validated.

    Raises:
        ValueError: If ``container`` does not provide ``name``.
    """
    try:
        return getattr(container, name)
    except AttributeError as exc:
        raise ValueError(f"{name} must be a Warp array.") from exc


def _array(
    container: Any, name: str, dtype: Any, shape: tuple[int, ...], device: Any
) -> Any:
    """Validate one same-device contiguous Warp-array schema.

    Args:
        container: Object that owns the required field.
        name: Required field name.
        dtype: Required Warp dtype.
        shape: Required fixed array shape.
        device: Required Warp device.

    Returns:
        The validated array, retaining caller identity.

    Raises:
        ValueError: If the field is missing or has an invalid schema.
    """
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
    """Return byte bounds after rejecting unsafe noncontiguous views.

    Args:
        array: Warp array with a supported overlap-check dtype.

    Returns:
        Half-open byte range, or ``None`` for an empty array.

    Raises:
        ValueError: If the dtype is unsupported or the array is noncontiguous.
    """
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
    """Reject identity and byte-range aliases among future mutable state.

    Args:
        arrays: Validated arrays that later phases could mutate.

    Raises:
        ValueError: If two arrays are identical or overlap in byte storage.
    """
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
    """Count invalid one-dimensional floats without changing input values.

    Args:
        values: Device-resident values to scan.
        positive: Nonzero to require positive rather than nonnegative values.
        invalid: Device-resident one-element invalid-value counter.
    """
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
    """Count invalid two-dimensional floats without changing input values.

    Args:
        values: Device-resident values to scan.
        positive: Nonzero to require positive rather than nonnegative values.
        invalid: Device-resident one-element invalid-value counter.
    """
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
    """Count invalid three-dimensional floats without changing input values.

    Args:
        values: Device-resident values to scan.
        positive: Nonzero to require positive rather than nonnegative values.
        invalid: Device-resident one-element invalid-value counter.
    """
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
    """Count one-dimensional values outside an inclusive interval.

    Args:
        values: Device-resident values to scan.
        lower: Inclusive lower endpoint.
        upper: Inclusive upper endpoint.
        invalid: Device-resident one-element invalid-value counter.
    """
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
    """Count two-dimensional values outside an inclusive interval.

    Args:
        values: Device-resident values to scan.
        lower: Inclusive lower endpoint.
        upper: Inclusive upper endpoint.
        invalid: Device-resident one-element invalid-value counter.
    """
    row, column = wp.tid()
    if values[row, column] < lower or values[row, column] > upper:
        wp.atomic_add(invalid, 0, 1)


@wp.kernel
def _invalid_binary(
    values: wp.array2d(dtype=wp.int32),
    invalid: wp.array(dtype=wp.int32),
) -> None:
    """Count partitioning values other than zero or one.

    Args:
        values: Device-resident partitioning flags to scan.
        invalid: Device-resident one-element invalid-value counter.
    """
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
    """Count boxes whose selected precursor partitioning is disabled.

    Args:
        values: Device-resident binary partitioning flags.
        index: Selected precursor species index.
        invalid: Device-resident one-element invalid-value counter.
    """
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
    gated: wp.array(dtype=wp.int32),
) -> None:
    """Count zero and out-of-interval precursor concentrations by box.

    Args:
        concentration: Gas concentration [kg/m³], shaped ``(B, S)``.
        molar_mass: Gas molar mass [kg/mol], shaped ``(S,)``.
        precursor_index: Selected precursor species index.
        lower: Inclusive number-concentration lower endpoint [#/m³].
        upper: Inclusive number-concentration upper endpoint [#/m³].
        status: Device counters for invalid and zero concentrations.
    """
    box = wp.tid()
    number_concentration = (
        concentration[box, precursor_index]
        * wp.float64(AVOGADRO_NUMBER)
        / molar_mass[precursor_index]
    )
    if number_concentration == 0.0:
        wp.atomic_add(status, 1, 1)
        gated[box] = 1
    elif number_concentration < lower or number_concentration > upper:
        wp.atomic_add(status, 0, 1)


@wp.kernel
def _saturation_status(
    saturation: wp.array2d(dtype=wp.float64),
    precursor_index: wp.int32,
    lower: wp.float64,
    upper: wp.float64,
    status: wp.array(dtype=wp.int32),
    gated: wp.array(dtype=wp.int32),
) -> None:
    """Count saturation lower gates and upper-domain failures by box.

    Args:
        saturation: Dimensionless saturation ratios shaped ``(B, S)``.
        precursor_index: Selected precursor species index.
        lower: Inclusive saturation lower endpoint.
        upper: Inclusive saturation upper endpoint.
        status: Device counters for invalid and lower-gated values.
    """
    box = wp.tid()
    value = saturation[box, precursor_index]
    if value < lower:
        wp.atomic_add(status, 1, 1)
        gated[box] = 1
    elif value > upper:
        wp.atomic_add(status, 0, 1)


@wp.kernel
def _gate_union_count(
    first: wp.array(dtype=wp.int32),
    second: wp.array(dtype=wp.int32),
    count: wp.array(dtype=wp.int32),
) -> None:
    """Count boxes gated by either of two per-box read-only gate masks."""
    box = wp.tid()
    if first[box] != 0 or second[box] != 0:
        wp.atomic_add(count, 0, 1)


def _scan(values: Any, name: str, *, positive: bool = False) -> None:
    """Perform a read-only finite-domain scan with scalar status readback.

    Args:
        values: One-, two-, or three-dimensional ``wp.float64`` array.
        name: Input name used in validation errors.
        positive: Require positive rather than nonnegative values.

    Raises:
        ValueError: If any value is nonfinite or outside the requested domain.
    """
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
    """Count nonfinite two-dimensional values without changing input values.

    Args:
        values: Device-resident values to scan.
        invalid: Device-resident one-element invalid-value counter.
    """
    row, column = wp.tid()
    if not wp.isfinite(values[row, column]):
        wp.atomic_add(invalid, 0, 1)


def _scan_finite_2d(values: Any, name: str) -> None:
    """Perform a read-only scan that permits finite signed values.

    Args:
        values: Two-dimensional ``wp.float64`` array to scan.
        name: Input name used in validation errors.

    Raises:
        ValueError: If any value is nonfinite.
    """
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
    """Perform a read-only inclusive-interval scan with status readback.

    Args:
        values: One- or two-dimensional ``wp.float64`` array to scan.
        name: Input name used in validation errors.
        lower: Inclusive lower endpoint.
        upper: Inclusive upper endpoint.

    Raises:
        ValueError: If any value falls outside the configured interval.
    """
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
    """Validate binary partitioning with a device-resident read-only scan.

    Args:
        values: ``wp.int32`` partitioning flags shaped ``(B, S)``.

    Raises:
        ValueError: If any partitioning value is not zero or one.
    """
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
    """Validate supplied sidecar schemas without reading stale contents.

    Args:
        scratch: Optional caller-owned planning sidecars.
        finalized: Optional caller-owned finalized-demand sidecars.
        diagnostics: Optional caller-owned diagnostic sidecars.
        n_boxes: Fixed box count ``B``.
        n_particles: Fixed particle capacity ``N``.
        n_species: Fixed species count ``S``.
        device: Required shared Warp device.

    Returns:
        Validated sidecar arrays retaining caller identity.

    Raises:
        ValueError: If a supplied record or sidecar schema is invalid.
    """
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
    """Validate the deferred P1 boundary without writing caller-owned arrays.

    This private, concrete-only seam performs ordered schema, ownership,
    physical-domain, and species checks. It accepts caller-owned same-device
    fixed-shape Warp arrays, performs no hidden transfer or CPU fallback, and
    leaves particles, gas, environment, and stale sidecars unchanged on success
    and rejection. A scalar status readback is used solely for validation.

    Args:
        particles: Warp particle container with fixed ``(B, N, S)`` mass data.
        gas: Warp gas container with fixed ``(B, S)`` species data.
        config: Scalar nucleation controls and scientific bounds.
        time_step: Finite nonnegative step duration [s].
        temperature: Positive scalar [K] or same-device ``wp.float64 (B,)``.
        saturation: Configured same-device dimensionless ``wp.float64 (B, S)``.
        environment: Optional owner of temperature and configured saturation.
        scratch: Optional caller-owned planning sidecars.
        finalized_demand: Optional caller-owned finalized-demand sidecars.
        diagnostics: Optional caller-owned diagnostic sidecars.

    Returns:
        Private normalized metadata, including eligibility and all-box gate
        diagnostics. Returned zero diagnostics never clear supplied sidecars.

    Raises:
        TypeError: If a direct scalar has an unsupported type.
        ValueError: If schemas, ownership, physical values, or species settings
            violate the P1 contract.
    """
    if not isinstance(config, NucleationConfig):
        raise ValueError("config must be NucleationConfig.")
    try:
        config.__post_init__()
    except (TypeError, ValueError) as exc:
        raise ValueError("config scalar values are invalid.") from exc
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
    if (
        config.precursor_index >= n_species
        or len(config.molecule_counts) != n_species
    ):
        raise ValueError(
            "precursor index and molecule_counts must match species."
        )
    if any(count > np.iinfo(np.int32).max for count in config.molecule_counts):
        raise ValueError("molecule_counts must fit int32.")
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
    if not isinstance(temperature_value, float):
        _scan(temperature_value, "temperature", positive=True)
    if float(config.coefficient) == 0.0:
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
            "zero_coefficient",
            normalized_time_step,
        )
    if float(config.survival_factor) == 0.0:
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
            "zero_survival",
            normalized_time_step,
        )
    if isinstance(temperature_value, float):
        if (
            not config.temperature_lower
            <= temperature_value
            <= config.temperature_upper
        ):
            raise ValueError("temperature is outside configured bounds.")
    else:
        _scan_interval(
            temperature_value,
            "temperature",
            config.temperature_lower,
            config.temperature_upper,
        )
    if saturation_value is not None:
        _scan(saturation_value, "saturation")
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
            normalized_time_step,
        )
    # P1 has no rate work.  These conservative aggregate gates preserve the
    # required all-box zero diagnostics while later phases own mixed-box work.
    reason = None
    if normalized_time_step == 0.0:
        reason = "zero_time"
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
            normalized_time_step,
        )
    precursor_status = wp.zeros(2, dtype=wp.int32, device=device)
    precursor_gates = wp.zeros(n_boxes, dtype=wp.int32, device=device)
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
            precursor_gates,
        ],
        device=device,
    )
    precursor_values = precursor_status.numpy()
    if int(precursor_values[0]):
        raise ValueError(
            "precursor_number_concentration is outside configured bounds."
        )
    zero_precursor_boxes = int(precursor_values[1])
    saturation_gates = wp.zeros(n_boxes, dtype=wp.int32, device=device)
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
                saturation_gates,
            ],
            device=device,
        )
        saturation_values = saturation_status.numpy()
        if int(saturation_values[0]):
            raise ValueError("saturation is outside configured bounds.")
    gate_count = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(
        _gate_union_count,
        dim=n_boxes,
        inputs=[precursor_gates, saturation_gates, gate_count],
        device=device,
    )
    gated_boxes = int(gate_count.numpy()[0])
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
            normalized_time_step,
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
        normalized_time_step,
    )


@wp.kernel
def _plan_demand_work(  # noqa: C901
    concentration: wp.array2d(dtype=wp.float64),
    molar_mass: wp.array(dtype=wp.float64),
    saturation: wp.array2d(dtype=wp.float64),
    molecule_counts: wp.array(dtype=wp.int32),
    species_count: wp.int32,
    precursor_index: wp.int32,
    has_saturation: wp.int32,
    saturation_lower: wp.float64,
    coefficient: wp.float64,
    survival_factor: wp.float64,
    time_step: wp.float64,
    kinetic: wp.int32,
    number_concentration: wp.array(dtype=wp.float64),
    rate: wp.array(dtype=wp.float64),
    potential: wp.array(dtype=wp.float64),
    admitted: wp.array(dtype=wp.float64),
    removal: wp.array2d(dtype=wp.float64),
    gate_codes: wp.array(dtype=wp.int32),
    invalid: wp.array(dtype=wp.int32),
) -> None:
    """Plan one box of P2 demand in private device-resident work storage."""
    box = wp.tid()
    precursor = (
        concentration[box, precursor_index]
        * wp.float64(AVOGADRO_NUMBER)
        / molar_mass[precursor_index]
    )
    number_concentration[box] = precursor
    for species in range(species_count):
        removal[box, species] = wp.float64(0.0)

    code = wp.int32(_P2_GATE_ELIGIBLE)
    demand = wp.float64(0.0)
    formation_rate = wp.float64(0.0)
    if time_step == 0.0:
        code = wp.int32(_P2_GATE_ZERO_TIME)
    elif coefficient == 0.0:
        code = wp.int32(_P2_GATE_ZERO_COEFFICIENT)
    elif survival_factor == 0.0:
        code = wp.int32(_P2_GATE_ZERO_SURVIVAL)
    elif precursor == 0.0:
        code = wp.int32(_P2_GATE_ZERO_PRECURSOR)
    elif (
        has_saturation != 0
        and saturation[box, precursor_index] < saturation_lower
    ):
        code = wp.int32(_P2_GATE_LOW_SATURATION)
    else:
        formation_rate = survival_factor * coefficient * precursor
        if kinetic != 0:
            formation_rate = formation_rate * precursor
        demand = formation_rate * time_step
        minimum_ratio = wp.float64(0.0)
        limiting_species = wp.int32(-1)
        has_participant = wp.int32(0)
        for species in range(species_count):
            if molecule_counts[species] > 0:
                event_mass = (
                    wp.float64(molecule_counts[species])
                    * molar_mass[species]
                    / wp.float64(AVOGADRO_NUMBER)
                )
                if not wp.isfinite(event_mass) or event_mass <= 0.0:
                    wp.atomic_add(invalid, 0, 1)
                ratio = concentration[box, species] / event_mass
                if has_participant == 0 or ratio < minimum_ratio:
                    minimum_ratio = ratio
                    limiting_species = wp.int32(species)
                has_participant = wp.int32(1)
        if minimum_ratio == 0.0:
            code = wp.int32(_P2_GATE_ZERO_INVENTORY)
            demand = wp.float64(0.0)
        elif minimum_ratio < demand:
            demand = minimum_ratio
            code = wp.int32(_P2_GATE_GAS_LIMITED_OFFSET) + limiting_species
        for _correction in range(4):
            safe = wp.int32(1)
            for species in range(species_count):
                if molecule_counts[species] > 0:
                    event_mass = (
                        wp.float64(molecule_counts[species])
                        * molar_mass[species]
                        / wp.float64(AVOGADRO_NUMBER)
                    )
                    if demand * event_mass > concentration[box, species]:
                        safe = wp.int32(0)
            if safe == 0:
                demand = demand * wp.float64(0.9999999999999999)
        for species in range(species_count):
            if molecule_counts[species] > 0:
                event_mass = (
                    wp.float64(molecule_counts[species])
                    * molar_mass[species]
                    / wp.float64(AVOGADRO_NUMBER)
                )
                removal[box, species] = demand * event_mass
                if (
                    not wp.isfinite(removal[box, species])
                    or removal[box, species] < 0.0
                ):
                    wp.atomic_add(invalid, 0, 1)
                elif removal[box, species] > concentration[box, species]:
                    wp.atomic_add(invalid, 1, 1)
    rate[box] = formation_rate
    potential[box] = formation_rate * time_step
    admitted[box] = demand
    gate_codes[box] = code
    if (
        not wp.isfinite(precursor)
        or precursor < 0.0
        or not wp.isfinite(formation_rate)
        or formation_rate < 0.0
        or not wp.isfinite(potential[box])
        or potential[box] < 0.0
        or not wp.isfinite(demand)
        or demand < 0.0
    ):
        wp.atomic_add(invalid, 0, 1)


@wp.kernel
def _commit_demand_plan(
    number_concentration: wp.array(dtype=wp.float64),
    rate: wp.array(dtype=wp.float64),
    potential: wp.array(dtype=wp.float64),
    admitted: wp.array(dtype=wp.float64),
    removal: wp.array2d(dtype=wp.float64),
    gate_codes: wp.array(dtype=wp.int32),
    species_count: wp.int32,
    scratch_number_concentration: wp.array(dtype=wp.float64),
    scratch_rate: wp.array(dtype=wp.float64),
    scratch_demand: wp.array(dtype=wp.float64),
    finalized_demand: wp.array(dtype=wp.float64),
    finalized_removal: wp.array2d(dtype=wp.float64),
    diagnostics: wp.array(dtype=wp.int32),
) -> None:
    """Commit exactly the P2-owned sidecar fields for one box."""
    box = wp.tid()
    scratch_number_concentration[box] = number_concentration[box]
    scratch_rate[box] = rate[box]
    scratch_demand[box] = potential[box]
    finalized_demand[box] = admitted[box]
    diagnostics[box] = gate_codes[box]
    for species in range(species_count):
        finalized_removal[box, species] = removal[box, species]


def _plan_nucleation_demand(
    particles: Any,
    gas: Any,
    config: NucleationConfig,
    time_step: Any,
    *,
    scratch: NucleationScratchBuffers,
    finalized_demand: NucleationFinalizedDemandBuffers,
    diagnostics: NucleationDiagnosticBuffers,
    temperature: Any | None = None,
    saturation: Any | None = None,
    environment: Any | None = None,
) -> None:
    """Privately plan inventory-safe nucleation demand without state mutation.

    P2 writes only its defined planning/finalized-demand/diagnostic sidecars.
    Particle state, gas inventory, P3 counts, and P3 slot-index diagnostics are
    never read for planning ownership and are never changed.
    """
    if not isinstance(scratch, NucleationScratchBuffers):
        raise ValueError("scratch must be NucleationScratchBuffers.")
    if not isinstance(finalized_demand, NucleationFinalizedDemandBuffers):
        raise ValueError(
            "finalized_demand must be NucleationFinalizedDemandBuffers."
        )
    if not isinstance(diagnostics, NucleationDiagnosticBuffers):
        raise ValueError("diagnostics must be NucleationDiagnosticBuffers.")
    preflight = _preflight_nucleation(
        particles,
        gas,
        config,
        time_step,
        temperature=temperature,
        saturation=saturation,
        environment=environment,
        scratch=scratch,
        finalized_demand=finalized_demand,
        diagnostics=diagnostics,
    )
    if preflight.n_boxes == 0:
        return
    device = preflight.device
    boxes, species = preflight.n_boxes, preflight.n_species
    number_concentration = wp.zeros(boxes, dtype=wp.float64, device=device)
    rate = wp.zeros(boxes, dtype=wp.float64, device=device)
    potential = wp.zeros(boxes, dtype=wp.float64, device=device)
    admitted = wp.zeros(boxes, dtype=wp.float64, device=device)
    removal = wp.zeros((boxes, species), dtype=wp.float64, device=device)
    gate_codes = wp.zeros(boxes, dtype=wp.int32, device=device)
    invalid = wp.zeros(2, dtype=wp.int32, device=device)
    saturation_work = preflight.saturation
    if saturation_work is None:
        saturation_work = wp.zeros(
            (boxes, species), dtype=wp.float64, device=device
        )
    molecule_counts: Any = wp.array(
        np.asarray(config.molecule_counts, dtype=np.int32),
        dtype=wp.int32,
        device=device,
    )
    wp.launch(
        _plan_demand_work,
        dim=boxes,
        inputs=[
            gas.concentration,
            gas.molar_mass,
            saturation_work,
            molecule_counts,
            species,
            config.precursor_index,
            int(preflight.saturation is not None),
            (
                config.saturation_lower
                if config.saturation_lower is not None
                else 0.0
            ),
            config.coefficient,
            config.survival_factor,
            preflight.normalized_time_step,
            int(config.rate_law == "kinetic"),
            number_concentration,
            rate,
            potential,
            admitted,
            removal,
            gate_codes,
            invalid,
        ],
        device=device,
    )
    status = invalid.numpy()
    if int(status[0]):
        raise ValueError(
            "Derived nucleation demand must be finite and nonnegative."
        )
    if int(status[1]):
        raise ValueError("Nucleation demand cannot be made inventory-safe.")
    wp.launch(
        _commit_demand_plan,
        dim=boxes,
        inputs=[
            number_concentration,
            rate,
            potential,
            admitted,
            removal,
            gate_codes,
            species,
            scratch.precursor_number_concentration,
            scratch.potential_rate,
            scratch.potential_demand,
            finalized_demand.accepted_demand,
            finalized_demand.precursor_mass_change,
            diagnostics.gate_codes,
        ],
        device=device,
    )
