"""Define direct-Warp GPU nucleation planning seams and one public step.

This unexported module validates fixed-capacity particle, gas, environment,
and caller-owned sidecar schemas for a direct-Warp nucleation path. P1 is
read-only. Private P2 calculates source demand [#/m³], admits one shared
inventory-safe demand per box, and commits only its designated sidecars. P2
does not select or activate slots, resolve exhaustion, resize storage, mutate
particle fields or ``gas.concentration``, transfer data to the host, or use a
CPU physics fallback. Private P3 converts admitted demand times particle volume
to representable ``int32`` provisional event counts, reuses E6-F5 slot
diagnostics, and records only the deterministic free-slot prefix that is
currently selectable. It retains counts beyond free capacity and performs no
activation or particle/gas transaction. Private P4 consumes immutable P2/P3
handoffs, applies resampling before optional representative-volume scaling, and
writes only its caller-owned finalized diagnostics. The public
``nucleation_step_gpu`` then validates P4's handoff and performs one fused P5
particle/gas commit.

The preflight accepts only same-device, contiguous Warp arrays with fixed
shapes. Frozen records prevent rebinding their fields but do not copy, freeze,
or otherwise transfer their caller-owned arrays. Only ``nucleation_step_gpu``
is lazily exported through ``particula.gpu.kernels``; all records and helpers
remain concrete-module-only.
"""

# mypy: disable-error-code="valid-type, misc, operator"

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
from particula.gpu.kernels.exhaustion import (
    ResamplingBuffers,
    representative_volume_scaling_step_gpu,
    resampling_step_gpu,
)
from particula.gpu.kernels.slot_management import (
    _classify_slots,
    _write_diagnostics,
    _write_empty_diagnostics,
)
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
        precursor_number_concentration: P2-owned, same-device contiguous
            ``wp.float64`` array shaped ``(B,)`` for selected precursor number
            concentration [#/m³].
        potential_rate: Same-device contiguous ``wp.float64`` array shaped
            ``(B,)`` for P2 potential formation rate [#/m³/s].
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
        accepted_counts: P3-owned, same-device contiguous ``wp.int32`` array
            shaped ``(B,)`` for full provisional event counts. Counts retain
            demand beyond current free capacity; P2 leaves this field
            byte-for-byte unchanged.
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
        gate_codes: P2-owned, same-device contiguous ``wp.int32`` array shaped
            ``(B,)`` for gate diagnostics. Zero denotes eligible; codes 1--6
            denote the documented zero/low gates; and ``7 + s`` records an
            inventory-limited box whose limiting species index is ``s``.
        selected_slot_indices: Same-device contiguous ``wp.int32`` array
            shaped ``(B, N)`` for P3's deterministic selectable free-slot
            prefix, limited by current free capacity rather than provisional
            demand. P2 leaves it byte-for-byte unchanged. P3 reserves ``-1``
            for unused tails; P1 does not inspect, initialize, or otherwise
            alter stale output values.
        free_slot_indices: P3/E6-F5-owned, same-device contiguous ``wp.int32``
            array shaped ``(B, N)`` for ascending free slot indices with ``-1``
            tails.
        active_slot_counts: P3/E6-F5-owned, same-device contiguous ``wp.int32``
            array shaped ``(B,)`` for active slot counts.
        free_slot_counts: P3/E6-F5-owned, same-device contiguous ``wp.int32``
            array shaped ``(B,)`` for free slot counts.
    """

    gate_codes: Any
    selected_slot_indices: Any
    free_slot_indices: Any
    active_slot_counts: Any
    free_slot_counts: Any


@dataclass(frozen=True)
class NucleationExhaustionControls:
    """Store exact-boolean private P4 capacity-policy controls.

    ``resampling`` is considered first for a row only when it can release its
    entire P3 deficit. ``representative_volume_scaling`` is then considered for
    any remaining exhausted row. Exact Python booleans deliberately prevent
    integer or NumPy scalar substitutes at this concrete-only seam.

    Attributes:
        resampling: Enable fully viable resampling before scaling fallback.
        representative_volume_scaling: Enable representative-volume scaling for
            exhausted rows not selected for resampling.
    """

    resampling: bool
    representative_volume_scaling: bool

    def __post_init__(self) -> None:
        """Reject non-exact Boolean policy controls."""
        if type(self.resampling) is not bool:
            raise TypeError("resampling must be an exact Python bool.")
        if type(self.representative_volume_scaling) is not bool:
            raise TypeError(
                "representative_volume_scaling must be an exact Python bool."
            )


@dataclass(frozen=True)
class NucleationExhaustionBuffers:
    """Reference caller-owned P4 policy workspace and finalized diagnostics.

    P2 admitted demand and P3 provisional counts are historical immutable
    handoffs. P4 copies admitted demand into ``demand_workspace`` and derives
    ``final_counts`` from its post-policy demand and current box volume.
    Expected P4 all-box preflight rejections occur before workspace writes or
    either primitive and preserve every supplied sidecar. Once a selected
    primitive begins, its own planning and commit failure contract applies;
    P4 does not provide cross-primitive rollback. This concrete-only record is
    not exported through ``particula.gpu.kernels`` or ``particula.gpu``.

    Attributes:
        resampling_buffers: Caller-owned E6-F6 resampling scratch and
            diagnostic storage.
        demand_workspace: Mutable ``wp.float64`` ``(B,)`` copy of P2 admitted
            demand [#/m³].
        final_demand: ``wp.float64`` ``(B,)`` P4-finalized demand [#/m³].
        requested_scale: ``wp.float64`` ``(B,)`` requested volume scales.
        minimum_scale: ``wp.float64`` ``(B,)`` lower bounds for volume scales.
        minimum_volume: ``wp.float64`` ``(B,)`` minimum allowed box volumes
            [m³].
        resolved_scale: ``wp.float64`` ``(B,)`` scale selected by P4; unscaled
            rows receive ``1.0``.
        resampling_releasable_counts: ``wp.int32`` ``(B,)`` maximum counts
            resampling may release.
        required_release_counts: ``wp.int32`` ``(B,)`` P4 output deficits
            before policy application.
        scaling_required: ``wp.int32`` ``(B,)`` P4 output flags for rows
            selected for scaling.
        final_counts: ``wp.int32`` ``(B,)`` P4-finalized event counts.
        final_selected_slot_indices: ``wp.int32`` ``(B, N)`` P4-finalized
            ascending free-slot prefixes with ``-1`` tails.
    """

    resampling_buffers: ResamplingBuffers
    demand_workspace: Any
    final_demand: Any
    requested_scale: Any
    minimum_scale: Any
    minimum_volume: Any
    resolved_scale: Any
    resampling_releasable_counts: Any
    required_release_counts: Any
    scaling_required: Any
    final_counts: Any
    final_selected_slot_indices: Any


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
    finalized_demand: NucleationFinalizedDemandBuffers | None = None
    diagnostics: NucleationDiagnosticBuffers | None = None
    p3_sidecars: tuple[Any, ...] = ()
    protected_sidecars: tuple[Any, ...] = ()
    particle_arrays: tuple[Any, ...] = ()
    gas_arrays: tuple[Any, ...] = ()
    protected_inputs: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        """Capture P1 storage identities for later mutable P4 validation."""
        if not self.particle_arrays:
            object.__setattr__(
                self,
                "particle_arrays",
                (
                    self.particles.masses,
                    self.particles.concentration,
                    self.particles.charge,
                    self.particles.density,
                    self.particles.volume,
                ),
            )
        if not self.protected_inputs:
            external = tuple(
                value
                for value in (self.temperature, self.saturation)
                if _is_warp_array_like(value)
            )
            object.__setattr__(
                self,
                "protected_inputs",
                self.particle_arrays
                + (
                    self.gas.molar_mass,
                    self.gas.concentration,
                    self.gas.partitioning,
                )
                + external
                + self.p3_sidecars
                + tuple(
                    value
                    for value in self.protected_sidecars
                    if not any(
                        value is p3_sidecar for p3_sidecar in self.p3_sidecars
                    )
                ),
            )
        if not self.gas_arrays:
            object.__setattr__(
                self,
                "gas_arrays",
                (
                    self.gas.molar_mass,
                    self.gas.concentration,
                    self.gas.partitioning,
                ),
            )


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
                _array(
                    diagnostics,
                    "free_slot_indices",
                    wp.int32,
                    (n_boxes, n_particles),
                    device,
                ),
                _array(
                    diagnostics,
                    "active_slot_counts",
                    wp.int32,
                    (n_boxes,),
                    device,
                ),
                _array(
                    diagnostics,
                    "free_slot_counts",
                    wp.int32,
                    (n_boxes,),
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
    fixed-shape Warp arrays, performs no caller-state transfer or CPU physics
    fallback, and leaves particles, gas, environment, and stale sidecars
    unchanged. A private scalar status readback is used solely for validation.

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
    # P3 receives only finalized-demand and diagnostic records. Scratch remains
    # protected through ``sidecars`` but cannot be identity-validated by P3.
    p3_sidecars: tuple[Any, ...] = ()
    if finalized_demand is not None and diagnostics is not None:
        p3_sidecars += (
            finalized_demand.accepted_counts,
            finalized_demand.accepted_demand,
            finalized_demand.precursor_mass_change,
            diagnostics.gate_codes,
            diagnostics.selected_slot_indices,
            diagnostics.free_slot_indices,
            diagnostics.active_slot_counts,
            diagnostics.free_slot_counts,
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
            finalized_demand=finalized_demand,
            diagnostics=diagnostics,
            p3_sidecars=p3_sidecars,
            protected_sidecars=sidecars,
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
            finalized_demand=finalized_demand,
            diagnostics=diagnostics,
            p3_sidecars=p3_sidecars,
            protected_sidecars=sidecars,
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
    if n_boxes == 0:
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
            finalized_demand=finalized_demand,
            diagnostics=diagnostics,
            p3_sidecars=p3_sidecars,
            protected_sidecars=sidecars,
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
            finalized_demand=finalized_demand,
            diagnostics=diagnostics,
            p3_sidecars=p3_sidecars,
            protected_sidecars=sidecars,
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
    if n_particles == 0:
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
            finalized_demand=finalized_demand,
            diagnostics=diagnostics,
            p3_sidecars=p3_sidecars,
            protected_sidecars=sidecars,
        )
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
            finalized_demand=finalized_demand,
            diagnostics=diagnostics,
            p3_sidecars=p3_sidecars,
            protected_sidecars=sidecars,
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
        finalized_demand=finalized_demand,
        diagnostics=diagnostics,
        p3_sidecars=p3_sidecars,
        protected_sidecars=sidecars,
        protected_inputs=particle_arrays
        + gas_arrays
        + tuple(external)
        + sidecars,
    )


@wp.func
def _float64_predecessor(value: wp.float64) -> wp.float64:
    """Return the exact finite float64 predecessor of a positive value.

    Warp 1.11 exposes neither ``bitcast`` nor ``nextafter``/``frexp`` to
    device code. Powers of two and multiplication/division by two are exact in
    binary64, so determine the binade with supported scalar arithmetic and
    subtract its exact ULP. Subnormals retain the minimum binary64 spacing.

    Args:
        value: Finite positive binary64 value.

    Returns:
        Largest representable binary64 value strictly less than ``value``.
    """
    minimum_normal = wp.float64(2.2250738585072014e-308)
    minimum_subnormal = wp.float64(4.9406564584124654e-324)
    if value < minimum_normal:
        return value - minimum_subnormal

    binade = minimum_normal
    for _exponent in range(2046):
        doubled = binade * wp.float64(2.0)
        if value >= doubled:
            binade = doubled
    ulp = binade
    for _fraction_bit in range(52):
        ulp = ulp * wp.float64(0.5)
    if value == binade and binade > minimum_normal:
        ulp = ulp * wp.float64(0.5)
    return value - ulp


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
    """Plan one box of P2 demand in private device-resident work storage.

    Calculates ``C`` [#/m³], ``J`` [#/m³/s], and ``E_pot = J × dt`` [#/m³],
    then limits one common admitted demand by participating precursor
    inventories. The result preserves per-event precursor composition; planned
    removal is ``E_admit × m_event`` [kg/m³]. This kernel never writes
    caller-owned sidecars, particles, or gas inventory.

    Args:
        concentration: Gas concentration [kg/m³], shaped ``(B, S)``.
        molar_mass: Species molar masses [kg/mol], shaped ``(S,)``.
        saturation: Dimensionless saturation ratios shaped ``(B, S)``.
        molecule_counts: Dimensionless molecules per formed event, shaped
            ``(S,)``.
        species_count: Fixed number of species ``S``.
        precursor_index: Selected precursor species index.
        has_saturation: Nonzero when saturation gating is configured.
        saturation_lower: Inclusive saturation gate threshold.
        coefficient: Nonnegative activation [1/s] or kinetic [m³/s]
            coefficient.
        survival_factor: Nonnegative dimensionless survival factor.
        time_step: Validated nonnegative step duration [s].
        kinetic: Nonzero when the kinetic rate law is selected.
        number_concentration: Private output for selected precursor ``C``
            [#/m³], shaped ``(B,)``.
        rate: Private output for gated formation rate ``J`` [#/m³/s], shaped
            ``(B,)``.
        potential: Private output for potential demand ``E_pot`` [#/m³], shaped
            ``(B,)``.
        admitted: Private output for inventory-admitted demand [#/m³], shaped
            ``(B,)``.
        removal: Private output for planned species removal [kg/m³], shaped
            ``(B, S)``.
        gate_codes: Private output for P2 gate and limiter codes, shaped
            ``(B,)``.
        invalid: Private two-lane counters for derived-domain and
            inventory-safety failures.
    """
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
        # Warp has no device-side nextafter primitive. Use the exact directed
        # float64 predecessor after each unsafe inventory check.
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
                demand = _float64_predecessor(demand)
                code = wp.int32(_P2_GATE_GAS_LIMITED_OFFSET) + limiting_species
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
    """Commit exactly the P2-owned sidecar fields for one box.

    Args:
        number_concentration: Validated private precursor concentration [#/m³].
        rate: Validated private formation rate [#/m³/s].
        potential: Validated private potential demand [#/m³].
        admitted: Validated private inventory-admitted demand [#/m³].
        removal: Validated private planned species removal [kg/m³].
        gate_codes: Validated private P2 gate and limiter codes.
        species_count: Fixed number of species ``S``.
        scratch_number_concentration: P2-owned precursor-concentration sidecar.
        scratch_rate: P2-owned potential-rate sidecar.
        scratch_demand: P2-owned potential-demand sidecar.
        finalized_demand: P2-owned admitted-demand sidecar.
        finalized_removal: P2-owned planned-removal sidecar.
        diagnostics: P2-owned gate-code sidecar.
    """
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

    This concrete-only direct-Warp P2 seam calculates selected precursor number
    concentration ``C`` [#/m³], formation rate ``J`` [#/m³/s], and potential
    demand ``E_pot = J × dt`` [#/m³]. It admits one common demand per box from
    participating gas inventories, so every planned removal is
    ``E_admit × m_event`` [kg/m³]. Survival is already included in ``J`` and is
    not applied again. It is intentionally not exported through
    ``particula.gpu.kernels`` or ``particula.gpu``.

    Private scalar status readback is permitted for validation; no caller-state
    transfer or CPU physics fallback occurs. After read-only P1 and
    derived-state validation, P2 commits only
    ``scratch.precursor_number_concentration``, ``scratch.potential_rate``,
    ``scratch.potential_demand``, ``finalized_demand.accepted_demand``,
    ``finalized_demand.precursor_mass_change``, and ``diagnostics.gate_codes``.
    It reads gas state for planning but never mutates particle state,
    ``gas.concentration``, P3 ``accepted_counts``, or P3 selected-slot indices.
    It neither selects/activates slots nor resolves exhaustion; P3 only stages
    metadata for a later capacity-policy and activation phase.

    Args:
        particles: Caller-owned Warp particle container used only for P1 schema
            and physical-state validation.
        gas: Caller-owned Warp gas container; its concentration [kg/m³] and
            molar mass [kg/mol] determine P2 rates and inventory admission.
        config: Immutable rate-law controls, event composition, and bounds.
        time_step: Finite nonnegative source-planning interval [s].
        scratch: Exact caller-owned P2 planning-sidecar record.
        finalized_demand: Exact caller-owned P2/P3 demand-sidecar record; P2
            writes only admitted demand and planned precursor removal.
        diagnostics: Exact caller-owned P2/P3 diagnostic-sidecar record; P2
            writes only gate codes.
        temperature: Positive scalar [K] or same-device ``wp.float64`` array
            shaped ``(B,)`` for P1 validation.
        saturation: Configured same-device dimensionless ``wp.float64`` array
            shaped ``(B, S)`` for P1 validation and P2 gating.
        environment: Optional owner of temperature and configured saturation.

    Raises:
        TypeError: If a direct scalar has an unsupported type.
        ValueError: If P1 schemas, ownership, physical inputs, sidecars, or
            derived P2 demand are invalid, or inventory cannot be made safe.
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
    _plan_nucleation_demand_from_preflight(
        preflight, scratch, finalized_demand, diagnostics
    )


def _plan_nucleation_demand_from_preflight(
    preflight: _NucleationPreflight,
    scratch: NucleationScratchBuffers,
    finalized_demand: NucleationFinalizedDemandBuffers,
    diagnostics: NucleationDiagnosticBuffers,
) -> None:
    """Run P2 using an already validated P1 handoff.

    This internal helper keeps the standalone P2 seam intact while allowing the
    public P5 boundary to perform P1 exactly once.
    """
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
        np.asarray(preflight.config.molecule_counts, dtype=np.int32),
        dtype=wp.int32,
        device=device,
    )
    wp.launch(
        _plan_demand_work,
        dim=boxes,
        inputs=[
            preflight.gas.concentration,
            preflight.gas.molar_mass,
            saturation_work,
            molecule_counts,
            species,
            preflight.config.precursor_index,
            int(preflight.saturation is not None),
            (
                preflight.config.saturation_lower
                if preflight.config.saturation_lower is not None
                else 0.0
            ),
            preflight.config.coefficient,
            preflight.config.survival_factor,
            preflight.normalized_time_step,
            int(preflight.config.rate_law == "kinetic"),
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


@wp.kernel
def _convert_admitted_demand_to_counts(
    accepted_demand: wp.array(dtype=wp.float64),
    volume: wp.array(dtype=wp.float64),
    maximum_count: wp.float64,
    counts: wp.array(dtype=wp.int32),
    invalid: wp.array(dtype=wp.int32),
    invalid_index: wp.int32,
) -> None:
    """Convert per-volume admitted demand to integral provisional counts.

    This private P3 kernel writes only private workspace. It rejects derived
    counts that are nonfinite, negative, nonintegral, or outside the inclusive
    ``int32`` range before any caller-owned P3 or E6-F5 sidecar is written.

    Args:
        accepted_demand: P2-admitted demand [#/m³], shaped ``(B,)``.
        volume: Validated particle-box volume [m³], shaped ``(B,)``.
        maximum_count: Largest representable provisional ``int32`` count.
        counts: Private output for provisional counts, shaped ``(B,)``.
        invalid: Private status counters shared with slot classification.
        invalid_index: Status lane reserved for invalid conversions.
    """
    box = wp.tid()
    events = accepted_demand[box] * volume[box]
    if (
        not wp.isfinite(accepted_demand[box])
        or accepted_demand[box] < 0.0
        or not wp.isfinite(volume[box])
        or volume[box] <= 0.0
        or not wp.isfinite(events)
        or events < 0.0
        or events != wp.floor(events)
        or events > maximum_count
    ):
        wp.atomic_add(invalid, invalid_index, 1)
    else:
        counts[box] = wp.int32(events)


@wp.kernel
def _commit_staged_nucleation_slots(
    counts: wp.array(dtype=wp.int32),
    free_slot_indices: wp.array2d(dtype=wp.int32),
    free_slot_counts: wp.array(dtype=wp.int32),
    accepted_counts: wp.array(dtype=wp.int32),
    selected_slot_indices: wp.array2d(dtype=wp.int32),
    n_particles: int,
) -> None:
    """Commit full P3 counts and the selectable E6-F5 free-slot prefix.

    Full provisional counts are retained even when they exceed free capacity.
    Every selected-index lane outside the deterministic selectable prefix is
    overwritten with ``-1``.

    Args:
        counts: Validated private provisional event counts, shaped ``(B,)``.
        free_slot_indices: E6-F5 ascending free-slot indices, shaped ``(B, N)``.
        free_slot_counts: E6-F5 free-slot counts, shaped ``(B,)``.
        accepted_counts: P3 output for full provisional counts, shaped ``(B,)``.
        selected_slot_indices: P3 output for selectable free-slot indices,
            shaped ``(B, N)``.
        n_particles: Fixed particle capacity ``N`` per box.
    """
    box = wp.tid()
    count = counts[box]
    accepted_counts[box] = count
    selectable = wp.min(count, free_slot_counts[box])
    for rank in range(n_particles):
        if rank < selectable:
            selected_slot_indices[box, rank] = free_slot_indices[box, rank]
        else:
            selected_slot_indices[box, rank] = -1


def _validate_staged_nucleation_sidecars(
    preflight: _NucleationPreflight,
    finalized_demand: NucleationFinalizedDemandBuffers,
    diagnostics: NucleationDiagnosticBuffers,
) -> None:
    """Validate exact P3 handoff records before staging writes."""
    if preflight.finalized_demand is not finalized_demand:
        raise ValueError(
            "finalized_demand must be the preflight-validated record."
        )
    if preflight.diagnostics is not diagnostics:
        raise ValueError("diagnostics must be the preflight-validated record.")
    supplied_sidecars = (
        finalized_demand.accepted_counts,
        finalized_demand.accepted_demand,
        finalized_demand.precursor_mass_change,
        diagnostics.gate_codes,
        diagnostics.selected_slot_indices,
        diagnostics.free_slot_indices,
        diagnostics.active_slot_counts,
        diagnostics.free_slot_counts,
    )
    if len(preflight.p3_sidecars) != len(supplied_sidecars) or any(
        expected is not supplied
        for expected, supplied in zip(
            preflight.p3_sidecars, supplied_sidecars, strict=True
        )
    ):
        raise ValueError("P3 sidecars must be the preflight-validated storage.")


def _convert_staged_nucleation_counts(
    accepted_demand: Any,
    volume: Any,
    device: Any,
) -> tuple[Any, Any]:
    """Convert admitted demand to private counts and a shared invalid flag."""
    counts = wp.zeros(accepted_demand.shape[0], dtype=wp.int32, device=device)
    invalid = wp.zeros(2, dtype=wp.int32, device=device)
    wp.launch(
        _convert_admitted_demand_to_counts,
        dim=accepted_demand.shape[0],
        inputs=[
            accepted_demand,
            volume,
            float(np.iinfo(np.int32).max),
            counts,
            invalid,
            1,
        ],
        device=device,
    )
    return counts, invalid


def _write_staged_nucleation_diagnostics(
    preflight: _NucleationPreflight,
    categories: Any,
    diagnostics: NucleationDiagnosticBuffers,
) -> None:
    """Write E6-F5 classification outputs to the supplied diagnostics."""
    if preflight.n_particles:
        wp.launch(
            _write_diagnostics,
            dim=preflight.n_boxes,
            inputs=[
                categories,
                diagnostics.free_slot_indices,
                diagnostics.active_slot_counts,
                diagnostics.free_slot_counts,
            ],
            device=preflight.device,
        )
    else:
        wp.launch(
            _write_empty_diagnostics,
            dim=preflight.n_boxes,
            inputs=[
                diagnostics.active_slot_counts,
                diagnostics.free_slot_counts,
            ],
            device=preflight.device,
        )


def _stage_nucleation_slots(
    preflight: _NucleationPreflight,
    finalized_demand: NucleationFinalizedDemandBuffers,
    diagnostics: NucleationDiagnosticBuffers,
) -> None:
    """Privately stage P3 event counts and selectable free-slot metadata.

    P3 consumes P2's inventory-admitted demand [#/m³] and validated particle
    volume [m³] to calculate full ``int32`` provisional event counts. It
    delegates active/free classification and ascending free-slot ordering to
    E6-F5, retains counts beyond free capacity, and writes only the supplied
    count and slot-diagnostic sidecars. It does not activate slots, resolve
    exhaustion, resize storage, or mutate particles or gas.

    P3 shares one private scalar status readback between its conversion and the
    E6-F5 slot-state classification before either output writer launches.
    After a successful E6-F5 or P3 writer launch, callers must synchronize
    before consuming outputs; asynchronous writer failures do not promise
    rollback.

    Args:
        preflight: Exact private P1 result reused without repeated validation.
        finalized_demand: Exact P2/P3 sidecar record containing admitted demand
            and receiving full provisional counts.
        diagnostics: Exact P2/P3 diagnostic record receiving E6-F5 layouts and
            P3's capacity-limited selectable prefix.

    Raises:
        ValueError: If private handoff records are invalid, slots are invalid,
            or event conversion does not produce finite, nonnegative, integral
            ``int32`` counts.
    """
    if not isinstance(preflight, _NucleationPreflight):
        raise ValueError("preflight must be _NucleationPreflight.")
    if not isinstance(finalized_demand, NucleationFinalizedDemandBuffers):
        raise ValueError(
            "finalized_demand must be NucleationFinalizedDemandBuffers."
        )
    if not isinstance(diagnostics, NucleationDiagnosticBuffers):
        raise ValueError("diagnostics must be NucleationDiagnosticBuffers.")
    _validate_staged_nucleation_sidecars(
        preflight, finalized_demand, diagnostics
    )
    if preflight.n_boxes == 0:
        return

    counts, invalid = _convert_staged_nucleation_counts(
        finalized_demand.accepted_demand,
        preflight.particles.volume,
        preflight.device,
    )
    categories = wp.empty(
        (preflight.n_boxes, preflight.n_particles),
        dtype=wp.int32,
        device=preflight.device,
    )
    if preflight.n_particles:
        wp.launch(
            _classify_slots,
            dim=(preflight.n_boxes, preflight.n_particles),
            inputs=[
                preflight.particles.masses,
                preflight.particles.concentration,
                preflight.particles.charge,
                categories,
                invalid,
            ],
            device=preflight.device,
        )
    status = invalid.numpy()
    if int(status[0]):
        raise ValueError("Invalid particle slot state.")
    if int(status[1]):
        raise ValueError(
            "accepted_demand times particle volume must be a finite, "
            "nonnegative integral int32 count."
        )
    _write_staged_nucleation_diagnostics(preflight, categories, diagnostics)
    wp.launch(
        _commit_staged_nucleation_slots,
        dim=preflight.n_boxes,
        inputs=[
            counts,
            diagnostics.free_slot_indices,
            diagnostics.free_slot_counts,
            finalized_demand.accepted_counts,
            diagnostics.selected_slot_indices,
            preflight.n_particles,
        ],
        device=preflight.device,
    )


@wp.kernel
def _validate_p4_handoff(  # noqa: C901
    accepted_demand: wp.array(dtype=wp.float64),
    accepted_counts: wp.array(dtype=wp.int32),
    volume: wp.array(dtype=wp.float64),
    categories: wp.array2d(dtype=wp.int32),
    free_indices: wp.array2d(dtype=wp.int32),
    active_counts: wp.array(dtype=wp.int32),
    free_counts: wp.array(dtype=wp.int32),
    selected_indices: wp.array2d(dtype=wp.int32),
    releasable: wp.array(dtype=wp.int32),
    requested: wp.array(dtype=wp.float64),
    minimum: wp.array(dtype=wp.float64),
    minimum_volume: wp.array(dtype=wp.float64),
    invalid: wp.array(dtype=wp.int32),
) -> None:
    """Validate immutable P2/P3 handoffs and P4 policy inputs per box."""
    box = wp.tid()
    events = accepted_demand[box] * volume[box]
    if (
        not wp.isfinite(accepted_demand[box])
        or accepted_demand[box] < 0.0
        or not wp.isfinite(volume[box])
        or volume[box] <= 0.0
        or not wp.isfinite(events)
        or events < 0.0
        or events != wp.floor(events)
        or events > wp.float64(2147483647.0)
        or accepted_counts[box] != wp.int32(events)
        or releasable[box] < 0
        or releasable[box] > wp.max(active_counts[box] - 1, 0)
        or not wp.isfinite(requested[box])
        or not wp.isfinite(minimum[box])
        or not wp.isfinite(minimum_volume[box])
        or minimum[box] <= 0.0
        or requested[box] < minimum[box]
        or requested[box] > 1.0
        or minimum_volume[box] <= 0.0
    ):
        wp.atomic_add(invalid, 0, 1)
    active = wp.int32(0)
    free = wp.int32(0)
    selectable = wp.min(accepted_counts[box], free_counts[box])
    for particle in range(categories.shape[1]):
        if categories[box, particle] == 1:
            active += 1
        elif categories[box, particle] == 2:
            if free_indices[box, free] != particle:
                wp.atomic_add(invalid, 0, 1)
            free += 1
        elif categories[box, particle] != 3:
            wp.atomic_add(invalid, 0, 1)
        expected = -1
        if particle < selectable:
            expected = free_indices[box, particle]
        if selected_indices[box, particle] != expected:
            wp.atomic_add(invalid, 0, 1)
    if active != active_counts[box] or free != free_counts[box]:
        wp.atomic_add(invalid, 0, 1)


@wp.kernel
def _select_p4_policy(
    demand: wp.array(dtype=wp.float64),
    counts: wp.array(dtype=wp.int32),
    free_counts: wp.array(dtype=wp.int32),
    releasable: wp.array(dtype=wp.int32),
    requested: wp.array(dtype=wp.float64),
    minimum_volume: wp.array(dtype=wp.float64),
    volume: wp.array(dtype=wp.float64),
    resampling_enabled: wp.int32,
    scaling_enabled: wp.int32,
    release: wp.array(dtype=wp.int32),
    scaling: wp.array(dtype=wp.int32),
    invalid: wp.array(dtype=wp.int32),
) -> None:
    """Select all-or-nothing resampling before scaling for one box."""
    box = wp.tid()
    deficit = wp.max(counts[box] - free_counts[box], 0)
    release[box] = 0
    scaling[box] = 0
    if deficit > 0:
        if resampling_enabled != 0 and releasable[box] >= deficit:
            release[box] = deficit
        elif scaling_enabled != 0:
            scaling[box] = 1
            scaled_events = (demand[box] * requested[box]) * (
                volume[box] * requested[box]
            )
            if (
                not wp.isfinite(scaled_events)
                or scaled_events < 0.0
                or scaled_events != wp.floor(scaled_events)
                or scaled_events > wp.float64(2147483647.0)
                or scaled_events > wp.float64(free_counts[box])
                or volume[box] * requested[box] < minimum_volume[box]
            ):
                wp.atomic_add(invalid, 0, 1)
        else:
            wp.atomic_add(invalid, 0, 1)


@wp.kernel
def _copy_p4_workspace(
    source: wp.array(dtype=wp.float64), destination: wp.array(dtype=wp.float64)
) -> None:
    """Copy immutable P2 admitted demand into private mutable P4 workspace."""
    box = wp.tid()
    destination[box] = source[box]


@wp.kernel
def _validate_p4_final(  # noqa: C901
    demand: wp.array(dtype=wp.float64),
    volume: wp.array(dtype=wp.float64),
    free_counts: wp.array(dtype=wp.int32),
    invalid: wp.array(dtype=wp.int32),
) -> None:
    """Validate final P4 counts before finalized diagnostic writes."""
    box = wp.tid()
    events = demand[box] * volume[box]
    if (
        not wp.isfinite(demand[box])
        or demand[box] < 0.0
        or not wp.isfinite(events)
        or events < 0.0
        or events != wp.floor(events)
        or events > wp.float64(2147483647.0)
        or events > wp.float64(free_counts[box])
    ):
        wp.atomic_add(invalid, 0, 1)


@wp.kernel
def _write_p4_final(
    demand: wp.array(dtype=wp.float64),
    volume: wp.array(dtype=wp.float64),
    free_indices: wp.array2d(dtype=wp.int32),
    final_demand: wp.array(dtype=wp.float64),
    final_counts: wp.array(dtype=wp.int32),
    final_indices: wp.array2d(dtype=wp.int32),
) -> None:
    """Commit finalized P4 demand, counts, and ascending free-slot prefixes."""
    box = wp.tid()
    count = wp.int32(demand[box] * volume[box])
    final_demand[box] = demand[box]
    final_counts[box] = count
    for rank in range(final_indices.shape[1]):
        if rank < count:
            final_indices[box, rank] = free_indices[box, rank]
        else:
            final_indices[box, rank] = -1


@wp.kernel
def _write_current_p4_diagnostics(
    categories: wp.array2d(dtype=wp.int32),
    free_indices: wp.array2d(dtype=wp.int32),
    free_counts: wp.array(dtype=wp.int32),
) -> None:
    """Write private ascending current free-slot diagnostics for P4."""
    box = wp.tid()
    free = wp.int32(0)
    for particle in range(categories.shape[1]):
        if categories[box, particle] == 2:
            free_indices[box, free] = particle
            free += 1
    for particle in range(free, categories.shape[1]):
        free_indices[box, particle] = -1
    free_counts[box] = free


@wp.kernel
def _write_unscaled_p4_resolution(
    scaling: wp.array(dtype=wp.int32),
    resolved_scale: wp.array(dtype=wp.float64),
) -> None:
    """Set the P4 resolved scale to one for unselected policy rows."""
    box = wp.tid()
    if scaling[box] == 0:
        resolved_scale[box] = wp.float64(1.0)


@wp.kernel
def _write_p4_policy_diagnostics(
    counts: wp.array(dtype=wp.int32),
    free_counts: wp.array(dtype=wp.int32),
    scaling: wp.array(dtype=wp.int32),
    required_release: wp.array(dtype=wp.int32),
    scaling_required: wp.array(dtype=wp.int32),
) -> None:
    """Commit validated private policy selection to P4 diagnostic sidecars."""
    box = wp.tid()
    required_release[box] = wp.max(counts[box] - free_counts[box], 0)
    scaling_required[box] = scaling[box]


@wp.kernel
def _aggregate_p4_preflight_status(
    invalid: wp.array(dtype=wp.int32),
    release: wp.array(dtype=wp.int32),
    scaling: wp.array(dtype=wp.int32),
    status: wp.array(dtype=wp.int32),
) -> None:
    """Aggregate all P4 preflight results into one scalar status word."""
    box = wp.tid()
    if invalid[0] != 0:
        status[0] = 1
    if release[box] != 0:
        wp.atomic_or(status, 0, 2)
    if scaling[box] != 0:
        wp.atomic_or(status, 0, 4)


def _validate_p4_buffers(
    preflight: _NucleationPreflight,
    buffers: NucleationExhaustionBuffers,
) -> tuple[Any, ...]:
    """Validate P4 and nested E6-F6 storage before any P4 writer runs."""
    if not isinstance(buffers, NucleationExhaustionBuffers):
        raise ValueError("buffers must be NucleationExhaustionBuffers.")
    if not isinstance(buffers.resampling_buffers, ResamplingBuffers):
        raise ValueError("resampling_buffers must be ResamplingBuffers.")
    b, n, s, device = (
        preflight.n_boxes,
        preflight.n_particles,
        preflight.n_species,
        preflight.device,
    )
    schema = (
        ("demand_workspace", wp.float64, (b,)),
        ("final_demand", wp.float64, (b,)),
        ("requested_scale", wp.float64, (b,)),
        ("minimum_scale", wp.float64, (b,)),
        ("minimum_volume", wp.float64, (b,)),
        ("resolved_scale", wp.float64, (b,)),
        ("resampling_releasable_counts", wp.int32, (b,)),
        ("required_release_counts", wp.int32, (b,)),
        ("scaling_required", wp.int32, (b,)),
        ("final_counts", wp.int32, (b,)),
        ("final_selected_slot_indices", wp.int32, (b, n)),
    )
    arrays = tuple(
        _array(buffers, name, dtype, shape, device)
        for name, dtype, shape in schema
    )
    nested_schema = (
        ("retained_counts", wp.int32, (b,)),
        ("released_counts", wp.int32, (b,)),
        ("retained_indices", wp.int32, (b, n)),
        ("released_indices", wp.int32, (b, n)),
        ("sorted_indices", wp.int32, (b, n)),
        ("replacement_masses", wp.float64, (b, n, s)),
        ("replacement_concentration", wp.float64, (b, n)),
        ("replacement_charge", wp.float64, (b, n)),
        ("source_radii", wp.float64, (b, n)),
        ("radius_cubed_relative_error", wp.float64, (b,)),
        ("mean_radius_relative_error", wp.float64, (b,)),
        ("surface_relative_error", wp.float64, (b,)),
        ("diversity_absolute_error", wp.float64, (b,)),
        ("planning_status", wp.int32, (b,)),
    )
    nested = tuple(
        _array(buffers.resampling_buffers, name, dtype, shape, device)
        for name, dtype, shape in nested_schema
    )
    protected = preflight.protected_inputs
    _no_overlap(protected + arrays + nested)
    return arrays + nested


def _validate_public_p4_inputs(
    preflight: _NucleationPreflight,
    controls: NucleationExhaustionControls,
    buffers: NucleationExhaustionBuffers,
) -> None:
    """Validate public P4 controls and storage before any no-work return."""
    if not isinstance(controls, NucleationExhaustionControls):
        raise ValueError("controls must be NucleationExhaustionControls.")
    controls.__post_init__()
    _validate_p4_buffers(preflight, buffers)


def _revalidate_p4_particle_state(preflight: _NucleationPreflight) -> None:
    """Revalidate mutable P1 particle storage before P4 writes or primitives."""
    b, n, s, device = (
        preflight.n_boxes,
        preflight.n_particles,
        preflight.n_species,
        preflight.device,
    )
    current = (
        _array(preflight.particles, "masses", wp.float64, (b, n, s), device),
        _array(
            preflight.particles,
            "concentration",
            wp.float64,
            (b, n),
            device,
        ),
        _array(preflight.particles, "charge", wp.float64, (b, n), device),
        _array(preflight.particles, "density", wp.float64, (s,), device),
        _array(preflight.particles, "volume", wp.float64, (b,), device),
    )
    if any(
        expected is not supplied
        for expected, supplied in zip(
            preflight.particle_arrays, current, strict=True
        )
    ):
        raise ValueError("P4 particle fields must be the P1-validated storage.")
    _scan(current[0], "particles.masses")
    _scan(current[1], "particles.concentration")
    _scan_finite_2d(current[2], "particles.charge")
    _scan(current[3], "particles.density", positive=True)
    _scan(current[4], "particles.volume", positive=True)


def _orchestrate_nucleation_exhaustion(  # noqa: C901
    preflight: _NucleationPreflight,
    finalized_demand: NucleationFinalizedDemandBuffers,
    diagnostics: NucleationDiagnosticBuffers,
    controls: NucleationExhaustionControls,
    buffers: NucleationExhaustionBuffers,
) -> tuple[Any, ...]:
    """Privately apply P4 capacity policy and write finalized diagnostics.

    P4 validates immutable P2/P3 handoffs, then selects resampling only when it
    can release a row's complete deficit. It selects representative-volume
    scaling only for remaining exhausted rows, invokes each selected E6-F6
    primitive once, and derives final demand/count/free-slot diagnostics from
    the resulting device state. P2/P3 handoffs remain unchanged.

    All expected P4 rejections are detected before P4 workspace writes or an
    E6-F6 primitive call, preserving particle, gas, P2/P3/P4, and nested
    sidecar state. Once a selected primitive begins, its independent
    planning/commit failure contract applies; P4 intentionally provides no
    cross-primitive rollback. This concrete-only, unexported seam does not
    activate slots, mutate gas or source mass, resize storage, transfer data to
    the host, use a CPU fallback, or provide E6-F9 integration.

    Args:
        preflight: Exact private P1 metadata with validated particle and gas
            containers.
        finalized_demand: Exact immutable P2/P3 record containing admitted
            demand and provisional counts.
        diagnostics: Exact immutable P2/P3 record containing slot diagnostics
            and the provisional selected prefix.
        controls: Exact-Boolean resampling-first/scaling-fallback controls.
        buffers: Exact caller-owned P4 and nested E6-F6 workspace/diagnostic
            record.

    Raises:
        ValueError: If handoffs, policy inputs, sidecars, slot state, or final
            demand capacity are invalid.
    """
    if not isinstance(preflight, _NucleationPreflight):
        raise ValueError("preflight must be _NucleationPreflight.")
    if not isinstance(controls, NucleationExhaustionControls):
        raise ValueError("controls must be NucleationExhaustionControls.")
    controls.__post_init__()
    _validate_staged_nucleation_sidecars(
        preflight, finalized_demand, diagnostics
    )
    _revalidate_p4_particle_state(preflight)
    p4_storage = _validate_p4_buffers(preflight, buffers)
    if preflight.n_boxes == 0:
        return p4_storage
    categories = wp.empty(
        (preflight.n_boxes, preflight.n_particles),
        dtype=wp.int32,
        device=preflight.device,
    )
    invalid = wp.zeros(1, dtype=wp.int32, device=preflight.device)
    if preflight.n_particles:
        wp.launch(
            _classify_slots,
            dim=(preflight.n_boxes, preflight.n_particles),
            inputs=[
                preflight.particles.masses,
                preflight.particles.concentration,
                preflight.particles.charge,
                categories,
                invalid,
            ],
            device=preflight.device,
        )
    wp.launch(
        _validate_p4_handoff,
        dim=preflight.n_boxes,
        inputs=[
            finalized_demand.accepted_demand,
            finalized_demand.accepted_counts,
            preflight.particles.volume,
            categories,
            diagnostics.free_slot_indices,
            diagnostics.active_slot_counts,
            diagnostics.free_slot_counts,
            diagnostics.selected_slot_indices,
            buffers.resampling_releasable_counts,
            buffers.requested_scale,
            buffers.minimum_scale,
            buffers.minimum_volume,
            invalid,
        ],
        device=preflight.device,
    )
    release = wp.zeros(
        preflight.n_boxes, dtype=wp.int32, device=preflight.device
    )
    scaling = wp.zeros(
        preflight.n_boxes, dtype=wp.int32, device=preflight.device
    )
    wp.launch(
        _select_p4_policy,
        dim=preflight.n_boxes,
        inputs=[
            finalized_demand.accepted_demand,
            finalized_demand.accepted_counts,
            diagnostics.free_slot_counts,
            buffers.resampling_releasable_counts,
            buffers.requested_scale,
            buffers.minimum_volume,
            preflight.particles.volume,
            int(controls.resampling),
            int(controls.representative_volume_scaling),
            release,
            scaling,
            invalid,
        ],
        device=preflight.device,
    )
    preflight_status = wp.zeros(1, dtype=wp.int32, device=preflight.device)
    wp.launch(
        _aggregate_p4_preflight_status,
        dim=preflight.n_boxes,
        inputs=[invalid, release, scaling, preflight_status],
        device=preflight.device,
    )
    status = int(preflight_status.numpy()[0])
    if status & 1:
        raise ValueError("P4 handoff, policy input, or capacity is invalid.")
    wp.launch(
        _write_p4_policy_diagnostics,
        dim=preflight.n_boxes,
        inputs=[
            finalized_demand.accepted_counts,
            diagnostics.free_slot_counts,
            scaling,
            buffers.required_release_counts,
            buffers.scaling_required,
        ],
        device=preflight.device,
    )
    wp.launch(
        _copy_p4_workspace,
        dim=preflight.n_boxes,
        inputs=[finalized_demand.accepted_demand, buffers.demand_workspace],
        device=preflight.device,
    )
    if status & 2:
        resampling_step_gpu(
            preflight.particles, release, buffers.resampling_buffers
        )
    if status & 4:
        representative_volume_scaling_step_gpu(
            preflight.particles,
            buffers.demand_workspace,
            scaling,
            buffers.requested_scale,
            buffers.minimum_scale,
            buffers.minimum_volume,
            buffers.resolved_scale,
        )
    else:
        wp.launch(
            _write_unscaled_p4_resolution,
            dim=preflight.n_boxes,
            inputs=[scaling, buffers.resolved_scale],
            device=preflight.device,
        )
    final_invalid = wp.zeros(1, dtype=wp.int32, device=preflight.device)
    current_free = diagnostics.free_slot_indices
    current_count = diagnostics.free_slot_counts
    if status & 2:
        # Resampling is the only selected primitive that changes slot state.
        final_categories = wp.empty(
            (preflight.n_boxes, preflight.n_particles),
            dtype=wp.int32,
            device=preflight.device,
        )
        if preflight.n_particles:
            wp.launch(
                _classify_slots,
                dim=(preflight.n_boxes, preflight.n_particles),
                inputs=[
                    preflight.particles.masses,
                    preflight.particles.concentration,
                    preflight.particles.charge,
                    final_categories,
                    final_invalid,
                ],
                device=preflight.device,
            )
        # Keep P3 records immutable by writing current state into private work.
        current_free = wp.empty(
            (preflight.n_boxes, preflight.n_particles),
            dtype=wp.int32,
            device=preflight.device,
        )
        current_count = wp.empty(
            preflight.n_boxes, dtype=wp.int32, device=preflight.device
        )
        wp.launch(
            _write_current_p4_diagnostics,
            dim=preflight.n_boxes,
            inputs=[final_categories, current_free, current_count],
            device=preflight.device,
        )
    wp.launch(
        _validate_p4_final,
        dim=preflight.n_boxes,
        inputs=[
            buffers.demand_workspace,
            preflight.particles.volume,
            current_count,
            final_invalid,
        ],
        device=preflight.device,
    )
    if int(final_invalid.numpy()[0]):
        raise ValueError("P4 final demand does not fit available capacity.")
    wp.launch(
        _write_p4_final,
        dim=preflight.n_boxes,
        inputs=[
            buffers.demand_workspace,
            preflight.particles.volume,
            current_free,
            buffers.final_demand,
            buffers.final_counts,
            buffers.final_selected_slot_indices,
        ],
        device=preflight.device,
    )
    return p4_storage


@wp.kernel
def _validate_participating_molecule_counts(
    partitioning: wp.array2d(dtype=wp.int32),
    molecule_counts: wp.array(dtype=wp.int32),
    invalid: wp.array(dtype=wp.int32),
) -> None:
    """Reject positive molecule counts for non-participating gas lanes."""
    box = wp.tid()
    for species in range(molecule_counts.shape[0]):
        if molecule_counts[species] > 0 and partitioning[box, species] != 1:
            wp.atomic_or(invalid, 0, 1)


def _validate_public_molecule_eligibility(
    preflight: _NucleationPreflight,
) -> None:
    """Validate P5 gas-removal eligibility before P4 can mutate particles."""
    if preflight.n_boxes == 0:
        return
    molecule_counts: Any = wp.array(
        np.asarray(preflight.config.molecule_counts, dtype=np.int32),
        dtype=wp.int32,
        device=preflight.device,
    )
    invalid = wp.zeros(1, dtype=wp.int32, device=preflight.device)
    wp.launch(
        _validate_participating_molecule_counts,
        dim=preflight.n_boxes,
        inputs=[preflight.gas.partitioning, molecule_counts, invalid],
        device=preflight.device,
    )
    if int(invalid.numpy()[0]):
        raise ValueError(
            "Positive molecule counts require participating gas species."
        )


@wp.kernel
def _validate_p5_handoff(  # noqa: C901
    masses: wp.array3d(dtype=wp.float64),
    concentration: wp.array2d(dtype=wp.float64),
    charge: wp.array2d(dtype=wp.float64),
    volume: wp.array(dtype=wp.float64),
    gas_concentration: wp.array2d(dtype=wp.float64),
    molar_mass: wp.array(dtype=wp.float64),
    partitioning: wp.array2d(dtype=wp.int32),
    molecule_counts: wp.array(dtype=wp.int32),
    final_demand: wp.array(dtype=wp.float64),
    final_counts: wp.array(dtype=wp.int32),
    selected_indices: wp.array2d(dtype=wp.int32),
    status: wp.array(dtype=wp.int32),
) -> None:
    """Validate P4's final handoff before the sole P5 writer launches."""
    box = wp.tid()
    count = final_counts[box]
    events = final_demand[box] * volume[box]
    invalid = bool(
        not wp.isfinite(final_demand[box])
        or final_demand[box] < 0.0
        or count < 0
        or count > selected_indices.shape[1]
        or not wp.isfinite(events)
        or events < 0.0
        or events != wp.floor(events)
        or events > wp.float64(2147483647)
        or count != wp.int32(events)
    )
    previous = wp.int32(-1)
    for rank in range(selected_indices.shape[1]):
        selected = selected_indices[box, rank]
        if rank < count:
            if (
                selected < 0
                or selected >= concentration.shape[1]
                or selected <= previous
                or concentration[box, selected] != 0.0
                or charge[box, selected] != 0.0
            ):
                invalid = True
            else:
                for species in range(masses.shape[2]):
                    if masses[box, selected, species] != 0.0:
                        invalid = True
            previous = selected
        elif selected != -1:
            invalid = True
    for species in range(molecule_counts.shape[0]):
        if molecule_counts[species] > 0:
            event_mass = (
                wp.float64(molecule_counts[species])
                * molar_mass[species]
                / wp.float64(AVOGADRO_NUMBER)
            )
            if (
                partitioning[box, species] != 1
                or not wp.isfinite(event_mass)
                or event_mass <= 0.0
                or not wp.isfinite(gas_concentration[box, species])
                or final_demand[box] * event_mass
                > gas_concentration[box, species]
            ):
                invalid = True
    if invalid:
        wp.atomic_or(status, 0, 1)
    if count > 0:
        wp.atomic_or(status, 0, 2)


@wp.kernel
def _commit_nucleation_p5_kernel(
    masses: wp.array3d(dtype=wp.float64),
    concentration: wp.array2d(dtype=wp.float64),
    charge: wp.array2d(dtype=wp.float64),
    volume: wp.array(dtype=wp.float64),
    gas_concentration: wp.array2d(dtype=wp.float64),
    molar_mass: wp.array(dtype=wp.float64),
    molecule_counts: wp.array(dtype=wp.int32),
    final_demand: wp.array(dtype=wp.float64),
    final_counts: wp.array(dtype=wp.int32),
    selected_indices: wp.array2d(dtype=wp.int32),
    resolved_scale: wp.array(dtype=wp.float64),
) -> None:
    """Fuse finalized free-slot activation and matching gas removal."""
    box = wp.tid()
    for rank in range(final_counts[box]):
        particle = selected_indices[box, rank]
        for species in range(masses.shape[2]):
            masses[box, particle, species] = (
                wp.float64(molecule_counts[species])
                * molar_mass[species]
                / wp.float64(AVOGADRO_NUMBER)
            )
        concentration[box, particle] = wp.float64(1.0) / volume[box]
        charge[box, particle] = wp.float64(0.0)
    for species in range(molecule_counts.shape[0]):
        gas_concentration[box, species] *= resolved_scale[box]
        if molecule_counts[species] > 0:
            gas_concentration[box, species] -= final_demand[box] * (
                wp.float64(molecule_counts[species])
                * molar_mass[species]
                / wp.float64(AVOGADRO_NUMBER)
            )


def _revalidate_p5_storage(preflight: _NucleationPreflight) -> None:
    """Reject rebound P1 storage before P5's device handoff validation."""
    b, n, s, device = (
        preflight.n_boxes,
        preflight.n_particles,
        preflight.n_species,
        preflight.device,
    )
    particles = (
        _array(preflight.particles, "masses", wp.float64, (b, n, s), device),
        _array(
            preflight.particles,
            "concentration",
            wp.float64,
            (b, n),
            device,
        ),
        _array(preflight.particles, "charge", wp.float64, (b, n), device),
        _array(preflight.particles, "density", wp.float64, (s,), device),
        _array(preflight.particles, "volume", wp.float64, (b,), device),
    )
    gas = (
        _array(preflight.gas, "molar_mass", wp.float64, (s,), device),
        _array(preflight.gas, "concentration", wp.float64, (b, s), device),
        _array(preflight.gas, "partitioning", wp.int32, (b, s), device),
    )
    if any(
        expected is not supplied
        for expected, supplied in zip(
            preflight.particle_arrays,
            particles,
            strict=True,
        )
    ) or any(
        expected is not supplied
        for expected, supplied in zip(preflight.gas_arrays, gas, strict=True)
    ):
        raise ValueError("P5 fields must be the P1-validated storage.")


def _commit_nucleation_p5(
    preflight: _NucleationPreflight,
    buffers: NucleationExhaustionBuffers,
    p4_storage: tuple[Any, ...],
) -> bool:
    """Validate the final P4 record and launch one fused P5 commit if needed."""
    _revalidate_p5_storage(preflight)
    if any(
        expected is not supplied
        for expected, supplied in zip(
            p4_storage,
            _validate_p4_buffers(preflight, buffers),
            strict=True,
        )
    ):
        raise ValueError("P5 fields must be the P4-validated storage.")
    if preflight.n_boxes == 0:
        return False
    molecule_counts: Any = wp.array(
        np.asarray(preflight.config.molecule_counts, dtype=np.int32),
        dtype=wp.int32,
        device=preflight.device,
    )
    status = wp.zeros(1, dtype=wp.int32, device=preflight.device)
    wp.launch(
        _validate_p5_handoff,
        dim=preflight.n_boxes,
        inputs=[
            preflight.particles.masses,
            preflight.particles.concentration,
            preflight.particles.charge,
            preflight.particles.volume,
            preflight.gas.concentration,
            preflight.gas.molar_mass,
            preflight.gas.partitioning,
            molecule_counts,
            buffers.final_demand,
            buffers.final_counts,
            buffers.final_selected_slot_indices,
            status,
        ],
        device=preflight.device,
    )
    result = int(status.numpy()[0])
    if result & 1:
        raise ValueError("P5 finalized nucleation handoff is invalid.")
    if not result & 2:
        return False
    wp.launch(
        _commit_nucleation_p5_kernel,
        dim=preflight.n_boxes,
        inputs=[
            preflight.particles.masses,
            preflight.particles.concentration,
            preflight.particles.charge,
            preflight.particles.volume,
            preflight.gas.concentration,
            preflight.gas.molar_mass,
            molecule_counts,
            buffers.final_demand,
            buffers.final_counts,
            buffers.final_selected_slot_indices,
            buffers.resolved_scale,
        ],
        device=preflight.device,
    )
    return True


def nucleation_step_gpu(
    particles: Any,
    gas: Any,
    config: NucleationConfig,
    time_step: Any,
    *,
    scratch: NucleationScratchBuffers,
    finalized_demand: NucleationFinalizedDemandBuffers,
    diagnostics: NucleationDiagnosticBuffers,
    exhaustion_controls: NucleationExhaustionControls,
    exhaustion_buffers: NucleationExhaustionBuffers,
    temperature: Any | None = None,
    saturation: Any | None = None,
    environment: Any | None = None,
) -> tuple[Any, Any]:
    """Execute one fixed-capacity direct-Warp nucleation step.

    This direct-device boundary sequences P1 preflight, P2 inventory admission,
    P3 slot staging, and P4 resampling-first/scaling-fallback resolution before
    one fused P5 writer activates finalized free slots and removes matching gas
    mass. The caller owns fixed-capacity, fixed-shape, contiguous same-device
    Warp particle, gas, input, and sidecar storage, and must synchronize the
    active Warp device before observing successful asynchronous writes.

    Public validation rejections before P4 primitive entry leave particle and
    gas state unchanged. P2--P4 may write their documented caller-owned
    sidecars before a later phase rejects. Once a P4 primitive begins, its
    documented no-rollback boundary applies; P5 likewise has no rollback after
    its writer launches. Successful calls return the identical ``(particles,
    gas)`` objects. CPU fallback, host/device transfer helpers, resizing,
    compaction, a Runnable API, backend selection, E6-F9 integration, graph
    capture, autodiff, and performance guarantees are deliberately deferred.

    Args:
        particles: Caller-owned fixed-capacity Warp particle container.
        gas: Caller-owned Warp gas container on the particle-data device.
        config: Immutable scalar nucleation controls.
        time_step: Finite nonnegative step duration [s].
        scratch: Caller-owned P2 planning sidecars.
        finalized_demand: Caller-owned P2/P3 finalized-demand sidecars.
        diagnostics: Caller-owned P2/P3 slot and gate diagnostic sidecars.
        exhaustion_controls: Concrete P4 resampling and scaling policy controls.
        exhaustion_buffers: Caller-owned P4 workspace and finalized diagnostics.
        temperature: Positive scalar [K] or same-device ``wp.float64 (B,)``.
        saturation: Configured same-device dimensionless ``wp.float64 (B, S)``.
        environment: Optional same-device owner of temperature and saturation.

    Returns:
        The identical caller-owned ``(particles, gas)`` containers.

    Raises:
        TypeError: If a direct scalar input has an unsupported type.
        ValueError: If P1--P5 validation, storage ownership, physical state, or
            finalized handoff validation fails.
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
    _validate_public_p4_inputs(
        preflight, exhaustion_controls, exhaustion_buffers
    )
    if preflight.n_boxes == 0:
        return particles, gas
    _plan_nucleation_demand_from_preflight(
        preflight, scratch, finalized_demand, diagnostics
    )
    _stage_nucleation_slots(preflight, finalized_demand, diagnostics)
    _validate_public_molecule_eligibility(preflight)
    p4_storage = _orchestrate_nucleation_exhaustion(
        preflight,
        finalized_demand,
        diagnostics,
        exhaustion_controls,
        exhaustion_buffers,
    )
    _commit_nucleation_p5(preflight, exhaustion_buffers, p4_storage)
    return particles, gas
