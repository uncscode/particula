"""Plan and atomically commit CPU particle-source transactions.

The concrete-module-only P2 boundary converts survival-adjusted potential event
rates [#/m³/s] and a duration [s] into immutable, gas-admitted source-demand
records without accessing particle capacity or mutating caller state. The P3
boundary consumes those records, stages slot activation and exhaustion policy on
a private ``ParticleData`` copy, and writes validated particle and gas arrays
atomically. P3 scales pre-existing particle and gas concentrations for
selected representative-volume rows before removing finalized source mass.
Neither P2 nor P3 is exported through public package namespaces.
"""

from dataclasses import dataclass
from numbers import Real
from typing import cast

import numpy as np
from numpy.typing import NDArray

from particula.dynamics.nucleation.nucleation_strategies import (
    InjectionComposition,
)
from particula.gas.gas_data import GasData
from particula.particles.exhaustion import (
    POLICY_RESAMPLE_DEFERRED,
    POLICY_SCALE_DEFERRED,
    ExhaustionControls,
    ExhaustionInputs,
    apply_representative_volume_scaling,
    apply_resampling,
    plan_resampling,
    resolve_exhaustion,
)
from particula.particles.particle_data import ParticleData
from particula.particles.slot_management import (
    activate_slots,
    get_slot_diagnostics,
)
from particula.util.constants import AVOGADRO_NUMBER


def _readonly_copy(
    values: object,
    dtype: type[np.generic],
    name: str,
    ndim: int | None = None,
) -> NDArray[np.generic]:
    """Copy an array payload into owned, read-only storage.

    This helper ensures that frozen record attributes cannot be changed through
    an aliased caller-owned NumPy array.

    Args:
        values: Array-compatible payload.
        dtype: Required output NumPy dtype.
        name: Field name for validation errors.
        ndim: Optional required number of dimensions.

    Returns:
        A fresh, owned, read-only NumPy array with the requested dtype.

    Raises:
        TypeError: If the value cannot be converted to the requested dtype.
        ValueError: If the array rank differs from ``ndim``.
    """
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be float64-compatible") from error
    if raw.dtype.kind in "bOSUc":
        raise TypeError(f"{name} must be float64-compatible")
    try:
        array = np.array(values, dtype=dtype, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"{name} must be float64-compatible") from error
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must be rank {ndim}")
    array.setflags(write=False)
    return array


def _readonly_vector(
    values: object,
    dtype: type[np.generic],
    name: str,
) -> NDArray[np.generic]:
    """Copy a rank-one record payload into owned, read-only storage.

    Args:
        values: Array-compatible rank-one payload.
        dtype: Required output NumPy dtype.
        name: Field name for validation errors.

    Returns:
        A fresh, owned, read-only rank-one NumPy array.

    Raises:
        TypeError: If ``values`` cannot be converted to ``dtype``.
        ValueError: If ``values`` is not rank one.
    """
    return _readonly_copy(values, dtype, name, ndim=1)


def _is_real_scalar(value: object) -> bool:
    """Return whether a value is a non-Boolean real scalar."""
    return (
        isinstance(value, (Real, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and not isinstance(value, np.ndarray)
    )


@dataclass(frozen=True)
class PotentialEventData:
    """Potential particle-formation rate and source duration.

    Attributes:
        potential_rate: Survival-adjusted event rate [#/m³/s] with shape
            ``(n_boxes,)``. The record owns a fresh read-only float64 copy.
        duration: Finite nonnegative source duration [s].
    """

    potential_rate: NDArray[np.float64]
    duration: float

    def __post_init__(self) -> None:
        """Validate and defensively own the potential-source inputs."""
        rate = cast(
            NDArray[np.float64],
            _readonly_copy(
                self.potential_rate,
                np.float64,
                "potential_rate",
                ndim=1,
            ),
        )
        if not np.all(np.isfinite(rate)):
            raise ValueError("potential_rate must be finite")
        if np.any(rate < 0.0):
            raise ValueError("potential_rate must be nonnegative")
        if not _is_real_scalar(self.duration):
            raise TypeError("duration must be a real scalar")
        try:
            duration = float(self.duration)
        except OverflowError as error:
            raise ValueError("duration must be finite") from error
        if not np.isfinite(duration):
            raise ValueError("duration must be finite")
        if duration < 0.0:
            raise ValueError("duration must be nonnegative")
        object.__setattr__(self, "potential_rate", rate)
        object.__setattr__(self, "duration", duration)


@dataclass(frozen=True)
class SourceDemandData:
    """Immutable P2 gas mass demand for one common event count per box.

    Attributes:
        per_event_mass: Species mass per formed event [kg/event] with shape
            ``(n_species,)``. The record owns a fresh read-only float64 copy.
        gas_mass_removed: Provisional gas demand [kg/m³], shape
            ``(n_boxes, n_species)``. The record owns a fresh read-only
            float64 copy. P2 does not apply this demand to gas.
    """

    per_event_mass: NDArray[np.float64]
    gas_mass_removed: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Defensively own immutable float64 output payloads."""
        object.__setattr__(
            self,
            "per_event_mass",
            _readonly_vector(self.per_event_mass, np.float64, "per_event_mass"),
        )
        object.__setattr__(
            self,
            "gas_mass_removed",
            _readonly_copy(
                self.gas_mass_removed,
                np.float64,
                "gas_mass_removed",
                ndim=2,
            ),
        )


@dataclass(frozen=True)
class SourceDiagnostics:
    """Immutable source counts and limiting-species diagnostics.

    All payloads are fresh, read-only arrays owned by this record. A
    limiting-species index of ``-1`` means no limiter is reported, including
    zero-potential and zero-admitted rows.

    Attributes:
        potential_event_count: Potential events [#/m³] with shape
            ``(n_boxes,)``.
        gas_admitted_event_count: Inventory-admitted events [#/m³] with shape
            ``(n_boxes,)``.
        gas_limited_event_count: Potential events not admitted by gas [#/m³]
            with shape ``(n_boxes,)``.
        limiting_species_index: Participating gas species that limits each box,
            with shape ``(n_boxes,)`` and ``-1`` as the no-limiting sentinel.
    """

    potential_event_count: NDArray[np.float64]
    gas_admitted_event_count: NDArray[np.float64]
    gas_limited_event_count: NDArray[np.float64]
    limiting_species_index: NDArray[np.int32]

    def __post_init__(self) -> None:
        """Defensively own immutable output payloads."""
        for name, dtype in (
            ("potential_event_count", np.float64),
            ("gas_admitted_event_count", np.float64),
            ("gas_limited_event_count", np.float64),
            ("limiting_species_index", np.int32),
        ):
            object.__setattr__(
                self,
                name,
                _readonly_vector(getattr(self, name), dtype, name),
            )


def _validate_gas(  # noqa: C901
    gas: GasData, n_boxes: int, n_species: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate gas schema and physical inventory inputs without mutation.

    Args:
        gas: Gas inventory whose molar masses are [kg/mol] and concentrations
            are [kg/m³].
        n_boxes: Required number of gas concentration rows.
        n_species: Required number of gas species columns.

    Returns:
        Float64 views or conversions of validated molar masses [kg/mol] and
        concentrations [kg/m³]. Neither result is written by this module.

    Raises:
        ValueError: If gas schemas, dtypes, or physical values are invalid.
    """
    if (
        not isinstance(gas.name, list)
        or len(gas.name) != n_species
        or not all(isinstance(name, str) for name in gas.name)
    ):
        raise ValueError("gas name must be a list of n_species strings")
    if not isinstance(gas.molar_mass, np.ndarray) or gas.molar_mass.shape != (
        n_species,
    ):
        raise ValueError("gas molar_mass must have shape (n_species,)")
    if not isinstance(
        gas.concentration, np.ndarray
    ) or gas.concentration.shape != (n_boxes, n_species):
        raise ValueError("gas concentration shape must match potential_rate")
    if not isinstance(
        gas.partitioning, np.ndarray
    ) or gas.partitioning.shape != (n_species,):
        raise ValueError("gas partitioning must have shape (n_species,)")
    if gas.partitioning.dtype != np.bool_:
        raise ValueError("gas partitioning must be a boolean array")
    if (
        gas.molar_mass.dtype.kind in "bOSUc"
        or gas.concentration.dtype.kind in "bOSUc"
    ):
        raise ValueError("gas fields must be float64-compatible")
    try:
        molar_mass = np.asarray(gas.molar_mass, dtype=np.float64)
        concentration = np.asarray(gas.concentration, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("gas fields must be float64-compatible") from error
    if not np.all(np.isfinite(molar_mass)):
        raise ValueError("gas molar_mass must be finite")
    if not np.all(np.isfinite(concentration)):
        raise ValueError("gas concentration must be finite")
    if np.any(concentration < 0.0):
        raise ValueError("gas concentration must be nonnegative")
    return molar_mass, concentration


def finalize_particle_source(  # noqa: C901
    potential_events: PotentialEventData,
    composition: InjectionComposition,
    gas: GasData,
) -> tuple[SourceDemandData, SourceDiagnostics]:
    """Finalize immutable, inventory-limited source demand records.

    Every box receives one common admitted event count constrained by its
    tightest participating gas inventory. The returned records own fresh,
    read-only arrays. This concrete P2 boundary only plans demand: it neither
    mutates inputs or ``gas`` nor accesses particle slots or exhaustion state.

    Args:
        potential_events: Survival-adjusted potential source rate [#/m³/s] and
            source duration [s]. Survival is not applied again.
        composition: Nonnegative molecule counts by gas species for each
            formed event.
        gas: Read-only source inventory with molar masses [kg/mol] and
            concentrations [kg/m³].

    Returns:
        Immutable per-event mass [kg/event] and provisional gas demand [kg/m³],
        followed by event-count diagnostics [#/m³] and limiting species indices.

    Raises:
        TypeError: If a top-level input has an invalid type.
        ValueError: If schemas, physical values, or derived quantities are
            invalid or cannot be made inventory-safe after four ULP corrections.
    """
    if not isinstance(potential_events, PotentialEventData):
        raise TypeError("potential_events must be PotentialEventData")
    if not isinstance(composition, InjectionComposition):
        raise TypeError("composition must be InjectionComposition")
    if not isinstance(gas, GasData):
        raise TypeError("gas must be GasData")

    potential_rate = potential_events.potential_rate
    with np.errstate(over="raise", invalid="raise"):
        try:
            potential_count = potential_rate * potential_events.duration
        except FloatingPointError as error:
            raise ValueError("potential_event_count must be finite") from error
    if not np.all(np.isfinite(potential_count)):
        raise ValueError("potential_event_count must be finite")

    n_boxes = potential_count.size
    n_species = len(composition.molecule_counts)
    molar_mass, concentration = _validate_gas(gas, n_boxes, n_species)
    if any(count > 2**53 for count in composition.molecule_counts):
        raise ValueError("molecule_counts must be representable as float64")
    try:
        counts = np.asarray(composition.molecule_counts, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "molecule_counts must be representable as float64"
        ) from error
    if not np.all(np.isfinite(counts)):
        raise ValueError("molecule_counts must be representable as float64")
    participating = np.flatnonzero(counts > 0.0)
    if np.any(molar_mass[participating] <= 0.0):
        raise ValueError("participating gas molar_mass must be positive")

    with np.errstate(over="raise", invalid="raise", under="ignore"):
        try:
            per_event_mass = counts * molar_mass / AVOGADRO_NUMBER
        except FloatingPointError as error:
            raise ValueError("per_event_mass must be finite") from error
    if not np.all(np.isfinite(per_event_mass)):
        raise ValueError("per_event_mass must be finite")
    if np.any(per_event_mass[participating] <= 0.0):
        raise ValueError(
            "per_event_mass must be positive for participating gas"
        )

    if n_boxes == 0:
        admitted = np.empty(0, dtype=np.float64)
        limiting_indices = np.empty(0, dtype=np.intp)
    elif participating.size == 0:
        admitted = potential_count.copy()
        limiting_indices = np.full(n_boxes, -1, dtype=np.intp)
    else:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            try:
                ratios = (
                    concentration[:, participating]
                    / per_event_mass[participating]
                )
            except FloatingPointError as error:
                raise ValueError(
                    "gas inventory ratio must be finite"
                ) from error
        if not np.all(np.isfinite(ratios)):
            raise ValueError("gas inventory ratio must be finite")
        minimum_ratio = np.min(ratios, axis=1)
        limiting_indices = participating[np.argmin(ratios, axis=1)]
        admitted = np.minimum(potential_count, minimum_ratio)
    if not np.all(np.isfinite(admitted)):
        raise ValueError("gas_admitted_event_count must be finite")

    for _ in range(4):
        with np.errstate(over="raise", invalid="raise"):
            try:
                demand = admitted[:, None] * per_event_mass[None, :]
            except FloatingPointError as error:
                raise ValueError("gas_mass_removed must be finite") from error
        overshot = np.any(
            demand[:, participating] > concentration[:, participating], axis=1
        )
        if not np.any(overshot):
            break
        admitted[overshot] = np.nextafter(admitted[overshot], -np.inf)
    with np.errstate(over="raise", invalid="raise"):
        try:
            demand = admitted[:, None] * per_event_mass[None, :]
        except FloatingPointError as error:
            raise ValueError("gas_mass_removed must be finite") from error
    if np.any(demand[:, participating] > concentration[:, participating]):
        raise ValueError("gas demand remains out of inventory after correction")
    if not np.all(np.isfinite(demand)) or np.any(demand < 0.0):
        raise ValueError("gas_mass_removed must be finite and nonnegative")

    gas_limited = potential_count - admitted
    reduced = potential_count > admitted
    limiting = np.full(n_boxes, -1, dtype=np.int32)
    limiting[reduced & (admitted > 0.0)] = limiting_indices[
        reduced & (admitted > 0.0)
    ]

    return (
        SourceDemandData(per_event_mass, demand),
        SourceDiagnostics(potential_count, admitted, gas_limited, limiting),
    )


@dataclass(frozen=True)
class ParticleSourceCommitConfig:
    """Own controls and representative-volume inputs for a P3 transaction.

    The configuration owns read-only float64 P4 sidecars. Their length is
    checked against the P2 box count by ``commit_particle_source``. The four
    resampling bounds intentionally require exact Python ``float`` values to
    match the exhaustion-policy boundary.

    Attributes:
        maximum_slot_weight: Positive maximum represented events per activated
            slot [#/m³].
        source_charge: Finite charge assigned to each activated source slot
            [elementary-charge counts].
        exhaustion_controls: Immutable controls selecting exhaustion policies.
        requested_scale: Requested representative-volume scale per box,
            shape ``(n_boxes,)``. This record owns a read-only float64 copy.
        minimum_scale: Minimum permitted representative-volume scale per box,
            shape ``(n_boxes,)``. This record owns a read-only float64 copy.
        minimum_volume: Minimum particle representative volume per box [m³],
            shape ``(n_boxes,)``. This record owns a read-only float64 copy.
        radius_cubed_relative_error: Permitted resampling relative error for
            radius cubed.
        mean_radius_relative_error: Permitted resampling relative error for
            mean radius.
        surface_relative_error: Permitted resampling relative error for surface
            area.
        diversity_absolute_error: Permitted resampling absolute diversity error.
    """

    maximum_slot_weight: float
    source_charge: float
    exhaustion_controls: ExhaustionControls
    requested_scale: NDArray[np.float64]
    minimum_scale: NDArray[np.float64]
    minimum_volume: NDArray[np.float64]
    radius_cubed_relative_error: float = 1.0
    mean_radius_relative_error: float = 1.0
    surface_relative_error: float = 1.0
    diversity_absolute_error: float = 1.0

    def __post_init__(self) -> None:
        """Validate scalar controls and defensively own P4 inputs."""
        for name, positive in (
            ("maximum_slot_weight", True),
            ("source_charge", False),
        ):
            value = getattr(self, name)
            if not _is_real_scalar(value):
                raise TypeError(f"{name} must be a real scalar")
            value = float(value)
            if not np.isfinite(value) or (positive and value <= 0.0):
                qualifier = "positive" if positive else "finite"
                raise ValueError(f"{name} must be finite and {qualifier}")
            object.__setattr__(self, name, value)
        if not isinstance(self.exhaustion_controls, ExhaustionControls):
            raise TypeError("exhaustion_controls must be ExhaustionControls")
        for name in (
            "radius_cubed_relative_error",
            "mean_radius_relative_error",
            "surface_relative_error",
            "diversity_absolute_error",
        ):
            value = getattr(self, name)
            if type(value) is not float:
                raise TypeError(f"{name} must be an exact Python float")
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("requested_scale", "minimum_scale", "minimum_volume"):
            object.__setattr__(
                self,
                name,
                _readonly_vector(getattr(self, name), np.float64, name),
            )


@dataclass(frozen=True)
class FinalizedSourceDiagnostics:
    """Own immutable diagnostics from a committed P3 source transaction.

    All arrays are fresh read-only copies. Event counts are [#/m³], gas mass
    fields are [kg/m³], and slot counts and policy codes are ``int32``. The
    conservation residual is ``particle_post + gas_post - scale *
    (particle_pre + gas_pre)`` for every box and species.

    Attributes:
        potential_event_count: P2 potential event count, shape ``(n_boxes,)``.
        gas_admitted_event_count: P2 gas-admitted event count, shape
            ``(n_boxes,)``.
        represented_event_count: Final event count represented in particle
            storage, shape ``(n_boxes,)``.
        gas_limited_event_count: P2 event count excluded by gas inventory,
            shape ``(n_boxes,)``.
        representation_reduction_event_count: Event count removed by
            representative-volume scaling, shape ``(n_boxes,)``.
        residual_event_count: Unpackaged final source demand, shape
            ``(n_boxes,)``; successful transactions report zero.
        limiting_species_index: P2 limiting gas-species index or ``-1``, shape
            ``(n_boxes,)``.
        gas_mass_removed: Final source mass removed from gas, shape
            ``(n_boxes, n_species)``.
        requested_slot_count: Final equal-weight source-slot request count,
            shape ``(n_boxes,)``.
        activated_slot_count: Number of source slots activated, shape
            ``(n_boxes,)``.
        released_slot_count: Number of slots released by resampling, shape
            ``(n_boxes,)``.
        exhaustion_policy_code: Resolved exhaustion-policy code, shape
            ``(n_boxes,)``.
        representative_volume_scale: Applied representative-volume scale,
            shape ``(n_boxes,)``.
        conservation_residual: Final particle-plus-gas scaled-domain mass
            residual, shape ``(n_boxes, n_species)``.
    """

    potential_event_count: NDArray[np.float64]
    gas_admitted_event_count: NDArray[np.float64]
    represented_event_count: NDArray[np.float64]
    gas_limited_event_count: NDArray[np.float64]
    representation_reduction_event_count: NDArray[np.float64]
    residual_event_count: NDArray[np.float64]
    limiting_species_index: NDArray[np.int32]
    gas_mass_removed: NDArray[np.float64]
    requested_slot_count: NDArray[np.int32]
    activated_slot_count: NDArray[np.int32]
    released_slot_count: NDArray[np.int32]
    exhaustion_policy_code: NDArray[np.int32]
    representative_volume_scale: NDArray[np.float64]
    conservation_residual: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Defensively own all finalized diagnostic arrays."""
        vectors = (
            "potential_event_count",
            "gas_admitted_event_count",
            "represented_event_count",
            "gas_limited_event_count",
            "representation_reduction_event_count",
            "residual_event_count",
            "limiting_species_index",
            "requested_slot_count",
            "activated_slot_count",
            "released_slot_count",
            "exhaustion_policy_code",
            "representative_volume_scale",
        )
        int_names = {
            "limiting_species_index",
            "requested_slot_count",
            "activated_slot_count",
            "released_slot_count",
            "exhaustion_policy_code",
        }
        for name in vectors:
            dtype = np.int32 if name in int_names else np.float64
            object.__setattr__(
                self, name, _readonly_vector(getattr(self, name), dtype, name)
            )
        for name in ("gas_mass_removed", "conservation_residual"):
            object.__setattr__(
                self,
                name,
                _readonly_copy(getattr(self, name), np.float64, name, ndim=2),
            )


def _validate_commit_particle_schema(  # noqa: C901
    particles: object,
) -> ParticleData:
    """Validate writable, physically valid, non-overlapping P3 particle storage.

    Args:
        particles: Candidate fixed-capacity particle container.

    Returns:
        The validated particle container without modifying its arrays.

    Raises:
        TypeError: If ``particles`` or a required field has an invalid type.
        ValueError: If field schemas, mutability, aliasing, or physical values
            violate the P3 transaction boundary.
    """
    if not isinstance(particles, ParticleData):
        raise TypeError("particles must be a ParticleData")
    fields = (
        ("masses", particles.masses, 3),
        ("concentration", particles.concentration, 2),
        ("charge", particles.charge, 2),
        ("density", particles.density, 1),
        ("volume", particles.volume, 1),
    )
    for name, value, rank in fields:
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{name} must be a numpy array")
        if value.dtype != np.float64:
            raise ValueError(f"{name} must have dtype float64")
        if value.ndim != rank:
            raise ValueError(f"{name} must have rank {rank}")
        if not value.flags.writeable:
            raise ValueError(f"{name} must be writable")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be contiguous")
    boxes, capacity, species = particles.masses.shape
    if capacity == 0 or species == 0:
        raise ValueError(
            "particle capacity and species capacity must be positive"
        )
    if (
        particles.concentration.shape != (boxes, capacity)
        or particles.charge.shape != (boxes, capacity)
        or particles.density.shape != (species,)
        or particles.volume.shape != (boxes,)
    ):
        raise ValueError("particle fields have incompatible shapes")
    arrays = tuple(value for _, value, _ in fields)
    if any(
        np.shares_memory(left, right)
        for index, left in enumerate(arrays)
        for right in arrays[index + 1 :]
    ):
        raise ValueError("particle fields must not share storage")
    if (
        not np.all(np.isfinite(particles.masses))
        or np.any(particles.masses < 0.0)
        or not np.all(np.isfinite(particles.concentration))
        or np.any(particles.concentration < 0.0)
        or not np.all(np.isfinite(particles.charge))
        or not np.all(np.isfinite(particles.density))
        or np.any(particles.density <= 0.0)
        or not np.all(np.isfinite(particles.volume))
        or np.any(particles.volume <= 0.0)
    ):
        raise ValueError("particle fields must contain finite physical values")
    return particles


def _validate_commit_inputs(  # noqa: C901
    demand: object,
    diagnostics: object,
    particles: object,
    gas: object,
    config: object,
) -> tuple[
    SourceDemandData,
    SourceDiagnostics,
    ParticleData,
    GasData,
    ParticleSourceCommitConfig,
]:
    """Validate immutable P2 records and mutable P3 state before staging.

    This read-only preflight verifies record consistency, writable particle and
    gas schemas, nonaliasing, and representative-volume sidecars. It performs
    no caller-visible mutation, preserving the transaction's all-or-nothing
    boundary if validation fails.

    Args:
        demand: Candidate P2 per-event mass and provisional gas-demand record.
        diagnostics: Candidate P2 event-count diagnostic record.
        particles: Candidate writable fixed-capacity particle container.
        gas: Candidate writable gas inventory.
        config: Candidate P3 capacity and scaling configuration.

    Returns:
        Validated, type-narrowed P2 records, particle data, gas data, and
        transaction configuration.

    Raises:
        TypeError: If a top-level boundary object has an invalid type.
        ValueError: If records, schemas, aliasing, physical values, or scaling
            controls violate the P3 transaction boundary.
    """
    if not isinstance(demand, SourceDemandData):
        raise TypeError("demand must be SourceDemandData")
    if not isinstance(diagnostics, SourceDiagnostics):
        raise TypeError("diagnostics must be SourceDiagnostics")
    particles = _validate_commit_particle_schema(particles)
    if not isinstance(gas, GasData):
        raise TypeError("gas must be GasData")
    if not isinstance(config, ParticleSourceCommitConfig):
        raise TypeError("config must be ParticleSourceCommitConfig")
    boxes, _, species = particles.masses.shape
    if gas.concentration.shape != (boxes, species):
        raise ValueError("gas concentration shape must match particles")
    _validate_gas(gas, boxes, species)
    if (
        gas.concentration.dtype != np.float64
        or not gas.concentration.flags.writeable
        or not gas.concentration.flags.c_contiguous
    ):
        raise ValueError(
            "gas concentration must be writable contiguous float64"
        )
    if any(
        np.shares_memory(gas.concentration, field)
        for field in (
            particles.masses,
            particles.concentration,
            particles.charge,
            particles.density,
            particles.volume,
        )
    ):
        raise ValueError("gas concentration must not share particle storage")
    arrays = (
        demand.per_event_mass,
        demand.gas_mass_removed,
        diagnostics.potential_event_count,
        diagnostics.gas_admitted_event_count,
        diagnostics.gas_limited_event_count,
        diagnostics.limiting_species_index,
    )
    if any(array.flags.writeable for array in arrays):
        raise ValueError("P2 records must have read-only arrays")
    if (
        demand.per_event_mass.dtype != np.float64
        or demand.gas_mass_removed.dtype != np.float64
        or diagnostics.potential_event_count.dtype != np.float64
        or diagnostics.gas_admitted_event_count.dtype != np.float64
        or diagnostics.gas_limited_event_count.dtype != np.float64
        or diagnostics.limiting_species_index.dtype != np.int32
    ):
        raise ValueError("P2 records must have documented dtypes")
    if (
        demand.per_event_mass.shape != (species,)
        or demand.gas_mass_removed.shape != (boxes, species)
        or any(
            array.shape != (boxes,)
            for array in (
                diagnostics.potential_event_count,
                diagnostics.gas_admitted_event_count,
                diagnostics.gas_limited_event_count,
                diagnostics.limiting_species_index,
            )
        )
    ):
        raise ValueError("P2 record shapes are inconsistent")
    if (
        not np.all(np.isfinite(demand.per_event_mass))
        or np.any(demand.per_event_mass < 0.0)
        or not np.all(np.isfinite(demand.gas_mass_removed))
        or np.any(demand.gas_mass_removed < 0.0)
        or any(
            not np.all(np.isfinite(array)) or np.any(array < 0.0)
            for array in (
                diagnostics.potential_event_count,
                diagnostics.gas_admitted_event_count,
                diagnostics.gas_limited_event_count,
            )
        )
    ):
        raise ValueError("P2 records must be finite and nonnegative")
    if not np.allclose(
        diagnostics.gas_admitted_event_count
        + diagnostics.gas_limited_event_count,
        diagnostics.potential_event_count,
        rtol=1e-12,
        atol=1e-30,
    ) or not np.array_equal(
        demand.gas_mass_removed,
        diagnostics.gas_admitted_event_count[:, None]
        * demand.per_event_mass[None, :],
    ):
        raise ValueError("P2 records are mutually inconsistent")
    if np.any(
        (diagnostics.limiting_species_index < -1)
        | (diagnostics.limiting_species_index >= species)
    ):
        raise ValueError("P2 limiting species indices are invalid")
    for name in ("requested_scale", "minimum_scale", "minimum_volume"):
        values = getattr(config, name)
        if values.shape != (boxes,):
            raise ValueError(f"{name} must have shape (B,)")
    if (
        not np.all(np.isfinite(config.requested_scale))
        or not np.all(np.isfinite(config.minimum_scale))
        or not np.all(np.isfinite(config.minimum_volume))
        or np.any(config.minimum_scale <= 0.0)
        or np.any(config.minimum_scale > config.requested_scale)
        or np.any(config.requested_scale > 1.0)
        or np.any(config.minimum_volume <= 0.0)
    ):
        raise ValueError("invalid representative-volume scaling configuration")
    return demand, diagnostics, particles, gas, config


def _request_counts(
    event_count: NDArray[np.float64],
    maximum_slot_weight: float,
    capacity: int,
    *,
    enforce_capacity: bool = True,
) -> NDArray[np.int32]:
    """Calculate checked equal-weight source-slot requests per box.

    A positive event count requests ``ceil(event_count / maximum_slot_weight)``
    slots; zero count requests zero slots.

    Args:
        event_count: Final or provisional represented event count [#/m³].
        maximum_slot_weight: Positive maximum represented events per slot
            [#/m³].
        capacity: Fixed particle-slot capacity available in each box.
        enforce_capacity: Whether to reject requests above ``capacity``. Policy
            resolution uses unchecked provisional requests before scaling;
            activation always enforces capacity.

    Returns:
        Int32 source-slot request count for each box.

    Raises:
        ValueError: If the derived request count is nonfinite, negative, or
            exceeds fixed particle capacity when ``enforce_capacity`` is true.
    """
    with np.errstate(over="raise", invalid="raise"):
        try:
            requested = np.ceil(event_count / maximum_slot_weight)
        except FloatingPointError as error:
            raise ValueError("requested slot count must be finite") from error
    if (
        not np.all(np.isfinite(requested))
        or np.any(requested < 0.0)
        or np.any(requested > np.iinfo(np.int32).max)
        or (enforce_capacity and np.any(requested > capacity))
    ):
        raise ValueError("requested slot count exceeds particle capacity")
    return requested.astype(np.int32)


def _weighted_particle_mass(particles: ParticleData) -> NDArray[np.float64]:
    """Calculate per-box, per-species concentration-weighted particle mass.

    Args:
        particles: Particle data with masses [kg/event] and concentrations
            [#/m³].

    Returns:
        Particle mass concentration [kg/m³] with shape
        ``(n_boxes, n_species)``.
    """
    return np.einsum(
        "bn,bns->bs",
        particles.concentration,
        particles.masses,
        dtype=np.float64,
        optimize=True,
    )


def commit_particle_source(  # noqa: C901, PLR0914, PLR0915
    demand: SourceDemandData,
    diagnostics: SourceDiagnostics,
    particles: ParticleData,
    gas: GasData,
    config: ParticleSourceCommitConfig,
) -> FinalizedSourceDiagnostics:
    """Commit a gas-admitted source into staged fixed-capacity particle storage.

    P2 event counts and represented event counts are [#/m³]. P2 demand and
    final gas removal are mass concentrations [kg/m³], while per-event particle
    mass is [kg/event] and ``source_charge`` is in elementary-charge counts.
    Policy resolution, resampling, representative-volume scaling, and
    activation operate on a private ``ParticleData.copy()``.
    A selected representative-volume row scales pre-existing particle and gas
    concentrations before final source mass is subtracted. Returned diagnostics
    own read-only arrays. Only caller ``masses``, ``concentration``, ``charge``,
    ``volume``, and gas concentration arrays are mutated, after all validation.

    Args:
        demand: Immutable P2 per-event mass and provisional gas-demand record.
        diagnostics: Immutable P2 event-count diagnostics.
        particles: Writable fixed-capacity particle data.
        gas: Writable gas concentration inventory.
        config: Immutable capacity, resampling, and P4 scaling controls.

    Returns:
        Immutable final event, slot, policy, gas-transfer, and conservation
        diagnostics.

    Raises:
        TypeError: If a boundary object has the wrong type.
        ValueError: If schemas, physical values, P2 records, capacity policy,
            scaling, activation, gas inventory, or conservation are invalid.
    """
    demand, diagnostics, particles, gas, config = _validate_commit_inputs(
        demand, diagnostics, particles, gas, config
    )
    particle_pre = _weighted_particle_mass(particles)
    gas_pre = gas.concentration.copy()
    staged_particles = particles.copy()
    boxes, capacity, species = staged_particles.masses.shape

    provisional_counts = _request_counts(
        diagnostics.gas_admitted_event_count,
        config.maximum_slot_weight,
        capacity,
        enforce_capacity=False,
    )
    free_indices, active_counts, free_counts = get_slot_diagnostics(
        staged_particles
    )
    releasable = np.maximum(active_counts - 1, 0).astype(np.int32)
    exhaustion_plan = resolve_exhaustion(
        ExhaustionInputs(
            provisional_counts,
            free_counts,
            releasable,
            free_indices,
        ),
        config.exhaustion_controls,
        allow_oversized_requests=True,
    )
    selected_resampling = any(
        plan.policy_code == POLICY_RESAMPLE_DEFERRED
        for plan in exhaustion_plan.box_plans
    )
    if selected_resampling:
        resampling_plan = plan_resampling(
            staged_particles,
            exhaustion_plan,
            radius_cubed_relative_error=config.radius_cubed_relative_error,
            mean_radius_relative_error=config.mean_radius_relative_error,
            surface_relative_error=config.surface_relative_error,
            diversity_absolute_error=config.diversity_absolute_error,
        )
        apply_resampling(staged_particles, resampling_plan)
        released_counts = np.asarray(
            [len(plan.released_indices) for plan in resampling_plan.box_plans],
            dtype=np.int32,
        )
    else:
        released_counts = np.zeros(boxes, dtype=np.int32)
    policy_codes = np.asarray(
        [plan.policy_code for plan in exhaustion_plan.box_plans], dtype=np.int32
    )

    provisional_demand = np.ascontiguousarray(
        diagnostics.gas_admitted_event_count.copy(), dtype=np.float64
    )
    scaling_required = np.ascontiguousarray(
        policy_codes == POLICY_SCALE_DEFERRED, dtype=np.bool_
    )
    requested_scale = np.array(
        config.requested_scale, dtype=np.float64, copy=True
    )
    minimum_scale = np.array(config.minimum_scale, dtype=np.float64, copy=True)
    minimum_volume = np.array(
        config.minimum_volume, dtype=np.float64, copy=True
    )
    resolved_scale = np.zeros(boxes, dtype=np.float64)
    _, represented_events, resolved_scale = apply_representative_volume_scaling(
        staged_particles,
        provisional_demand,
        scaling_required,
        requested_scale,
        minimum_scale,
        minimum_volume,
        resolved_scale,
    )
    reduction = diagnostics.gas_admitted_event_count - represented_events
    if (
        not np.all(np.isfinite(represented_events))
        or np.any(represented_events < 0.0)
        or not np.all(np.isfinite(reduction))
        or np.any(reduction < 0.0)
    ):
        raise ValueError(
            "represented source demand must be finite and nonnegative"
        )
    requested_counts = _request_counts(
        represented_events, config.maximum_slot_weight, capacity
    )
    request_capacity = max(1, int(np.max(requested_counts, initial=0)))
    request_masses = np.zeros(
        (boxes, request_capacity, species), dtype=np.float64
    )
    request_concentration = np.zeros(
        (boxes, request_capacity), dtype=np.float64
    )
    request_charge = np.zeros((boxes, request_capacity), dtype=np.float64)
    for box_index, count in enumerate(requested_counts):
        prefix = int(count)
        if prefix:
            request_masses[box_index, :prefix] = demand.per_event_mass
            request_concentration[box_index, :prefix] = (
                represented_events[box_index] / prefix
            )
            request_charge[box_index, :prefix] = config.source_charge
    activated_counts = activate_slots(
        staged_particles,
        request_masses,
        request_concentration,
        request_charge,
        requested_counts,
    )
    gas_mass_removed = (
        represented_events[:, None] * demand.per_event_mass[None, :]
    )
    staged_gas = resolved_scale[:, None] * gas_pre - gas_mass_removed
    if (
        not np.all(np.isfinite(staged_gas))
        or np.any(staged_gas < 0.0)
        or not np.all(np.isfinite(gas_mass_removed))
    ):
        raise ValueError(
            "final gas concentration must be finite and nonnegative"
        )
    particle_post = _weighted_particle_mass(staged_particles)
    residual = (
        particle_post
        + staged_gas
        - resolved_scale[:, None] * (particle_pre + gas_pre)
    )
    if not np.all(np.isfinite(residual)) or not np.allclose(
        residual, 0.0, rtol=1e-12, atol=1e-30
    ):
        raise ValueError(
            "particle and gas source transaction must conserve mass"
        )
    residual_events = np.zeros(boxes, dtype=np.float64)
    finalized = FinalizedSourceDiagnostics(
        potential_event_count=diagnostics.potential_event_count,
        gas_admitted_event_count=diagnostics.gas_admitted_event_count,
        represented_event_count=represented_events,
        gas_limited_event_count=diagnostics.gas_limited_event_count,
        representation_reduction_event_count=reduction,
        residual_event_count=residual_events,
        limiting_species_index=diagnostics.limiting_species_index,
        gas_mass_removed=gas_mass_removed,
        requested_slot_count=requested_counts,
        activated_slot_count=activated_counts,
        released_slot_count=released_counts,
        exhaustion_policy_code=policy_codes,
        representative_volume_scale=resolved_scale,
        conservation_residual=residual,
    )
    particles.masses[:] = staged_particles.masses
    particles.concentration[:] = staged_particles.concentration
    particles.charge[:] = staged_particles.charge
    particles.volume[:] = staged_particles.volume
    gas.concentration[:] = staged_gas
    return finalized
