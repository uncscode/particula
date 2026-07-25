"""Finalize gas-admitted particle-source demand without mutating state.

This CPU-only P2 boundary converts survival-adjusted potential event rates into
immutable gas mass-demand records.  It does not activate particle slots, plan
capacity exhaustion, or mutate :class:`~particula.gas.gas_data.GasData`.
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
from particula.util.constants import AVOGADRO_NUMBER


def _readonly_copy(
    values: object,
    dtype: type[np.generic],
    name: str,
    ndim: int | None = None,
) -> NDArray[np.generic]:
    """Copy an array payload and make its owned copy read-only.

    Args:
        values: Array-compatible payload.
        dtype: Required output NumPy dtype.
        name: Field name for validation errors.
        ndim: Optional required number of dimensions.

    Returns:
        A fresh, read-only NumPy array.

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
    """Copy a rank-one record payload into immutable owned storage."""
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
        potential_rate: Survival-adjusted event rate [#/m³/s], shape
            ``(n_boxes,)``. The record owns a read-only float64 copy.
        duration: Source duration [s].
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
        per_event_mass: Species mass per formed event [kg/event], shape
            ``(n_species,)``.
        gas_mass_removed: Provisional gas demand [kg/m³], shape
            ``(n_boxes, n_species)``. P2 does not apply this demand to gas.
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

    A limiting-species index of ``-1`` means gas did not reduce a positive
    potential count, including zero-potential and zero-admitted rows.
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


def _validate_gas(
    gas: GasData, n_boxes: int, n_species: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate gas schemas and physical inputs without mutating gas."""
    if len(gas.name) != n_species:
        raise ValueError("composition width must match gas species")
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
    tightest participating gas inventory. Inputs and ``gas`` are read-only from
    this boundary; particle slots and exhaustion state are intentionally absent.

    Args:
        potential_events: Survival-adjusted potential source rate and duration.
        composition: Molecule counts by gas species for each formed event.
        gas: Read-only source inventory data.

    Returns:
        Immutable provisional gas demand and source diagnostics.

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
    counts = np.asarray(composition.molecule_counts, dtype=np.float64)
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
