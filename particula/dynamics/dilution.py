"""Validated, vectorized CPU calculations for chamber dilution.

This module uses chamber volume ``V`` [m³], inlet flow ``Q`` [m³/s], and
dilution coefficient ``alpha`` [s⁻¹], where ``alpha = Q / V``. For
particle-number or gas-mass concentration ``c`` [1/m³ or kg/m³], it evaluates
``dc/dt = -alpha * c`` and the exact finite update
``c_new = c * exp(-alpha * time_step)`` for elapsed time [s].

The helpers validate finite physical domains, use ordinary NumPy broadcasting,
and do not mutate caller-owned arrays. All-scalar inputs return a scalar;
inputs including an array return a broadcast-shape array. The concrete
``dilute_aerosol`` intentionally mutates its ``Aerosol`` argument in place.
It completely validates supported particle and gas concentration state and all
updates before its first write, then restores prior backing state if a later
write fails. ``DilutionStrategy`` is the supported public strategy and
delegates this atomic update to the primitive. The low-level
``dilute_aerosol`` and ``get_dilution_step`` helpers remain concrete-module
APIs and are not exported from ``particula.dynamics``. This module provides no
GPU behavior.
"""

from numbers import Number
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from particula.util.validate_inputs import validate_inputs

if TYPE_CHECKING:
    from particula.aerosol import Aerosol


def _return_scalar_if_appropriate(
    result: NDArray[np.float64],
    *operands: float | NDArray[np.float64],
) -> float | NDArray[np.float64]:
    """Return a scalar result only when every operand is scalar.

    Args:
        result: NumPy result array from a broadcast calculation.
        *operands: Original calculation operands used to determine scalar mode.

    Returns:
        The scalar value in ``result`` if all operands are scalars; otherwise,
        ``result`` unchanged.
    """
    if all(np.isscalar(operand) for operand in operands):
        return result.item()
    return result


def _require_finite_result(
    result: NDArray[np.float64],
    name: str,
) -> NDArray[np.float64]:
    """Raise when a computed dilution quantity is nonfinite.

    Args:
        result: Computed dilution quantity.
        name: Quantity name included in the validation error.

    Returns:
        The validated result.

    Raises:
        ValueError: If any computed value is nonfinite.
    """
    if not np.all(np.isfinite(result)):
        raise ValueError(f"Computed {name} must be finite.")
    return result


@validate_inputs({"volume": "positive", "input_flow_rate": "nonnegative"})
def get_volume_dilution_coefficient(
    volume: float | NDArray[np.float64],
    input_flow_rate: float | NDArray[np.float64],
) -> float | NDArray[np.float64]:
    """Calculate the volume dilution coefficient ``alpha = Q / V``.

    Finite inputs are validated and broadcast with NumPy before division. The
    function does not mutate inputs; all-scalar inputs return a scalar, and any
    array input returns an array with the broadcast shape.

    Args:
        volume: Chamber volume ``V`` [m³], strictly positive and finite.
        input_flow_rate: Inlet flow rate ``Q`` [m³/s], nonnegative and finite.

    Returns:
        Dilution coefficient [s⁻¹] as a scalar for scalar inputs, otherwise an
        array with the broadcast shape.

    Raises:
        TypeError: If an operand is ``None`` or is not numeric.
        ValueError: If an operand is outside its finite physical domain or the
            operand shapes cannot be broadcast.
    """
    if volume is None:
        raise TypeError("Argument 'volume' must not be None.")
    if input_flow_rate is None:
        raise TypeError("Argument 'input_flow_rate' must not be None.")

    volume_array, input_flow_rate_array = np.broadcast_arrays(
        np.asarray(volume, dtype=np.float64),
        np.asarray(input_flow_rate, dtype=np.float64),
    )
    try:
        with np.errstate(over="raise", invalid="raise"):
            result = np.asarray(input_flow_rate_array / volume_array)
    except FloatingPointError as error:
        raise ValueError(
            "Computed dilution coefficient must be finite."
        ) from error
    result = _require_finite_result(result, "dilution coefficient")
    return _return_scalar_if_appropriate(result, volume, input_flow_rate)


@validate_inputs({"coefficient": "nonnegative", "concentration": "nonnegative"})
def get_dilution_rate(
    coefficient: float | NDArray[np.float64],
    concentration: float | NDArray[np.float64],
) -> float | NDArray[np.float64]:
    """Calculate the instantaneous dilution rate ``dc/dt = -alpha * c``.

    Finite inputs are validated and broadcast with NumPy before multiplication.
    The function does not mutate inputs; all-scalar inputs return a scalar, and
    any array input returns an array with the broadcast shape.

    Args:
        coefficient: Dilution coefficient ``alpha`` [s⁻¹], nonnegative and
            finite.
        concentration: Particle-number or gas-mass concentration [1/m³ or
            kg/m³], nonnegative and finite.

    Returns:
        Concentration rate of change [1/(m³ s) or kg/(m³ s)] as a scalar for
        scalar inputs, otherwise an array with the broadcast shape.

    Raises:
        TypeError: If an operand is ``None`` or is not numeric.
        ValueError: If an operand is outside its finite physical domain or the
            operand shapes cannot be broadcast.
    """
    if coefficient is None:
        raise TypeError("Argument 'coefficient' must not be None.")
    if concentration is None:
        raise TypeError("Argument 'concentration' must not be None.")

    coefficient_array, concentration_array = np.broadcast_arrays(
        np.asarray(coefficient, dtype=np.float64),
        np.asarray(concentration, dtype=np.float64),
    )
    try:
        with np.errstate(over="raise", invalid="raise"):
            result = np.asarray(-coefficient_array * concentration_array)
    except FloatingPointError as error:
        raise ValueError("Computed dilution rate must be finite.") from error
    result = _require_finite_result(result, "dilution rate")
    return _return_scalar_if_appropriate(result, coefficient, concentration)


@validate_inputs(
    {
        "coefficient": "nonnegative",
        "concentration": "nonnegative",
        "time_step": "nonnegative",
    }
)
def get_dilution_step(
    coefficient: float | NDArray[np.float64],
    concentration: float | NDArray[np.float64],
    time_step: float | NDArray[np.float64],
) -> float | NDArray[np.float64]:
    """Calculate the exact finite-step concentration, rather than a delta.

    This evaluates ``c_new = c * exp(-alpha * time_step)``, not an Euler
    approximation, so zero coefficient, concentration, or elapsed time is an
    exact no-op. Finite inputs are validated and broadcast with NumPy without
    mutating them. All-scalar inputs return a scalar, and any array input
    returns an array with the broadcast shape. This helper is deliberately
    module-only and is not exported from ``particula.dynamics``.

    Args:
        coefficient: Dilution coefficient ``alpha`` [s⁻¹], nonnegative and
            finite.
        concentration: Initial concentration [1/m³ or kg/m³], nonnegative and
            finite.
        time_step: Elapsed time [s], nonnegative and finite.

    Returns:
        Updated concentration ``c * exp(-alpha * time_step)`` as a scalar for
        scalar inputs, otherwise an array with the broadcast shape.

    Raises:
        TypeError: If an operand is ``None`` or is not numeric.
        ValueError: If an operand is outside its finite physical domain or the
            operand shapes cannot be broadcast.
    """
    if coefficient is None:
        raise TypeError("Argument 'coefficient' must not be None.")
    if concentration is None:
        raise TypeError("Argument 'concentration' must not be None.")
    if time_step is None:
        raise TypeError("Argument 'time_step' must not be None.")

    (
        coefficient_array,
        concentration_array,
        time_step_array,
    ) = np.broadcast_arrays(
        np.asarray(coefficient, dtype=np.float64),
        np.asarray(concentration, dtype=np.float64),
        np.asarray(time_step, dtype=np.float64),
    )
    with np.errstate(over="ignore", under="ignore"):
        result = np.asarray(
            concentration_array * np.exp(-coefficient_array * time_step_array)
        )
    result = _require_finite_result(result, "dilution step")
    return _return_scalar_if_appropriate(
        result,
        coefficient,
        concentration,
        time_step,
    )


def _validate_nonnegative_scalar(value: object, argument: str) -> np.float64:
    """Validate and convert a finite, nonnegative numeric scalar.

    Zero-dimensional NumPy numeric arrays are accepted; arrays with other
    shapes are rejected so container dilution always has one shared decay.

    Args:
        value: Python or NumPy numeric scalar to validate.
        argument: Argument name included in validation error messages.

    Returns:
        Validated scalar converted to ``np.float64``.

    Raises:
        TypeError: If ``value`` is ``None`` or is not numeric.
        ValueError: If ``value`` is nonscalar, nonfinite, or negative.
    """
    if value is None:
        raise TypeError(f"Argument '{argument}' must be numeric, not None.")

    value_array = np.asarray(value)
    if value_array.ndim != 0:
        raise ValueError(
            f"Argument '{argument}' must be a finite nonnegative scalar."
        )
    if isinstance(value, (bool, np.bool_)) or np.issubdtype(
        value_array.dtype, np.bool_
    ):
        raise TypeError(f"Argument '{argument}' must be numeric, not boolean.")
    if not isinstance(value, Number) and not np.issubdtype(
        value_array.dtype, np.number
    ):
        raise TypeError(f"Argument '{argument}' must be numeric.")
    try:
        scalar = np.float64(value_array)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"Argument '{argument}' must be numeric.") from error
    if not np.isfinite(scalar):
        raise ValueError(f"Argument '{argument}' must be finite.")
    if scalar < 0.0:
        raise ValueError(f"Argument '{argument}' must be nonnegative.")
    return scalar


def _preflight_concentration(
    candidate: object,
    source: object,
    name: str,
) -> NDArray[np.float64]:
    """Validate a concentration candidate before an in-place commit.

    Args:
        candidate: Proposed updated concentration.
        source: Original concentration that determines the required shape.
        name: Concentration name used in a validation error message.

    Returns:
        The finite, nonnegative candidate as a float64 array.

    Raises:
        ValueError: If the candidate is nonfinite, negative, or has a shape
            different from ``source``.
    """
    try:
        candidate_array = np.asarray(candidate, dtype=np.float64)
        source_array = np.asarray(source, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be numeric.") from error
    if (
        candidate_array.shape != source_array.shape
        or not np.all(np.isfinite(candidate_array))
        or np.any(candidate_array < 0.0)
    ):
        raise ValueError(
            f"{name} candidate must be finite, nonnegative, and match "
            "the source shape."
        )
    return candidate_array


def _preflight_dilution_aerosol(
    aerosol: "Aerosol",
    coefficient: np.float64,
    time_step: np.float64,
) -> dict[str, Any]:
    """Validate supported dilution state and candidates before any write.

    This concrete-path preflight validates physical particle and gas sources,
    particle storage, and particle volume; it also computes shape-compatible
    finite candidates and captures rollback state. Therefore a malformed
    supported aerosol fails without changing a concentration, including for a
    zero elapsed time or dilution coefficient.

    Args:
        aerosol: Aerosol whose supported concentration state is inspected.
        coefficient: Validated nonnegative dilution coefficient [s⁻¹].
        time_step: Validated nonnegative elapsed time [s].

    Returns:
        Sources, candidates, and rollback snapshots for a subsequent atomic
        commit.

    Raises:
        TypeError: If the aerosol does not provide the supported concentration
            protocol or a required concentration or storage value is not
            numeric.
        ValueError: If a source, particle volume, stored concentration, or
            candidate is invalid or has an incompatible shape.
    """
    try:
        particle = aerosol.particles
        gas_containers = (
            aerosol.atmosphere.partitioning_species,
            aerosol.atmosphere.gas_only_species,
        )
        particle_storage = particle.concentration
        particle_volume = particle.get_volume()
    except AttributeError as error:
        raise TypeError(
            "Aerosol dilution requires supported concentration state."
        ) from error
    try:
        particle_source = particle.get_concentration()
    except (AttributeError, TypeError, ValueError) as error:
        raise TypeError("particle concentration must be numeric.") from error
    gas_sources: list[object] = []
    for gas, name in zip(
        gas_containers,
        ("partitioning gas concentration", "gas-only concentration"),
        strict=True,
    ):
        try:
            gas_sources.append(gas.get_concentration())
        except (AttributeError, TypeError, ValueError) as error:
            raise TypeError(f"{name} must be numeric.") from error

    particle_source_array = _preflight_concentration(
        particle_source, particle_source, "particle concentration"
    )
    partitioning_source = _preflight_concentration(
        gas_sources[0], gas_sources[0], "partitioning gas concentration"
    )
    gas_only_source = _preflight_concentration(
        gas_sources[1], gas_sources[1], "gas-only concentration"
    )
    particle_storage_array = _preflight_concentration(
        particle_storage,
        particle_storage,
        "particle concentration storage",
    )
    try:
        volume = np.asarray(particle_volume, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError("particle volume must be numeric.") from error
    if volume.ndim != 0 or not np.isfinite(volume) or volume <= 0.0:
        raise ValueError("particle volume must be a finite positive scalar.")

    particle_candidate = _preflight_concentration(
        get_dilution_step(coefficient, particle_source_array, time_step),
        particle_source_array,
        "particle concentration",
    )
    partitioning_candidate = _preflight_concentration(
        get_dilution_step(coefficient, partitioning_source, time_step),
        partitioning_source,
        "partitioning gas concentration",
    )
    gas_only_candidate = _preflight_concentration(
        get_dilution_step(coefficient, gas_only_source, time_step),
        gas_only_source,
        "gas-only concentration",
    )
    with np.errstate(over="ignore", invalid="ignore"):
        stored_particle_candidate = particle_candidate * volume
    stored_particle_candidate = _preflight_concentration(
        stored_particle_candidate,
        particle_storage_array,
        "stored particle concentration",
    )

    try:
        gas_snapshots = tuple(
            _snapshot_gas_backing_state(gas) for gas in gas_containers
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise TypeError(
            "Aerosol dilution requires supported gas concentration storage."
        ) from error
    return {
        "particle": particle,
        "gas_containers": gas_containers,
        "particle_candidate": stored_particle_candidate,
        "gas_candidates": (partitioning_candidate, gas_only_candidate),
        "particle_snapshot": _snapshot_particle_backing_state(particle),
        "gas_snapshots": gas_snapshots,
    }


def _snapshot_gas_backing_state(
    gas: Any,
) -> tuple[Any, NDArray[np.float64], Any | None]:
    """Capture direct gas backing state for setter-independent rollback.

    Args:
        gas: Gas container providing a ``data.concentration`` backing array.

    Returns:
        Backing data object, copied concentration, and concentration-mode state.
    """
    data = gas.data
    return (
        data,
        np.copy(np.asarray(data.concentration)),
        getattr(gas, "_single_species_concentration_mode", None),
    )


def _snapshot_particle_backing_state(
    particle: Any,
) -> tuple[Any, NDArray[np.float64], Any | None]:
    """Capture direct particle backing state for setter-independent rollback.

    Args:
        particle: Particle container providing a ``data.concentration`` backing
            array.

    Returns:
        Backing data object, copied concentration, and private data state.
    """
    data = particle.data
    return (
        data,
        np.copy(np.asarray(data.concentration)),
        getattr(particle, "_data", None),
    )


def _restore_particle_backing_state(
    particle: Any,
    snapshot: tuple[Any, NDArray[np.float64], Any | None],
) -> None:
    """Restore particle backing state without invoking its public setter.

    Args:
        particle: Particle container whose state should be restored.
        snapshot: Backing state captured before commit.
    """
    data, concentration, private_data = snapshot
    data.concentration[...] = concentration
    if hasattr(particle, "_data"):
        particle._data = private_data if private_data is not None else data


def _restore_gas_backing_state(
    gas: Any,
    snapshot: tuple[Any, NDArray[np.float64], Any | None],
) -> None:
    """Restore gas backing state without invoking its public setter.

    Args:
        gas: Gas container whose state should be restored.
        snapshot: Backing state captured before commit.
    """
    data, concentration, concentration_mode = snapshot
    data.concentration[...] = concentration
    if hasattr(gas, "_data"):
        gas._data = data
    if hasattr(gas, "_single_species_concentration_mode"):
        gas._single_species_concentration_mode = concentration_mode


def dilute_aerosol(
    aerosol: "Aerosol",
    coefficient: float | np.number,
    time_step: float | np.number,
) -> "Aerosol":
    """Dilute an aerosol's particle and gas concentrations in place.

    Applies ``c_new = c * exp(-coefficient * time_step)`` to physical
    particle-number concentration and, in order, the atmosphere's partitioning
    and gas-only concentrations. Before writing, it validates every supported
    source, particle-storage representation, volume, and candidate. Thus,
    preflight failures are atomic; an unexpected later setter failure restores
    concentrations already written from snapshots before reraising the
    original exception. This low-level helper is deliberately concrete-module
    only; construct :class:`DilutionStrategy` for the public API.

    Args:
        aerosol: Aerosol whose concentrations are updated in place.
        coefficient: Scalar chamber dilution coefficient [s⁻¹], finite and
            nonnegative.
        time_step: Scalar elapsed time [s], finite and nonnegative.

    Returns:
        The identical, mutated ``aerosol`` instance.

    Raises:
        TypeError: If a scalar or required supported aerosol concentration or
            storage value is not numeric.
        ValueError: If either scalar is nonfinite, negative, or nonscalar, or
            supported aerosol state or a concentration candidate cannot be
            safely committed.
    """
    coefficient_scalar = _validate_nonnegative_scalar(
        coefficient,
        "coefficient",
    )
    time_step_scalar = _validate_nonnegative_scalar(time_step, "time_step")

    preflight = _preflight_dilution_aerosol(
        aerosol,
        coefficient_scalar,
        time_step_scalar,
    )
    if coefficient_scalar == 0.0 or time_step_scalar == 0.0:
        return aerosol
    particle = preflight["particle"]
    gas_containers = preflight["gas_containers"]
    stored_particle_candidate = preflight["particle_candidate"]
    gas_candidates = preflight["gas_candidates"]
    particle_snapshot = preflight["particle_snapshot"]
    gas_snapshots = preflight["gas_snapshots"]
    written: list[Any] = []
    try:
        written.append(particle)
        particle.concentration = stored_particle_candidate
        for gas, candidate in zip(
            gas_containers,
            gas_candidates,
            strict=True,
        ):
            written.append(gas)
            gas.set_concentration(candidate)
    except Exception as original_error:
        rollback_error: Exception | None = None
        for container in reversed(written):
            try:
                if container is particle:
                    _restore_particle_backing_state(
                        container, particle_snapshot
                    )
                else:
                    gas_index = gas_containers.index(container)
                    _restore_gas_backing_state(
                        container,
                        gas_snapshots[gas_index],
                    )
            except Exception as error:  # pragma: no cover - diagnostic path
                rollback_error = error
        if rollback_error is not None:
            raise original_error from rollback_error
        raise
    return aerosol


class DilutionStrategy:
    """Apply a shared, validated chamber dilution coefficient to an aerosol.

    The strategy reports particle-number rates and delegates atomic particle
    and gas concentration updates to the concrete dilution primitive. A step
    validates its elapsed time and the supported aerosol state before the first
    concentration write; malformed state therefore leaves it unchanged. This
    strategy is exported from :mod:`particula.dynamics`.

    Args:
        coefficient: Scalar chamber dilution coefficient [s⁻¹], finite and
            nonnegative.
    """

    def __init__(self, coefficient: float | np.number):
        """Initialize the strategy with a validated dilution coefficient.

        Args:
            coefficient: Scalar chamber dilution coefficient [s⁻¹], finite and
                nonnegative.
        """
        self.coefficient = _validate_nonnegative_scalar(
            coefficient, "coefficient"
        )

    def rate(self, aerosol: "Aerosol") -> float | NDArray[np.float64]:
        """Calculate the particle-number dilution rate.

        Args:
            aerosol: Aerosol providing physical particle-number concentration.

        Returns:
            Particle-number concentration rate [1/(m³ s)] with the scalar or
            array shape of the physical particle-number concentration.
        """
        concentration = aerosol.particles.get_concentration()
        return get_dilution_rate(self.coefficient, concentration)

    def _preflight(self, aerosol: "Aerosol", time_step: np.float64) -> None:
        """Validate supported aerosol state before runnable substeps.

        Args:
            aerosol: Aerosol whose dilution state is validated without writes.
            time_step: Validated total elapsed time [s].

        Raises:
            TypeError: If required supported aerosol state is missing or not
                numeric.
            ValueError: If supported aerosol state cannot be safely diluted.
        """
        _preflight_dilution_aerosol(aerosol, self.coefficient, time_step)

    def step(
        self,
        aerosol: "Aerosol",
        time_step: float | np.number,
    ) -> "Aerosol":
        """Apply one validated, atomic in-place aerosol dilution step.

        Validates the concrete aerosol state and every candidate before its
        first write. A preflight failure leaves the aerosol unchanged; a later
        setter failure restores previously written concentration backing state.

        Args:
            aerosol: Aerosol whose particle and gas concentrations are diluted.
            time_step: Elapsed time [s], finite, nonnegative, and scalar.

        Returns:
            The identical, mutated aerosol instance.

        Raises:
            TypeError: If ``time_step`` is not numeric.
            ValueError: If ``time_step`` is nonfinite, negative, or nonscalar,
                or supported aerosol state cannot be safely committed.
        """
        return dilute_aerosol(aerosol, self.coefficient, time_step)
