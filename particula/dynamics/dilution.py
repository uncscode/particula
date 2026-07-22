"""Validated, vectorized CPU calculations for chamber dilution.

This module uses chamber volume ``V`` [m³], inlet flow ``Q`` [m³/s], and
dilution coefficient ``alpha`` [s⁻¹], where ``alpha = Q / V``. For
particle-number or gas-mass concentration ``c`` [1/m³ or kg/m³], it evaluates
``dc/dt = -alpha * c`` and the exact finite update
``c_new = c * exp(-alpha * time_step)`` for elapsed time [s].

The helpers validate finite physical domains, use ordinary NumPy broadcasting,
and do not mutate caller-owned arrays. All-scalar inputs return a scalar;
inputs including an array return a broadcast-shape array. The concrete
``dilute_aerosol`` primitive intentionally mutates its ``Aerosol`` argument in
place. Neither it nor ``get_dilution_step`` is exported from
``particula.dynamics``, and this module provides no GPU behavior.
"""

from numbers import Number
from typing import TYPE_CHECKING

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
    result = np.asarray(input_flow_rate_array / volume_array)
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
    result = np.asarray(-coefficient_array * concentration_array)
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
    if not isinstance(value, Number) and not np.issubdtype(
        value_array.dtype, np.number
    ):
        raise TypeError(f"Argument '{argument}' must be numeric.")
    try:
        scalar = np.float64(value_array)
    except (TypeError, ValueError) as error:
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
    candidate_array = np.asarray(candidate, dtype=np.float64)
    source_array = np.asarray(source, dtype=np.float64)
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


def dilute_aerosol(
    aerosol: "Aerosol",
    coefficient: float | np.number,
    time_step: float | np.number,
) -> "Aerosol":
    """Dilute an aerosol's particle and gas concentrations in place.

    Applies ``c_new = c * exp(-coefficient * time_step)`` to physical
    particle-number concentration and, in order, the atmosphere's partitioning
    and gas-only concentrations. It preflights every candidate before writing.
    If an unexpected write fails, it restores concentrations already written
    from snapshots before reraising the original exception.

    Args:
        aerosol: Aerosol whose concentrations are updated in place.
        coefficient: Scalar chamber dilution coefficient [s⁻¹], finite and
            nonnegative.
        time_step: Scalar elapsed time [s], finite and nonnegative.

    Returns:
        The identical, mutated ``aerosol`` instance.

    Raises:
        TypeError: If ``coefficient`` or ``time_step`` is not numeric.
        ValueError: If either scalar is nonfinite, negative, or nonscalar, or
            a concentration candidate cannot be safely committed.
    """
    coefficient_scalar = _validate_nonnegative_scalar(
        coefficient,
        "coefficient",
    )
    time_step_scalar = _validate_nonnegative_scalar(time_step, "time_step")

    particle = aerosol.particles
    gas_containers = (
        aerosol.atmosphere.partitioning_species,
        aerosol.atmosphere.gas_only_species,
    )
    particle_source = particle.get_concentration()
    gas_sources = tuple(gas.get_concentration() for gas in gas_containers)

    particle_candidate = _preflight_concentration(
        get_dilution_step(
            coefficient_scalar,
            particle_source,
            time_step_scalar,
        ),
        particle_source,
        "particle concentration",
    )
    gas_candidates = tuple(
        _preflight_concentration(
            get_dilution_step(coefficient_scalar, source, time_step_scalar),
            source,
            "gas concentration",
        )
        for source in gas_sources
    )
    volume = np.asarray(particle.get_volume(), dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        stored_particle_candidate = particle_candidate * volume
    stored_particle_candidate = _preflight_concentration(
        stored_particle_candidate,
        particle_source,
        "stored particle concentration",
    )

    particle_snapshot = np.copy(np.asarray(particle.concentration))
    gas_snapshots = tuple(np.copy(np.asarray(source)) for source in gas_sources)
    written: list[tuple[object, NDArray[np.float64]]] = []
    try:
        written.append((particle, particle_snapshot))
        particle.concentration = stored_particle_candidate
        for gas, candidate, snapshot in zip(
            gas_containers,
            gas_candidates,
            gas_snapshots,
            strict=True,
        ):
            written.append((gas, snapshot))
            gas.set_concentration(candidate)
    except Exception as original_error:
        rollback_error: Exception | None = None
        for container, snapshot in reversed(written):
            try:
                if container is particle:
                    container.concentration = snapshot
                else:
                    container.set_concentration(snapshot)
            except Exception as error:  # pragma: no cover - diagnostic path
                rollback_error = error
        if rollback_error is not None:
            raise original_error from rollback_error
        raise
    return aerosol
