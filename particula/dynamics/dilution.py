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
from typing import Any

import numpy as np
from numpy.typing import NDArray

from particula.aerosol import Aerosol
from particula.aerosol_validation import validate_aerosol_structure
from particula.util.validate_inputs import validate_inputs


def _preflight_native_dilution(
    aerosol: Aerosol,
    coefficient: np.float64,
    time_step: np.float64,
    *,
    writable: bool = True,
) -> dict[str, Any]:
    """Stage native concentration updates without changing caller storage.

    Every gas lane is treated equally. Raw resolved counts and PMF/PDF density
    storage receive the same decay; volume normalization is only for rates.
    Protected-field aliases reject even for zero-work calls.
    """
    if not isinstance(aerosol, Aerosol):
        raise TypeError("dilution requires a native Aerosol")
    particles, gas, environment = (
        aerosol.particles,
        aerosol.gas,
        aerosol.environment,
    )
    validate_aerosol_structure(particles, gas, environment)
    particles.validate_representation()
    if particles.n_boxes != 1:
        raise ValueError("CPU dilution requires exactly one box.")
    coefficient = _validate_nonnegative_scalar(coefficient, "coefficient")
    targets = (particles.concentration, gas.concentration)
    protected: tuple[np.ndarray, ...] = (
        particles.masses,
        particles.charge,
        particles.density,
        particles.volume,
        gas.molar_mass,
        gas.partitioning,
        environment.temperature,
        environment.pressure,
        environment.saturation_ratio,
    )
    if particles.radius_grid is not None:
        protected += (particles.radius_grid,)
    candidates = []
    for index, target in enumerate(targets):
        if writable and not target.flags.writeable:
            raise ValueError("native dilution destinations must be writable")
        if any(
            np.shares_memory(target, other)
            for other in (*targets[index + 1 :], *protected)
        ):
            raise ValueError(
                "native dilution destinations must not alias state"
            )
        _preflight_concentration(target, target, "concentration")
        candidates.append(
            _preflight_concentration(
                get_dilution_step(coefficient, target, time_step),
                target,
                "concentration",
            )
        )
    return {
        "targets": targets,
        "candidates": tuple(candidates),
        "snapshots": tuple(target.copy() for target in targets),
    }


def _commit_native_dilution(preflight: dict[str, Any]) -> None:
    """Publish concentrations, restoring writes on an unexpected error."""
    written = []
    try:
        for target, candidate, snapshot in zip(
            preflight["targets"],
            preflight["candidates"],
            preflight["snapshots"],
            strict=True,
        ):
            written.append((target, snapshot))
            np.copyto(target, candidate)
    except Exception as original_error:
        rollback_error = None
        for target, snapshot in reversed(written):
            try:
                target[...] = snapshot
            except Exception as error:
                rollback_error = error
        if rollback_error is not None:
            raise original_error from rollback_error
        raise


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
    if np.issubdtype(value_array.dtype, np.complexfloating):
        raise TypeError(f"Argument '{argument}' must be real-valued.")
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


def dilute_aerosol(
    aerosol: "Aerosol",
    coefficient: float | np.number,
    time_step: float | np.number,
) -> "Aerosol":
    """Dilute an aerosol's particle and gas concentrations in place.

    Applies ``c_new = c * exp(-coefficient * time_step)`` to physical
    particle concentration storage and every gas lane, including inert species.
    Before writing, it validates every supported
    source, particle-storage representation, volume, and candidate. Thus,
    preflight failures are atomic; an unexpected later write failure restores
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

    preflight = _preflight_native_dilution(
        aerosol,
        coefficient_scalar,
        time_step_scalar,
    )
    if coefficient_scalar == 0.0 or time_step_scalar == 0.0:
        return aerosol
    _commit_native_dilution(preflight)
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
        _preflight_native_dilution(
            aerosol, self.coefficient, np.float64(0), writable=False
        )
        concentration = aerosol.particles.concentration[0]
        if aerosol.particles.distribution_type == "particle_resolved":
            with np.errstate(over="ignore", invalid="ignore"):
                concentration = concentration / aerosol.particles.volume[0]
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
        _preflight_native_dilution(aerosol, self.coefficient, time_step)

    def step(
        self,
        aerosol: "Aerosol",
        time_step: float | np.number,
    ) -> "Aerosol":
        """Apply one validated, atomic in-place aerosol dilution step.

        Validates the concrete aerosol state and every candidate before its
        first write. A preflight failure leaves the aerosol unchanged; a later
        write failure restores previously written concentration backing state.

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
