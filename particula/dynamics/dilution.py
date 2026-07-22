"""Validated, vectorized CPU calculations for chamber dilution.

This module uses chamber volume ``V`` [m³], inlet flow ``Q`` [m³/s], and
dilution coefficient ``alpha`` [s⁻¹], where ``alpha = Q / V``. For
particle-number or gas-mass concentration ``c`` [1/m³ or kg/m³], it evaluates
``dc/dt = -alpha * c`` and the exact finite update
``c_new = c * exp(-alpha * time_step)`` for elapsed time [s].

The helpers validate finite physical domains, use ordinary NumPy broadcasting,
and do not mutate caller-owned arrays. All-scalar inputs return a scalar;
inputs including an array return a broadcast-shape array. This module neither
mutates containers nor provides GPU behavior. ``get_dilution_step`` is a
module-only helper, deliberately not exported from ``particula.dynamics``.
"""

import numpy as np
from numpy.typing import NDArray

from particula.util.validate_inputs import validate_inputs


def _return_scalar_if_appropriate(
    result: NDArray[np.float64],
    *operands: float | NDArray[np.float64],
) -> float | NDArray[np.float64]:
    """Return a scalar result when every input is scalar."""
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
