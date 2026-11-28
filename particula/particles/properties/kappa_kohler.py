"""Equilibrium activation of an insoluble or constant-κ dry particle."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _log_activity_penalty(t, log_kappa):
    # log(expm1(t)) remains finite even when exp(t) would overflow.
    log_excess = t + np.log(-np.expm1(-t))
    return np.logaddexp(0.0, log_kappa - log_excess)


def _peak(left, right, log_kappa, log_s):
    """Find a maximum on a bracket where the slope changes from + to −."""
    # Logarithmic bisection resolves roots far smaller than machine epsilon.
    for _ in range(60):
        middle = (left + right) * 0.5
        t = np.exp(middle)
        wet = np.log(-np.expm1(-t))
        log_slope = np.log(3.0) + wet + np.logaddexp(0.0, t + wet - log_kappa)
        increasing = log_slope < np.log(t + log_s)
        left = np.where(increasing, middle, left)
        right = np.where(increasing, right, middle)
    t = np.exp((left + right) * 0.5)
    return np.exp(-_log_activity_penalty(t, log_kappa) / 3) / (log_s + t)


def _threshold_scale(kappa, log_s):
    """Return critical diameter per unit Kelvin length for positive κ."""
    log_kappa = np.log(kappa)
    # For finite binary64 inputs, F(exp(-420)) < 0 and F(1024) > 0.
    low, high = -420.0, np.log(1024.0)
    split = kappa > 18.0 + np.sqrt(288.0)
    right = np.full_like(kappa, high)
    if np.any(split):
        k = kappa[split]
        disc = np.sqrt(np.maximum(0, 1 - (36 / k) * (1 - 1 / k)))
        right[split] = np.log(np.log(6 * (1 - 1 / k) / (1 + disc)))
    result = _peak(low, right, log_kappa, log_s)
    if np.any(split):
        # Each outer interval contains at most one maximum. If the root
        # lies outside it, bisection returns its endpoint, a valid lower
        # bound on the global maximum of the diameter curve.
        left = np.log(np.log(k) + np.log((1 + disc) / 6))
        result[split] = np.maximum(
            result[split], _peak(left, high, log_kappa[split], log_s[split])
        )
    return result


def get_kappa_kohler_critical_dry_diameter(
    kappa: ArrayLike, saturation_ratio: ArrayLike, kelvin_diameter: ArrayLike
) -> float | NDArray[np.float64]:
    """Invert the global maximum of the equilibrium κ-Köhler curve.

    Args:
        kappa: Dimensionless nonnegative hygroscopicity, constant during growth.
        saturation_ratio: Water vapor pressure ratio over flat water (> 1).
        kelvin_diameter: Positive diameter-basis Kelvin length in m, twice
            the Kelvin radius. Surface tension stays constant during growth.

    Returns:
        Minimum dry diameter in m, as a float or a float64 array with the
        NumPy broadcast shape. At zero κ the result is A / log(S).

    Raises:
        ValueError: For nonfinite, incompatible or out-of-domain inputs.
        FloatingPointError: If the result cannot be represented in float64.

    References:
        Petters and Kreidenweis (2007), Eq. 6, doi:10.5194/acp-7-1961-2007.
    """
    inputs = []
    for value, lower, inclusive in (
        (kappa, 0, True),
        (saturation_ratio, 1, False),
        (kelvin_diameter, 0, False),
    ):
        if np.iscomplexobj(value):
            raise ValueError(
                "Expected real κ, saturation ratio and Kelvin length"
            )
        array = np.asarray(value, dtype=np.float64)
        valid = array >= lower if inclusive else array > lower
        if not np.all(np.isfinite(array) & valid):
            raise ValueError("Require finite κ >= 0, S > 1 and A > 0")
        inputs.append(array)
    k, ratio, length = np.broadcast_arrays(*inputs)
    log_s = np.log(ratio)
    scale = np.array(1.0 / log_s)
    positive = k > 0
    if np.any(positive):
        scale[positive] = _threshold_scale(k[positive], log_s[positive])
    with np.errstate(over="ignore", under="ignore"):
        diameter = length * scale
    if not np.all(np.isfinite(diameter) & (diameter > 0)):
        raise FloatingPointError("Critical dry diameter outside float64 range")
    return float(diameter) if diameter.ndim == 0 else diameter
