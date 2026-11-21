"""Subsaturated homogeneous-particle water uptake for optical closure."""

import numpy as np

from particula.optics.state import mix_refractive_index


def _real_arrays(*values):
    if any(np.iscomplexobj(x) for x in values):
        raise ValueError("Growth inputs must be real")
    return np.broadcast_arrays(
        *(np.asarray(x, dtype=np.float64) for x in values)
    )


def subsaturated_growth(dry_radius, relative_humidity, kappa, kelvin_length):
    """Return the wet radius and water/dry volume ratio at equilibrium.

    Args:
        dry_radius: Positive dry radius [m].
        relative_humidity: Water saturation ratio in [0, 1).
        kappa: Nonnegative effective hygroscopicity.
        kelvin_length: Nonnegative radius-form Kelvin coefficient [m],
            determined by temperature, water density and surface tension.
            A zero length selects the flat-interface model.

    Returns:
        Wet radius [m] and dimensionless water/dry volume ratio.
        Inputs use NumPy broadcasting.

    The model follows the liquid branch of the kappa-Köhler relation.
    A prescribed dry solid population can use kappa zero. Phase selection and
    uptake kinetics are outside this equilibrium calculation.
    """
    dry, rh, k, kelvin = _real_arrays(
        dry_radius, relative_humidity, kappa, kelvin_length
    )
    if (
        any(np.any(~np.isfinite(x)) for x in (dry, rh, k, kelvin))
        or np.any(dry <= 0)
        or np.any(rh < 0)
        or np.any(rh >= 1)
        or np.any(k < 0)
        or np.any(kelvin < 0)
    ):
        raise ValueError("Unphysical subsaturated growth coordinates")

    # Solve w = kappa*RH/(exp(A/r_wet) - RH); the flat result brackets
    # the first root. Bracket logarithmically in 1+w at large growth.
    rhs = k * rh
    flat = rhs / (1 - rh)
    if not np.all(np.isfinite(flat)):
        raise ValueError("Subsaturated water uptake exceeds float64 range")
    bend = kelvin / dry
    upper = np.log1p(flat)
    lower = np.zeros_like(upper)
    # Bisection in log(1+w) resolves small uptake as well as large growth.
    # For finite RH<1, the first root is unique below the flat limit.
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        for _ in range(64):
            h = (lower + upper) / 2
            current = np.expm1(h)
            term = np.expm1(bend * np.exp(-h / 3)) + (1 - rh)
            needs_water = current < rhs / term
            lower = np.where(needs_water, h, lower)
            upper = np.where(needs_water, upper, h)
        water = np.where(kelvin == 0, flat, np.expm1((lower + upper) / 2))
        wet = dry * np.cbrt(1 + water)
    if not np.all(np.isfinite(wet)):
        raise ValueError("Wet radius exceeds float64 range")
    return wet, water


def hydrate_index(dry_index, water_index, water_to_dry_volume):
    """Mix the dry and water indices at a prescribed water/dry volume ratio."""
    if np.iscomplexobj(water_to_dry_volume):
        raise ValueError("Water/dry volume ratio must be real")
    dry, water, ratio = np.broadcast_arrays(
        np.asarray(dry_index, dtype=np.complex128),
        np.asarray(water_index, dtype=np.complex128),
        np.asarray(water_to_dry_volume, dtype=np.float64),
    )
    if np.any(~np.isfinite(ratio)) or np.any(ratio.real < 0):
        raise ValueError("Require finite nonnegative water/dry volume ratio")
    dry_fraction = 1 / (1 + ratio.real)
    water_fraction = ratio.real / (1 + ratio.real)
    return mix_refractive_index(
        np.stack((dry, water), axis=-1),
        np.stack((dry_fraction, water_fraction), axis=-1),
    )
