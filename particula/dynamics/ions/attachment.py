"""LF2013 air-ion attachment at 298.15 K and 101325 Pa.

Coefficients describe conductive particles and are in m^3/s. Polynomial fits
are valid only over the per-charge radius intervals in the published tables.
They omit fluxes below 1e-15 m^3/s; omitted values are not extrapolated.

Reference: Lopez-Yglesias and Flagan (2013), Aerosol Science and Technology,
47, 688–704, https://doi.org/10.1080/02786826.2013.783684.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np

from particula.dynamics.ions.attachment_table import centered_attachment_tables

ATTACHMENT_CUTOFF = 1e-15


@lru_cache(maxsize=1)
def _tables():
    """Load immutable published coefficients, preserving decimal precision."""
    directory = (
        Path(__file__).resolve().parents[2] / "util" / "lf2013_coagulation"
    )
    tables = tuple(
        np.loadtxt(directory / name, skiprows=1, dtype=np.longdouble)
        for name in ("S1.txt", "S2.txt")
    )
    for table in tables:
        table.flags.writeable = False
    return tables


def attachment_coefficient(radius, charge, polarity, *, omitted="reject"):
    """Return LF2013 air-ion attachment rates in m^3/s.

    Args:
        radius: Aerosol radius in m, scalar or broadcastable array.
        charge: Integer aerosol charge in elementary charges, from -100 to 100.
        polarity: Free ion polarity, exactly -1 or +1.
        omitted: ``reject`` (default), ``lower`` or ``upper``. Bounds apply
            only to repulsive rates omitted below the table's lower radius
            at radii >= 0.2 nm. Lower returns zero, upper returns 1e-15 m^3/s.
            These are uncertainty bounds, not fitted rate estimates.

    Returns:
        Broadcast NumPy array of finite positive rates (or zero lower bounds).

    Raises:
        ValueError: If input is outside the supported fit/bound domain.
    """
    if omitted not in ("reject", "lower", "upper"):
        raise ValueError("omitted must be reject, lower or upper")
    radius, charge, polarity = np.broadcast_arrays(
        np.asarray(radius, dtype=float),
        np.asarray(charge, dtype=float),
        np.asarray(polarity, dtype=float),
    )
    if not np.all(np.isfinite(radius) & (radius >= 2e-10) & (radius <= 1e-5)):
        raise ValueError(
            "LF2013 radius must be within [0.2 nm, 10 micrometers]"
        )
    if not np.all(
        np.isfinite(charge)
        & (charge == np.floor(charge))
        & (np.abs(charge) <= 100)
    ):
        raise ValueError(
            "LF2013 particle charge must be an integer in [-100, 100]"
        )
    if not np.all((polarity == -1) | (polarity == 1)):
        raise ValueError("ion polarity must be -1 or +1")
    negative, positive = centered_attachment_tables()
    rows = np.where(
        (polarity < 0)[..., None],
        negative[charge.astype(int) + 100],
        positive[charge.astype(int) + 100],
    )
    below = radius < np.asarray(rows[..., -2], dtype=float)
    above = radius > np.asarray(rows[..., -1], dtype=float)
    if np.any(above) or np.any(below & (charge * polarity <= 0)):
        raise ValueError("LF2013 input outside the published fit domain")
    if omitted == "reject" and np.any(below):
        raise ValueError("LF2013 repulsive rate omitted below published cutoff")
    # The centered basis removes cancellation without fitting a new model.
    # Evaluate only valid lanes; bounds must never evaluate an extrapolation.
    safe_radius = np.maximum(radius, rows[..., -2])
    x = (np.log10(safe_radius) - rows[..., -4]) / rows[..., -3]
    exponent = rows[..., 23].copy()
    for index in range(22, -1, -1):
        exponent = exponent * x + rows[..., index]
    result = np.asarray(10**exponent, dtype=float)
    if not np.all(np.isfinite(result) & (result > 0)):
        raise ValueError("LF2013 fit produced invalid attachment rates")
    return np.where(
        below, 0.0 if omitted == "lower" else ATTACHMENT_CUTOFF, result
    )
