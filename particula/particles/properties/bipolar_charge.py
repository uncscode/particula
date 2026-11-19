"""Equilibrium charge fractions of submicron particles in bipolar air ions."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Rows correspond to signed charge -2 through +2, columns to powers 0–5.
_COEFFICIENTS = np.array(
    [
        [-26.3328, 35.9044, -21.4608, 7.0867, -1.3088, 0.1051],
        [-2.3197, 0.6175, 0.6201, -0.1105, -0.1260, 0.0297],
        [-0.0003, -0.1014, 0.3073, -0.3372, 0.1023, -0.0105],
        [-2.3484, 0.6044, 0.4800, 0.0013, -0.1553, 0.0320],
        [-44.4756, 79.3772, -62.8900, 26.4492, -5.7480, 0.5049],
    ]
)


def get_bipolar_charge_probability(
    particle_diameter: ArrayLike, charge: ArrayLike
) -> np.float64 | NDArray[np.float64]:
    """Estimate the fraction carrying a signed charge in bipolar equilibrium.

    Evaluate the Wiedensohler polynomial in log10(diameter / nm). Each fit
    is independent. A sum over selected states may differ from one. Sampling
    a population requires choosing the charge tails and normalizing it.

    Args:
        particle_diameter: Mobility diameter in metres, 1–1000 nm for
            neutral or singly charged particles and 20–1000 nm for ±2.
        charge: Signed number of elementary charges, one of -2, -1, 0, 1, 2.
            Diameter and charge follow NumPy broadcasting rules.

    Returns:
        Dimensionless fitted fraction for each diameter and charge pair.

    Raises:
        ValueError: A diameter or charge lies outside its fitted domain.

    References:
        Wiedensohler (1988), https://doi.org/10.1016/0021-8502(88)90278-9.
        Wiedensohler et al. (2012), Table A7,
        https://doi.org/10.5194/amt-5-657-2012.
    """
    diameter, state = np.broadcast_arrays(particle_diameter, charge)
    if state.dtype.kind not in "iuf" or not np.all(
        np.isin(state, [-2, -1, 0, 1, 2])
    ):
        raise ValueError("charge must be an integer between -2 and 2")
    lower = np.where(np.abs(state) == 2, 20e-9, 1e-9)
    if diameter.dtype.kind not in "iuf" or not np.all(
        (diameter >= lower) & (diameter <= 1e-6)
    ):
        raise ValueError("particle_diameter outside the fitted charge domain")
    coefficients = _COEFFICIENTS[state.astype(np.intp) + 2]
    log_diameter = np.log10(diameter.astype(np.float64) * 1e9)
    return 10.0 ** np.polynomial.polynomial.polyval(
        log_diameter, np.moveaxis(coefficients, -1, 0), tensor=False
    )
