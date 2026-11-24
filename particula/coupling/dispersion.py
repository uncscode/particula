"""Flat, reflecting-ground reference solutions for passive aerosol mass.

The steady plume uses Briggs open-country spreads (PNNL-20990, Table A-1).
The Gaussian puff solves constant-diffusivity advection and diffusion. A
uniform ellipsoid provides a finite-width alternative with the same free-air
second moments. All three assume uniform horizontal wind and a ground image.
"""

import numpy as np

# Horizontal slope, vertical slope, vertical curvature (1/m), exponent.
_BRIGGS = {
    "A": (0.22, 0.20, 0.0, 0.0),
    "B": (0.16, 0.12, 0.0, 0.0),
    "C": (0.11, 0.08, 0.0002, -0.5),
    "D": (0.08, 0.06, 0.0015, -0.5),
    "E": (0.06, 0.03, 0.0003, -1.0),
    "F": (0.04, 0.016, 0.0003, -1.0),
}


def _coordinates(x, y, z, height):
    coords = np.broadcast_arrays(
        *[np.asarray(a, dtype=float) for a in (x, y, z)]
    )
    if not all(np.all(np.isfinite(a)) for a in coords):
        raise ValueError("coordinates must be finite")
    if not np.isfinite(height) or height < 0 or np.any(coords[2] < 0):
        raise ValueError("heights above flat ground must be nonnegative")
    return coords


def _normal(distance, sigma):
    return np.exp(-0.5 * (distance / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)


def gaussian_plume(x, y, z, *, rate, wind, height, stability):
    """Return kg/m³ at downwind x, crosswind y and height z (metres).

    rate is the continuous source (kg/s), wind the downwind speed (m/s),
    height the effective source height (m), and stability the Pasquill class
    A–F. The point-source plane and upwind half-space return zero.
    """
    x, y, z = _coordinates(x, y, z, height)
    if not np.isfinite([rate, wind]).all() or rate < 0 or wind <= 0:
        raise ValueError("require nonnegative rate and positive wind")
    if stability not in _BRIGGS:
        raise ValueError("stability must be A through F")
    ay, az, bz, exponent = _BRIGGS[stability]
    downwind = x > 0
    distance = x[downwind]
    sy = ay * distance / np.sqrt(1 + 0.0001 * distance)
    sz = az * distance * (1 + bz * distance) ** exponent
    result = np.zeros_like(x)
    result[downwind] = (
        rate
        / wind
        * _normal(y[downwind], sy)
        * (
            _normal(z[downwind] - height, sz)
            + _normal(z[downwind] + height, sz)
        )
    )
    return result


def puff_concentration(
    x, y, z, *, mass, age, wind, height, diffusivity, shape="gaussian"
):
    """Return kg/m³ from one instantaneous release at (0, 0, height) in m.

    mass is kg, age is seconds after release, wind is signed x velocity
    (m/s), and diffusivity gives positive x, y, z diffusivities (m²/s).
    A top_hat has uniform density inside an ellipsoid of semi-axes
    sqrt(5) times the Gaussian widths, so both have variance 2 K t in
    free air. Ground reflection folds the vertical distribution.
    """
    x, y, z = _coordinates(x, y, z, height)
    k = np.asarray(diffusivity, dtype=float)
    if k.shape != (3,) or not np.isfinite(k).all() or np.any(k <= 0):
        raise ValueError("give three positive finite diffusivities")
    if not np.isfinite([mass, age, wind]).all() or mass < 0 or age <= 0:
        raise ValueError(
            "require nonnegative mass, positive age and finite wind"
        )
    sx, sy, sz = np.sqrt(2 * k * age)
    if shape == "top_hat":
        axes = np.sqrt(5.0) * np.array([sx, sy, sz])
        horizontal = ((x - wind * age) / axes[0]) ** 2 + (y / axes[1]) ** 2
        direct = horizontal + ((z - height) / axes[2]) ** 2 <= 1
        image = horizontal + ((z + height) / axes[2]) ** 2 <= 1
        volume = (4 * np.pi / 3) * np.prod(axes)
        return (mass / volume) * (direct.astype(float) + image)
    if shape != "gaussian":
        raise ValueError("shape must be gaussian or top_hat")
    return (
        mass
        * _normal(x - wind * age, sx)
        * _normal(y, sy)
        * (_normal(z - height, sz) + _normal(z + height, sz))
    )
