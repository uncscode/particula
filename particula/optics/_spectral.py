"""Shared coordinates for spectral radiation and its boundary conditions."""

import numpy as np


def finite_array(values, name):
    """Copy a real-valued physical field into a read-only float64 array.

    Args:
        values (ArrayLike): Array-like field. Its axes and units are preserved.
        name (str): Field name used in validation errors.

    Returns:
        (ndarray): Independent, read-only NumPy array with the input shape.

    Raises:
        ValueError: A value is complex, nonfinite or cannot convert to float.
    """
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be real")
    array = np.array(values, dtype=float, copy=True)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    array.setflags(write=False)
    return array


def wavelengths(values):
    """Validate the vacuum wavelength coordinate shared by shortwave fields.

    Args:
        values (ArrayLike): One-dimensional sequence of positive wavelengths
            [m].

    Returns:
        (ndarray): Read-only wavelength array [m] in strictly increasing order.

    Raises:
        ValueError: The input is empty, invalid or out of order.
    """
    wave = finite_array(values, "Wavelengths")
    if (
        wave.ndim != 1
        or not wave.size
        or np.any(wave <= 0)
        or np.any(np.diff(wave) <= 0)
    ):
        raise ValueError("Require increasing positive wavelengths in metres")
    return wave


def interfaces(values):
    """Validate the top-first vertical grid used by optical layers.

    Args:
        values (ArrayLike): One-dimensional sequence of interface altitudes
            [m].

    Returns:
        (ndarray): Read-only strictly descending altitudes [m]. Adjacent pairs
            bound homogeneous layers.

    Raises:
        ValueError: The input is invalid or has fewer than two interfaces.
    """
    height = finite_array(values, "Interface heights")
    if height.ndim != 1 or height.size < 2 or np.any(np.diff(height) >= 0):
        raise ValueError("Require strictly descending interface heights")
    return height


def spectral_values(values, wave, name):
    """Broadcast a finite spectral field onto a validated wavelength grid.

    Args:
        values (ArrayLike): Scalar or array broadcastable to the
            one-dimensional grid.
        wave (ndarray): Wavelength coordinate with the required output shape.
        name (str): Name used in validation errors.

    Returns:
        (ndarray): Read-only float64 values with the shape of wave and units of
            values.

    Raises:
        ValueError: The field is invalid or cannot broadcast to the grid.
    """
    return finite_array(np.broadcast_to(values, wave.shape), name)


def require_same_grid(left, right, name):
    """Require identical coordinate values before combining spectral fields.

    Args:
        left (ArrayLike | None): First coordinate, or None if absent.
        right (ArrayLike | None): Second coordinate in the same units and
            axis order.
        name (str): Coordinate name used in validation errors.

    Raises:
        ValueError: The shapes or values differ.
    """
    if not np.array_equal(left, right):
        raise ValueError(f"{name} must match")
