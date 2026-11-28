"""Shared coordinates for spectral radiation and its boundary conditions."""

import numpy as np


def finite_array(values, name):
    """Copy finite real values into an immutable float64 array."""
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be real")
    array = np.array(values, dtype=float, copy=True)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    array.setflags(write=False)
    return array


def wavelengths(values):
    """Validate strictly increasing vacuum wavelengths in metres."""
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
    """Validate top-first layer interface altitudes in metres."""
    height = finite_array(values, "Interface heights")
    if height.ndim != 1 or height.size < 2 or np.any(np.diff(height) >= 0):
        raise ValueError("Require strictly descending interface heights")
    return height


def spectral_values(values, wave, name):
    """Broadcast a scalar or one value per wavelength to a frozen array."""
    return finite_array(np.broadcast_to(values, wave.shape), name)


def require_same_grid(left, right, name):
    """Require exact coordinate alignment before combining physical fields."""
    if not np.array_equal(left, right):
        raise ValueError(f"{name} must match")
