"""Tests for mixing state index calculation."""

import numpy as np
import pytest

from particula.particles.properties.mixing_state_index import (
    get_mixing_state_index,
)

# --------------------------------------------------------------------------- #
# Simple “sanity‑check” cases
# --------------------------------------------------------------------------- #


def test_internal_mixture_returns_one():
    """All particles have identical composition ⇒ χ should be 1."""
    masses = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    chi = get_mixing_state_index(masses)
    assert pytest.approx(chi, rel=1e-12) == 1.0


def test_external_mixture_returns_zero():
    """Each particle contains only one species ⇒ χ should be 0."""
    masses = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]])
    chi = get_mixing_state_index(masses)
    assert pytest.approx(chi, rel=1e-12) == 0.0


def test_mixture_identical_particles_returns_one():
    """All particles have identical composition and mass ⇒ χ should be 1."""
    masses = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    chi = get_mixing_state_index(masses)
    assert pytest.approx(chi, rel=1e-12) == 1.0


def test_single_particle_returns_one():
    """For a single particle, D̄ᵅ == Dᵞ ⇒ χ == 1."""
    masses = np.array([[4.0, 1.0, 5.0]])
    chi = get_mixing_state_index(masses)
    assert pytest.approx(chi, rel=1e-12) == 1.0


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #


def test_zero_total_mass_returns_nan():
    """No aerosol mass ⇒ function must return NaN."""
    masses = np.zeros((3, 2))
    chi = get_mixing_state_index(masses)
    assert np.isnan(chi)


def test_negative_mass_raises_value_error():
    """Negative inputs are invalid and should raise."""
    masses = np.array([[1.0, -0.1]])
    with pytest.raises(ValueError):
        _ = get_mixing_state_index(masses)
