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


def test_complementary_compositions_use_shannon_diversity():
    """Equal bulk fractions give D_gamma = 2 and identical row entropies."""
    masses = np.array([[9.0, 1.0], [1.0, 9.0]])
    expected = 0.9**-0.9 * 0.1**-0.1 - 1.0
    assert get_mixing_state_index(masses) == pytest.approx(expected)


@pytest.mark.parametrize("scale", [1e-300, 1.0, 1e300])
def test_weight_entropy_before_exponentiating(scale):
    """Two thirds of dry mass has entropy ln 2, one third has zero."""
    masses = scale * np.array([[2, 2], [1, 0], [0, 1], [0, 0]])
    assert get_mixing_state_index(masses) == pytest.approx(2 ** (2 / 3) - 1)


@pytest.mark.parametrize(
    "masses", [np.array([[3, 0], [1, 0]]), np.empty((0, 2))]
)
def test_undefined_bulk_diversity(masses):
    """No bulk species contrast gives an undefined index without division."""
    with np.errstate(all="raise"):
        assert np.isnan(get_mixing_state_index(masses))


def test_partially_mixed_population_uses_asymmetric_bulk_entropy():
    """One mixed and one pure particle give bulk fractions 3/4 and 1/4."""
    masses = np.array([[1.0, 1.0], [2.0, 0.0]])
    bulk_diversity = 0.75**-0.75 * 0.25**-0.25
    expected = (2**0.5 - 1) / (bulk_diversity - 1)
    assert get_mixing_state_index(masses) == pytest.approx(expected)


def test_trace_species_preserves_internal_mixture():
    """Bulk entropy can be positive even when exp(H) rounds to one."""
    masses = np.array([[1.0, 1e-20], [3.0, 3e-20], [0.0, 0.0]])
    assert get_mixing_state_index(masses) == pytest.approx(1.0, rel=1e-12)
