"""Tests for activity module exports and convenience imports."""

from importlib import reload

import numpy as np

import particula.activity as activity


def test_activity_module_exports_present_and_non_none():
    """All names in __all__ should be present and non-None."""
    reload(activity)
    for name in activity.__all__:
        assert hasattr(activity, name), f"Missing export: {name}"
        assert getattr(activity, name) is not None, f"Export {name} is None"


def test_activity_convenience_imports():
    """Key public APIs should import cleanly from particula.activity."""
    from particula.activity import (
        bat_activity_coefficients,
        bat_blending_weights,
        coefficients_c,
        find_phase_separation,
        find_phase_sep_index,
        gibbs_free_energy,
        gibbs_mix_weight,
        gibbs_of_mixing,
        MIN_SPREAD_IN_AW,
        Q_ALPHA_AT_1PHASE_AW,
        q_alpha,
        to_molar_mass_ratio,
        from_molar_mass_ratio,
    )

    assert callable(bat_activity_coefficients)
    assert callable(bat_blending_weights)
    assert callable(coefficients_c)
    assert callable(find_phase_separation)
    assert callable(find_phase_sep_index)
    assert callable(gibbs_free_energy)
    assert callable(gibbs_mix_weight)
    assert callable(gibbs_of_mixing)
    assert isinstance(MIN_SPREAD_IN_AW, float)
    assert isinstance(Q_ALPHA_AT_1PHASE_AW, float)
    assert callable(q_alpha)
    assert callable(to_molar_mass_ratio)
    assert callable(from_molar_mass_ratio)


def test_q_alpha_min_spread_shape_and_finiteness():
    """q_alpha should be finite and preserve shape with minimum spread."""
    activities = np.linspace(0.5, 1.0, 5)
    result = activity.q_alpha(activity.MIN_SPREAD_IN_AW, activities)
    assert result.shape == activities.shape
    assert np.all(np.isfinite(result))
    assert np.all(result >= 0)
    assert np.all(result <= 1)
