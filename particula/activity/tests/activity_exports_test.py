"""Tests verifying activity package exports and q_alpha helpers.

This suite makes sure the activity package exposes the intended public
APIs, that convenience imports resolve correctly, and that the q_alpha
helper behaves over the minimum allowed spread.
"""

from importlib import reload

import numpy as np
import particula.activity as activity


def test_activity_module_exports_present_and_non_none():
    """Ensure each export listed in __all__ exists and is not None.

    Reloading the activity package refreshes module attributes to catch
    dynamic exports that might be added or modified during import-time
    side effects.

    Raises:
        AssertionError: If an export is missing or resolves to None.
    """
    reload(activity)
    for name in activity.__all__:
        assert hasattr(activity, name), f"Missing export: {name}"
        assert getattr(activity, name) is not None, f"Export {name} is None"


def test_activity_convenience_imports():
    """Validate that core helpers import cleanly via particula.activity.

    Downstream consumers should be able to rely on the tops-level activity
    namespace rather than internal module layout.
    """
    from particula.activity import (
        MIN_SPREAD_IN_AW,
        Q_ALPHA_AT_1PHASE_AW,
        bat_activity_coefficients,
        bat_blending_weights,
        coefficients_c,
        find_phase_sep_index,
        find_phase_separation,
        from_molar_mass_ratio,
        gibbs_free_energy,
        gibbs_mix_weight,
        gibbs_of_mixing,
        q_alpha,
        to_molar_mass_ratio,
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
    """Check q_alpha stays finite and preserves shape near MIN_SPREAD.

    Passing the MIN_SPREAD_IN_AW value ensures the transition width stays above
    the enforced minimum while exercising the logistic curve.

    Raises:
        AssertionError: If the result contains infinities, changes shape, or
            leaves the expected bounds.
    """
    activities = np.linspace(0.5, 1.0, 5)
    separation_activity = np.array(activity.MIN_SPREAD_IN_AW, dtype=np.float64)
    result = activity.q_alpha(separation_activity, activities)
    assert result.shape == activities.shape
    assert np.all(np.isfinite(result))
    assert np.all(result >= 0)
    assert np.all(result <= 1)
