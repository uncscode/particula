"""Physical units, controlled growth and independent rare-event intervals."""

from dataclasses import replace

import numpy as np
import pytest
from scripts.gpu_studies.lucky_few import (
    SurvivalConfig,
    build_survival_case,
    difference_interval,
    run_survival,
    success_interval,
    whole_steps,
)


def test_survival_case_has_one_seed_and_explicit_volume_per_box():
    """One independent original lineage per box supports binomial intervals."""
    config = SurvivalConfig(boxes=3)
    initial, _ = build_survival_case(config)
    p = initial.particles
    assert p.masses.shape == (3, 128, 5)
    np.testing.assert_array_equal(p.concentration.sum(axis=1), 33)
    np.testing.assert_array_equal((p.masses[..., 1] > 0).sum(axis=1), 1)
    radii = (3 / (4 * np.pi) * (p.masses / p.density).sum(axis=2)) ** (1 / 3)
    np.testing.assert_allclose(2 * radii[:, 0], 3e-9, rtol=2e-15)
    np.testing.assert_allclose(2 * radii[:, 1:33], 100e-9, rtol=2e-15)
    assert config.background_particles / config.box_volume == 1e11


@pytest.mark.parametrize(
    "changes",
    [
        {"capacity": 32},
        {"boxes": 0},
        {"ion_pairs_cm3_s": -1},
        {"vapor_concentration": np.nan},
        {"time_step": 3},
        {"cleanup_seconds": 0.5},
        {"omitted": "extrapolate"},
    ],
)
def test_survival_config_rejects_invalid_physical_or_numerical_values(changes):
    """Do not silently alter physical durations or fit-omission choices."""
    with pytest.raises(ValueError):
        build_survival_case(replace(SurvivalConfig(), **changes))


@pytest.mark.parametrize("interval, step", [(1, 0), (np.inf, 1), (1, 0.3)])
def test_whole_steps_rejects_invalid_subdivision(interval, step):
    """Preparation does not silently round or divide by zero."""
    with pytest.raises(ValueError):
        whole_steps(interval, step)


def test_binomial_extremes_and_zero_event_upper_bound():
    """Compare exact beta endpoints with their closed-form extreme limits."""
    low, high, upper = success_interval(0, 100)
    assert low == 0
    np.testing.assert_allclose(high, 1 - 0.025 ** (1 / 100), rtol=2e-15)
    np.testing.assert_allclose(upper, 1 - 0.05 ** (1 / 100), rtol=2e-15)
    low, high, upper = success_interval(100, 100)
    np.testing.assert_allclose(low, 0.025 ** (1 / 100), rtol=2e-15)
    assert high == 1 and upper is None


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)]
)
def test_controlled_neutral_growth_reaches_target_and_closes_feed_budget(
    device,
):
    """Without background capture every original seed crosses a nearby target."""
    wp = pytest.importorskip("warp")
    if device != "cpu" and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    config = SurvivalConfig(
        boxes=4,
        capacity=8,
        background_particles=0,
        target_diameter=3.001e-9,
        time_step=0.1,
        duration=2,
        vapor_refresh_seconds=1,
        cleanup_seconds=0.5,
    )
    result = run_survival(config, device=device, capture=device != "cpu")
    np.testing.assert_array_equal(result["outcomes"][0], [0, 0, 4])
    np.testing.assert_array_equal(result["outcomes"][-1], [4, 0, 0])
    assert result["maximum_mass_residual"] < 2e-12
    assert result["maximum_observed_slots"] == 1
    np.testing.assert_allclose(result["time"], [0, 1, 2], rtol=1e-14, atol=0)


@pytest.mark.parametrize("confidence", [0, 1, -0.1, np.nan, np.inf, True])
def test_success_interval_rejects_invalid_confidence(confidence):
    """Coverage probabilities must define genuine finite intervals."""
    with pytest.raises(ValueError, match="confidence"):
        success_interval(0, 100, confidence=confidence)
    with pytest.raises(ValueError, match="confidence"):
        difference_interval(0, 100, 0, 100, confidence=confidence)


def test_difference_interval_has_joint_coverage_and_reverses_sign():
    """Check the Bonferroni endpoints against exact all-zero/all-one formulas."""
    low, high = difference_interval(0, 100, 100, 100)
    expected_low = 2 * 0.0125 ** (1 / 100) - 1
    np.testing.assert_allclose(low, expected_low, rtol=2e-15)
    assert high == 1
    np.testing.assert_allclose(
        difference_interval(100, 100, 0, 100), [-high, -low], rtol=2e-15
    )
    zero = difference_interval(0, 100, 0, 100)
    expected_high = 1 - 0.0125 ** (1 / 100)
    np.testing.assert_allclose(
        zero, [-expected_high, expected_high], rtol=2e-15
    )
