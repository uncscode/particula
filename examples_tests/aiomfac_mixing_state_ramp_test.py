"""Check the published particle-resolved ramp without requiring a CUDA device."""

import json
from pathlib import Path

import numpy as np


def test_resolved_population_and_first_observed_splits():
    """Reconstruct every first-split bracket directly from the phase history."""
    result = _saved_result()
    ids = result["particle_id"]
    dry = np.array(result["dry_mass_kg"])
    assert len(set(ids)) == len(ids) == 1000
    assert dry.shape == (len(ids), 3)
    assert np.all(dry > 0)
    fractions = dry / dry.sum(axis=1, keepdims=True)
    assert np.unique(fractions.round(12), axis=0).shape[0] == len(ids)
    phase = np.asarray(result["phase_count"])
    assert phase.shape == (len(result["time_s"]), len(ids))
    assert np.all(phase[0] == 1)
    assert np.all(np.isin(phase, (1, 2)))
    observed = phase == 2
    first = np.argmax(observed, axis=0)
    expected = [
        [int(idx - 1), int(idx)] if seen else None
        for idx, seen in zip(first, observed.any(axis=0), strict=True)
    ]
    assert expected == result["onset_sample_bracket"]
    assert len(np.unique(first[observed.any(axis=0)])) > 1


def test_water_exchange_balances_the_external_reservoir():
    """Count each simulated droplet once and include externally assigned vapor."""
    result = _saved_result()
    water = np.asarray(result["water_kg"])
    gas = np.asarray(result["gas_water_kg"])
    imposed = np.asarray(result["cumulative_imposed_water_kg"])
    assert water.shape == (len(gas), len(result["particle_id"]))
    assert np.all(np.isfinite(water)) and np.all(water >= 0)
    transfer = (water - water[0]).sum(axis=1)
    balance = transfer + (gas - gas[0]) - imposed
    assert abs(transfer[-1]) > 0
    np.testing.assert_allclose(balance, 0, atol=1e-9 * abs(transfer[-1]))
    np.testing.assert_array_equal(result["maximum_dry_mass_error_kg"], 0)


def test_equilibrium_boundary_predicts_each_sampled_liquid_count():
    """Compare actual particle compositions with the independent CPU binodal."""
    result = _saved_result()
    cloud = result["equilibrium_control"]["particle_cloud_points"]
    assert cloud["particle_id"] == result["particle_id"]
    dry = np.sum(result["dry_mass_kg"], axis=1)
    water = np.asarray(result["water_kg"])
    water_fraction = water / (water + dry)
    threshold = np.asarray(cloud["water_mass_fraction"])
    assert threshold.shape == dry.shape
    assert np.all((threshold > 0) & (threshold < 1))
    np.testing.assert_array_equal(
        water_fraction < threshold, np.asarray(result["phase_count"]) == 2
    )


def _saved_result():
    folder = Path(__file__).resolve().parents[1] / "website" / "results"
    return json.loads(
        (folder / "aiomfac-mixing-state-ramp/data.json").read_text()
    )
