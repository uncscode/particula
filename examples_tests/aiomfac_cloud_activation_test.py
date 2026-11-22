"""Qualify haze equilibrium, Köhler thresholds and the closed 1,500 m ascent."""

import numpy as np
import pytest
from scripts.docs_examples import import_study

pytest.importorskip("particula_aiomfac")
study = import_study("aiomfac-multivapor-parcel")


def test_critical_radius_uses_instantaneous_organic_mass():
    """Increasing dissolved butanediol changes the water activation barrier."""
    state = study.initial_haze()
    masses = np.stack([state.masses, state.masses.copy()])
    masses[1, :, 2] += masses[1, :, 3] * 0.1
    radii, excess = study.critical_water_curve(
        state, masses, np.full(2, state.temperature)
    )
    assert np.all(radii[1] > radii[0])
    assert np.all(excess[1] < excess[0])


def test_closed_cloud_activation():
    """Ambient saturation, two critical crossings, energy and inventories resolve."""
    data = study.cloud_activation("cpu")
    heights = [event["height_m"] for event in data["class_events"]]
    assert data["initial_rh_pct"] == pytest.approx(82.0)
    assert 395 < data["cloud_base_m"] < 405
    assert (
        data["cloud_base_m"]
        < data["peak_height_m"]
        < heights[0]
        < heights[1]
        < 1500
    )
    assert heights == pytest.approx([463.47, 707.24], abs=0.2)
    assert max(data["refined_crossing_shift_m"]) < 0.01
    assert data["terminal_activated_fraction"] == 1
    assert data["liquid_water_g_m3"][-1] == pytest.approx(2.5433, rel=1e-4)
    assert max(data["max_relative_inventory_error"]) < 1e-10
    assert (
        data["max_first_law_error_K"] < data["coarse_first_law_error_K"] < 0.01
    )
