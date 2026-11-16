"""Check full-case reuse and species-resolved refinement diagnostics."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import numpy.testing as npt
import pytest
from scripts.assess_respiratory_refinement import (
    ROOT,
    SOURCE,
    compare,
    load_setup,
)


def state():
    """Use an independent two-material inventory with non-unit counts/volume."""
    return SimpleNamespace(
        particles=SimpleNamespace(
            masses=np.array([[[1.0, 1e-12], [2.0, 2e-12]]]),
            concentration=np.array([[1.0, 2.0]]),
            volume=np.array([4.0]),
            radii=np.array([[1.0, 2.0]]),
        ),
        gas=SimpleNamespace(concentration=np.array([[0.5, 1e-13]])),
    )


def test_minor_lane_is_not_hidden_by_major_reservoir():
    """A 1% tiny-solute difference remains visible beside kilogram masses."""
    initial = state()
    coarse, fine = deepcopy(initial), deepcopy(initial)
    coarse.particles.masses[..., 1] -= 1e-14
    coarse.gas.concentration[..., 1] += 3e-14 / 4
    coarse.particles.radii *= 0.999
    result = compare(initial, coarse, fine)
    assert result["radius_relative"] == pytest.approx(0.001)
    npt.assert_allclose(result["slot_material_initial_relative"], [0, 0.01])
    for key in (
        "particle_phase_inventory_fraction",
        "gas_phase_inventory_fraction",
    ):
        npt.assert_allclose(result[key], [0, 3e-14 / 5.4e-12])


def test_minor_species_budget_failure_is_detected():
    """An uncompensated tiny-lane change fails its independent inventory."""
    initial = state()
    coarse = deepcopy(initial)
    coarse.gas.concentration[..., 1] += 1e-14
    with pytest.raises(AssertionError):
        compare(initial, coarse, deepcopy(initial))


def test_full_published_setup_is_reused_without_advancing():
    """Use all actual properties and samples, stopping before the costly loop."""
    import warp as wp

    if not wp.is_cuda_available() or not wp.is_conditional_graph_supported():
        pytest.skip("native CUDA conditional graphs unavailable")
    namespace, controls = load_setup((ROOT / SOURCE).read_text(), SOURCE)
    assert namespace["simulation"].time == 0
    assert namespace["simulation"].statistics.accepted == 0
    assert namespace["aerosol"].particles.masses.shape == (1, 1000, 12)
    assert controls == {
        "device": "cuda:0",
        "method": "ros3",
        "rtol": 1e-5,
        "particle_atol": 1e-25,
        "gas_atol": 1e-12,
        "first_step": 1e-6,
        "max_step": 0.1,
    }
    npt.assert_array_equal(
        namespace["simulation"].snapshot().particles.masses,
        namespace["aerosol"].particles.masses,
    )
