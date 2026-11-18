"""Check equilibrium pressure and finite-time response in the binary BAT pulse."""

import numpy as np
from docs.examples.thermodynamics.activity.bat_humidity_pulse import (
    M_WATER,
    SAMPLES_PER_LEG,
    TEMPERATURE,
    chamber,
    gas_water,
    trajectory,
)
from particula.util.constants import GAS_CONSTANT


def test_pulse_recovers_both_closed_reservoir_endpoints():
    """The flat activity times the Kelvin factor predicts endpoint gas RH."""
    aerosol, process = chamber()
    state = trajectory(None, 4)
    endpoints = state[[SAMPLES_PER_LEG, -1]]
    _, radius, vapor, particle_water = endpoints.T
    dry_mass = aerosol.particles.masses[0, 0, 1]
    number_density = (
        aerosol.particles.concentration[0, 0] / aerosol.particles.volume[0]
    )
    masses = np.column_stack(
        (particle_water / number_density, np.full(2, dry_mass))
    )
    activity = process.condensation_strategy.activity_strategy.activity(masses)[
        :, 0
    ]
    kelvin = np.exp(
        2 * 0.072 * M_WATER / (radius * 1000 * GAS_CONSTANT * TEMPERATURE)
    )
    np.testing.assert_allclose(
        vapor / gas_water(1), activity * kelvin, atol=1e-7, rtol=0
    )
    np.testing.assert_array_equal(
        state[SAMPLES_PER_LEG, [1, 3]], state[SAMPLES_PER_LEG + 1, [1, 3]]
    )
    assert state[1, 1] < endpoints[0, 1]
