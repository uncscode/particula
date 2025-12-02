"""Test the sedimentation kernel functions."""

# pylint: disable=duplicate-code

import numpy as np

from particula.dynamics.coagulation.sedimentation_kernel import (
    get_sedimentation_kernel_sp2016,
    get_sedimentation_kernel_sp2016_via_system_state,
)


def test_sedimentation_kernel():
    """Test just the sedimentation kernel."""
    particle_raidus = np.array([1e-6, 2e-6, 3e-6, 4e-6])
    settling_velocities = np.array([0.01, 0.02, 0.03, 0.04])
    expected_kernel = np.array(
        [
            [0.0, 1.25663706e-13, 2.51327412e-13, 3.76991118e-13],
            [1.25663706e-13, 0.0, 1.25663706e-13, 2.51327412e-13],
            [2.51327412e-13, 1.25663706e-13, 0.0, 1.25663706e-13],
            [3.76991118e-13, 2.51327412e-13, 1.25663706e-13, 0.0],
        ]
    )

    kernel = get_sedimentation_kernel_sp2016(
        particle_radius=particle_raidus,
        settling_velocities=settling_velocities,
        calculate_collision_efficiency=False,
    )

    assert np.allclose(kernel, expected_kernel, rtol=1e-6)


def test_sedimentation_kernel_via_system_state():
    """Test the sedimentation kernel via system state."""
    particle_radius = np.array([1e-6, 2e-6, 3e-6, 4e-6])
    particle_density = np.array([1000, 1000, 1000, 1000])
    temperature = 298.15
    pressure = 101325
    expected_kernel = np.array(
        [
            [0.0, 1.25663706e-13, 2.51327412e-13, 3.76991118e-13],
            [1.25663706e-13, 0.0, 1.25663706e-13, 2.51327412e-13],
            [2.51327412e-13, 1.25663706e-13, 0.0, 1.25663706e-13],
            [3.76991118e-13, 2.51327412e-13, 1.25663706e-13, 0.0],
        ]
    )

    kernel = get_sedimentation_kernel_sp2016_via_system_state(
        particle_radius=particle_radius,
        particle_density=particle_density,
        temperature=temperature,
        pressure=pressure,
        calculate_collision_efficiency=False,
    )

    assert np.allclose(kernel, expected_kernel, rtol=1e-6)
