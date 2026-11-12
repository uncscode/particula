"""Data-only v0.3.0 inventories across ordered particle/gas/environment lanes."""

import numpy as np
import numpy.testing as npt
import pytest
from particula.aerosol_validation import validate_aerosol_structure
from particula.gas import EnvironmentData, GasData
from particula.particles import ParticleData


def _containers(
    partitioning: list[bool],
) -> tuple[ParticleData, GasData, EnvironmentData]:
    """Build unequal-width containers with distinct species and box values."""
    particles = ParticleData(
        masses=np.array(
            [[[0.5, 1.5], [2, 3]], [[1, 2], [4, 1]]], dtype=np.float64
        )
        * 1e-18,
        concentration=np.array([[8, 0.5], [3, 2]], dtype=np.float64),
        charge=np.zeros((2, 2), dtype=np.float64),
        density=np.array([1000, 1500], dtype=np.float64),
        volume=np.array([0.25, 4], dtype=np.float64),
        distribution_type="particle_resolved",
    )
    # Particle order is [solute_a, solute_b]; gas order is deliberately different.
    gas = GasData(
        name=["solute_b", "inert", "solute_a"],
        molar_mass=np.array([0.02, 0.03, 0.04], dtype=np.float64),
        concentration=np.array([[2, 7, 3], [5, 11, 13]], dtype=np.float64)
        * 1e-18,
        partitioning=np.array(partitioning, dtype=np.bool_),
    )
    environment = EnvironmentData(
        temperature=np.array([280, 300], dtype=np.float64),
        pressure=np.array([90000, 101325], dtype=np.float64),
        saturation_ratio=np.array(
            [[0.2, 0.7, 0.3], [0.5, 1.1, 1.3]], dtype=np.float64
        ),
    )
    return particles, gas, environment


@pytest.mark.parametrize("partitioning", [[True, False, True], [False] * 3])
def test_ordered_nonunit_volume_inventories(partitioning: list[bool]) -> None:
    """Check each extensive species independently, including unmapped gas."""
    particles, gas, environment = _containers(partitioning)
    particles.validate_representation()
    validate_aerosol_structure(particles, gas, environment)

    # Hand arithmetic, independent of population helpers:
    # box 0: [8*.5 + .5*2, 8*1.5 + .5*3]; box 1: [3*1 + 2*4, 3*2 + 2*1].
    expected_particles = np.array([[5, 13.5], [11, 8]]) * 1e-18
    # Gas density times physical volume, retaining every ordered lane.
    expected_gas = np.array([[0.5, 1.75, 0.75], [20, 44, 52]]) * 1e-18
    actual_particles = particles.species_mass_inventory
    actual_gas = gas.concentration * particles.volume[:, None]
    # Absolute tolerance is twelve orders below the smallest fixture mass;
    # default NumPy absolute tolerances would hide count/density mistakes.
    npt.assert_allclose(
        actual_particles, expected_particles, rtol=1e-12, atol=1e-30
    )
    npt.assert_allclose(actual_gas, expected_gas, rtol=1e-12, atol=1e-30)
    npt.assert_allclose(
        particles.species_mass_density,
        np.array([[20, 54], [2.75, 2]]) * 1e-18,
        rtol=1e-12,
        atol=1e-30,
    )
    npt.assert_allclose(particles.number_density, [34, 1.25], rtol=1e-12)

    # Chemical correspondence is explicit even when transfer is disabled.
    # The eligible process map is empty for an all-false mask.
    correspondence = [(0, 1), (2, 0)]
    mapping = [(g, p) for g, p in correspondence if partitioning[g]]
    expected_combined = {0: [14, 28], 2: [5.75, 63]}
    for gas_index, particle_index in correspondence:
        npt.assert_allclose(
            actual_particles[:, particle_index] + actual_gas[:, gas_index],
            np.array(expected_combined[gas_index]) * 1e-18,
            rtol=1e-12,
            atol=1e-30,
        )
    unmapped = [i for i in range(3) if i not in {g for g, _ in mapping}]
    npt.assert_allclose(
        actual_gas[:, unmapped],
        expected_gas[:, unmapped],
        rtol=1e-12,
        atol=1e-30,
    )
    assert gas.name == ["solute_b", "inert", "solute_a"]
    npt.assert_array_equal(gas.partitioning, partitioning)
    npt.assert_array_equal(gas.molar_mass, [0.02, 0.03, 0.04])
    npt.assert_array_equal(
        environment.saturation_ratio, [[0.2, 0.7, 0.3], [0.5, 1.1, 1.3]]
    )


@pytest.mark.parametrize("partitioning", [[True, False, True], [False] * 3])
def test_current_reads_and_detached_container_copies(
    partitioning: list[bool],
) -> None:
    """Raw edits affect current inventories but never detached snapshots."""
    particles, gas, environment = _containers(partitioning)
    particle_copy, gas_copy, environment_copy = (
        particles.copy(),
        gas.copy(),
        environment.copy(),
    )
    old_inventory = particles.species_mass_inventory
    for original, copied, fields in (
        (
            particles,
            particle_copy,
            ("masses", "concentration", "charge", "density", "volume"),
        ),
        (gas, gas_copy, ("concentration", "molar_mass", "partitioning")),
        (
            environment,
            environment_copy,
            ("temperature", "pressure", "saturation_ratio"),
        ),
    ):
        for field in fields:
            assert not np.shares_memory(
                getattr(original, field), getattr(copied, field)
            )
    assert gas_copy.name is not gas.name
    assert particle_copy.distribution_type == "particle_resolved"

    particles.concentration[0, 0] = 4
    gas.concentration[0, 2] = 9e-18
    environment.saturation_ratio[0, 2] = 0.9
    validate_aerosol_structure(particles, gas, environment)
    npt.assert_allclose(
        particles.species_mass_inventory,
        np.array([[3, 7.5], [11, 8]]) * 1e-18,
        rtol=1e-12,
        atol=1e-30,
    )
    npt.assert_allclose(
        particles.species_mass_inventory[:, 0]
        + gas.concentration[:, 2] * particles.volume,
        np.array([5.25, 63]) * 1e-18,
        rtol=1e-12,
        atol=1e-30,
    )
    npt.assert_allclose(
        old_inventory,
        np.array([[5, 13.5], [11, 8]]) * 1e-18,
        rtol=1e-12,
        atol=1e-30,
    )
    npt.assert_array_equal(particle_copy.species_mass_inventory, old_inventory)
    npt.assert_array_equal(
        gas_copy.concentration[0], np.array([2, 7, 3]) * 1e-18
    )
    npt.assert_array_equal(
        environment_copy.saturation_ratio[0], [0.2, 0.7, 0.3]
    )
    gas.name[0] = "renamed_b"
    assert gas_copy.name == ["solute_b", "inert", "solute_a"]
