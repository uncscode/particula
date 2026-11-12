"""Run the published native process snippets and check physical outcomes."""

import re
from pathlib import Path

import numpy as np
import particula as par
import pytest

FEATURES = Path(__file__).resolve().parents[1] / "docs/Features"


def snippets(name):
    """Read exact Python fences in publication order."""
    page = FEATURES / f"{name}_strategy_system.md"
    return page, re.findall(
        r"```python\n(.*?)\n```", page.read_text(), re.DOTALL
    )


@pytest.mark.parametrize("name", ["wall_loss", "coagulation"])
def test_cpu_process_guide_snippets(name):
    """CPU construction, builders, factories and composed workflows execute."""
    page, blocks = snippets(name)
    namespace = {}
    for block in blocks:
        if "particula.gpu" in block or "WarpBrownian" in block:
            continue  # Optional imports are tested separately.
        exec(compile(block, str(page), "exec"), namespace)  # noqa: S102
    aerosol = namespace["aerosol"]
    assert isinstance(aerosol, par.Aerosol)
    assert aerosol.particles.distribution_type == "discrete"
    assert np.all(np.isfinite(aerosol.particles.masses))
    assert np.all(aerosol.particles.concentration >= 0)
    assert 0 < aerosol.particles.number_density.item() < 1e8
    assert 0 <= aerosol.gas.concentration.item() < 1e-6
    assert aerosol.gas is namespace["gas"]
    assert aerosol.environment is namespace["environment"]


@pytest.mark.parametrize("name", ["wall_loss", "coagulation", "condensation"])
def test_optional_process_guide_imports(name):
    """Optional GPU declarations remain valid with Warp installed."""
    pytest.importorskip("warp")
    page, blocks = snippets(name)
    namespace = {}
    for block in blocks:
        if "particula.gpu" in block or "WarpBrownian" in block:
            exec(compile(block, str(page), "exec"), namespace)  # noqa: S102


def test_wall_loss_quick_start_discrete_decay():
    """Forty explicit Euler steps match their analytical finite-step decay."""
    page, blocks = snippets("wall_loss")
    block = next(b for b in blocks if "# 3. Compute instantaneous wall" in b)
    namespace = {}
    exec(compile(block, str(page), "exec"), namespace)  # noqa: S102
    initial = par.particles.create_lognormal_binned_particle_data(
        distribution_type="discrete"
    )
    coefficient = namespace["wall_loss"].loss_coefficient(
        initial, temperature=298.15, pressure=101325.0
    )
    expected = initial.concentration * (1.0 - 0.25 * coefficient) ** 40
    final = namespace["particle"]
    np.testing.assert_allclose(final.concentration, expected, rtol=1e-12)
    np.testing.assert_array_equal(final.masses, initial.masses)
    np.testing.assert_array_equal(final.charge, initial.charge)
    assert final.number_density.item() < initial.number_density.item()


def test_brownian_quick_start_number_loss():
    """The published single-material PMF evolves number without changing bins."""
    page, blocks = snippets("coagulation")
    block = next(b for b in blocks if "# 3. Compute instantaneous kernel" in b)
    namespace = {}
    exec(compile(block, str(page), "exec"), namespace)  # noqa: S102
    initial = par.particles.create_lognormal_binned_particle_data(
        distribution_type="discrete"
    )
    final = namespace["particle"]
    assert 0 < final.number_density.item() < initial.number_density.item()
    assert np.all(final.concentration >= 0)
    np.testing.assert_array_equal(final.masses, initial.masses)
    # This finite-grid PMF method is not an exact mass-conserving merge.
    assert np.all(np.isfinite(final.species_mass_inventory))
    assert np.all(final.species_mass_inventory > 0)


def test_condensation_guide_all_snippets():
    """Run all CPU recipes, including native factories and composed calls."""
    page, blocks = snippets("condensation")
    namespace = {}
    for block in blocks:
        if "particula.gpu" not in block:
            exec(compile(block, str(page), "exec"), namespace)  # noqa: S102
    aerosol = namespace["aerosol"]
    assert aerosol.particles.distribution_type == "particle_resolved"
    assert aerosol.gas.concentration[0, 1] == 1e-6
    assert 0 <= aerosol.gas.concentration[0, 0] < 1e-6


@pytest.mark.parametrize("index", [2, 3, 4, 7, 8, 10, 11, 14, 16, 17])
def test_condensation_recipe_inventories(index):
    """Each closed recipe conserves mapped extensive particle-plus-gas mass."""
    page, blocks = snippets("condensation")
    namespace = {}
    exec(compile(blocks[0], str(page), "exec"), namespace)  # noqa: S102
    prefix, body = blocks[index].split(
        "particle, gas = aerosol.particles, aerosol.gas\n", 1
    )
    exec(compile(prefix, str(page), "exec"), namespace)  # noqa: S102
    aerosol = namespace["aerosol"]
    if index == 10:
        # The discrete quick start deliberately selects the original defaults.
        aerosol.particles = par.particles.create_lognormal_binned_particle_data(
            distribution_type="discrete"
        )
    particle, gas = aerosol.particles, aerosol.gas
    namespace.update(particle=particle, gas=gas)
    initial_particle = np.sum(
        particle.masses * particle.concentration[..., None], axis=1
    )
    initial_gas = gas.concentration.copy() * particle.volume[:, None]
    exec(compile(body, str(page), "exec"), namespace)  # noqa: S102
    final = namespace["particle"]
    final_particle = np.sum(
        final.masses * final.concentration[..., None], axis=1
    )
    np.testing.assert_allclose(
        final_particle + gas.concentration * final.volume[:, None],
        initial_particle + initial_gas,
        rtol=1e-12,
        atol=1e-30,
    )
    assert np.all(final.masses >= 0)
    assert np.all(gas.concentration >= 0)
    if index in [7, 17]:
        np.testing.assert_array_equal(
            gas.concentration[:, 1] * final.volume,
            initial_gas[:, 1],
        )
    if index == 3:
        np.testing.assert_allclose(
            namespace["energy_released"],
            np.sum(final_particle - initial_particle) * 2.4e6,
            rtol=1e-10,
            atol=1e-20,
        )


def test_dilution_guide_decay_and_held_state():
    """Exact published decay scales counts and every gas lane at fixed volume."""
    page, blocks = snippets("dilution")
    namespace = {}
    for block in blocks:
        exec(compile(block, str(page), "exec"), namespace)  # noqa: S102
    aerosol = namespace["result"]
    assert aerosol is namespace["aerosol"]
    assert aerosol.particles is namespace["particles"]
    assert aerosol.gas is namespace["gas"]
    assert aerosol.environment is namespace["environment"]
    np.testing.assert_allclose(
        aerosol.particles.concentration, [[4 / np.e, 8 / np.e]], rtol=1e-12
    )
    np.testing.assert_allclose(
        aerosol.particles.number_density, [6 / np.e], rtol=1e-12
    )
    np.testing.assert_allclose(
        aerosol.gas.concentration, [[3e-6 / np.e, 5e-6 / np.e]], rtol=1e-12
    )
    np.testing.assert_array_equal(aerosol.particles.volume, [2.0])
    np.testing.assert_array_equal(
        aerosol.particles.masses, [[[1e-18], [2e-18]]]
    )
    np.testing.assert_array_equal(aerosol.environment.temperature, [298.15])
    np.testing.assert_array_equal(
        aerosol.environment.saturation_ratio, [[1.0, 0.0]]
    )


def test_nucleation_guide_mapped_inventory():
    """Published fixed-slot source conserves precursor mass and leaves inert gas."""
    page, blocks = snippets("nucleation")
    namespace = {}
    for block in blocks:
        if "particula.gpu" not in block:
            exec(compile(block, str(page), "exec"), namespace)  # noqa: S102
    aerosol = namespace["result"]
    assert aerosol is namespace["aerosol"]
    assert aerosol.particles is namespace["particles"]
    assert aerosol.gas is namespace["gas"]
    assert aerosol.environment is namespace["environment"]
    particles = aerosol.particles
    assert particles.masses.shape == (1, 3, 1)
    assert np.any(particles.concentration > 0)
    np.testing.assert_array_equal(particles.volume, [0.25])
    assert 0 <= aerosol.gas.concentration[0, 0] < 1e-12
    assert aerosol.gas.concentration[0, 1] == 2e-6
    total = np.sum(particles.masses * particles.concentration[..., None])
    total += aerosol.gas.concentration[0, 0] * 0.25
    np.testing.assert_allclose(total, 2.5e-13, rtol=1e-12, atol=1e-30)


def test_native_scientific_direct_calls():
    """Direct native composition preserves inert gas and accounts for sinks."""
    page = FEATURES / "native_scientific_processes.md"
    blocks = re.findall(r"```python\n(.*?)\n```", page.read_text(), re.DOTALL)
    namespace = {}
    for block in blocks:
        exec(compile(block, str(page), "exec"), namespace)  # noqa: S102
    aerosol = namespace["aerosol"]
    particles = aerosol.particles
    np.testing.assert_array_equal(
        aerosol.gas.concentration[:, 0], namespace["initial_gas"][:, 0]
    )
    initial = namespace["initial_particle_mass"] + (
        namespace["initial_gas"][:, [2, 1]] * 0.25
    )
    final = (
        np.sum(particles.masses * particles.concentration[..., None], axis=1)
        + aerosol.gas.concentration[:, [2, 1]] * 0.25
    )
    assert np.all(final <= initial * (1 + 1e-12))
    assert np.all(final >= 0)
    assert 0 <= particles.number_density.item() <= 16.0
    lost = particles.concentration == 0
    assert np.all(particles.masses[lost] == 0)
    assert np.all(particles.charge[lost] == 0)
