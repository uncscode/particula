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


@pytest.mark.parametrize("name", ["wall_loss", "coagulation"])
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
