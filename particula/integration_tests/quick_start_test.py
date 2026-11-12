"""Execute the published native quick start with independent inventory checks."""

import re
from pathlib import Path

import numpy as np
import particula as par


def test_quick_start_runs():
    """README construction and composed dynamics retain native conservation."""
    readme = Path(__file__).resolve().parents[2] / "readme.md"
    snippets = re.findall(
        r"```python\n(.*?)\n```", readme.read_text(), re.DOTALL
    )
    assert len(snippets) == 1
    namespace = {}
    exec(compile(snippets[0], str(readme), "exec"), namespace)  # noqa: S102
    aerosol = namespace["aerosol"]
    assert isinstance(aerosol, par.Aerosol)
    assert aerosol.particles is namespace["particles"]
    assert aerosol.gas is namespace["gas"]
    assert aerosol.environment is namespace["environment"]
    assert aerosol.particles.distribution_type == "particle_resolved"
    assert aerosol.gas.name == ["organic"]
    assert 0 <= aerosol.gas.concentration[0, 0] < 1e-6
    inventory = (
        np.sum(
            aerosol.particles.masses
            * aerosol.particles.concentration[..., None]
        )
        + aerosol.gas.concentration[0, 0] * 1e-6
    )
    np.testing.assert_allclose(inventory, 1e-12 + 3e-18, rtol=1e-12, atol=1e-30)
    assert np.all(np.isfinite(aerosol.particles.masses))
    assert np.all(aerosol.particles.masses >= 0)


def test_documentation_quick_start_runs():
    """The published PMF example preserves number and material inventory."""
    page = Path(__file__).resolve().parents[2] / "docs" / "index.md"
    snippets = re.findall(r"```python\n(.*?)\n```", page.read_text(), re.DOTALL)
    assert len(snippets) == 1
    namespace = {}
    exec(compile(snippets[0], str(page), "exec"), namespace)  # noqa: S102
    aerosol = namespace["result"]
    assert aerosol is namespace["aerosol"]
    particles = aerosol.particles
    assert particles.distribution_type == "discrete"
    np.testing.assert_allclose(particles.number_density, [1e8], rtol=1e-12)
    np.testing.assert_array_equal(particles.volume, [1.0])
    assert 0 <= aerosol.gas.concentration[0, 0] < 100.0
    final_particle_mass = np.sum(
        particles.masses * particles.concentration[..., None]
    )
    initial_particle_mass = namespace["initial_particle_mass"].item()
    assert final_particle_mass > initial_particle_mass
    np.testing.assert_allclose(
        final_particle_mass + aerosol.gas.concentration[0, 0],
        initial_particle_mass + 100.0,
        rtol=1e-12,
        atol=1e-30,
    )
    assert np.all(np.isfinite(particles.masses))
    assert np.all(particles.masses >= 0)
