"""Executable native migration guidance: ownership and rejection boundaries."""

import re
import runpy
from pathlib import Path

import numpy as np
import pytest

SOURCE = (
    Path(__file__).resolve().parents[1] / "docs/Examples/native_state_access.py"
)


@pytest.fixture
def example():
    """Load the published functions without running its command entry point."""
    return runpy.run_path(str(SOURCE))


def snapshot(*containers):
    """Freeze raw public fields, including names and population metadata."""
    return [
        {
            name: value.copy()
            if isinstance(value, (np.ndarray, list))
            else value
            for name, value in vars(container).items()
        }
        for container in containers
    ]


def assert_unchanged(containers, saved):
    """Compare every original field without deriving potentially invalid state."""
    for container, fields in zip(containers, saved, strict=True):
        for name, expected in fields.items():
            np.testing.assert_equal(getattr(container, name), expected)


def test_access_has_explicit_count_volume_and_copy_semantics(example):
    """Raw storage aliases; all derived population properties are detached."""
    aerosol = example["build_aerosol"]()
    particles = aerosol.particles
    np.testing.assert_allclose(particles.number_density, [16.0])
    np.testing.assert_allclose(
        particles.species_mass_inventory,
        [[12e-18, 16e-18]],
        rtol=1e-15,
        atol=1e-30,
    )
    for name in (
        "radii",
        "total_mass",
        "mass_fractions",
        "effective_density",
        "slot_concentration_density",
        "number_density",
        "species_mass_density",
        "species_mass_inventory",
    ):
        derived = getattr(particles, name)
        assert not np.shares_memory(derived, particles.masses)
        assert not np.shares_memory(derived, particles.concentration)
    copied = particles.copy()
    copied.concentration[0, 0] = 20.0
    assert particles.concentration[0, 0] == 2.0
    alias = aerosol.particles.concentration
    alias[0, 0] = 4.0
    np.testing.assert_array_equal(particles.number_density, [24.0])


@pytest.mark.parametrize("field", ["particles", "gas", "environment"])
def test_individual_replacement_rejects_without_mutating_either_side(
    example, field
):
    """Invalid candidates preserve held identities and all candidate bytes."""
    aerosol = example["build_aerosol"]()
    old = (aerosol.particles, aerosol.gas, aerosol.environment)
    candidates = example["replacement_layout"](aerosol)
    candidate = candidates[("particles", "gas", "environment").index(field)]
    if field == "particles":
        candidate.concentration[0, 0] = -1.0
    containers = (*old, *candidates)
    saved = snapshot(*containers)
    with pytest.raises(ValueError):
        setattr(aerosol, field, candidate)
    assert all(
        current is original
        for current, original in zip(
            (aerosol.particles, aerosol.gas, aerosol.environment),
            old,
            strict=True,
        )
    )
    assert_unchanged(containers, saved)


def test_coordinated_replacement_admits_layout_but_does_not_reconfigure_process(
    example,
):
    """Valid new widths require explicit new process provenance."""
    aerosol = example["build_aerosol"]()
    process = example["build_condensation"](aerosol.gas)
    particles, gas, environment = example["replacement_layout"](aerosol)
    old = (aerosol.particles, aerosol.gas, aerosol.environment)
    all_containers = (*old, particles, gas, environment)
    saved = snapshot(*all_containers)
    with pytest.raises(ValueError):
        aerosol.replace_data(particles=particles, gas=gas, environment=old[2])
    assert aerosol.particles is old[0] and aerosol.gas is old[1]
    assert aerosol.environment is old[2]
    assert_unchanged(all_containers, saved)
    assert (
        aerosol.replace_data(
            particles=particles, gas=gas, environment=environment
        )
        is None
    )
    assert aerosol.particles is particles and aerosol.gas is gas
    assert aerosol.environment is environment
    with pytest.raises(ValueError):
        process.execute(aerosol, 1.0)
    assert_unchanged(all_containers, saved)


@pytest.mark.parametrize("mutation", ["temperature", "gas_order"])
def test_next_process_revalidates_direct_mutation(example, mutation):
    """Held writable data does not bypass physical or species-order checks."""
    aerosol = example["build_aerosol"]()
    process = example["build_condensation"](aerosol.gas)
    if mutation == "temperature":
        aerosol.environment.temperature[0] = np.nan
    else:
        aerosol.gas.name[:] = list(reversed(aerosol.gas.name))
    containers = (aerosol.particles, aerosol.gas, aerosol.environment)
    saved = snapshot(*containers)
    with pytest.raises(ValueError):
        process.execute(aerosol, 1.0)
    assert_unchanged(containers, saved)


def test_published_example_runs_real_mapped_condensation(example):
    """Both mapped materials gain conserved mass while inert gases stay fixed."""
    result = example["run_example"]()
    np.testing.assert_array_equal(
        result.gas.concentration[:, [0, 3]], [[9e-15, 7e-15]]
    )
    assert np.all(
        result.particles.masses > example["build_aerosol"]().particles.masses
    )


@pytest.mark.parametrize("page", ["construction", "containers", "dynamics"])
def test_native_migration_page_python_snippets_execute(page):
    """Execute the exact supported snippets in each page's documented order."""
    path = (
        SOURCE.parents[2]
        / "docs/Features/particle-data-migration"
        / f"{page}.md"
    )
    snippets = re.findall(r"```python\n(.*?)\n```", path.read_text(), re.DOTALL)
    assert snippets
    namespace = {}
    for snippet in snippets:
        exec(compile(snippet, str(path), "exec"), namespace)  # noqa: S102
    assert (
        namespace["aerosol"].particles.distribution_type == "particle_resolved"
    )
