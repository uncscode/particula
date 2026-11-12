"""Check native construction and unchanged functional activity guide snippets."""

import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def execute_page(relative_path):
    """Execute actual Python fences, excluding explicitly textual pseudocode."""
    page = ROOT / relative_path
    namespace = {}
    snippets = re.findall(r"```python\n(.*?)\n```", page.read_text(), re.DOTALL)
    assert snippets
    for snippet in snippets:
        exec(compile(snippet, str(page), "exec"), namespace)  # noqa: S102
    return namespace


def test_object_oriented_native_builder_units():
    """The three-bin tutorial retains its physical sizes and concentration."""
    namespace = execute_page(
        "docs/Theory/Code_Concepts/Details/Object_Oriented_Patterns.md"
    )
    particles = namespace["particles"]
    radii = np.array([100e-9, 200e-9, 300e-9])
    masses = 4 * np.pi / 3 * radii**3 * 2500
    concentration = np.array([1e8, 1e9, 1e10])
    assert particles.distribution_type == "discrete"
    np.testing.assert_allclose(particles.radii, [radii], rtol=1e-12)
    np.testing.assert_allclose(particles.masses[0, :, 0], masses, rtol=1e-12)
    np.testing.assert_allclose(particles.concentration, [concentration])
    np.testing.assert_allclose(
        particles.species_mass_density, [[masses @ concentration]], rtol=1e-12
    )
    np.testing.assert_array_equal(particles.volume, [1.0])
    np.testing.assert_array_equal(particles.charge, [[0.0, 0.0, 0.0]])
    derived = particles.radii
    derived[:] = 0
    np.testing.assert_allclose(particles.radii, [radii], rtol=1e-12)


def test_activity_index_molar_fractions():
    """The functional activity entrypoint remains independent of native state."""
    namespace = execute_page("docs/Examples/Activity/index.md")
    amounts = np.array([0.5e-9 / 18.015e-3, 0.5e-9 / 200e-3])
    np.testing.assert_allclose(
        namespace["activity"], amounts / amounts.sum(), rtol=1e-12
    )


def test_design_pattern_function_and_strategy_examples():
    """The functional example keeps symmetric cubic-radius kernel scaling."""
    namespace = execute_page(
        "docs/Theory/Code_Concepts/Details/Design_Patterns.md"
    )
    kernel = namespace["kernel"]
    assert kernel.shape == (2, 2)
    assert np.all(np.isfinite(kernel)) and np.all(kernel > 0)
    np.testing.assert_allclose(kernel, kernel.T, rtol=1e-12)
    np.testing.assert_allclose(kernel[1, 1] / kernel[0, 0], 8, rtol=1e-12)
    assert (
        namespace["coagulation_process"].coagulation_strategy
        is namespace["combined_strategy"]
    )
