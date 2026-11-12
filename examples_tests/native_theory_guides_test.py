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


def test_activity_feature_equilibrium_organic_balance():
    """The completed recipes separate reservoir water from organic inventory."""
    namespace = execute_page("docs/Features/activity_system.md")
    result = namespace["result"]
    organic = np.array([1.0, 5.0, 10.0])
    fractions = result.partition_coefficients
    assert np.all(np.isfinite(fractions))
    assert np.all((fractions >= 0) & (fractions <= 1))
    phases = [result.alpha_phase, result.beta_phase]
    liquid_organic = sum(
        phase.total_concentration - phase.water_concentration
        for phase in phases
    )
    np.testing.assert_allclose(
        liquid_organic, np.sum(organic * fractions), rtol=1e-12
    )
    np.testing.assert_allclose(
        liquid_organic + np.sum(organic * (1 - fractions)),
        organic.sum(),
        rtol=1e-12,
    )
    assert sum(phase.water_concentration for phase in phases) > 0


def test_activity_theory_coefficient_identity():
    """BAT activity equals its coefficient times the appropriate mole fraction."""
    namespace = execute_page(
        "docs/Theory/Activity_Calculations/activity_theory.md"
    )
    np.testing.assert_allclose(
        namespace["activity_water"],
        namespace["gamma_water"] * 0.7,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        namespace["activity_organic"],
        namespace["gamma_organic"] * 0.3,
        rtol=1e-12,
    )


def test_equilibrium_theory_species_balance():
    """The functional solver preserves each organic species with water separate."""
    namespace = execute_page(
        "docs/Theory/Activity_Calculations/equilibria_theory.md"
    )
    assert namespace["result"].success
    alpha, beta, system = (
        namespace["alpha"],
        namespace["beta"],
        namespace["system"],
    )
    organic = np.array([1.0, 2.0, 3.0])
    fraction = system[2]
    assert np.all((fraction >= 0) & (fraction <= 1))
    np.testing.assert_allclose(
        alpha[0] + beta[0] + organic * (1 - fraction), organic, rtol=1e-12
    )
    np.testing.assert_allclose(
        alpha[2] + beta[2] - alpha[3] - beta[3], system[0], rtol=1e-12
    )


def require_explicit_launch_device(monkeypatch, wp):
    """Reject implicit device selection even when only Warp CPU is available."""
    launch = wp.launch

    def checked_launch(*args, **kwargs):
        assert kwargs.get("device") is not None
        return launch(*args, **kwargs)

    monkeypatch.setattr(wp, "launch", checked_launch)


def test_warp_theory_quick_start(tmp_path, monkeypatch):
    """The exact file-backed Warp primer runs explicitly on its CPU backend."""
    import runpy

    import pytest

    wp = pytest.importorskip("warp")
    require_explicit_launch_device(monkeypatch, wp)
    page = ROOT / "docs/Theory/nvidia-warp/index.md"
    blocks = re.findall(r"```python\n(.*?)\n```", page.read_text(), re.DOTALL)
    source = tmp_path / "warp_primer.py"
    source.write_text("\n\n".join(blocks))
    namespace = runpy.run_path(str(source))
    assert namespace["positions"].device.is_cpu
    assert namespace["result"].shape == (10000, 3)
    assert np.all(np.isfinite(namespace["result"]))
    positions = wp.array(np.zeros((2, 3)), dtype=wp.vec3, device="cpu")
    velocities = wp.array(np.full((2, 3), 2.0), dtype=wp.vec3, device="cpu")
    wp.launch(
        namespace["update_particle_positions"],
        dim=2,
        inputs=[positions, velocities, 0.25],
        device="cpu",
    )
    wp.synchronize()
    np.testing.assert_allclose(positions.numpy(), np.full((2, 3), 0.5))


def test_warp_theory_native_environment_round_trip():
    """The actual native helper example preserves all thermodynamic lanes."""
    import pytest

    pytest.importorskip("warp")
    page = ROOT / "docs/Theory/nvidia-warp/datastructures.md"
    blocks = re.findall(r"```python\n(.*?)\n```", page.read_text(), re.DOTALL)
    block = next(b for b in blocks if "from_warp_environment_data" in b)
    namespace = {}
    exec(compile(block, str(page), "exec"), namespace)  # noqa: S102
    result = namespace["restored"]
    np.testing.assert_array_equal(result.temperature, [298.15, 305.0])
    np.testing.assert_array_equal(result.pressure, [101325.0, 90000.0])
    np.testing.assert_array_equal(
        result.saturation_ratio, [[0.95, 1.05, 0.75], [0.8, 0.9, 1.1]]
    )


def test_warp_roadmap_toy_gradient(tmp_path, monkeypatch):
    """The teaching Tape example differentiates its quadratic on Warp CPU."""
    import runpy

    import pytest

    wp = pytest.importorskip("warp")
    require_explicit_launch_device(monkeypatch, wp)
    page = ROOT / "docs/Features/Roadmap/warp-autodiff-limitations.md"
    blocks = re.findall(r"```python\n(.*?)\n```", page.read_text(), re.DOTALL)
    source = tmp_path / "warp_tape_example.py"
    source.write_text(blocks[0])
    namespace = runpy.run_path(str(source))
    assert namespace["x"].device.is_cpu
    np.testing.assert_allclose(namespace["gradient"], [5.0, 7.0, 9.0])
    np.testing.assert_array_equal(namespace["x"].grad.numpy(), [0, 0, 0])
