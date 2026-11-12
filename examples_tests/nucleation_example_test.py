"""Runtime regressions for the CPU nucleation example."""

import ast
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "docs/Examples/Nucleation/cpu_nucleation.py"
EXAMPLE_TIMEOUT_SECONDS = 30


def test_cpu_nucleation_example_uses_public_imports_without_source_helpers():
    """The runnable example stays on public APIs rather than demand
    planning/source transaction helpers.
    """
    tree = ast.parse(EXAMPLE.read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert {
        "particula.dynamics",
        "particula.gas",
        "particula.particles.exhaustion",
    } <= imports
    assert imports <= {
        "__future__",
        "particula.dynamics.nucleation.species_mapping",
        "particula",
        "particula.dynamics",
        "particula.gas",
        "particula.particles",
        "particula.particles.exhaustion",
    }
    referenced_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            referenced_names.add(node.id)
        elif isinstance(node, ast.Attribute):
            referenced_names.add(node.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                referenced_names.update(alias.name.split("."))
    assert not referenced_names.intersection(
        {
            "particle_source",
            "finalize_particle_source",
            "commit_particle_source",
            "ParticleSourceCommitConfig",
        }
    )


def test_cpu_nucleation_example_uses_public_api_and_conserves_mass() -> None:
    """The one-box example transfers gas while conserving total mass."""
    namespace = runpy.run_path(str(EXAMPLE))
    aerosol = namespace["run_example"]()
    particles = aerosol.particles
    gas = aerosol.gas
    assert particles.distribution_type == "particle_resolved"
    npt.assert_array_equal(particles.volume, [0.25])

    assert particles.masses.shape == (1, 3, 1)
    assert particles.masses.dtype == np.float64
    assert gas.concentration.shape == (1, 2)
    assert gas.concentration.dtype == np.float64
    assert np.any(particles.concentration > 0.0)
    assert gas.concentration[0, 0] < 1.0e-12
    npt.assert_allclose(
        gas.concentration[:, 1:], [[2.0e-6]], rtol=0.0, atol=0.0
    )
    total = np.sum(particles.masses * particles.concentration[..., None])
    total /= particles.volume[0]
    npt.assert_allclose(
        total + gas.concentration[0, 0], 1.0e-12, rtol=1e-12, atol=1e-30
    )


def _run_cpu_nucleation_example() -> subprocess.CompletedProcess[str]:
    """Run the example with a finite execution limit."""
    command = [sys.executable, "-Werror", str(EXAMPLE)]
    try:
        return subprocess.run(  # noqa: S603 - fixed repository-local script
            command,
            check=False,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=EXAMPLE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        raise AssertionError(
            "CPU nucleation example timed out after "
            f"{EXAMPLE_TIMEOUT_SECONDS} seconds: {' '.join(command)}"
        ) from error


def test_cpu_nucleation_example_main_is_warning_clean() -> None:
    """The command executes successfully with warnings as errors."""
    completed = _run_cpu_nucleation_example()
    assert completed.returncode == 0, completed.stderr


def test_cpu_nucleation_example_timeout_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stalled command reports its command and timeout."""
    command = [sys.executable, "-Werror", str(EXAMPLE)]

    def raise_timeout(*_args: object, **_kwargs: object) -> None:
        raise subprocess.TimeoutExpired(command, EXAMPLE_TIMEOUT_SECONDS)

    monkeypatch.setattr(subprocess, "run", raise_timeout)
    with pytest.raises(
        AssertionError, match="timed out after 30 seconds"
    ) as error:
        _run_cpu_nucleation_example()
    assert " ".join(command) in str(error.value)
