"""Publication regressions for the supported CPU nucleation documentation."""

import ast
import re
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "docs/Examples/Nucleation/cpu_nucleation.py"
FEATURE = ROOT / "docs/Features/nucleation_strategy_system.md"
THEORY = ROOT / "docs/Theory/Technical/Dynamics/Nucleation_Equations.md"
EXAMPLE_INDEX = ROOT / "docs/Examples/Nucleation/index.md"
ROADMAP = ROOT / "docs/Features/Roadmap/data-oriented-gpu.md"
AGENTS = ROOT / "AGENTS.md"
ARCHITECTURE = ROOT / ".opencode/guides/architecture_reference.md"
ARCHITECTURE_OUTLINE = (
    ROOT / ".opencode/guides/architecture/architecture_outline.md"
)
EXAMPLE_TIMEOUT_SECONDS = 30


def _assert_local_links_resolve(document: Path) -> None:
    """Assert local Markdown destinations and anchors resolve from a document."""
    for line in document.read_text(encoding="utf-8").splitlines():
        if "](" not in line:
            continue
        destination = line.split("](", maxsplit=1)[1].split(")", maxsplit=1)[0]
        if destination.startswith(("http://", "https://", "#")):
            continue
        target, _, anchor = destination.partition("#")
        target_path = document.parent / target
        assert target_path.is_file(), f"Missing link target: {target_path}"
        if anchor:
            headings = target_path.read_text(encoding="utf-8").lower()
            normalized_anchor = " ".join(re.split(r"[^a-z0-9]+", anchor))
            normalized_headings = " ".join(re.split(r"[^a-z0-9]+", headings))
            assert normalized_anchor in normalized_headings


def test_cpu_nucleation_example_uses_public_api_and_conserves_mass() -> None:
    """The published one-box example transfers gas without changing identity."""
    namespace = runpy.run_path(str(EXAMPLE))
    aerosol = namespace["run_example"]()
    particles = aerosol.particles.data
    gas = aerosol.atmosphere.partitioning_species.data
    gas_only = aerosol.atmosphere.gas_only_species.data

    assert particles.masses.shape == (1, 3, 1)
    assert particles.masses.dtype == np.float64
    assert gas.concentration.shape == (1, 1)
    assert gas.concentration.dtype == np.float64
    assert np.any(particles.concentration > 0.0)
    assert gas.concentration[0, 0] < 1.0e-12
    npt.assert_allclose(gas_only.concentration, [[2.0e-6]], rtol=0.0, atol=0.0)
    total = np.sum(particles.masses * particles.concentration[..., None])
    npt.assert_allclose(
        total + gas.concentration.sum(),
        1.0e-12,
        rtol=1e-12,
        atol=1e-30,
    )


def _run_cpu_nucleation_example() -> subprocess.CompletedProcess[str]:
    """Run the published example with a finite execution limit."""
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
        joined_command = " ".join(command)
        raise AssertionError(
            "CPU nucleation example timed out after "
            f"{EXAMPLE_TIMEOUT_SECONDS} seconds: {joined_command}"
        ) from error


def test_cpu_nucleation_example_main_is_warning_clean() -> None:
    """The documented command executes successfully with warnings as errors."""
    completed = _run_cpu_nucleation_example()
    assert completed.returncode == 0, completed.stderr
    assert "CPU nucleation example completed." in completed.stdout


def test_cpu_nucleation_example_timeout_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stalled documented command reports its command and timeout."""
    command = [sys.executable, "-Werror", str(EXAMPLE)]

    def raise_timeout(*_args: object, **_kwargs: object) -> None:
        raise subprocess.TimeoutExpired(command, EXAMPLE_TIMEOUT_SECONDS)

    monkeypatch.setattr(subprocess, "run", raise_timeout)

    with pytest.raises(
        AssertionError, match="timed out after 30 seconds"
    ) as error:
        _run_cpu_nucleation_example()

    assert " ".join(command) in str(error.value)


def test_example_ast_does_not_reference_concrete_source_helpers() -> None:
    """The runnable example stays on the approved public P4/P5 surface."""
    source = EXAMPLE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }

    assert "particula.dynamics" in imports
    assert "particula.gas" in imports
    assert "particula.particles.exhaustion" in imports
    for forbidden in (
        "particle_source",
        "finalize_particle_source",
        "commit_particle_source",
        "ParticleSourceCommitConfig",
    ):
        assert forbidden not in source


def test_nucleation_documentation_exposes_supported_and_deferred_scope() -> (
    None
):
    """Navigation and scientific boundaries remain visible to readers."""
    feature = FEATURE.read_text(encoding="utf-8")
    theory = THEORY.read_text(encoding="utf-8")
    index = EXAMPLE_INDEX.read_text(encoding="utf-8")
    roadmap = ROADMAP.read_text(encoding="utf-8")
    agents = AGENTS.read_text(encoding="utf-8")
    architecture = ARCHITECTURE.read_text(encoding="utf-8")
    architecture_outline = ARCHITECTURE_OUTLINE.read_text(encoding="utf-8")

    assert "cpu_nucleation.py" in index
    assert "illustrative custom workflow" in index.lower()
    assert "CPU-only" in feature
    assert "single-box" in feature
    assert "each attempted substep" in feature
    assert "rtol=1e-12, atol=1e-30" in feature
    assert "Vehkamäki" in theory and "not implemented physics" in theory
    assert "E6-F7 ships a CPU-only" in roadmap
    assert "nucleation_step_gpu" in feature
    assert "concrete-only" in feature
    assert "successful\nno-op" in feature
    assert "Direct-Warp P1--P5 correspondence" in theory
    assert "accepted_demand * volume" in theory
    assert "representable as an `int32` event count" in theory
    assert "after P2 has written its documented\nsidecars" in theory
    assert "gpu_direct_nucleation.py" in index
    assert "nucleation_step_gpu" in roadmap
    assert "CPU-only runnable, never a\n  GPU fallback" in agents
    assert "P1 preflights" in agents and "P5 commits" in agents
    assert "fused P5" in architecture
    assert "Direct GPU Nucleation" in architecture_outline
    assert "no hidden transfer" in architecture_outline
    for document in (FEATURE, THEORY, EXAMPLE_INDEX, ROADMAP):
        _assert_local_links_resolve(document)
