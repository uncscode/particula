"""Distribution contracts, including relocated archive execution."""

import importlib.util
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "build_particula_plugins.py"
SPEC = importlib.util.spec_from_file_location("plugin_builder", SCRIPT)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


@pytest.fixture
def example():
    """Load the standalone example without importing optional plotting."""
    path = builder.PLUGIN / "scripts/run_example.py"
    spec = importlib.util.spec_from_file_location("plugin_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mass_fraction", [0.99, 1.01])
def test_coagulation_rejects_particle_drift_hidden_by_gas(
    example, mass_fraction
):
    """A gas-dominated budget cannot hide a 1% particle loss or gain."""
    particle_mass = 4.864826562917495e-13
    gas_mass = 1e-8
    rows = [
        {
            "particle_mass_kg_m-3": mass,
            "total_mass_kg_m-3": mass + gas_mass,
        }
        for mass in (
            particle_mass,
            particle_mass * mass_fraction,
            particle_mass,
        )
    ]
    checks = example.mass_balance_checks(rows, "coagulation")
    assert checks["max_relative_total_mass_error"] < 1e-6
    assert checks["max_relative_particle_mass_error"] == pytest.approx(0.01)
    assert not checks["mass_balance_pass"]


def test_coagulation_accepts_conserved_particle_mass(example):
    """A closed coagulation budget with unchanged particle mass passes."""
    row = {"particle_mass_kg_m-3": 1e-13, "total_mass_kg_m-3": 1e-8}
    checks = example.mass_balance_checks([row, dict(row)], "coagulation")
    assert checks["mass_balance_pass"]
    assert checks["max_relative_particle_mass_error"] == 0.0


@pytest.mark.parametrize("total_fraction, passed", [(1.0, True), (0.99, False)])
def test_condensation_checks_combined_budget(example, total_fraction, passed):
    """Vapor uptake may change particle mass but must conserve total mass."""
    rows = [
        {"particle_mass_kg_m-3": 1e-10, "total_mass_kg_m-3": 1e-8},
        {
            "particle_mass_kg_m-3": 2e-10,
            "total_mass_kg_m-3": 1e-8 * total_fraction,
        },
    ]
    checks = example.mass_balance_checks(rows, "condensation")
    assert checks["mass_balance_pass"] is passed
    assert "max_relative_particle_mass_error" not in checks


def test_generated_resources_and_manifests():
    """Repository installation uses the current canonical payload."""
    _, catalog = builder.validate()
    builder.sync_or_check(catalog)


def test_generated_drift_rejected(tmp_path, monkeypatch):
    """A stale generated payload cannot be shipped silently."""
    import shutil

    target = tmp_path / "plugin"
    shutil.copytree(
        builder.PLUGIN, target, ignore=shutil.ignore_patterns("node_modules")
    )
    monkeypatch.setattr(builder, "PLUGIN", target)
    _, catalog = builder.validate()
    (target / "opencode/payload/skills/particula-run/SKILL.md").write_text(
        "stale"
    )
    with pytest.raises(ValueError, match="drift"):
        builder.sync_or_check(catalog)


@pytest.mark.skipif(
    importlib.util.find_spec("matplotlib") is None,
    reason="Plotting smoke runs in assistant-plugins CI with Matplotlib",
)
def test_archives_reproducible_and_relocated_smoke(tmp_path):
    """Every shipped helper works after extraction without the source tree."""
    manifest, _ = builder.validate()
    first, second = tmp_path / "first", tmp_path / "second"
    builder.build(first, manifest)
    builder.build(second, manifest)
    for archive_path in first.glob("*.zip"):
        assert (
            archive_path.read_bytes()
            == (second / archive_path.name).read_bytes()
        )
        destination = tmp_path / archive_path.stem
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            assert archive.testzip() is None
            assert not any(
                Path(n).is_absolute() or ".." in Path(n).parts for n in names
            )
            assert not any(
                "node_modules" in n or "__pycache__" in n or n.endswith(".pyc")
                for n in names
            )
            archive.extractall(destination)
        if "opencode" in archive_path.name:
            roots = [
                destination / "particula-helper/payload",
                destination / ".opencode/skills/particula-run",
            ]
        else:
            roots = [destination]
        for index, root in enumerate(roots):
            doctor = subprocess.run(  # noqa: S603 - extracted, locally built test helpers
                [sys.executable, str(root / "scripts/doctor.py")],
                cwd=tmp_path,
                capture_output=True,
                text=True,
            )
            assert doctor.returncode == 0, doctor.stdout + doctor.stderr
            assert json.loads(doctor.stdout)["starter_ready"]
            output = tmp_path / f"{archive_path.stem}-{index}-run"
            command = [
                sys.executable,
                str(root / "scripts/run_example.py"),
                "--output",
                str(output),
            ]
            result = subprocess.run(  # noqa: S603 - extracted, locally built test helpers
                command,
                cwd=tmp_path,
                env={**os.environ, "MPLCONFIGDIR": str(tmp_path / "mpl")},
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, result.stdout + result.stderr
            report = json.loads((output / "run.json").read_text())
            assert report["checks"]["mass_balance_pass"]
            assert report["checks"]["number_conserved"]
            for name in ("time_series.csv", "distribution.csv", "result.png"):
                assert (output / name).stat().st_size > 0
            # Existing scientific results must never be silently overwritten.
            retry = subprocess.run(  # noqa: S603 - extracted, locally built test helpers
                command, cwd=tmp_path, capture_output=True, text=True
            )
            assert retry.returncode != 0
            assert "already exists" in retry.stderr
