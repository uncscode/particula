"""Smoke tests for the latent-heat condensation docs example."""

from __future__ import annotations

import math
import runpy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
EXAMPLE_PATH = (
    ROOT / "docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py"
)


def test_condensation_latent_heat_example_runs_as_main_entrypoint(
    capsys,
) -> None:
    """Published example path runs as ``__main__`` and prints labels."""
    runpy.run_path(str(EXAMPLE_PATH), run_name="__main__")

    output = capsys.readouterr().out

    assert "CPU-only example" in output
    assert "Bookkeeping only" in output
    assert "Gas concentration [kg/m^3]" in output
    assert "Particle mass change [kg/m^3]" in output
    assert "Per-step latent heat energy [J]" in output
    assert "Cumulative latent heat energy [J]" in output


def test_condensation_latent_heat_as_float_returns_scalar_for_scalars_and_arrays() -> (
    None
):
    """Private scalar helper normalizes scalar-like numeric inputs."""
    namespace = runpy.run_path(str(EXAMPLE_PATH))
    as_float = namespace["_as_float"]

    assert as_float(1.25) == 1.25
    assert as_float(np.array([[2.5]], dtype=np.float64)) == 2.5


def test_condensation_latent_heat_build_aerosol_returns_positive_state() -> (
    None
):
    """Private aerosol builder returns a finite supersaturated test state."""
    namespace = runpy.run_path(str(EXAMPLE_PATH))
    aerosol = namespace["_build_aerosol"]()

    gas_concentration = (
        aerosol.atmosphere.partitioning_species.get_concentration()
    )
    particle_mass_concentration = aerosol.particles.get_mass_concentration()

    assert np.isfinite(gas_concentration).all()
    assert np.isfinite(particle_mass_concentration).all()
    assert np.all(gas_concentration > 0.0)
    assert np.all(particle_mass_concentration > 0.0)


def test_condensation_latent_heat_run_example_returns_finite_structured_results() -> (
    None
):
    """Helper returns finite latent-heat bookkeeping diagnostics."""
    namespace = runpy.run_path(str(EXAMPLE_PATH))
    result = namespace["run_example"]()

    assert math.isfinite(result["initial_gas_concentration"])
    assert math.isfinite(result["final_gas_concentration"])
    assert math.isfinite(result["initial_particle_mass_concentration"])
    assert math.isfinite(result["final_particle_mass_concentration"])
    assert math.isfinite(result["cumulative_latent_heat_energy"])
    assert result["cpu_only_note"].startswith("CPU-only example")
    assert result["bookkeeping_only_note"].startswith("Bookkeeping only")

    per_step = result["per_step_latent_heat_energies"]
    assert per_step
    assert all(math.isfinite(value) for value in per_step)
    assert np.isclose(
        result["cumulative_latent_heat_energy"],
        sum(per_step),
        rtol=1e-14,
        atol=0.0,
    )


def test_condensation_latent_heat_example_reports_condensation_or_explicit_zero_transfer() -> (
    None
):
    """Example shows condensation trend or explains a zero-transfer case."""
    namespace = runpy.run_path(str(EXAMPLE_PATH))
    result = namespace["run_example"]()

    cumulative = result["cumulative_latent_heat_energy"]
    if cumulative != 0.0:
        assert cumulative > 0.0
        assert (
            result["final_gas_concentration"]
            < result["initial_gas_concentration"]
        )
        assert (
            result["final_particle_mass_concentration"]
            > result["initial_particle_mass_concentration"]
        )
    else:
        explanation = result.get("zero_transfer_explanation", "")
        assert explanation
        assert "latent-heat signal" in explanation
