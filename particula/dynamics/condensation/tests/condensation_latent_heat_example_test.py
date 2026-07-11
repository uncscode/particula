"""Runtime smoke tests for the latent-heat condensation docs example."""

from __future__ import annotations

import math
import runpy
from pathlib import Path
from typing import Any, Callable, TypedDict, cast

import numpy as np
import particula as par
import pytest

ROOT = Path(__file__).resolve().parents[4]
EXAMPLE_PATH = (
    ROOT / "docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py"
)
EFFECTIVE_ZERO_LATENT_HEAT_ENERGY_TOLERANCE = 1.0e-18


class ExampleResult(TypedDict, total=False):
    """Structured result returned by the latent-heat example."""

    initial_gas_concentration: float
    final_gas_concentration: float
    initial_particle_mass_concentration: float
    final_particle_mass_concentration: float
    particle_mass_change: float
    per_call_latent_heat_energy_densities: list[float]
    cumulative_latent_heat_energy_density: float
    latent_heat_reference: float
    cpu_only_note: str
    bookkeeping_only_note: str
    iteration_count: int
    sub_steps_per_call: int
    zero_transfer_explanation: str


class ExampleNamespace(TypedDict):
    """Typed namespace loaded from the example script."""

    run_example: Callable[[], ExampleResult]
    _as_float: Callable[[object], float]
    _build_aerosol: Callable[[], par.Aerosol]
    main: Callable[[], None]


@pytest.fixture(scope="module")
def example_namespace() -> ExampleNamespace:
    """Load the published example module once for runtime assertions."""
    return cast(ExampleNamespace, runpy.run_path(str(EXAMPLE_PATH)))


@pytest.fixture(scope="module")
def example_result(example_namespace: ExampleNamespace) -> ExampleResult:
    """Run the example once and share the structured result."""
    run_example = example_namespace["run_example"]
    return run_example()


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
    assert "Per-call latent heat energy density [J/m^3]" in output
    assert "Cumulative latent heat energy density [J/m^3]" in output


def test_condensation_latent_heat_run_example_returns_finite_structured_results(
    example_result: ExampleResult,
) -> None:
    """Helper returns finite public latent-heat bookkeeping diagnostics."""
    result = example_result

    assert math.isfinite(result["initial_gas_concentration"])
    assert math.isfinite(result["final_gas_concentration"])
    assert math.isfinite(result["initial_particle_mass_concentration"])
    assert math.isfinite(result["final_particle_mass_concentration"])
    assert math.isfinite(result["particle_mass_change"])
    assert math.isfinite(result["cumulative_latent_heat_energy_density"])
    assert result["initial_gas_concentration"] > 0.0
    assert result["final_gas_concentration"] > 0.0
    assert result["initial_particle_mass_concentration"] > 0.0
    assert result["final_particle_mass_concentration"] > 0.0
    assert result["cpu_only_note"].startswith("CPU-only example")
    assert result["bookkeeping_only_note"].startswith("Bookkeeping only")
    assert result["iteration_count"] == 5
    assert result["sub_steps_per_call"] == 1

    per_call = result["per_call_latent_heat_energy_densities"]
    assert per_call
    assert len(per_call) == result["iteration_count"]
    assert all(math.isfinite(value) for value in per_call)
    assert np.isclose(
        result["cumulative_latent_heat_energy_density"],
        sum(per_call),
        rtol=1e-14,
        atol=0.0,
    )


def test_condensation_latent_heat_example_runs_without_pint(
    monkeypatch,
) -> None:
    """Already-SI example inputs do not require the optional Pint package."""
    monkeypatch.setattr("particula.util.convert_units.unit_registry", None)

    namespace = runpy.run_path(str(EXAMPLE_PATH))
    result = namespace["run_example"]()

    assert math.isfinite(result["cumulative_latent_heat_energy_density"])


def test_condensation_latent_heat_as_float_uses_first_scalar_value(
    example_namespace: ExampleNamespace,
) -> None:
    """Private scalar helper returns the first finite numeric entry as float."""
    as_float = example_namespace["_as_float"]
    assert callable(as_float)

    result = as_float(np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64))

    assert isinstance(result, float)
    assert result == 1.5


def test_condensation_latent_heat_build_aerosol_creates_single_box_state(
    example_namespace: ExampleNamespace,
) -> None:
    """Private builder helper creates the documented supersaturated CPU setup."""
    build_aerosol = example_namespace["_build_aerosol"]
    assert callable(build_aerosol)

    aerosol = build_aerosol()
    gas_species = aerosol.atmosphere.partitioning_species

    assert gas_species.get_name() == "H2O"
    assert aerosol.atmosphere.temperature == pytest.approx(298.15)
    assert aerosol.atmosphere.total_pressure == pytest.approx(101325.0)
    particle_mass = aerosol.particles.get_mass()

    assert particle_mass.shape == (4,)
    assert np.all(particle_mass > 0.0)
    assert np.all(gas_species.get_concentration() > 0.0)


def test_condensation_latent_heat_main_path_matches_run_example_contract(
    capsys,
    example_result: ExampleResult,
) -> None:
    """Main entrypoint prints the public notes and structured diagnostics."""
    runpy.run_path(str(EXAMPLE_PATH), run_name="__main__")

    output = capsys.readouterr().out
    assert example_result["cpu_only_note"] in output
    assert example_result["bookkeeping_only_note"] in output
    if "zero_transfer_explanation" in example_result:
        assert example_result["zero_transfer_explanation"] in output


def test_condensation_latent_heat_example_reports_condensation_or_explicit_zero_transfer(
    example_result: ExampleResult,
) -> None:
    """Example shows condensation trend or explains a zero-transfer case."""
    result = example_result

    cumulative = result["cumulative_latent_heat_energy_density"]
    is_effectively_zero = math.isclose(
        cumulative,
        0.0,
        rel_tol=0.0,
        abs_tol=EFFECTIVE_ZERO_LATENT_HEAT_ENERGY_TOLERANCE,
    )
    if not is_effectively_zero:
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


def test_condensation_latent_heat_example_energy_matches_mass_transfer_contract(
    example_result: ExampleResult,
) -> None:
    """Cumulative energy density matches concentration change times latent heat."""
    expected_energy = (
        example_result["particle_mass_change"]
        * example_result["latent_heat_reference"]
    )

    assert np.isclose(
        example_result["cumulative_latent_heat_energy_density"],
        expected_energy,
        rtol=1e-12,
        atol=0.0,
    )


def test_condensation_latent_heat_run_example_adds_zero_transfer_explanation() -> (
    None
):
    """Zero-transfer branch adds the documented explanation string."""
    namespace = cast(ExampleNamespace, runpy.run_path(str(EXAMPLE_PATH)))
    run_example = cast(Any, namespace["run_example"])
    run_example.__globals__["EFFECTIVE_ZERO_LATENT_HEAT_ENERGY_TOLERANCE"] = 1.0

    result = namespace["run_example"]()

    assert "zero_transfer_explanation" in result
    assert (
        "did not transfer measurable vapor mass"
        in result["zero_transfer_explanation"]
    )


def test_condensation_latent_heat_main_prints_zero_transfer_explanation(
    capsys,
) -> None:
    """Main prints the zero-transfer explanation when present in results."""
    namespace = cast(ExampleNamespace, runpy.run_path(str(EXAMPLE_PATH)))
    main = cast(Any, namespace["main"])
    main.__globals__["run_example"] = lambda: {
        "cpu_only_note": "CPU-only example",
        "bookkeeping_only_note": "Bookkeeping only",
        "initial_gas_concentration": 1.0,
        "final_gas_concentration": 1.0,
        "initial_particle_mass_concentration": 2.0,
        "final_particle_mass_concentration": 2.0,
        "particle_mass_change": 0.0,
        "per_call_latent_heat_energy_densities": [0.0],
        "cumulative_latent_heat_energy_density": 0.0,
        "zero_transfer_explanation": "example explanation",
    }

    namespace["main"]()

    output = capsys.readouterr().out
    assert "Zero-transfer explanation: example explanation" in output
