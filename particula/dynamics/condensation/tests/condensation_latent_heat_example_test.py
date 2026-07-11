"""Smoke tests for the latent-heat condensation docs example."""

from __future__ import annotations

import json
import math
import runpy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
EXAMPLE_PATH = (
    ROOT / "docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py"
)
NOTEBOOK_PATH = (
    ROOT / "docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.ipynb"
)
DYNAMICS_INDEX_PATH = ROOT / "docs/Examples/Dynamics/index.md"
CONDENSATION_FEATURE_PATH = (
    ROOT / "docs/Features/condensation_strategy_system.md"
)
PUBLISHED_NOTEBOOK_RELATIVE_PATH = "Condensation/Condensation_Latent_Heat.ipynb"
FEATURE_NOTEBOOK_RELATIVE_PATH = (
    "../Examples/Dynamics/Condensation/Condensation_Latent_Heat.ipynb"
)
INDEX_NOTEBOOK_LABEL = (
    "[Condensation: Latent Heat Bookkeeping]"
    "(Condensation/Condensation_Latent_Heat.ipynb)"
)
INDEX_REVIEWED_DESCRIPTION_SNIPPET = (
    "Published CPU-only latent-heat bookkeeping walkthrough"
)
INDEX_BOOKKEEPING_CONTRACT_SNIPPET = "diagnostic only and does not feed back"
DOCS_INDEX_LATENT_HEAT_HEADING = (
    "**Supporting CPU latent-heat-corrected condensation diagnostics**"
)
DOCS_INDEX_LATENT_HEAT_CONTRACT_SNIPPET = (
    "without claiming temperature-feedback runtime support"
)
EFFECTIVE_ZERO_LATENT_HEAT_ENERGY_TOLERANCE = 1.0e-18


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


def test_condensation_latent_heat_run_example_returns_finite_structured_results() -> (
    None
):
    """Helper returns finite public latent-heat bookkeeping diagnostics."""
    namespace = runpy.run_path(str(EXAMPLE_PATH))
    result = namespace["run_example"]()

    assert math.isfinite(result["initial_gas_concentration"])
    assert math.isfinite(result["final_gas_concentration"])
    assert math.isfinite(result["initial_particle_mass_concentration"])
    assert math.isfinite(result["final_particle_mass_concentration"])
    assert math.isfinite(result["particle_mass_change"])
    assert math.isfinite(result["cumulative_latent_heat_energy"])
    assert result["initial_gas_concentration"] > 0.0
    assert result["final_gas_concentration"] > 0.0
    assert result["initial_particle_mass_concentration"] > 0.0
    assert result["final_particle_mass_concentration"] > 0.0
    assert result["cpu_only_note"].startswith("CPU-only example")
    assert result["bookkeeping_only_note"].startswith("Bookkeeping only")
    assert result["iteration_count"] == 5

    per_step = result["per_step_latent_heat_energies"]
    assert per_step
    assert len(per_step) == result["iteration_count"]
    assert all(math.isfinite(value) for value in per_step)
    assert np.isclose(
        result["cumulative_latent_heat_energy"],
        sum(per_step),
        rtol=1e-14,
        atol=0.0,
    )


def test_condensation_latent_heat_example_runs_without_pint(monkeypatch) -> None:
    """Already-SI example inputs do not require the optional Pint package."""
    monkeypatch.setattr("particula.util.convert_units.unit_registry", None)

    namespace = runpy.run_path(str(EXAMPLE_PATH))
    result = namespace["run_example"]()

    assert math.isfinite(result["cumulative_latent_heat_energy"])


def test_condensation_latent_heat_main_path_matches_run_example_contract(
    capsys,
) -> None:
    """Main entrypoint prints the public notes and structured diagnostics."""
    namespace = runpy.run_path(str(EXAMPLE_PATH))
    expected = namespace["run_example"]()

    runpy.run_path(str(EXAMPLE_PATH), run_name="__main__")

    output = capsys.readouterr().out
    assert expected["cpu_only_note"] in output
    assert expected["bookkeeping_only_note"] in output
    if "zero_transfer_explanation" in expected:
        assert expected["zero_transfer_explanation"] in output


def test_condensation_latent_heat_example_reports_condensation_or_explicit_zero_transfer() -> (
    None
):
    """Example shows condensation trend or explains a zero-transfer case."""
    namespace = runpy.run_path(str(EXAMPLE_PATH))
    result = namespace["run_example"]()

    cumulative = result["cumulative_latent_heat_energy"]
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


def test_condensation_latent_heat_run_example_adds_zero_transfer_explanation() -> (
    None
):
    """Zero-transfer branch adds the documented explanation string."""
    namespace = runpy.run_path(str(EXAMPLE_PATH))
    namespace["run_example"].__globals__[
        "EFFECTIVE_ZERO_LATENT_HEAT_ENERGY_TOLERANCE"
    ] = 1.0

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
    namespace = runpy.run_path(str(EXAMPLE_PATH))
    namespace["main"].__globals__["run_example"] = lambda: {
        "cpu_only_note": "CPU-only example",
        "bookkeeping_only_note": "Bookkeeping only",
        "initial_gas_concentration": 1.0,
        "final_gas_concentration": 1.0,
        "initial_particle_mass_concentration": 2.0,
        "final_particle_mass_concentration": 2.0,
        "particle_mass_change": 0.0,
        "per_step_latent_heat_energies": [0.0],
        "cumulative_latent_heat_energy": 0.0,
        "zero_transfer_explanation": "example explanation",
    }

    namespace["main"]()

    output = capsys.readouterr().out
    assert "Zero-transfer explanation: example explanation" in output


def test_condensation_latent_heat_notebook_exists_at_published_path() -> None:
    """Published latent-heat notebook exists at the documented example path."""
    assert NOTEBOOK_PATH.exists()


def test_condensation_latent_heat_notebook_keeps_paired_publication_metadata() -> (
    None
):
    """Published notebook keeps the reviewed paired-notebook contract."""
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

    assert notebook["metadata"]["jupytext"]["text_representation"] == {
        "extension": ".py",
        "format_name": "percent",
        "format_version": "1.3",
        "jupytext_version": "1.17.3",
    }
    assert notebook["cells"][0]["cell_type"] == "markdown"
    assert "CPU-only condensation workflow" in "".join(
        notebook["cells"][0]["source"]
    )
    assert "diagnostic only" in "".join(notebook["cells"][1]["source"])


def test_dynamics_index_links_published_latent_heat_notebook() -> None:
    """Dynamics index links the published latent-heat notebook artifact."""
    content = DYNAMICS_INDEX_PATH.read_text(encoding="utf-8")

    assert PUBLISHED_NOTEBOOK_RELATIVE_PATH in content
    assert INDEX_NOTEBOOK_LABEL in content
    assert INDEX_REVIEWED_DESCRIPTION_SNIPPET in content
    assert INDEX_BOOKKEEPING_CONTRACT_SNIPPET in content


def test_dynamics_index_drops_raw_latent_heat_python_command() -> None:
    """Dynamics index no longer advertises the raw python command entry."""
    content = DYNAMICS_INDEX_PATH.read_text(encoding="utf-8")

    assert (
        "python docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py"
        not in content
    )
    assert "Condensation_Latent_Heat.py" not in content


def test_condensation_feature_page_contains_single_latent_heat_example_link() -> (
    None
):
    """Feature page keeps exactly one direct latent-heat example cross-link."""
    content = CONDENSATION_FEATURE_PATH.read_text(encoding="utf-8")

    assert content.count(FEATURE_NOTEBOOK_RELATIVE_PATH) == 1


def test_docs_index_latent_heat_summary_stays_diagnostic_only() -> None:
    """Top-level docs keep the reviewed latent-heat wording contract."""
    content = ROOT.joinpath("docs/index.md").read_text(encoding="utf-8")

    assert DOCS_INDEX_LATENT_HEAT_HEADING in content
    assert "temperature-feedback runtime support" in content
    assert DOCS_INDEX_LATENT_HEAT_CONTRACT_SNIPPET in content
    assert "**Supporting non-isothermal condensation**" not in content
