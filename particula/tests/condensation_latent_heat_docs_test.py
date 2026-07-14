"""Docs publication tests for the latent-heat condensation example."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = (
    ROOT / "docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.ipynb"
)
DYNAMICS_INDEX_PATH = ROOT / "docs/Examples/Dynamics/index.md"
CONDENSATION_FEATURE_PATH = (
    ROOT / "docs/Features/condensation_strategy_system.md"
)
DOCS_INDEX_PATH = ROOT / "docs/index.md"
ROADMAP_PATH = ROOT / "docs/Features/Roadmap/data-oriented-gpu.md"
PUBLISHED_NOTEBOOK_RELATIVE_PATH = "Condensation/Condensation_Latent_Heat.ipynb"
FEATURE_NOTEBOOK_RELATIVE_PATH = (
    "../Examples/Dynamics/Condensation/Condensation_Latent_Heat.ipynb"
)
CPU_BASELINE_TEST_PATH = (
    "particula/integration_tests/condensation_latent_heat_conservation_test.py"
)
INDEX_NOTEBOOK_LABEL = (
    "[Condensation: Latent Heat Bookkeeping]"
    "(Condensation/Condensation_Latent_Heat.ipynb)"
)
INDEX_REVIEWED_DESCRIPTION_SNIPPET = (
    "Published CPU-only latent-heat bookkeeping walkthrough"
)
INDEX_BOOKKEEPING_CONTRACT_SNIPPET = "diagnostic only and does not feed back"
INDEX_CPU_BASELINE_SNIPPET = "The executable CPU integration baseline remains"
DOCS_INDEX_LATENT_HEAT_HEADING = (
    "**Supporting CPU latent-heat-corrected condensation diagnostics**"
)
DOCS_INDEX_DIRECT_GPU_CONTRACT_SNIPPETS = (
    "optionally applies a latent-rate correction during each of its four",
    "CPU-oracle/Warp parity coverage.",
    "Omitted latent heat,",
    "or a zero per-species value, retains that species' isothermal rate path.",
    "Broader temperature feedback, gas coupling/conservation, energy",
    "bookkeeping, and strategy/runnable-level support remain deferred.",
)
FEATURE_CPU_BASELINE_SNIPPET = (
    "This baseline is CPU-only and diagnostic/reference only"
)
FEATURE_DIRECT_GPU_CONTRACT_SNIPPETS = (
    "optional per-species latent-rate correction in each of its four equal",
    "substeps, with CPU-oracle/Warp parity coverage.",
    "using a zero entry for a species, retains that species' isothermal rate",
    "This does not provide broader temperature feedback, gas coupling or",
    "conservation, energy bookkeeping, or strategy/runnable-level latent-heat",
    "support; those remain deferred.",
)
ROADMAP_CPU_BASELINE_SNIPPET = "current executable CPU integration baseline"
ROADMAP_DIRECT_GPU_CONTRACT_SNIPPETS = (
    "Bounded direct GPU condensation applies an optional Warp-backed latent-heat",
    "rate correction during each fixed substep.",
    "It does not provide temperature",
    "feedback, gas coupling or conservation, or energy bookkeeping.",
)


def test_condensation_latent_heat_notebook_exists_at_published_path() -> None:
    """Published latent-heat notebook exists at the documented example path."""
    assert NOTEBOOK_PATH.exists()


def test_condensation_latent_heat_notebook_keeps_paired_executed_state() -> (
    None
):
    """Published notebook stays synced with the paired script and executed."""
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

    assert notebook["metadata"]["jupytext"]["text_representation"] == {
        "extension": ".py",
        "format_name": "percent",
        "format_version": "1.3",
        "jupytext_version": "1.17.3",
    }
    assert notebook["cells"][0]["cell_type"] == "markdown"
    assert "sub_steps=1" in "".join(notebook["cells"][0]["source"])

    code_cells = [
        cell for cell in notebook["cells"] if cell.get("cell_type") == "code"
    ]
    assert code_cells
    assert any(cell.get("execution_count") is not None for cell in code_cells)
    assert any(cell.get("outputs") for cell in code_cells)
    assert "Per-call latent heat energy density [J/m^3]" in "".join(
        code_cells[-1]["outputs"][0].get("text", [])
    )


def test_dynamics_index_links_published_latent_heat_notebook() -> None:
    """Dynamics index links the published latent-heat notebook artifact."""
    content = DYNAMICS_INDEX_PATH.read_text(encoding="utf-8")

    assert PUBLISHED_NOTEBOOK_RELATIVE_PATH in content
    assert INDEX_NOTEBOOK_LABEL in content
    assert INDEX_REVIEWED_DESCRIPTION_SNIPPET in content
    assert INDEX_BOOKKEEPING_CONTRACT_SNIPPET in content
    assert INDEX_CPU_BASELINE_SNIPPET in content
    assert CPU_BASELINE_TEST_PATH in content


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
    content = DOCS_INDEX_PATH.read_text(encoding="utf-8")

    assert DOCS_INDEX_LATENT_HEAT_HEADING in content
    for snippet in DOCS_INDEX_DIRECT_GPU_CONTRACT_SNIPPETS:
        assert snippet in content
    assert "thermal_work" not in content
    assert "**Supporting non-isothermal condensation**" not in content


def test_condensation_feature_page_keeps_direct_gpu_latent_heat_boundary() -> (
    None
):
    """Feature page states the bounded direct GPU latent-heat contract."""
    content = CONDENSATION_FEATURE_PATH.read_text(encoding="utf-8")

    assert CPU_BASELINE_TEST_PATH in content
    assert FEATURE_CPU_BASELINE_SNIPPET in content
    for snippet in FEATURE_DIRECT_GPU_CONTRACT_SNIPPETS:
        assert snippet in content
    assert "thermal_work" not in content


def test_roadmap_records_bounded_direct_gpu_latent_heat_support() -> None:
    """Roadmap distinguishes direct GPU support from deferred integration."""
    content = ROADMAP_PATH.read_text(encoding="utf-8")

    assert ROADMAP_CPU_BASELINE_SNIPPET in content
    assert CPU_BASELINE_TEST_PATH in content
    for snippet in ROADMAP_DIRECT_GPU_CONTRACT_SNIPPETS:
        assert snippet in content
    assert "thermal_work" not in content
    assert "with per-box temperature feedback through the" not in content
