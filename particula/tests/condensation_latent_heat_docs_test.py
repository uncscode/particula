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
FOUNDATIONS_PATH = ROOT / "docs/Features/data-containers-and-gpu-foundations.md"
MIGRATION_PATH = ROOT / "docs/Features/particle-data-migration.md"
FOUNDATIONS_CONFIGURATION_SNIPPETS = (
    "from particula.gpu.kernels import condensation_step_gpu",
    "constant `wp.int32(0)`",
    "canonical Buck `wp.int32(1)`",
    "Activity mode `0` is ideal and mode `1` is kappa",
    "surface-tension mode `0` is static and mode `1` is composition-weighted",
    "ordered-molar-mass compatibility contract",
)
FOUNDATIONS_SCHEMA_INPUT_SNIPPETS = (
    "`(n_boxes, n_particles, n_species)`",
    "`(n_boxes, n_species)`",
    "`(n_boxes,)`",
    "`(n_species,)`",
    "each may be omitted independently",
    "direct scalar",
    "direct same-device Warp array",
    "hybrid direct scalar/Warp-array inputs",
    "`environment=`",
    "non-`wp.float64` temperature arrays are normalized",
)
FOUNDATIONS_LIFECYCLE_SNIPPETS = (
    "exactly four equal `time_step / 4.0` substeps",
    "P2 finalizes that proposal against particle and gas inventory limits",
    "Later proposals read the coupled gas concentration",
    "mutate particle masses and gas concentration in place",
    "**P2-finalized, inventory-limited** transfer",
    "returned by identity",
    "supplied work storage retains only the final gated raw proposal",
    "The two-item return is `(particles, mass_transfer)`",
    "overwrite the derived GPU-only vapor-pressure buffer",
    "`energy_transfer` is a caller-owned active-device\n"
    "`wp.float64`, `(n_boxes, n_species)`, write-only output",
    "not a third return value",
    "`thermal_work` has the same validated sidecar shape but remains deferred "
    "and\nunused",
)
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
    "The direct hook couples each P2-finalized particle transfer to gas",
    "Broader temperature feedback and CPU-strategy/runnable-level support remain",
)
FEATURE_CPU_BASELINE_SNIPPET = (
    "This baseline is CPU-only and diagnostic/reference only"
)
FEATURE_DIRECT_GPU_CONTRACT_SNIPPETS = (
    "optional per-species latent-rate correction in each of its four equal",
    "substeps, with CPU-oracle/Warp parity coverage.",
    "using a zero entry for a species, retains that species' isothermal rate",
    "Issue #1272 also ships optional keyword-only caller-owned active-device",
    "This is diagnostic bookkeeping, not temperature feedback, gas mutation or",
    "gas/full-system conservation.",
)
ROADMAP_CPU_BASELINE_SNIPPET = "current executable CPU integration baseline"
ROADMAP_DIRECT_GPU_CONTRACT_SNIPPETS = (
    "The direct step applies an optional latent-heat rate correction in each"
    " of its",
    "four fixed substeps, with CPU-oracle/Warp parity coverage.",
    "E4-F4's #1272 signed diagnostic is shipped: optional keyword-only caller-owned",
    "This leaves temperature feedback, gas mutation",
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
    assert "`thermal_work` is validated but remains" in content


def test_roadmap_records_bounded_direct_gpu_latent_heat_support() -> None:
    """Roadmap distinguishes direct GPU support from deferred integration."""
    content = ROADMAP_PATH.read_text(encoding="utf-8")

    assert ROADMAP_CPU_BASELINE_SNIPPET in content
    assert CPU_BASELINE_TEST_PATH in content
    for snippet in ROADMAP_DIRECT_GPU_CONTRACT_SNIPPETS:
        assert snippet in content
    assert "`thermal_work` remains" in content
    assert "with per-box temperature feedback through the" not in content


def test_foundations_page_publishes_gpu_condensation_configuration_contract() -> (
    None
):
    """Foundations page defines the canonical direct-step configuration contract."""
    content = FOUNDATIONS_PATH.read_text(encoding="utf-8")
    normalized = " ".join(content.split())

    for snippet in (
        *FOUNDATIONS_CONFIGURATION_SNIPPETS,
        *FOUNDATIONS_SCHEMA_INPUT_SNIPPETS,
    ):
        assert " ".join(snippet.split()) in normalized


def test_foundations_page_publishes_gpu_condensation_lifecycle_contract() -> (
    None
):
    """Foundations page distinguishes finalized output from mutable work state."""
    content = FOUNDATIONS_PATH.read_text(encoding="utf-8")
    normalized = " ".join(content.split())

    for snippet in FOUNDATIONS_LIFECYCLE_SNIPPETS:
        assert " ".join(snippet.split()) in normalized


def test_foundations_page_publishes_validation_and_shipped_boundaries() -> None:
    """Foundations page preserves bounded support and rollback guardrails."""
    content = FOUNDATIONS_PATH.read_text(encoding="utf-8")
    normalized = " ".join(content.split())
    boundaries = content.split("## Current shipped support boundaries", 1)[
        1
    ].split("## Guidance for current users", 1)[0]

    for snippet in (
        "aggregate invalid state, metadata, or ownership fails with\n`ValueError` before launches or caller mutation",
        "A later failure caused by a\nfresh raw proposal does not roll back completed substeps",
        "snapshot and restore particle masses, gas concentration,\nderived vapor pressure, and caller-owned output/work buffers",
        'Warp `device="cpu"` is the baseline',
        "CUDA is additive local evidence",
        "unavailable devices skip cleanly",
        "high-level `Aerosol`/`Runnable` path",
        "automatic backend selection or fallback",
        "implicit transfer or synchronization",
        "adaptive stepping",
        "BAT",
        "staggered/Gauss-Seidel support",
    ):
        assert " ".join(snippet.split()) in normalized
    assert "E4-F6" not in boundaries
    assert "E4-F7" not in boundaries
    for prohibited_claim in (
        "supports a high-level `Aerosol`/`Runnable` path",
        "automatic backend selection or fallback is supported",
        "implicit CPU↔GPU transfer or synchronization is supported",
        "adaptive stepping is supported",
        "BAT support is shipped",
        "staggered/Gauss-Seidel support is shipped",
    ):
        assert prohibited_claim not in boundaries


def test_migration_page_links_to_canonical_gpu_condensation_contract() -> None:
    """Migration guidance summarizes, rather than forks, the GPU contract."""
    content = MIGRATION_PATH.read_text(encoding="utf-8")
    section = content.split(
        "### `condensation_step_gpu` environment inputs", 1
    )[1].split("## Conversion helpers", 1)[0]
    normalized = " ".join(section.split())

    for snippet in (
        "[Data Containers and GPU Foundations](data-containers-and-gpu-foundations.md)",
        "from particula.gpu.kernels import condensation_step_gpu",
        "thermodynamics=thermodynamics",
        "mass_transfer",
        "`to_warp_*`",
        "`from_warp_*`",
        "ordered species names",
        "synchronization and checkpoint/snapshot responsibility",
        "`gas.concentration` mutate\nin place",
        "positive-finite scalars",
        "same-device Warp arrays with shape `(n_boxes,)`",
        "hybrid scalar/Warp-array",
        "`environment=WarpEnvironmentData`",
        "derived, non-authoritative state",
        "rather than duplicating that matrix here",
    ):
        assert " ".join(snippet.split()) in normalized
    assert "particle_out, gas_out = condensation_step_gpu(" not in section
    signature = section.split("The non-executable signature is", 1)[1].split(
        "Direct temperature", 1
    )[0]
    assert "gas_out" not in signature
