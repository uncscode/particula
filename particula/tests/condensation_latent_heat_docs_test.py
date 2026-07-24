"""Docs publication tests for the latent-heat condensation example."""

from __future__ import annotations

import json
import re
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
ROADMAP_INDEX_PATH = ROOT / "docs/Features/Roadmap/index.md"
FOUNDATIONS_PATH = ROOT / "docs/Features/data-containers-and-gpu-foundations.md"
MIGRATION_PATH = ROOT / "docs/Features/particle-data-migration.md"
README_PATH = ROOT / "readme.md"
EXAMPLES_INDEX_PATH = ROOT / "docs/Examples/index.md"
CANONICAL_CONTRACT_LABEL = "Canonical low-level direct-condensation contract"
EXAMPLES_CONTRACT_DESTINATION = (
    "../Features/data-containers-and-gpu-foundations.md"
)
README_CONTRACT_DESTINATION = (
    "./docs/Features/data-containers-and-gpu-foundations.md"
)
EXAMPLES_CONTRACT_LINK = (
    f"[{CANONICAL_CONTRACT_LABEL}]({EXAMPLES_CONTRACT_DESTINATION})"
)
README_CONTRACT_LINK = (
    f"[{CANONICAL_CONTRACT_LABEL}]({README_CONTRACT_DESTINATION})"
)
P2_QUICK_START_SOURCE = "Direct GPU kernels quick-start source"
P2_QUICK_START_SOURCE_LINK = (
    "[Direct GPU kernels quick-start source]"
    "(https://github.com/Gorkowski/particula/blob/main/docs/Examples/"
    "gpu_direct_kernels_quick_start.py)"
)
FOUNDATIONS_P3_HEADING = "### Focused reproduction commands"
MIGRATION_P3_HEADING = (
    "### Direct-condensation troubleshooting and reproduction"
)
P3_BASELINE_COMMANDS = (
    "python docs/Examples/gpu_direct_kernels_quick_start.py",
    "pytest particula/gpu/tests/gpu_direct_kernels_example_test.py -q",
    "pytest particula/gpu/kernels/tests/condensation_test.py -q -Werror",
    "pytest particula/gpu/kernels/tests/condensation_stiffness_test.py "
    "-q -Werror",
    "pytest particula/gpu/dynamics/tests/coagulation_funcs_test.py -q -Werror",
    "pytest particula/gpu/kernels/tests/coagulation_test.py -q -Werror",
    "pytest particula/gpu/kernels/tests/coagulation_validation_test.py -q "
    '-m "warp and gpu_parity" -Werror',
    "pytest particula/gpu/kernels/tests/"
    "coagulation_stochastic_validation_test.py -q "
    '-m "warp and stochastic and not cuda" -Werror',
    "pytest particula/tests/gpu_coagulation_docs_test.py -q -Werror",
    "pytest particula/integration_tests/"
    "condensation_latent_heat_conservation_test.py -q",
    "pytest particula/integration_tests/"
    "condensation_particle_resolved_test.py -q",
    "pytest particula/tests/condensation_latent_heat_docs_test.py -q -Werror",
)
P3_COMMAND_TARGETS = (
    ROOT / "docs/Examples/gpu_direct_kernels_quick_start.py",
    ROOT / "particula/gpu/tests/gpu_direct_kernels_example_test.py",
    ROOT / "particula/gpu/kernels/tests/condensation_test.py",
    ROOT / "particula/gpu/kernels/tests/condensation_stiffness_test.py",
    ROOT / "particula/gpu/dynamics/tests/coagulation_funcs_test.py",
    ROOT / "particula/gpu/kernels/tests/coagulation_test.py",
    ROOT / "particula/gpu/kernels/tests/coagulation_validation_test.py",
    ROOT
    / "particula/gpu/kernels/tests/coagulation_stochastic_validation_test.py",
    ROOT / "particula/tests/gpu_coagulation_docs_test.py",
    ROOT
    / "particula/integration_tests/condensation_latent_heat_conservation_test.py",
    ROOT / "particula/integration_tests/condensation_particle_resolved_test.py",
    ROOT / "particula/tests/condensation_latent_heat_docs_test.py",
)
P3_CUDA_COMMANDS = (
    "pytest particula/gpu/kernels/tests/condensation_test.py -q "
    '-m "warp and cuda" -Werror',
    "pytest particula/gpu/kernels/tests/"
    "coagulation_stochastic_validation_test.py -q "
    '-m "warp and cuda" -Werror',
)
P3_FOUNDATIONS_SNIPPETS = (
    'Warp `device="cpu"`',
    "**Optional/local CUDA evidence:**",
    "skips cleanly when CUDA is unavailable",
    "ordered CPU gas-name metadata",
    "thermodynamics-sidecar species order",
    "`(n_boxes, ...)`",
    "`wp.float64`",
    "`environment=`",
    "positive finite physical values",
    "P2 inventory-limited applied transfers",
    "Explicitly synchronize before observing it on the host",
    "particle-mass/gas-concentration parity matrix",
    "particle-plus-gas inventory conservation checks",
    "latent-heat energy/bookkeeping",
)
P3_MIGRATION_SNIPPETS = (
    "ordered gas names",
    "`(n_boxes, ...)`",
    "`wp.float64`",
    "`environment=`",
    "positive finite temperature/pressure",
    "P2 inventory limiting",
    "Synchronize explicitly",
    'Warp `device="cpu"`',
    "CUDA is optional/local",
    "skips cleanly when CUDA is unavailable",
    "valid water-species index",
    "caller-owned energy output",
)
P3_ANCHOR_LINK = (
    "./docs/Features/data-containers-and-gpu-foundations.md"
    "#focused-reproduction-commands"
)
EPIC_D_COMPLETED_PUBLICATION = "E4-F1--E4-F7 recorded evidence is complete"
EPIC_D_DECISION_RECORD_LINK = (
    "[fixed-four decision record](condensation-stiffness-study.md)"
)
EPIC_D_DEFERRED_BOUNDARIES = (
    "Temperature feedback",
    "high-level `Aerosol`/`Runnable` integration",
    "adaptive stepping",
    "unsupported physics",
    "graph capture/replay",
    "broad autodiff",
    "general CPU-strategy parity remain future work outside the shipped scope",
)
FOUNDATIONS_CONFIGURATION_SNIPPETS = (
    "from particula.gpu.kernels import condensation_step_gpu",
    "from particula.gpu.kernels.thermodynamics import ThermodynamicsConfig",
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
    "validates that delta and all pending commit values before mutating particle",
    "Later proposals read the coupled gas concentration",
    "mutate particle masses and gas concentration in place",
    "**P2-finalized, inventory-limited** transfer",
    "returned by identity",
    "supplied work storage retains only the final gated raw proposal",
    "The two-item return assignment is\n"
    "`particles_out, mass_transfer = condensation_step_gpu(...)`",
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


def _normalized_section(content: str, heading: str) -> str:
    """Return a heading section through the next same-or-higher heading."""
    start = content.index(heading)
    section = content[start:]
    level = len(heading) - len(heading.lstrip("#"))
    for line_index, line in enumerate(section.splitlines()[1:], start=1):
        if line.startswith("#") and len(line) - len(line.lstrip("#")) <= level:
            section = "\n".join(section.splitlines()[:line_index])
            break
    return " ".join(section.split())


def _canonical_contract_destinations(content: str) -> list[str]:
    """Return destinations for links bearing the canonical-contract label."""
    return re.findall(
        rf"\[{re.escape(CANONICAL_CONTRACT_LABEL)}\]\(([^)]+)\)",
        content,
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
    condensation_section = content.split(
        "### GPU thermodynamics and condensation refresh", 1
    )[1].split("### Condensation activity and surface sidecars", 1)[0]
    sidecar_section = content.split(
        "### Condensation activity and surface sidecars", 1
    )[1].split("### Latent-rate correction and energy diagnostic", 1)[0]
    lifecycle_section = content.split(
        "### Latent-rate correction and energy diagnostic", 1
    )[1].split("## Current shipped support boundaries", 1)[0]
    boundaries = content.split("## Current shipped support boundaries", 1)[
        1
    ].split("## Guidance for current users", 1)[0]

    for snippet in (
        "aggregate invalid state, metadata, or ownership fails with\n`ValueError` before mutable physics or caller-state mutation",
        "Preflight may run\nread-only validation kernels",
        "A later failure caused by a\nfresh raw proposal does not roll back completed substeps",
    ):
        assert " ".join(snippet.split()) in " ".join(sidecar_section.split())
    for snippet in (
        "snapshot and restore particle masses, gas concentration,\nderived vapor pressure, and caller-owned output/work buffers",
    ):
        assert " ".join(snippet.split()) in " ".join(lifecycle_section.split())
    for snippet in (
        "Particle mass, transfer, and scratch transfer arrays are active-device\n`wp.float64`",
        "gas concentration\nand energy arrays are active-device `wp.float64`",
        "scratch property arrays are active-device `wp.float64`",
        "latent heat and thermal work arrays are\nactive-device `wp.float64`",
        "P2 demand, release, and scale sidecars must likewise be caller-owned,\nactive-device, stable-shape `wp.float64` arrays",
    ):
        assert " ".join(snippet.split()) in " ".join(sidecar_section.split())
    for snippet in (
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
        assert " ".join(snippet.split()) in " ".join(boundaries.split())
    assert "before launches or caller mutation" not in sidecar_section
    assert "No hidden CPU↔GPU synchronization occurs" not in boundaries
    assert (
        "Callers remain responsible for synchronization before\n  host observation or restoration"
        in boundaries
    )
    assert (
        "CUDA preflight validation-flag readbacks may\n  synchronize"
        in boundaries
    )
    assert "does not perform hidden\ntransfers" in condensation_section
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
        "particles_out, mass_transfer = condensation_step_gpu(...,",
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


def test_p3_scoped_sections_publish_required_remedies() -> None:
    """P3 troubleshooting sections retain their bounded remediation text."""
    foundations = _normalized_section(
        FOUNDATIONS_PATH.read_text(encoding="utf-8"),
        "## Direct-kernel troubleshooting",
    )
    migration = _normalized_section(
        MIGRATION_PATH.read_text(encoding="utf-8"),
        MIGRATION_P3_HEADING,
    )

    for snippet in P3_FOUNDATIONS_SNIPPETS:
        assert " ".join(snippet.split()) in foundations
    for snippet in P3_MIGRATION_SNIPPETS:
        assert " ".join(snippet.split()) in migration


def test_p3_command_matrix_has_exact_commands_and_existing_targets() -> None:
    """Published command matrix uses delivered paths and warning flags."""
    content = FOUNDATIONS_PATH.read_text(encoding="utf-8")
    commands = _normalized_section(content, FOUNDATIONS_P3_HEADING)

    assert FOUNDATIONS_P3_HEADING in content
    assert content.count(FOUNDATIONS_P3_HEADING) == 1
    for command in P3_BASELINE_COMMANDS:
        assert command in commands
    for command in P3_CUDA_COMMANDS:
        assert command in commands
    assert '-m "warp and cuda"' in commands
    assert all("-q" in command for command in P3_BASELINE_COMMANDS[1:])
    assert commands.count("-Werror") == 11
    for target in P3_COMMAND_TARGETS:
        assert target.exists()


def test_p3_cross_links_keep_one_canonical_command_matrix() -> None:
    """README and migration page discover the one canonical command matrix."""
    readme = README_PATH.read_text(encoding="utf-8")
    migration = _normalized_section(
        MIGRATION_PATH.read_text(encoding="utf-8"),
        MIGRATION_P3_HEADING,
    )

    assert readme.count(P3_ANCHOR_LINK) == 1
    assert migration.count("#focused-reproduction-commands") == 1
    assert "pytest " not in migration
    assert (
        "python docs/Examples/gpu_direct_kernels_quick_start.py"
        not in migration
    )


def test_p3_command_evidence_and_scope_remain_bounded() -> None:
    """P3 separates evidence classes without expanding runtime support."""
    foundations = _normalized_section(
        FOUNDATIONS_PATH.read_text(encoding="utf-8"),
        "## Direct-kernel troubleshooting",
    )
    migration = _normalized_section(
        MIGRATION_PATH.read_text(encoding="utf-8"),
        MIGRATION_P3_HEADING,
    )

    assert (
        "none establishes either of the other evidence classes" in foundations
    )
    assert "no one class proves either of the others" in foundations
    for prohibited_claim in (
        "automatic migration is supported",
        "high-level `Aerosol`/`Runnable` support",
        "mandatory CUDA",
        "implicit transfer/synchronization",
        "general CPU-strategy/runnable parity",
    ):
        assert prohibited_claim not in foundations
        assert prohibited_claim not in migration


def test_readme_describes_the_two_call_direct_condensation_quick_start() -> (
    None
):
    """README describes current quick-start sidecars without stale RNG claims."""
    quick_start = " ".join(README_PATH.read_text(encoding="utf-8").split())

    assert (
        "two direct condensation calls with reused scratch buffers,"
        in quick_start
    )
    assert "latent-heat, and energy sidecars" in quick_start
    assert "one condensation step, one coagulation step" not in quick_start
    assert "caller-owned `rng_states`" not in quick_start


def test_p3_matrix_labels_cpu_integration_commands_as_cpu_evidence() -> None:
    """CPU integration commands remain distinct from direct-GPU evidence."""
    commands = _normalized_section(
        FOUNDATIONS_PATH.read_text(encoding="utf-8"),
        FOUNDATIONS_P3_HEADING,
    )

    assert (
        "CPU integration/inventory-conservation evidence (separate" in commands
    )
    assert (
        "inventory conservation checks); not direct-GPU validation" in commands
    )
    assert (
        "CPU integration evidence for particle-resolved condensation; not"
        in commands
    )


def test_migration_page_groups_sidecars_by_their_required_shapes() -> None:
    """Migration troubleshooting avoids a universal per-box sidecar claim."""
    migration = _normalized_section(
        MIGRATION_PATH.read_text(encoding="utf-8"),
        MIGRATION_P3_HEADING,
    )

    for snippet in (
        "species configuration uses `(n_species,)`",
        "scratch property fields use `(n_boxes,)`",
        "per-particle or per-species transfer shapes",
        "water-species index, remain scalar",
    ):
        assert snippet in migration


def test_example_index_links_canonical_low_level_condensation_contract() -> (
    None
):
    """Examples index has one resolving canonical contract discovery link."""
    content = EXAMPLES_INDEX_PATH.read_text(encoding="utf-8")

    assert content.count(EXAMPLES_CONTRACT_LINK) == 1
    assert content.count(EXAMPLES_CONTRACT_DESTINATION) == 1
    assert _canonical_contract_destinations(content) == [
        EXAMPLES_CONTRACT_DESTINATION
    ]
    assert (
        EXAMPLES_INDEX_PATH.parent / EXAMPLES_CONTRACT_DESTINATION
    ).resolve() == FOUNDATIONS_PATH
    assert P2_QUICK_START_SOURCE_LINK in content
    assert P2_QUICK_START_SOURCE != CANONICAL_CONTRACT_LABEL


def test_readme_links_canonical_low_level_condensation_contract() -> None:
    """README keeps distinct canonical-contract and P3 troubleshooting links."""
    content = README_PATH.read_text(encoding="utf-8")

    assert content.count(README_CONTRACT_LINK) == 1
    assert content.count(P3_ANCHOR_LINK) == 1
    assert README_CONTRACT_LINK != P3_ANCHOR_LINK
    assert _canonical_contract_destinations(content) == [
        README_CONTRACT_DESTINATION
    ]
    assert (README_PATH.parent / README_CONTRACT_DESTINATION).resolve() == (
        FOUNDATIONS_PATH
    )


def test_roadmap_marks_e4_low_level_condensation_publication_shipped() -> None:
    """Epic D records the bounded completed low-level publication."""
    content = ROADMAP_PATH.read_text(encoding="utf-8")
    epic_d_start = content.index("## Epic D: GPU Condensation Physics Parity")
    epic_e_start = content.index("## Epic E: GPU Coagulation Physics Coverage")
    roadmap = " ".join(content[epic_d_start:epic_e_start].split())

    assert (
        "| 4 | [Epic D: GPU Condensation Physics Parity]"
        "(#epic-d-gpu-condensation-physics-parity) | Shipped | E4 |"
    ) in content
    assert "Status: shipped." in roadmap
    for snippet in (
        EPIC_D_COMPLETED_PUBLICATION,
        "particula.gpu.kernels",
        "exactly four `time_step / 4.0` substeps",
        "P2-finalized",
        "fp64",
        "Warp `cpu` baseline",
        "Optional/local additive CUDA evidence",
        "skips cleanly when unavailable",
        "../data-containers-and-gpu-foundations.md",
        EPIC_D_DECISION_RECORD_LINK,
        *EPIC_D_DEFERRED_BOUNDARIES,
    ):
        assert " ".join(snippet.split()) in roadmap
    for stale_claim in (
        "In progress | E4-F5",
        "E4-F6 remains",
        "E4-F7 remains",
        "final support-contract",
        "Status: in progress",
    ):
        assert stale_claim not in roadmap
    for unsupported_claim in (
        "high-level backends are shipped",
        "automatic transfers are shipped",
        "adaptive stepping is shipped",
    ):
        assert unsupported_claim not in roadmap


def test_roadmap_index_labels_e4_as_bounded_low_level_support() -> None:
    """Roadmap index links shipped E4 without expanding its support claim."""
    content = " ".join(ROADMAP_INDEX_PATH.read_text(encoding="utf-8").split())

    assert "(ADW plan E4)" in content
    assert "bounded low-level direct-condensation publication" in content
    assert "fixed-four P2 inventory finalization" in content
    assert "It does not provide high-level runnable integration" in content
    assert "general CPU-strategy parity" in content
