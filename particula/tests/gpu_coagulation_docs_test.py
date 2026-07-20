"""Hardware-free publication tests for the GPU coagulation contract."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

ROOT = Path(__file__).resolve().parents[2]
FOUNDATIONS_PATH = ROOT / "docs/Features/data-containers-and-gpu-foundations.md"
STRATEGY_PATH = ROOT / "docs/Features/coagulation_strategy_system.md"
VALIDATION_PATH = ROOT / "docs/Features/Roadmap/coagulation-validation.md"
DETAILED_ROADMAP_PATH = ROOT / "docs/Features/Roadmap/data-oriented-gpu.md"
ROADMAP_INDEX_PATH = ROOT / "docs/Features/Roadmap/index.md"
TESTING_GUIDE_PATH = ROOT / ".opencode/guides/testing_guide.md"
EXAMPLES_INDEX_PATH = ROOT / "docs/Examples/index.md"
E5_PLAN_PATH = ROOT / ".opencode/plans/epics/E5.json"
E5_F2_PLAN_PATH = ROOT / ".opencode/plans/features/E5-F2.json"
E5_F9_PLAN_PATH = ROOT / ".opencode/plans/features/E5-F9.json"
FOUNDATIONS_HEADING = "### GPU coagulation configuration and sidecar ownership"
STRATEGY_HEADING = "### GPU direct-kernel foundations and limitations"
RELEASE_VALIDATION_HEADING = "### Release-validation command sets"
BASELINE_COMMANDS = (
    "pytest particula/gpu/kernels/tests/coagulation_validation_test.py -q "
    '-m "warp and gpu_parity" -Werror',
    "pytest particula/gpu/kernels/tests/"
    "coagulation_stochastic_validation_test.py -q "
    '-m "warp and stochastic and not cuda" -Werror',
    "pytest particula/tests/gpu_coagulation_docs_test.py -q -Werror",
)
OPTIONAL_CUDA_COMMAND = (
    "pytest particula/gpu/kernels/"
    "tests/coagulation_stochastic_validation_test.py -q "
    '-m "warp and cuda" -Werror'
)
DIRECT_COAGULATION_COMMAND = (
    "pytest particula/gpu/kernels/tests/coagulation_test.py -q -Werror"
)
EXPECTED_E5_ROWS = (
    (
        "E5",
        "GPU Coagulation Physics Coverage",
        "Shipped",
    ),
    ("E5-F1", "Mechanism Configuration and Sampling Contract", "Shipped"),
    (
        "E5-F2",
        "Charged Pair Physics and Charge-Conserving Merges",
        "Shipped",
    ),
    ("E5-F3", "Charged and Brownian-Plus-Charged GPU Execution", "Shipped"),
    ("E5-F4", "SP2016 Sedimentation GPU Execution", "Shipped"),
    ("E5-F5", "ST1956 Turbulent-Shear GPU Execution", "Shipped"),
    ("E5-F6", "Single-Pass Additive Multi-Mechanism Coagulation", "Shipped"),
    ("E5-F7", "Cross-Mechanism GPU Validation Matrix", "Shipped"),
    (
        "E5-F8",
        "Independent CPU-Warp Condensation Walkthrough",
        "Shipped",
    ),
    (
        "E5-F9",
        "GPU Coagulation Support Documentation and Epic Closeout",
        "Shipped",
    ),
)
EXPECTED_E5_ARTIFACTS = (
    ("GPU coagulation validation record", "coagulation-validation.md"),
    (
        "condensation parity walkthrough ownership record",
        "condensation-parity-walkthrough.md",
    ),
    (
        "GPU condensation parity walkthrough",
        "../../Examples/gpu_condensation_parity_walkthrough.py",
    ),
)


def _normalized(content: str) -> str:
    """Return content with consecutive whitespace normalized."""
    return " ".join(content.split())


def _section(content: str, heading: str) -> str:
    """Return a Markdown section through its next same-or-higher heading."""
    start = content.index(heading)
    section = content[start:]
    level = len(heading) - len(heading.lstrip("#"))
    in_code_block = False
    for line_index, line in enumerate(section.splitlines()[1:], start=1):
        if line.startswith("```"):
            in_code_block = not in_code_block
        if (
            not in_code_block
            and line.startswith("#")
            and len(line) - len(line.lstrip("#")) <= level
        ):
            return "\n".join(section.splitlines()[:line_index])
    return section


def _roadmap_record(content: str) -> str:
    """Return the unique E5 roadmap inventory section."""
    heading = "### E5 roadmap inventory"
    assert content.count(heading) == 1
    return _section(content, heading)


def _record_rows(record: str) -> list[tuple[str, str, str]]:
    """Return strict three-cell E5 Markdown table rows from a record."""
    rows = []
    for line in record.splitlines():
        cells = [cell.strip() for cell in line.strip().split("|")]
        if len(cells) != 5 or cells[0] or cells[-1]:
            continue
        identifier = re.fullmatch(r"`(E5(?:-F[1-9])?)`", cells[1])
        if identifier:
            rows.append((identifier.group(1), cells[2], cells[3]))
    return rows


def _labeled_destinations(record: str) -> list[tuple[str, str]]:
    """Return Markdown link labels and destinations from a record."""
    return re.findall(r"\[([^]]+)\]\(([^)]+)\)", record)


def _local_destinations(content: str) -> list[str]:
    """Return local Markdown link destinations without anchors or remote URLs."""
    destinations = re.findall(r"\[[^]]+\]\(([^)]+)\)", content)
    return [
        destination.split("#", maxsplit=1)[0]
        for destination in destinations
        if not destination.startswith(("http://", "https://", "#"))
    ]


def _data_containers_card(content: str) -> str:
    """Return the Data Containers tutorial card."""
    start = content.index("-   __[Data Containers]")
    return content[start : content.index("-   __[Particle Phase]", start)]


def _resolving_destinations(path: Path, content: str) -> list[str]:
    """Return local Markdown destinations that resolve from a document."""
    return [
        destination
        for destination in _local_destinations(content)
        if (path.parent / destination).resolve().exists()
    ]


@dataclass(frozen=True)
class _CloseoutInputs:
    """Independent evidence inputs for the fail-closed closeout gate."""

    child_states: Mapping[str, bool]
    artifacts_valid: bool
    references_valid: bool
    links_valid: bool
    focused_commands: Mapping[str, Literal["passed", "failed", "skipped"]]
    warp_installed: bool
    warp_cpu_result: Literal["passed", "failed", "skipped", "not_applicable"]
    cuda_result: Literal["passed", "failed", "skipped", "not_applicable"]


def _closeout_gate(
    inputs: _CloseoutInputs,
) -> Literal["blocked"] | tuple[str, str, str]:
    """Return the permitted closeout transition only for complete evidence."""
    if (
        not inputs.child_states
        or not all(inputs.child_states.values())
        or not inputs.artifacts_valid
        or not inputs.references_valid
        or not inputs.links_valid
        or not inputs.focused_commands
        or any(
            result != "passed" for result in inputs.focused_commands.values()
        )
        or (inputs.warp_installed and inputs.warp_cpu_result != "passed")
        or (
            not inputs.warp_installed
            and inputs.warp_cpu_result != "not_applicable"
        )
        or inputs.cuda_result == "failed"
    ):
        return "blocked"
    return ("E5 shipped", "E5-F9 shipped", "Epic F active")


def _detailed_epic_index_statuses(content: str) -> dict[str, str]:
    """Return the Epic E and F statuses from the detailed index table."""
    statuses = {}
    for number, epic in (("5", "E"), ("6", "F")):
        match = re.search(
            rf"^\|\s*{number}\s*\|\s*\[Epic {epic}:.*?\]\([^)]+\)"
            r"\s*\|\s*([^|]+?)\s*\|",
            content,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        assert match, f"Epic {epic} is missing from the detailed roadmap index"
        statuses[epic] = match.group(1).strip().lower()
    return statuses


def _roadmap_index_statuses(content: str) -> dict[str, str]:
    """Return Epic E and F statuses from their roadmap-index sections."""
    shipped = _section(content, "### Shipped")
    active = _section(content, "### Active")
    pending = _section(content, "### Pending")
    epic_e = "[Epic E: GPU Coagulation Physics Coverage]("
    epic_f = "[Epic F: GPU Process Completeness]("

    assert shipped.count(epic_e) == 1
    assert active.count(epic_e) == 0
    assert pending.count(epic_e) == 0
    assert active.count(epic_f) == 1
    assert pending.count(epic_f) == 0
    return {"E": "shipped", "F": "active"}


def _command_target(command: str) -> Path:
    """Return the repository target named by a published pytest command."""
    match = re.search(r"\b(particula/[^\s]+)", command)
    assert match, f"Command has no particula target: {command}"
    return ROOT / match.group(1)


def test_foundations_guide_publishes_canonical_direct_coagulation_contract() -> (
    None
):
    """Foundations guide keeps the direct coagulation boundary complete."""
    section = _normalized(
        _section(
            FOUNDATIONS_PATH.read_text(encoding="utf-8"), FOUNDATIONS_HEADING
        )
    )

    for snippet in (
        "from particula.gpu.kernels import coagulation_step_gpu",
        "from particula.gpu.kernels.coagulation import CoagulationMechanismConfig",
        "immutable `mechanism_config=CoagulationMechanismConfig(...)` as keyword-only host metadata",
        "not re-exported by `particula.gpu.kernels`",
        '`distribution_type="particle_resolved"',
        "`brownian`",
        "`charged_hard_sphere`",
        "`sedimentation_sp2016`",
        "`turbulent_shear_st1956`",
        "canonical Brownian, charged, sedimentation, then turbulent order",
        "singleton masks are `1`, `2`, `4`, and `8`",
        "two-way mask",
        "`3`",
        "`5`",
        "`6`",
        "`9`",
        "`10`",
        "`12`",
        "four-term mask is `15`",
        "Rejected three-way mask",
        "`7`",
        "`11`",
        "`13`",
        "`14`",
        "`turbulent_dissipation`",
        "`fluid_density`",
        "active-device `wp.float64` Warp arrays shaped `(n_boxes,)`",
        "ignored by non-turbulent masks",
        "The return tuple is exactly `(particles, collision_pairs, n_collisions)`",
        "Accepted collisions mutate caller-owned particle mass, concentration, and charge in place",
        "Supplied collision buffers are returned by identity",
        "reset only when `initialize_rng=True`",
        "Malformed configuration and an unsupported distribution fail before particle access",
        "Mask `7` fails at capability preflight before particle metadata",
        "Masks `11`, `13`, and `14` validate particle metadata",
        "no rollback is guaranteed after such a failure",
        "Transfers are explicit",
        "high-level `Runnable` integration",
        "CPU fallback",
        "hidden transfer",
        "performance or broad-accuracy claims",
    ):
        assert _normalized(snippet) in section, snippet


def test_strategy_guide_matches_the_shipped_direct_mask_boundary() -> None:
    """Strategy guide keeps the shipped direct-kernel boundary in sync."""
    section = _normalized(
        _section(STRATEGY_PATH.read_text(encoding="utf-8"), STRATEGY_HEADING)
    )

    for snippet in (
        "direct-kernel-only path is separate from CPU strategies, builders, factories, and `Runnable` APIs",
        "from particula.gpu.kernels import coagulation_step_gpu",
        "from particula.gpu.kernels.coagulation import CoagulationMechanismConfig",
        '`distribution_type="particle_resolved"',
        "`brownian`",
        "`charged_hard_sphere`",
        "`sedimentation_sp2016`",
        "`turbulent_shear_st1956`",
        "Brownian, charged, sedimentation, then turbulent order",
        "singleton masks `1`, `2`, `4`, `8`",
        "two-term masks `3`, `5`, `6`, `9`, `10`, `12`",
        "four-term mask `15`",
        "Three-term masks `7`, `11`, `13`, and `14` are deferred and fail closed",
        "`turbulent_dissipation` and `fluid_density`",
        "same-device `wp.float64` arrays with shape `(n_boxes,)`",
        "Non-turbulent masks ignore these inputs",
        "caller-owned Warp resources",
        "exact return tuple is `(particles, collision_pairs, n_collisions)`",
        "RNG state is not returned",
        "Malformed configuration and unsupported distributions fail before particle access",
        "later runtime failure has no rollback guarantee",
        "Transfers are explicit",
    ):
        assert _normalized(snippet) in section, snippet
    for command in BASELINE_COMMANDS:
        assert _normalized(command) in section
    assert _normalized(OPTIONAL_CUDA_COMMAND) in section
    assert "Brownian-plus-sedimentation combinations and other" not in section


def test_gpu_coagulation_commands_and_optional_device_policy_are_resolvable() -> (
    None
):
    """Published direct-coagulation commands target checked-in test files."""
    foundations = FOUNDATIONS_PATH.read_text(encoding="utf-8")
    strategy = STRATEGY_PATH.read_text(encoding="utf-8")
    validation = VALIDATION_PATH.read_text(encoding="utf-8")

    for command in BASELINE_COMMANDS:
        assert command in foundations
        assert command in strategy
        assert _command_target(command).exists()
        assert "-q" in command
        assert "-Werror" in command
    assert OPTIONAL_CUDA_COMMAND in foundations
    assert OPTIONAL_CUDA_COMMAND in strategy
    assert OPTIONAL_CUDA_COMMAND in validation
    assert _command_target(OPTIONAL_CUDA_COMMAND).exists()
    assert "-q" in OPTIONAL_CUDA_COMMAND
    assert "-Werror" in OPTIONAL_CUDA_COMMAND
    assert DIRECT_COAGULATION_COMMAND in foundations
    assert DIRECT_COAGULATION_COMMAND in validation
    assert _command_target(DIRECT_COAGULATION_COMMAND).exists()
    assert BASELINE_COMMANDS[-1] in validation

    normalized_foundations = _normalized(foundations)
    normalized_strategy = _normalized(strategy)
    normalized_validation = _normalized(validation)
    assert (
        'The required baseline is Warp `device="cpu"` when Warp is installed.'
        in normalized_foundations
    )
    assert (
        "CUDA is optional additive evidence and guarded suites skip cleanly"
        in normalized_foundations
    )
    assert "Warp CPU is the installed-Warp baseline." in normalized_strategy
    assert (
        "CUDA is optional/additive, never mandatory, and guarded suites skip cleanly"
        in normalized_strategy
    )
    assert (
        "Warp CPU is the required baseline when Warp is installed."
        in normalized_validation
    )
    assert (
        "CUDA is optional, local/manual additive evidence and skips cleanly"
        in normalized_validation
    )


def test_gpu_coagulation_guide_cross_links_resolve_once() -> None:
    """Required local ownership and evidence links resolve with fixed cardinality."""
    foundations = FOUNDATIONS_PATH.read_text(encoding="utf-8")
    strategy_section = _section(
        STRATEGY_PATH.read_text(encoding="utf-8"), STRATEGY_HEADING
    )
    strategy_destinations = _local_destinations(strategy_section)
    foundations_destinations = _local_destinations(
        _section(foundations, FOUNDATIONS_HEADING)
    )

    assert (
        strategy_destinations.count("./data-containers-and-gpu-foundations.md")
        == 1
    )
    assert (
        strategy_destinations.count("./Roadmap/coagulation-validation.md") == 1
    )
    assert (
        foundations_destinations.count("Roadmap/coagulation-validation.md") == 1
    )
    for path, destination in (
        (STRATEGY_PATH, "./data-containers-and-gpu-foundations.md"),
        (STRATEGY_PATH, "./Roadmap/coagulation-validation.md"),
        (FOUNDATIONS_PATH, "Roadmap/coagulation-validation.md"),
    ):
        assert (path.parent / destination).resolve().exists()


def test_data_containers_card_has_one_local_coagulation_link_each() -> None:
    """Gallery card exposes one resolving guide and direct-example link."""
    card = _data_containers_card(
        EXAMPLES_INDEX_PATH.read_text(encoding="utf-8")
    )
    destinations = _resolving_destinations(EXAMPLES_INDEX_PATH, card)

    assert destinations.count("gpu_coagulation_direct.py") == 1
    assert destinations.count("../Features/coagulation_strategy_system.md") == 1


def test_closeout_gate_is_fail_closed_and_cuda_skip_is_optional() -> None:
    """Gate fixtures model decisions, not execution evidence."""
    complete = _CloseoutInputs(
        child_states={"E5-F2": True, "E5-F9-P3": True},
        artifacts_valid=True,
        references_valid=True,
        links_valid=True,
        focused_commands={"docs": "passed", "repository": "passed"},
        warp_installed=True,
        warp_cpu_result="passed",
        cuda_result="skipped",
    )
    assert _closeout_gate(complete) == (
        "E5 shipped",
        "E5-F9 shipped",
        "Epic F active",
    )
    for replacement in (
        {},
        {"E5-F2": False},
        {"E5-F9-P3": False},
    ):
        assert (
            _closeout_gate(
                _CloseoutInputs(
                    **{**complete.__dict__, "child_states": replacement}
                )
            )
            == "blocked"
        )
    for field in ("artifacts_valid", "references_valid", "links_valid"):
        assert (
            _closeout_gate(
                _CloseoutInputs(**{**complete.__dict__, field: False})
            )
            == "blocked"
        )
    for result in ("failed", "skipped"):
        assert (
            _closeout_gate(
                _CloseoutInputs(
                    **{**complete.__dict__, "warp_cpu_result": result}
                )
            )
            == "blocked"
        )
    assert (
        _closeout_gate(
            _CloseoutInputs(
                **{
                    **complete.__dict__,
                    "focused_commands": {"docs": "failed"},
                }
            )
        )
        == "blocked"
    )
    assert (
        _closeout_gate(
            _CloseoutInputs(**{**complete.__dict__, "focused_commands": {}})
        )
        == "blocked"
    )
    assert _closeout_gate(
        _CloseoutInputs(
            **{
                **complete.__dict__,
                "warp_installed": False,
                "warp_cpu_result": "not_applicable",
            }
        )
    ) == ("E5 shipped", "E5-F9 shipped", "Epic F active")


def test_final_authoritative_records_are_completed() -> None:
    """Parsed records require all prerequisites and closeout completion."""
    feature_plans = [
        json.loads(
            (ROOT / f".opencode/plans/features/E5-F{number}.json").read_text()
        )
        for number in range(1, 9)
    ]
    assert all(plan["status"] == "Shipped" for plan in feature_plans)
    assert all(plan["lifecycle"] == "completed" for plan in feature_plans)
    feature_two = json.loads(E5_F2_PLAN_PATH.read_text())
    feature_nine = json.loads(E5_F9_PLAN_PATH.read_text())
    assert feature_two["phases"][-1]["status"] == "Shipped"
    assert all(
        phase["status"] == "Shipped" for phase in feature_nine["phases"][:3]
    )
    assert feature_nine["phases"][2]["issue_number"] == 1374
    assert feature_nine["status"] == "Shipped"
    assert feature_nine["lifecycle"] == "completed"
    assert all(phase["status"] == "Shipped" for phase in feature_nine["phases"])
    epic = json.loads(E5_PLAN_PATH.read_text())
    assert epic["status"] == "Shipped"
    assert epic["lifecycle"] == "completed"
    assert [child["id"] for child in epic["child_plans"]] == [
        f"E5-F{number}" for number in range(1, 10)
    ]


def test_guides_do_not_promote_deferred_runtime_capabilities() -> None:
    """Guides retain bounded negative claims without rejecting accurate negations."""
    content = _normalized(
        FOUNDATIONS_PATH.read_text(encoding="utf-8")
        + "\n"
        + _section(STRATEGY_PATH.read_text(encoding="utf-8"), STRATEGY_HEADING)
    ).lower()

    for affirmative_claim in (
        "high-level `aerosol`/`runnable` integration is supported",
        "automatic transfer is supported",
        "cpu fallback is supported",
        "cuda is mandatory",
        "performance is guaranteed",
        "dns/general turbulence is supported",
        "unsupported mechanisms are supported",
        "unsupported distributions are supported",
    ):
        assert affirmative_claim not in content


def test_guides_explicitly_exclude_deferred_runtime_capabilities() -> None:
    """Guides explicitly retain the bounded direct-kernel exclusions."""
    foundations = _normalized(
        _section(
            FOUNDATIONS_PATH.read_text(encoding="utf-8"), FOUNDATIONS_HEADING
        )
    ).lower()
    strategy = _normalized(
        _section(STRATEGY_PATH.read_text(encoding="utf-8"), STRATEGY_HEADING)
    ).lower()

    for snippet in (
        "no high-level `runnable` integration",
        "automatic transfer",
        "cpu fallback",
        "mandatory cuda requirement",
        "general-turbulence support",
    ):
        assert snippet in foundations
    for snippet in (
        "no automatic transfer, cpu fallback, or high-level `runnable` integration",
        "cuda is optional/additive, never mandatory",
        "dns/general-turbulence support",
    ):
        assert snippet in strategy


def test_e5_roadmap_records_match_and_resolve_artifacts() -> None:
    """Roadmap records retain one matching inventory and artifact set."""
    records = []
    for source_path in (DETAILED_ROADMAP_PATH, ROADMAP_INDEX_PATH):
        content = source_path.read_text(encoding="utf-8")
        record = _roadmap_record(content)

        record_rows = _record_rows(record)
        assert record_rows == list(EXPECTED_E5_ROWS)
        assert "## Supported Evidence Matrix" not in record
        assert "## Deferred capability ownership" not in record

        record_destinations = _labeled_destinations(record)
        for artifact in EXPECTED_E5_ARTIFACTS:
            assert record_destinations.count(artifact) == 1
            assert (source_path.parent / artifact[1]).resolve().exists()

        full_document_destinations = _labeled_destinations(content)
        for artifact in EXPECTED_E5_ARTIFACTS:
            assert full_document_destinations.count(artifact) == 1

        all_rows = _record_rows(content)
        for identifier, _, _ in EXPECTED_E5_ROWS:
            assert sum(row[0] == identifier for row in all_rows) == 1

        artifact_records = [
            artifact
            for artifact in record_destinations
            if artifact in EXPECTED_E5_ARTIFACTS
        ]
        records.append((record_rows, artifact_records))

    assert records[0] == records[1]


def test_e5_roadmaps_keep_final_epic_statuses() -> None:
    """Roadmaps retain shipped E5 and active Epic F status."""
    stale_epic_e_row = (
        r"\|\s*5\s*\|\s*\[Epic E:.*?\|\s*Active\s*\|\s*"
        r"not scheduled\s*\|"
    )
    detailed = DETAILED_ROADMAP_PATH.read_text(encoding="utf-8")
    index = ROADMAP_INDEX_PATH.read_text(encoding="utf-8")

    assert _detailed_epic_index_statuses(detailed) == {
        "E": "shipped",
        "F": "active",
    }
    assert _roadmap_index_statuses(index) == {"E": "shipped", "F": "active"}
    for content in (detailed, index):
        normalized = _normalized(content).lower()
        assert not re.search(stale_epic_e_row, content, flags=re.IGNORECASE)
        assert "e5-f9 p4 remains" not in normalized


def test_testing_guide_publishes_hardware_free_docs_validation() -> None:
    """Release-validation commands include the hardware-free contract test."""
    release_validation = _normalized(
        _section(
            TESTING_GUIDE_PATH.read_text(encoding="utf-8"),
            RELEASE_VALIDATION_HEADING,
        )
    )

    assert _normalized(BASELINE_COMMANDS[-1]) in release_validation
