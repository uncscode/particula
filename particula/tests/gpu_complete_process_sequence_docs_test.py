"""Hardware-free publication tests for the E6 direct-process documentation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[2]
ROADMAPS = (
    ROOT / "docs/Features/Roadmap/data-oriented-gpu.md",
    ROOT / "docs/Features/Roadmap/index.md",
)
FOUNDATIONS = ROOT / "docs/Features/data-containers-and-gpu-foundations.md"
EXAMPLE = ROOT / "docs/Examples/gpu_complete_process_sequence.py"
EXAMPLE_TEST = (
    ROOT / "particula/gpu/tests/gpu_complete_process_sequence_example_test.py"
)
P2_TEST = ROOT / "particula/gpu/tests/process_sequence_test.py"
TESTING_GUIDE = ROOT / ".opencode/guides/testing_guide.md"
AGENTS_GUIDE = ROOT / "AGENTS.md"
DOCS_INDEX = ROOT / "docs/index.md"
E6_PLAN = ROOT / ".opencode/plans/epics/E6.json"
FEATURES = {
    f"E6-F{number}": ROOT / f".opencode/plans/features/E6-F{number}.json"
    for number in range(1, 10)
}
DOCS_TEST_COMMAND = "pytest particula/tests/gpu_complete_process_sequence_docs_test.py -q -Werror"
P3_SOURCE = (
    "https://github.com/Gorkowski/particula/blob/main/docs/Examples/"
    "gpu_complete_process_sequence.py"
)
P4_SECTIONS = {
    name: ROOT / f".opencode/plans/sections/features/E6-F9/{name}.md"
    for name in (
        "phase_details",
        "dependencies",
        "documentation_updates",
        "success_criteria",
        "change_log",
    )
}
E6_SECTIONS = {
    name: ROOT / f".opencode/plans/sections/epics/E6/{name}.md"
    for name in (
        "child_plans",
        "dependency_map",
        "success_metrics",
        "change_log",
    )
}
COMPONENT_GUIDES = tuple(
    ROOT / f"docs/Features/{name}.md"
    for name in (
        "dilution_strategy_system",
        "wall_loss_strategy_system",
        "nucleation_strategy_system",
        "slot_exhaustion_policies",
        "condensation_strategy_system",
        "coagulation_strategy_system",
    )
)


def _section(content: str, heading: str) -> str:
    """Return a Markdown section through its next same-or-higher heading."""
    start = content.index(heading)
    section = content[start:]
    level = len(heading) - len(heading.lstrip("#"))
    for index, line in enumerate(section.splitlines()[1:], start=1):
        if line.startswith("#") and len(line) - len(line.lstrip("#")) <= level:
            return "\n".join(section.splitlines()[:index])
    return section


def _records() -> dict[str, dict[str, Any]]:
    """Load the canonical E6 epic and feature plan records."""
    return {
        "E6": json.loads(E6_PLAN.read_text(encoding="utf-8")),
        **{
            identifier: json.loads(path.read_text(encoding="utf-8"))
            for identifier, path in FEATURES.items()
        },
    }


def _closeout_gate(
    records: dict[str, dict[str, Any]],
    p4_evidence: str,
) -> tuple[str, str, str]:
    """Return the only permitted E6 closeout projection, otherwise blocked."""
    children = [records[f"E6-F{number}"] for number in range(1, 9)]
    p4_plan = records["E6-F9"]
    phases = cast(list[dict[str, Any]], p4_plan["phases"])
    p4 = phases[3]
    child_complete = all(
        child["status"] == "Shipped"
        and child["lifecycle"] == "completed"
        and child["completion_date"]
        and all(
            phase["status"] == "Shipped" and phase["completion_date"]
            for phase in child["phases"]
        )
        for child in children
    )
    p4_complete = (
        all(phase["status"] == "Shipped" for phase in phases)
        and p4["issue_number"] == 1449
        and p4["completion_date"]
        and all(
            f"`{command}`  # passed" in p4_evidence
            for command in (
                DOCS_TEST_COMMAND,
                "pytest particula/gpu/tests/gpu_complete_process_sequence_example_test.py -q -Werror",
                "mkdocs build --strict",
                "adw plans validate",
            )
        )
    )
    dates = [cast(str | None, phase["completion_date"]) for phase in phases]
    ordered = bool(p4["completion_date"]) and all(
        date and date <= p4["completion_date"] for date in dates[:3]
    )
    if not (child_complete and p4_complete and ordered):
        return ("E6 Draft", "E6-F9 Draft", "Epic G pending")
    return ("E6 Shipped", "E6-F9 Shipped", "Epic G active")


def _inventory_rows(content: str) -> list[tuple[str, str, str]]:
    """Return E6 inventory rows from a roadmap section."""
    section = _section(content, "### E6 roadmap inventory")
    rows = []
    for line in section.splitlines():
        match = re.match(
            r"\|\s*\[`(E6(?:-F[1-9])?)`\]\([^)]+\)\s*\|\s*"
            r"(.*?)\s*\|\s*(.*?)\s*\|",
            line,
        )
        if match:
            rows.append(cast(tuple[str, str, str], match.groups()))
    return rows


def test_closeout_gate_is_fail_closed_for_actual_and_synthetic_records() -> (
    None
):
    """E6 stays blocked until every child and P4 evidence gate is complete."""
    records = _records()
    evidence = P4_SECTIONS["phase_details"].read_text(encoding="utf-8")
    assert _closeout_gate(records, evidence) == (
        "E6 Draft",
        "E6-F9 Draft",
        "Epic G pending",
    )
    synthetic = json.loads(json.dumps(records))
    for number in range(1, 9):
        child = synthetic[f"E6-F{number}"]
        child["status"] = "Shipped"
        child["lifecycle"] = "completed"
        child["completion_date"] = "2026-07-26"
        for phase in child["phases"]:
            phase["status"] = "Shipped"
            phase["completion_date"] = "2026-07-26"
    p4 = synthetic["E6-F9"]["phases"][3]
    p4["status"] = "Shipped"
    p4["issue_number"] = 1449
    p4["completion_date"] = "2026-07-26"
    synthetic["E6-F9"]["phases"][0]["completion_date"] = "2026-07-27"
    assert _closeout_gate(synthetic, evidence) == (
        "E6 Draft",
        "E6-F9 Draft",
        "Epic G pending",
    )
    synthetic = json.loads(json.dumps(records))
    for number in range(1, 9):
        child = synthetic[f"E6-F{number}"]
        child["status"] = "Shipped"
        child["lifecycle"] = "completed"
        child["completion_date"] = "2026-07-26"
        for phase in child["phases"]:
            phase["status"] = "Shipped"
            phase["completion_date"] = "2026-07-26"
    synthetic["E6-F2"]["completion_date"] = None
    assert _closeout_gate(synthetic, evidence) == (
        "E6 Draft",
        "E6-F9 Draft",
        "Epic G pending",
    )
    synthetic["E6-F2"]["completion_date"] = "2026-07-26"
    synthetic["E6-F2"]["lifecycle"] = "active"
    assert _closeout_gate(synthetic, evidence) == (
        "E6 Draft",
        "E6-F9 Draft",
        "Epic G pending",
    )
    for phase in synthetic["E6-F9"]["phases"][:3]:
        phase["completion_date"] = "2026-07-25"
    synthetic["E6-F2"]["lifecycle"] = "completed"
    passed_evidence = "\n".join(
        f"`{command}`  # passed"
        for command in (
            DOCS_TEST_COMMAND,
            "pytest particula/gpu/tests/gpu_complete_process_sequence_example_test.py -q -Werror",
            "mkdocs build --strict",
            "adw plans validate",
        )
    )
    assert _closeout_gate(synthetic, passed_evidence) == (
        "E6 Shipped",
        "E6-F9 Shipped",
        "Epic G active",
    )


def test_p4_metadata_records_only_completed_work_evidence() -> None:
    """P4 records its own evidence while the incomplete parent remains blocked."""
    records = _records()
    p4 = records["E6-F9"]["phases"][3]
    evidence = P4_SECTIONS["phase_details"].read_text(encoding="utf-8")
    assert p4["status"] == "Shipped"
    assert p4["issue_number"] == 1449
    assert p4["completion_date"]
    for command in (
        DOCS_TEST_COMMAND,
        "pytest particula/gpu/tests/gpu_complete_process_sequence_example_test.py -q -Werror",
        "mkdocs build --strict",
        "adw plans validate",
    ):
        assert f"`{command}`  # passed" in evidence
    assert "tolerance" not in _section(evidence, "- [x] **E6-F9-P4:").lower()
    assert records["E6"]["status"] == "Draft"
    assert records["E6-F9"]["status"] == "Draft"


def test_inventory_matches_canonical_records_and_resolves_links() -> None:
    """Both roadmaps carry one identical, linked E6 inventory."""
    records = _records()
    expected = [
        (identifier, record["title"], record["status"])
        for identifier, record in records.items()
    ]
    inventories = []
    for path in ROADMAPS:
        content = path.read_text(encoding="utf-8")
        assert content.count("### E6 roadmap inventory") == 1
        rows = _inventory_rows(content)
        assert rows == expected
        for identifier in records:
            assert "](../../../.opencode/plans/" in _section(
                content, "### E6 roadmap inventory"
            )
            assert (
                ROOT
                / ".opencode/plans"
                / (
                    "epics/E6.json"
                    if identifier == "E6"
                    else f"features/{identifier}.json"
                )
            ).exists()
        inventories.append(rows)
    assert inventories[0] == inventories[1]


def test_direct_process_documentation_preserves_explicit_ownership() -> None:
    """Published ownership guidance names the direct boundary and exclusions."""
    content = _section(
        FOUNDATIONS.read_text(encoding="utf-8"),
        "### Complete direct-process illustration",
    )
    content = " ".join(content.split())
    for name in (
        "condensation_step_gpu",
        "coagulation_step_gpu",
        "dilution_step_gpu",
        "wall_loss_step_gpu",
        "nucleation_step_gpu",
        "to_warp_particle_data",
        "to_warp_gas_data",
        "to_warp_environment_data",
    ):
        assert name in content
    assert content.index("condensation_step_gpu") < content.index(
        "coagulation_step_gpu"
    )
    assert content.index("coagulation_step_gpu") < content.index(
        "dilution_step_gpu"
    )
    assert content.index("dilution_step_gpu") < content.index(
        "wall_loss_step_gpu"
    )
    assert content.index("wall_loss_step_gpu") < content.index(
        "nucleation_step_gpu"
    )
    for excluded in (
        "no hidden transfer",
        "CPU fallback",
        "high-level `Runnable`",
        "backend selector",
        "scheduler",
        "resident production loop",
        "transport API",
    ):
        assert excluded in content


def test_evidence_and_commands_are_real_and_hardware_free() -> None:
    """Published P2/P3 evidence points at checked-in artifacts only."""
    assert EXAMPLE.exists()
    assert EXAMPLE_TEST.exists()
    assert P2_TEST.exists()
    for path in ROADMAPS:
        content = path.read_text(encoding="utf-8")
        assert P3_SOURCE in content
        assert "process_sequence_test.py" in content
        assert "illustrative, not a production coordinator" in content
    assert DOCS_TEST_COMMAND in TESTING_GUIDE.read_text(encoding="utf-8")
    assert DOCS_TEST_COMMAND in AGENTS_GUIDE.read_text(encoding="utf-8")
    assert P3_SOURCE in DOCS_INDEX.read_text(encoding="utf-8")
    for path in (
        *P4_SECTIONS.values(),
        *E6_SECTIONS.values(),
        *COMPONENT_GUIDES,
    ):
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        if path in COMPONENT_GUIDES:
            assert P3_SOURCE in content
            assert "process_sequence_test.py" in content
