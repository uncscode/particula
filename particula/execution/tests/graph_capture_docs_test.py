"""Regression tests for the graph-capture developer documentation contract."""

import json
from pathlib import Path

ROOT = Path(__file__).parents[3]
DRIFT_ORDER = (
    "request",
    "session",
    "device",
    "dimensions",
    "primary_containers",
    "primary_arrays",
    "resource_views",
    "graph",
    "schedule",
    "schedule_order",
    "diagnostics",
    "communication",
    "configurations",
    "rng_resources",
)


def _read(relative_path: str) -> str:
    """Read one repository-relative documentation record."""
    return (ROOT / relative_path).read_text(encoding="utf-8")


def _require_statements(
    document: str, path: str, statements: tuple[str, ...]
) -> None:
    """Assert required contract statements occur in one document."""
    document = " ".join(document.split())
    for statement in statements:
        normalized_statement = " ".join(statement.split())
        assert normalized_statement in document, (
            f"Missing {statement!r} in {path}."
        )


def _section(document: str, heading: str) -> str:
    """Return one level-three Markdown section without later sections."""
    section = document[document.index(heading) + len(heading) :]
    next_heading = section.find("\n### ")
    return section if next_heading == -1 else section[:next_heading]


def test_developer_documents_preserve_graph_capture_contract_and_order() -> (
    None
):
    """Test developer docs state the bounded admission contract and drift order."""
    documents = {
        "AGENTS.md": _read("AGENTS.md"),
        "data-oriented-gpu.md": _read(
            "docs/Features/Roadmap/data-oriented-gpu.md"
        ),
    }
    required = (
        "particula.execution.graph_capture",
        "non-CPU Warp native devices require caller-provided availability/probe",
        "fallback/emulation",
        "payload-only compatibility",
        "complete_resident_graph_capture()",
        "explicit retirement then renewal",
        "E8-F2--E8-F8",
        "no automatic recapture",
        "cross-device replay",
        "hidden allocation/transfer/",
        "no retry or rollback guarantee",
        "checkpointed native graph handles",
        "native/full-loop capture or replay",
        "captured numerical parity",
        "benchmark/profiling/memory",
        "user examples",
    )
    for path, document in documents.items():
        _require_statements(document, path, required)
        heading = (
            "### Resident graph-capture admission lifecycle"
            if path == "AGENTS.md"
            else "### E8-F1 shipped contract"
        )
        section = _section(document, heading)
        comparison = (
            "Compatibility checks identity"
            if path == "AGENTS.md"
            else "Compatibility compares identity"
        )
        section = section[section.index(comparison) :]
        positions = [section.index(reason) for reason in DRIFT_ORDER]
        assert positions == sorted(positions), f"Drift order changed in {path}."


def test_planning_records_preserve_p4_validation_block_and_handoff_boundary() -> (
    None
):
    """Test P4 records its delivered validation and bounded handoff."""
    phase_details = _read(
        ".opencode/plans/sections/features/E8-F1/phase_details.md"
    )
    change_log = _read(".opencode/plans/sections/features/E8-F1/change_log.md")
    documentation = _read(
        ".opencode/plans/sections/features/E8-F1/documentation_updates.md"
    )
    criteria = _read(
        ".opencode/plans/sections/features/E8-F1/success_criteria.md"
    )
    risks = _read(".opencode/plans/sections/features/E8-F1/risk_register.md")
    children = _read(".opencode/plans/sections/epics/E8/child_plans.md")
    milestones = _read(
        ".opencode/plans/sections/epics/E8/milestones_timeline.md"
    )
    epic_changes = _read(".opencode/plans/sections/epics/E8/change_log.md")
    tasks = _read(
        ".opencode/plans/sections/features/E8-F1/implementation_tasks.md"
    )
    feature_record = json.loads(_read(".opencode/plans/features/E8-F1.json"))
    epic_record = json.loads(_read(".opencode/plans/epics/E8.json"))

    _require_statements(
        phase_details,
        "phase_details.md",
        (
            "E8-F1-P4",
            "Issue: #1550",
            "Status: Delivered",
            "2 graph-document tests",
        ),
    )
    _require_statements(
        documentation,
        "documentation_updates.md",
        (
            "Do not create a user-facing `docs/Examples/` graph-capture example",
            "#1550",
        ),
    )
    _require_statements(
        criteria,
        "success_criteria.md",
        (
            "[x] Recapture is explicit",
            "[x] Persistent coagulation",
            "strict-build criterion and delivery handoff are checked",
        ),
    )
    _require_statements(
        risks,
        "risk_register.md",
        ("hidden replay work", "no-retry/no-rollback", "captured-loop support"),
    )
    _require_statements(
        change_log,
        "change_log.md",
        ("Delivered E8-F1-P4", "#1550", "handoff to parent E8 is shipped"),
    )
    _require_statements(
        children,
        "child_plans.md",
        (
            "| E8-F1 | Graph-Capture Capability and Lifecycle Contracts | Shipped |",
        ),
    )
    _require_statements(
        milestones,
        "milestones_timeline.md",
        (
            "no captured fixed-loop smoke test has shipped",
            "2 graph-document tests",
            "6382 passed, 9 skipped, 94% coverage",
            "mkdocs build --strict` passed (exit 0",
        ),
    )
    _require_statements(
        epic_changes,
        "epics/E8/change_log.md",
        ("E8-F1", "#1550", "E8-F1 is complete"),
    )
    _require_statements(
        tasks,
        "implementation_tasks.md",
        (
            "[x] Run focused assertions with coverage disabled",
            "[x] Update the Epic H roadmap text",
            "[x] Record the full recapture-trigger table",
            "[x] Mark E8-F1 plan phases and changelog accurately",
        ),
    )
    assert feature_record["status"] == "Shipped"
    assert feature_record["lifecycle"] == "completed"
    assert all(
        phase["status"] == "Shipped" for phase in feature_record["phases"]
    )
    assert epic_record["status"] == "In Progress"
    assert {child["id"] for child in epic_record["child_plans"]} == {"E8-F1"}
    assert epic_record["milestones"] == [
        {
            "name": "Capture lifecycle established",
            "planned_date": None,
            "actual_date": "2026-08-30",
            "status": "Shipped",
        }
    ]
