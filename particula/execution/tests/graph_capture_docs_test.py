"""Regression tests for the graph-capture developer documentation contract."""

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
        "CUDA-only",
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
        section = document[document.index(heading) :]
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
    """Test P4 records unavailable validation without claiming delivery."""
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

    _require_statements(
        phase_details,
        "phase_details.md",
        ("E8-F1-P4", "Issue: #1550", "Status: In Progress", "17 passed"),
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
            "strict-build criterion and delivery handoff remain unchecked",
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
        ("Started E8-F1-P4", "#1550", "not delivered"),
    )
    _require_statements(
        children,
        "child_plans.md",
        (
            "| E8-F1 | Graph-Capture Capability and Lifecycle Contracts | Draft |",
        ),
    )
    _require_statements(
        milestones,
        "milestones_timeline.md",
        (
            "no captured fixed-loop smoke test has shipped",
            "validation is unavailable",
        ),
    )
    _require_statements(
        epic_changes,
        "epics/E8/change_log.md",
        ("E8-F1", "#1550", "handoff as incomplete"),
    )
