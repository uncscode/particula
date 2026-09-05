"""Regression tests for the graph-capture developer documentation contract."""

import ast
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
        "cross-device replay",
        "hidden allocation/transfer/",
        "no retry or rollback guarantee",
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


def test_p5_documents_three_way_native_capture_evidence_and_limits() -> None:
    """P5 records separate reference, Warp-CPU, and optional CUDA evidence."""
    documents = {
        "AGENTS.md": _read("AGENTS.md"),
        "roadmap": _read("docs/Features/Roadmap/data-oriented-gpu.md"),
        "foundations": _read(
            "docs/Features/data-containers-and-gpu-foundations.md"
        ),
    }
    statements = (
        "test-local NumPy reference",
        "uncaptured evidence",
        "optional pass-or-clean-skip evidence",
        "first physical timestep",
        "one-token replay",
        "opaque-handle provenance",
        "single- and multi-box",
        "no public exports",
        "no automatic recapture",
        "no fallback",
        "no hidden transfer/readback/synchronization",
        "no checkpointed handle",
        "cross-device replay",
        "deterministic parity",
        "independent conservation",
        "persistent RNG continuation",
        "aggregate stochastic",
        "cross-device RNG words",
        "no performance, profiling, or memory claim",
    )
    for path, document in documents.items():
        normalized = " ".join(document.split()).lower()
        missing = [
            statement
            for statement in statements
            if " ".join(statement.split()).lower() not in normalized
        ]
        assert not missing, f"Missing {missing!r} in {path}."
        assert "from particula.execution import graph_capture" not in document
        assert "from particula import graph_capture" not in document


def test_foundations_graph_capture_is_not_nested_under_environment() -> None:
    """Graph-capture validation remains a peer of the environment boundary."""
    document = _read("docs/Features/data-containers-and-gpu-foundations.md")
    environment = "### Environment transfer boundary"
    graph_capture = "### Resident native graph-capture validation (test-only)"

    environment_index = document.index(environment)
    graph_capture_index = document.index(graph_capture)
    next_environment_peer = document.index(
        "### Gas transfer boundary",
        environment_index,
    )

    assert graph_capture_index < environment_index < next_environment_peer
    assert (
        "### Resident native graph-capture validation boundary" not in document
    )


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
    assert {child["id"] for child in epic_record["child_plans"]} == {
        f"E8-F{number}" for number in range(1, 9)
    }
    assert epic_record["milestones"] == [
        {
            "name": "Capture lifecycle established",
            "planned_date": None,
            "actual_date": "2026-08-30",
            "status": "Shipped",
        }
    ]


def test_prepared_enqueue_docs_preserve_private_ready_contract() -> None:
    """Test developer docs distinguish setup, enqueue, and scheduler gating."""
    roadmap = _read("docs/Features/Roadmap/data-oriented-gpu.md")
    foundations = _read("docs/Features/data-containers-and-gpu-foundations.md")
    agents = _read("AGENTS.md")
    phrases = (
        "concrete-only",
        "READY",
        "validation",
        "dynamic",
        "RNG",
        "identity gate before token entry",
        "exactly one token",
        "twelve frozen operations in canonical order",
        "Empty or no-work bindings are write-free",
        "E8-F1",
        "writer",
        "E8-F3 owns resource work",
        "E8-F4 native capture/replay",
    )
    for path, document in (
        ("roadmap", roadmap),
        ("foundations", foundations),
        ("AGENTS.md", agents),
    ):
        _require_statements(document, path, phrases)
    assert "from particula.execution import Prepared" not in foundations
    assert "from particula import Prepared" not in foundations


def test_e8_f2_records_reconcile_shipped_phases_and_p8_evidence() -> None:
    """Test E8-F2 records retain P8 completion only with literal evidence."""
    feature = json.loads(_read(".opencode/plans/features/E8-F2.json"))
    phase_status = {phase["id"]: phase for phase in feature["phases"]}
    assert feature["status"] == "In Progress"
    assert feature["lifecycle"] == "active"
    for phase_number in range(1, 7):
        phase = phase_status[f"E8-F2-P{phase_number}"]
        assert phase["status"] == "Shipped"
    assert phase_status["E8-F2-P6"]["completion_date"] == "2026-09-01"
    assert phase_status["E8-F2-P7"]["status"] == "Not Started"
    assert phase_status["E8-F2-P8"]["status"] == "Shipped"
    assert phase_status["E8-F2-P8"]["issue_number"] == 1559
    assert phase_status["E8-F2-P8"]["completion_date"] == "2026-09-01"
    assert "22 passed" in phase_status["E8-F2-P8"]["notes_ref"]
    assert "mkdocs build --strict" in phase_status["E8-F2-P8"]["notes_ref"]
    for path in (
        "docs/Features/Roadmap/data-oriented-gpu.md",
        ".opencode/plans/sections/epics/E8/child_plans.md",
        ".opencode/plans/sections/epics/E8/milestones_timeline.md",
        ".opencode/plans/sections/features/E8-F2/scope.md",
        ".opencode/plans/sections/features/E8-F2/phase_details.md",
    ):
        document = _read(path)
        normalized_document = " ".join(document.split())
        assert (
            "P1--P6/P8" in normalized_document
            and "shipped" in normalized_document
        ), f"Missing final P8 reconciliation in {path}."
        assert (
            "P7" in normalized_document and "pending" in normalized_document
        ), f"P7 must remain pending in {path}."


def test_prepared_source_docstrings_state_setup_dispatch_boundary() -> None:
    """Test concrete E8-F2 modules document retained-reference enqueue work."""
    modules = (
        "particula/execution/resident_enqueue.py",
        "particula/execution/resident_scheduler.py",
        "particula/execution/state_updates.py",
        "particula/execution/thermodynamic_updates.py",
        "particula/execution/diagnostics.py",
        "particula/execution/resident_communication.py",
        "particula/execution/adapters/condensation.py",
        "particula/execution/adapters/coagulation.py",
        "particula/execution/process_adapters.py",
        "particula/gpu/kernels/communication.py",
        "particula/gpu/kernels/condensation.py",
        "particula/gpu/kernels/coagulation.py",
        "particula/gpu/kernels/dilution.py",
        "particula/gpu/kernels/wall_loss.py",
        "particula/gpu/kernels/nucleation.py",
        "particula/gpu/kernels/exhaustion.py",
    )
    for relative_path in modules:
        tree = ast.parse(_read(relative_path))
        docstring = ast.get_docstring(tree)
        assert docstring is not None, (
            f"Missing module docstring in {relative_path}."
        )
        normalized = " ".join(docstring.split()).lower()
        assert "prepared" in normalized, (
            f"Missing prepared setup/dispatch contract in {relative_path}."
        )
        assert "validation" in normalized, (
            f"Missing setup validation boundary in {relative_path}."
        )
        assert "allocation" in normalized, (
            f"Missing allocation boundary in {relative_path}."
        )

        prepared_seams = (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.ClassDef, ast.FunctionDef))
            and (
                node.name.startswith("Prepared")
                or node.name.startswith("setup_prepared_")
                or node.name.startswith("_prepare_")
                or node.name.startswith("_enqueue_prepared_")
            )
        )
        for seam in prepared_seams:
            seam_docstring = ast.get_docstring(seam)
            assert seam_docstring is not None, (
                f"Missing prepared-seam docstring for {seam.name} in "
                f"{relative_path}."
            )

    scheduler = ast.parse(_read("particula/execution/resident_scheduler.py"))
    enqueue = next(
        node
        for node in scheduler.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "enqueue_prepared_resident_simulation"
    )
    docstring = ast.get_docstring(enqueue)
    assert docstring is not None
    normalized = " ".join(docstring.split()).lower()
    assert "pre-token gate" in normalized
    assert "exactly one token" in normalized
    assert "fresh signature comparison" in normalized


def test_e8_f5_p5_closeout_docs_link_authoritative_evidence_and_preserve_limits() -> (
    None
):
    """Test E8-F5-P5 evidence is ordered, bounded, and cross-recorded."""
    roadmap = _read("docs/Features/Roadmap/data-oriented-gpu.md")
    evidence = _read(
        ".opencode/plans/sections/features/E8-F5/testing_strategy.md"
    )
    prior = "### E8-F4-P5 full-loop validation contract"
    closeout = "### E8-F5-P5 integrated validation closeout"
    evidence_heading = (
        "## E8-F5-P5 authoritative integrated validation evidence"
    )

    assert roadmap.index(prior) < roadmap.index(closeout)
    subsection = _section(roadmap, closeout)
    _require_statements(
        subsection,
        "E8-F5-P5 roadmap closeout",
        (
            "independent CPU/NumPy oracle",
            "required installed-Warp uncaptured rows",
            "optional native-CUDA capture/replay rows",
            "rtol=1e-12",
            "atol=1e-30",
            "concentration-weighted per-box/per-species conservation",
            "aggregate or sigma-bounded checks",
            "concrete-only",
            "no public graph-capture API or example",
            "no fallback, automatic recapture, retry, rollback",
            "hidden transfer/readback/synchronization",
            "checkpointed-native-handle",
            "cross-device replay claim",
            "no performance, memory, or profiling completion claim",
        ),
    )
    commands = (
        "pytest particula/execution/tests/captured_full_loop_test.py -q --no-cov",
        "pytest particula/execution/tests/graph_capture_test.py",
        "particula/execution/tests/rng_invariance_test.py",
        "particula/execution/tests/checkpoint_test.py -q --no-cov",
        '-m "warp and cuda" --no-cov',
        ".opencode/tools/run_pytest.py",
        "pytest particula/execution/tests/graph_capture_docs_test.py",
        "particula/tests/execution_selection_docs_test.py -q --no-cov",
        "mkdocs build --strict",
    )
    assert evidence_heading in evidence
    for command in commands:
        assert command in evidence
        assert command in subsection

    _require_statements(
        evidence,
        "E8-F5-P5 authoritative evidence",
        (
            "P5 (shipped, #1579)",
            "Evidence record date:** 2026-09-05",
            "approved strict-equivalent",
            "mkdocs build --strict` | 0",
            "P5 is shipped",
        ),
    )
    assert "Evidence record date (2026-09-05)" in subsection
    assert "P4 plan update" in subsection
    assert "dated 2026-09-05" in subsection
