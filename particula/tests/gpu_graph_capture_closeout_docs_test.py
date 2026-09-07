"""Validate the hardware-free graph-capture closeout publication boundary."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECORD_PATH = ROOT / "docs/Features/Roadmap/graph-capture-closeout.md"
HEADINGS = (
    "Scope and current disposition",
    "Closeout metadata",
    "Target-derivation ledger",
    "Required evidence matrix",
    "Command ledger",
    "Artifact provenance rules",
    "Blockers and promotion rule",
)
COMMANDS = (
    "pytest particula/tests/gpu_graph_capture_closeout_docs_test.py -q --no-cov",
    "pytest particula/execution/tests/graph_capture_test.py "
    "particula/execution/tests/captured_full_loop_test.py -q --no-cov",
    ".opencode/tools/run_linters.py",
    ".opencode/tools/run_pytest.py",
    "mkdocs build --strict",
    "pytest particula/execution/tests/captured_full_loop_test.py -q "
    '-m "warp and cuda" --no-cov',
    "pytest particula/gpu/tests/benchmark_test.py --benchmark -k resident -v "
    "-s --no-cov",
    "pytest particula/gpu/tests/profiling_smoke_test.py --benchmark -q --no-cov",
)
BLOCKING_STATUSES = {
    "STALE",
    "FAILED",
    "INFERRED",
    "UNAVAILABLE",
    "CLEAN-SKIP",
}
STDLIB_IMPORTS = {"__future__", "ast", "pathlib"}


def _normalized(content: str) -> str:
    """Return text with whitespace and case normalized."""
    return " ".join(content.lower().split())


def _section(content: str, heading: str) -> str:
    """Return the level-two section bounded by the next level-two heading."""
    marker = f"## {heading}"
    start = content.index(marker) + len(marker)
    end = content.find("\n## ", start)
    return content[start:] if end == -1 else content[start:end]


def _classify_status(
    status: str,
    final_revision: str,
    designated_device: str,
    command_output: str,
    artifact_reference: str | None,
    requires_artifact: bool,
) -> bool:
    """Return whether a synthetic criterion row can satisfy promotion."""
    if status in BLOCKING_STATUSES or status != "PASS":
        return False
    required = (final_revision, designated_device, command_output)
    if not all(value.strip() and value != "UNAVAILABLE" for value in required):
        return False
    if requires_artifact:
        return artifact_reference is not None and bool(
            artifact_reference.strip()
        )
    return artifact_reference is None or bool(artifact_reference.strip())


def _has_command_reference(row: str) -> bool:
    """Return whether a criterion row cites a command-ledger record."""
    return any(f"C{number}" in row for number in range(1, 9))


def _has_artifact_reference(row: str) -> bool:
    """Return whether a criterion row cites an artifact-ledger record."""
    return any(f"A{number}" in row for number in range(1, 3))


def _all_required_rows_pass(rows: tuple[str, ...]) -> bool:
    """Return whether every required synthetic H row is present and passing."""
    identifiers = {f"H{number}" for number in range(1, 12)}
    present = {
        identifier
        for identifier in identifiers
        if any(f"| {identifier} " in row for row in rows)
    }
    return present == identifiers and all("| PASS |" in row for row in rows)


def _valid_provenance_filename(
    filename: str, digest: str, byte_size: str
) -> bool:
    """Return whether synthetic provenance has a safe filename and digest."""
    unsafe = ("..", "latest", "symlink", "raw", "copy-summary")
    if not filename or filename.startswith("/"):
        return False
    if any(part in filename.lower() for part in unsafe):
        return False
    if not byte_size.isdigit() or int(byte_size) <= 0:
        return False
    return len(digest) == 64 and all(
        char in "0123456789abcdef" for char in digest
    )


def test_closeout_schema_and_frozen_target_placeholders() -> None:
    """Require the fixed schema, dated metadata, and static target inputs."""
    content = RECORD_PATH.read_text(encoding="utf-8")
    positions = [content.index(f"## {heading}") for heading in HEADINGS]
    assert positions == sorted(positions)
    metadata = _section(content, "Closeout metadata")
    for field in (
        "Record date | 2026-09-07",
        "Final source revision",
        "Python",
        "Warp",
        "CUDA driver/runtime",
        "Warp availability",
        "Designated qualified CUDA device",
        "Supplemental devices",
    ):
        assert field in metadata
    assert metadata.count("Designated qualified CUDA device") == 1
    targets = _section(content, "Target-derivation ledger")
    for feature in range(1, 8):
        assert f"E8-F{feature}" in targets
        assert f"features/E8-F{feature}/implementation_tasks.md" in targets
    assert (
        targets.count("UNAVAILABLE — final executable diff not collected") >= 3
    )
    assert (
        "exclude tests, docs, `.opencode/plans/`, and `.artifacts/`"
        in _normalized(targets)
    )


def test_evidence_matrix_fails_closed_and_traces_all_criteria() -> None:
    """Require H1--H11 mappings, coverage gates, and fail-closed references."""
    content = RECORD_PATH.read_text(encoding="utf-8")
    matrix = _section(content, "Required evidence matrix")
    for criterion in range(1, 12):
        row = next(
            line for line in matrix.splitlines() if f"| H{criterion} " in line
        )
        assert "UNAVAILABLE" in row
        assert _has_command_reference(row)
        if criterion in {1, 2, 3, 4, 6, 7, 8, 9}:
            assert _has_artifact_reference(row)
    targets = _section(content, "Target-derivation ledger")
    assert "Aggregate changed-module coverage" in targets
    assert "`>=80%`" in targets
    assert "P3 closeout document and its test are excluded" in targets
    assert not _classify_status(
        "PASS", "UNAVAILABLE", "device", "output", "A1", True
    )
    assert _classify_status("PASS", "abc123", "cuda:0", "literal", "A1", True)
    assert _classify_status("PASS", "abc123", "cuda:0", "literal", None, False)
    assert not _classify_status(
        "PASS", "abc123", "cuda:0", "literal", None, True
    )
    for status in BLOCKING_STATUSES:
        assert not _classify_status(
            status, "abc123", "cuda:0", "literal", "A1", True
        )
    synthetic_rows = (
        "| H1 | evidence | rev | device | PASS | C1; A1 |",
        "| H2 | evidence | rev | device | STALE | C2; A1 |",
    )
    assert len(synthetic_rows) == 2
    assert all(_has_command_reference(row) for row in synthetic_rows)
    assert not _all_required_rows_pass(synthetic_rows)


def test_command_ledger_preserves_order_and_literal_output_placeholders() -> (
    None
):
    """Require ordered command records without invented execution results."""
    content = RECORD_PATH.read_text(encoding="utf-8")
    ledger = _section(content, "Command ledger")
    positions = [ledger.index(command) for command in COMMANDS]
    assert positions == sorted(positions)
    assert (
        ledger.count("NOT RUN — output must be pasted verbatim after execution")
        == 8
    )
    normalized_ledger = _normalized(ledger)
    assert "assertion-only, not coverage evidence" in normalized_ledger
    assert "sole repository coverage command" in normalized_ledger
    assert "availability-only native-cuda evidence" in normalized_ledger
    assert (
        "pass-or-clean-skip is not required measured evidence"
        in normalized_ledger
    )


def test_provenance_rules_and_unshipped_state_are_safe() -> None:
    """Require safe synthetic provenance and a fully unavailable committed state."""
    content = RECORD_PATH.read_text(encoding="utf-8")
    provenance = _section(content, "Artifact provenance rules")
    for requirement in (
        "record ID",
        "manifest pointer",
        "schema/version",
        "final source revision",
        "workload ID",
        "machine/device provenance",
        "contained relative raw filename",
        "byte size",
        "lowercase SHA-256",
    ):
        assert requirement in provenance
    assert "absent reviewed F6 CUDA artifact" in provenance
    assert "absent reviewed F7 CUDA artifact" in provenance
    assert _valid_provenance_filename("records/row.json", "a" * 64, "12")
    for filename, digest, size in (
        ("/record.json", "a" * 64, "12"),
        ("../record.json", "a" * 64, "12"),
        ("latest.json", "a" * 64, "12"),
        ("symlink/record.json", "a" * 64, "12"),
        ("raw/record.json", "a" * 64, "12"),
        ("copy-summary.json", "a" * 64, "12"),
        ("record.json", "A" * 64, "12"),
        ("record.json", "a" * 64, "0"),
    ):
        assert not _valid_provenance_filename(filename, digest, size)
    assert "Disposition: UNSHIPPED/BLOCKED" in content
    assert "outstanding F2/F3 work block this" in content
    assert "UNAVAILABLE — no designated qualified CUDA device" in content
    normalized_content = _normalized(content)
    assert "measurement fields remain unavailable" in normalized_content
    assert "reviewed artifact fields remain unavailable" in normalized_content


def test_contract_test_uses_only_approved_stdlib_imports() -> None:
    """Keep this contract independent of test, GPU, artifact, and Git tooling."""
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(
                alias.name.split(".")[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            assert node.module is not None
            imported_modules.add(node.module.split(".")[0])
    assert imported_modules <= STDLIB_IMPORTS
