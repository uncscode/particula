"""Validate hardware-free graph-capture closeout and discovery contracts.

This documentation contract reads the committed closeout record and its scoped
discovery surfaces. It rejects incomplete, unsafe, or inferred promotion
evidence without importing pytest, Warp, CUDA, artifact tooling, Git tooling,
planning files, or runtime code.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = ROOT / "docs"
RECORD_PATH = ROOT / "docs/Features/Roadmap/graph-capture-closeout.md"
RUNBOOK_PATH = ROOT / "docs/Features/gpu_graph_capture.md"
EXAMPLE_PATH = ROOT / "docs/Examples/gpu_resident_graph_capture.py"
ROADMAP_PATH = ROOT / "docs/Features/Roadmap/data-oriented-gpu.md"
DISCOVERY_PATHS = (
    ROADMAP_PATH,
    ROOT / "docs/Features/Roadmap/index.md",
    ROOT / "docs/index.md",
    ROOT / "AGENTS.md",
)
DISCOVERY_LINKS = (
    (
        ROADMAP_PATH,
        "../../Examples/gpu_resident_graph_capture.py",
        EXAMPLE_PATH,
    ),
    (ROADMAP_PATH, "../gpu_graph_capture.md", RUNBOOK_PATH),
    (ROADMAP_PATH, "graph-capture-closeout.md", RECORD_PATH),
    (
        ROOT / "docs/Features/Roadmap/index.md",
        "../../Examples/gpu_resident_graph_capture.py",
        EXAMPLE_PATH,
    ),
    (
        ROOT / "docs/Features/Roadmap/index.md",
        "../gpu_graph_capture.md",
        RUNBOOK_PATH,
    ),
    (
        ROOT / "docs/Features/Roadmap/index.md",
        "graph-capture-closeout.md",
        RECORD_PATH,
    ),
    (
        ROOT / "docs/index.md",
        "Examples/gpu_resident_graph_capture.py",
        EXAMPLE_PATH,
    ),
    (ROOT / "docs/index.md", "Features/gpu_graph_capture.md", RUNBOOK_PATH),
    (
        ROOT / "docs/index.md",
        "Features/Roadmap/graph-capture-closeout.md",
        RECORD_PATH,
    ),
    (
        ROOT / "AGENTS.md",
        "docs/Examples/gpu_resident_graph_capture.py",
        EXAMPLE_PATH,
    ),
    (ROOT / "AGENTS.md", "docs/Features/gpu_graph_capture.md", RUNBOOK_PATH),
    (
        ROOT / "AGENTS.md",
        "docs/Features/Roadmap/graph-capture-closeout.md",
        RECORD_PATH,
    ),
    (
        ROOT / "AGENTS.md",
        "docs/Features/Roadmap/data-oriented-gpu.md",
        ROADMAP_PATH,
    ),
)
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
    "particula/execution/tests/captured_full_loop_test.py "
    "particula/execution/tests/rng_invariance_test.py "
    "particula/execution/tests/checkpoint_test.py -q --no-cov",
    ".opencode/tools/run_linters.py",
    ".opencode/tools/run_pytest.py",
    "mkdocs build --strict",
    "pytest particula/tests/gpu_graph_capture_runbook_docs_test.py -q --no-cov",
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
STDLIB_IMPORTS = {"__future__", "ast", "pathlib", "re"}
COMMAND_ID_PATTERN = re.compile(r"(?<![A-Za-z0-9])C[1-9](?![A-Za-z0-9])")
ARTIFACT_ID_PATTERN = re.compile(r"(?<![A-Za-z0-9])A[12](?![A-Za-z0-9])")


def _normalized(content: str) -> str:
    """Normalize text for case-insensitive whitespace comparisons.

    Args:
        content: Text to normalize.

    Returns:
        Lowercase text with consecutive whitespace replaced by one space.
    """
    return " ".join(content.lower().split())


def _section(content: str, heading: str) -> str:
    """Extract a level-two Markdown section from the closeout record.

    Args:
        content: Complete closeout-record text.
        heading: Exact level-two heading whose content is required.

    Returns:
        Text after the requested heading and before the next level-two heading,
        or through the end of the record.

    Raises:
        ValueError: If the requested heading is absent from ``content``.
    """
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
    """Determine whether synthetic criterion evidence permits promotion.

    A passing row needs populated final-revision, designated-device, and
    literal-command-output fields. Criteria that require artifacts also need a
    nonempty artifact reference; blocking statuses always fail.

    Args:
        status: Synthetic criterion status.
        final_revision: Revision associated with the criterion evidence.
        designated_device: Qualified device associated with the evidence.
        command_output: Literal output from the required command.
        artifact_reference: Optional artifact-ledger record reference.
        requires_artifact: Whether the criterion requires artifact evidence.

    Returns:
        ``True`` only when the synthetic row has complete passing evidence.
    """
    if status in BLOCKING_STATUSES or status != "PASS":
        return False
    required = (final_revision, designated_device, command_output)
    if not all(value.strip() and value != "UNAVAILABLE" for value in required):
        return False
    if requires_artifact:
        return artifact_reference is not None and bool(
            artifact_reference.strip(),
        )
    return artifact_reference is None or bool(artifact_reference.strip())


def _has_command_reference(row: str) -> bool:
    """Check whether a criterion row cites one required command record.

    Args:
        row: Serialized evidence-matrix row.

    Returns:
        ``True`` when the row contains a C1--C9 command-ledger identifier.
    """
    return COMMAND_ID_PATTERN.search(row) is not None


def _has_artifact_reference(row: str) -> bool:
    """Check whether a criterion row cites one artifact-ledger record.

    Args:
        row: Serialized evidence-matrix row.

    Returns:
        ``True`` when the row contains an A1 or A2 artifact-ledger identifier.
    """
    return ARTIFACT_ID_PATTERN.search(row) is not None


def _parse_evidence_matrix(matrix: str) -> dict[str, dict[str, str]]:
    """Parse the required-evidence matrix into named columns."""
    rows = [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in matrix.splitlines()
        if line.strip().startswith("|")
    ]
    headers, *body = rows
    assert headers == [
        "ID",
        "Required evidence",
        "Final revision",
        "Designated device",
        "Status",
        "Evidence / blocker",
    ]
    return {
        row[0]: dict(zip(headers, row, strict=True))
        for row in body
        if row[0] and not row[0].startswith("---")
    }


def _contains_unavailable_marker(value: str) -> bool:
    """Return whether a value contains an unavailable-status marker."""
    return "unavailable" in _normalized(value)


def _row_permits_promotion(row: dict[str, str]) -> bool:
    """Require complete non-unavailable column evidence for promotion."""
    required_fields = (
        "Final revision",
        "Designated device",
        "Evidence / blocker",
    )
    return row["Status"] == "PASS" and all(
        row[field].strip() and not _contains_unavailable_marker(row[field])
        for field in required_fields
    )


def _all_required_rows_pass(rows: tuple[dict[str, str], ...]) -> bool:
    """Check whether every required synthetic H1--H11 row passes."""
    identifiers = {f"H{number}" for number in range(1, 12)}
    return {row["ID"] for row in rows} == identifiers and all(
        _row_permits_promotion(row) for row in rows
    )


def _valid_provenance_filename(
    filename: str, digest: str, byte_size: str
) -> bool:
    """Validate a synthetic contained artifact filename and digest.

    Args:
        filename: Relative raw-artifact filename from a provenance record.
        digest: Expected lowercase hexadecimal SHA-256 digest.
        byte_size: Decimal byte size recorded for the artifact.

    Returns:
        ``True`` when the filename is contained and safe, the byte size is
        positive, and the digest is lowercase hexadecimal SHA-256.
    """
    unsafe = ("..", "latest", "symlink", "raw", "copy-summary")
    if (
        not filename
        or filename.startswith(("/", "\\"))
        or "\\" in filename
        or re.match(r"^[A-Za-z]:", filename) is not None
        or _is_nonlocal_markdown_target(filename)
    ):
        return False
    parts = filename.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        return False
    if any(part in filename.lower() for part in unsafe):
        return False
    if not byte_size.isdigit() or int(byte_size) <= 0:
        return False
    return len(digest) == 64 and all(
        char in "0123456789abcdef" for char in digest
    )


def _markdown_destination(link_content: str) -> str:
    """Extract a Markdown link destination without an optional title."""
    link_content = link_content.strip()
    if link_content.startswith("<"):
        end = link_content.find(">")
        return "" if end == -1 else link_content[1:end].strip()
    return link_content.split(maxsplit=1)[0] if link_content else ""


def _is_nonlocal_markdown_target(target: str) -> bool:
    """Return whether a target is a URI or protocol-relative reference."""
    return (
        target.startswith("//")
        or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", target) is not None
    )


def _local_markdown_targets(content: str) -> list[str]:
    """Return non-external Markdown targets without anchors.

    Args:
        content: Markdown source text.

    Returns:
        Local link targets with fragment identifiers removed.
    """
    targets: list[str] = []
    cursor = 0
    while (start := content.find("](", cursor)) != -1:
        target_start = start + 2
        end = content.find(")", target_start)
        if end == -1:
            break
        target = _markdown_destination(content[target_start:end])
        target = target.split("#", maxsplit=1)[0]
        if target and not _is_nonlocal_markdown_target(target):
            targets.append(target)
        cursor = end + 1
    return targets


def _resolve_local_markdown_target(
    target: str,
    source: Path,
    allowed_root: Path,
) -> Path:
    """Resolve one local target to a regular file under an allowed root.

    Args:
        target: Relative Markdown target, optionally with an anchor or title.
        source: Document containing the target.
        allowed_root: Root that contains permitted resolved files.

    Returns:
        Resolved regular-file target.

    Raises:
        AssertionError: If a target escapes, is missing, or is not a file.
    """
    target = _markdown_destination(target).split("#", maxsplit=1)[0]
    assert target and not _is_nonlocal_markdown_target(target)
    assert not target.startswith(("/", "\\"))
    assert "\\" not in target
    assert re.match(r"^[A-Za-z]:", target) is None
    resolved = (source.parent / target).resolve()
    assert resolved.is_relative_to(allowed_root.resolve())
    assert resolved.is_file()
    return resolved


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
    rows = _parse_evidence_matrix(matrix)
    for criterion in range(1, 12):
        row = rows[f"H{criterion}"]
        assert row["Status"] == "UNAVAILABLE"
        assert _contains_unavailable_marker(row["Final revision"])
        assert _contains_unavailable_marker(row["Designated device"])
        assert not _row_permits_promotion(row)
        assert _has_command_reference(row["Evidence / blocker"])
        if criterion in {1, 2, 3, 4, 6, 7, 8, 9}:
            assert _has_artifact_reference(row["Evidence / blocker"])
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
        {
            "ID": "H1",
            "Required evidence": "evidence",
            "Final revision": "rev",
            "Designated device": "device",
            "Status": "PASS",
            "Evidence / blocker": "C1; A1",
        },
        {
            "ID": "H2",
            "Required evidence": "evidence",
            "Final revision": "UNAVAILABLE — revision not collected",
            "Designated device": "device",
            "Status": "PASS",
            "Evidence / blocker": "C2; A1",
        },
    )
    assert len(synthetic_rows) == 2
    assert all(
        _has_command_reference(row["Evidence / blocker"])
        for row in synthetic_rows
    )
    assert _row_permits_promotion(synthetic_rows[0])
    assert not _row_permits_promotion(synthetic_rows[1])
    assert not _all_required_rows_pass(synthetic_rows)
    assert not _has_command_reference("C10 only")
    assert _has_command_reference("C1; C10")
    assert not _has_artifact_reference("A10 only")
    assert _has_artifact_reference("A1; A10")


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
        == 9
    )
    normalized_ledger = _normalized(ledger)
    assert "assertion-only, not coverage evidence" in normalized_ledger
    assert (
        "c4 is this record's sole repository coverage command"
        in normalized_ledger
    )
    assert (
        "c5 is its strict mkdocs documentation-rendering command"
        in normalized_ledger
    )
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
        ("https://example.com/record.json", "a" * 64, "12"),
        ("file:///record.json", "a" * 64, "12"),
        ("records/./record.json", "a" * 64, "12"),
        ("record.json", "A" * 64, "12"),
        ("record.json", "a" * 64, "0"),
    ):
        assert not _valid_provenance_filename(filename, digest, size)
    assert "Disposition: UNSHIPPED/BLOCKED" in content
    assert "outstanding E8-F2 and E8-F3 work block" in content
    assert "E8-F3 remains `In Progress`" in content
    assert "P2 and P5 phases are `Not Started`" in content
    assert "outstanding F2/F3 work block this" not in content
    assert "UNAVAILABLE — no designated qualified CUDA device" in content
    normalized_content = _normalized(content)
    assert "measurement fields remain unavailable" in normalized_content
    assert "reviewed artifact fields remain unavailable" in normalized_content


def test_discovery_links_resolve_to_the_canonical_graph_capture_sources() -> (
    None
):
    """Require only Step-1 discovery links to resolve to canonical files."""
    for source, target, expected in DISCOVERY_LINKS:
        allowed_root = ROOT if source == ROOT / "AGENTS.md" else DOCS_ROOT
        assert target in _local_markdown_targets(
            source.read_text(encoding="utf-8")
        )
        assert (
            _resolve_local_markdown_target(target, source, allowed_root)
            == expected.resolve()
        )


def test_link_resolution_rejects_escapes_directories_and_missing_files(
    tmp_path: Path,
) -> None:
    """Keep documentation and repository link-root policies independent."""
    repository_root = tmp_path / "repository"
    docs_root = repository_root / "docs"
    source = docs_root / "Features" / "source.md"
    source.parent.mkdir(parents=True)
    source.touch()
    target = docs_root / "Examples" / "example.md"
    target.parent.mkdir()
    target.touch()

    assert (
        _resolve_local_markdown_target(
            "../Examples/example.md#section", source, docs_root
        )
        == target.resolve()
    )
    invalid_targets = (
        "../../../outside.md",
        "../Examples",
        "../Examples/missing.md",
        "C:\\outside.md",
        "\\\\server\\share\\file.md",
        "//server/share/file.md",
        "https://example.com/file.md",
        "mailto:docs@example.com",
    )
    for invalid_target in invalid_targets:
        try:
            _resolve_local_markdown_target(invalid_target, source, docs_root)
        except AssertionError:
            pass
        else:
            raise AssertionError("invalid documentation link was accepted")

    root_source = repository_root / "AGENTS.md"
    root_source.touch()
    root_target = repository_root / "README.md"
    root_target.touch()
    assert _resolve_local_markdown_target(
        "README.md", root_source, repository_root
    )
    try:
        _resolve_local_markdown_target(
            "../outside.md", root_source, repository_root
        )
    except AssertionError:
        pass
    else:
        raise AssertionError("invalid repository-root link was accepted")


def test_local_markdown_parser_skips_uris_and_titles() -> None:
    """Keep non-file URLs and titles outside local filesystem resolution."""
    content = "\n".join(
        (
            "[anchored](guide.md#anchor)",
            '[titled](guide.md "Guide title")',
            "[angle](<guide with spaces.md> 'Guide title')",
            "[https](https://example.com/guide.md)",
            "[mailto](mailto:docs@example.com)",
            "[custom](vscode://file/guide.md)",
            "[protocol](//example.com/guide.md)",
        )
    )
    assert _local_markdown_targets(content) == [
        "guide.md",
        "guide.md",
        "guide with spaces.md",
    ]


def test_discovery_surfaces_preserve_ownership_and_blocked_limitations() -> (
    None
):
    """Require ownership, non-promotion, and no-fallback language in scope."""
    for source in DISCOVERY_PATHS:
        content = _normalized(source.read_text(encoding="utf-8"))
        assert "e8-f7/t7" in content
        assert "profiling and machine-bounded recommendations" in content
        assert "e8-f8" in content
        assert "example" in content
        assert "runbook" in content
        assert "limitations" in content
        assert "closeout" in content
        assert "unshipped" in content
        assert "no cpu or warp-cpu fallback" in content

    closeout = _normalized(RECORD_PATH.read_text(encoding="utf-8"))
    runbook = _normalized(RUNBOOK_PATH.read_text(encoding="utf-8"))
    assert "disposition: unshipped/blocked" in closeout
    assert "qualified cuda availability" in runbook
    assert "fixed inventory" in runbook
    assert "concrete, direct-import-only" in runbook
    assert "pass-or-clean-skip is not required measured evidence" in closeout


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
