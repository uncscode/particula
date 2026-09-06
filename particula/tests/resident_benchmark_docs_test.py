"""Validate the hardware-free resident benchmark evidence publication contract.

These tests read only the roadmap and resident benchmark report. They do not
load benchmark artifacts, import GPU dependencies, or execute benchmarks.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROADMAP_PATH = ROOT / "docs/Features/Roadmap/data-oriented-gpu.md"
REPORT_PATH = ROOT / "docs/Features/resident_benchmark_memory_budget.md"
ARTIFACT_PATH = ".artifacts/benchmarks/resident_capture_comparison.json"
ABSENT_REASON = "no reviewed schema-v3 resident artifact is checked in"
COMMAND = (
    "pytest particula/gpu/tests/benchmark_test.py --benchmark -k resident "
    "-v -s --no-cov"
)
STDLIB_IMPORTS = {"__future__", "ast", "pathlib", "re"}


def _section(content: str, heading: str) -> str:
    """Extract one exact level-two Markdown section.

    Args:
        content: Complete Markdown document content.
        heading: Exact level-two heading that starts the requested section.

    Returns:
        The requested heading and its content through the next level-one or
        level-two heading.
    """
    lines = content.splitlines()
    matches = [index for index, line in enumerate(lines) if line == heading]
    assert len(matches) == 1, f"Expected one heading: {heading}"
    start = matches[0]
    end = next(
        (
            index
            for index in range(start + 1, len(lines))
            if re.fullmatch(r"#{1,2} .+", lines[index])
        ),
        len(lines),
    )
    return "\n".join(lines[start:end])


def _split_row(line: str) -> list[str]:
    """Split a pipe-terminated Markdown table row into trimmed cells.

    Args:
        line: One Markdown table row.

    Returns:
        Table-cell text without surrounding whitespace or delimiter pipes.
    """
    assert line.rstrip().endswith("|"), f"Malformed table row: {line}"
    return [cell.strip() for cell in line.strip().split("|")[1:-1]]


def _current_evidence_rows(content: str) -> list[list[str]]:
    """Extract canonical case rows from the current-evidence table.

    Args:
        content: Complete resident benchmark report content.

    Returns:
        Parsed cells for each contiguous data row in the current-evidence
        table.
    """
    lines = _section(content, "## Current evidence status").splitlines()[1:]
    table_start = next(
        index for index, line in enumerate(lines) if line.startswith("|")
    )
    assert re.fullmatch(r"\s*\|(?:\s*:?-+:?\s*\|)+\s*", lines[table_start + 1])
    rows = []
    for line in lines[table_start + 2 :]:
        if not line.strip():
            break
        assert line.startswith("|"), f"Malformed table row: {line}"
        rows.append(_split_row(line))
    return rows


def _markdown_destinations(content: str, source_path: Path) -> list[Path]:
    """Extract normalized local Markdown-link destinations.

    Args:
        content: Markdown content whose links are inspected.
        source_path: Path of the Markdown file that contains ``content``.

    Returns:
        Absolute paths for local Markdown link destinations, excluding external
        URLs and fragment-only links.
    """
    destinations = []
    for destination in re.findall(r"(?<!!)\[[^]]*\]\(([^)]+)\)", content):
        destination = destination.split(maxsplit=1)[0].strip("<>")
        if "://" in destination or destination.startswith("#"):
            continue
        destinations.append(
            (source_path.parent / destination.split("#")[0]).resolve()
        )
    return destinations


def _resident_roadmap_section() -> str:
    """Return the resident benchmark publication section from the roadmap."""
    return _section(
        ROADMAP_PATH.read_text(encoding="utf-8"),
        "#### E8-F6-P6 resident benchmark and memory-budget evidence",
    )


def test_roadmap_links_once_to_resident_benchmark_report() -> None:
    """Validate the roadmap names the artifact and links once to the report."""
    roadmap = ROADMAP_PATH.read_text(encoding="utf-8")
    report = REPORT_PATH.read_text(encoding="utf-8")

    assert ARTIFACT_PATH in roadmap
    assert ARTIFACT_PATH in report
    assert (
        _markdown_destinations(roadmap, ROADMAP_PATH).count(
            REPORT_PATH.resolve()
        )
        == 1
    )


def test_report_reproduces_the_fixed_unavailable_benchmark_configuration() -> (
    None
):
    """Validate the report records fixed configuration and planning inputs."""
    section = _section(
        REPORT_PATH.read_text(encoding="utf-8"),
        "## Reproduction command and fixed matrix",
    )
    normalized = " ".join(section.split())

    for requirement in (
        COMMAND,
        "1, 10, 100, and 1000",
        "`(B, 16, 2)`",
        "100% activity",
        "communication, condensation, coagulation, dilution, wall loss, "
        "nucleation, and diagnostics",
        "gas communication",
        "gas/saturation diagnostics",
        "seed 1582",
        "two warmups",
        "three samples",
        "2 GiB budget",
        "64 MiB, 256 MiB, 1 GiB, and 4 GiB",
    ):
        assert requirement in normalized


def test_roadmap_reproduces_resident_artifact_and_configuration_boundary() -> (
    None
):
    """Validate roadmap publication preserves resident evidence safeguards."""
    normalized = " ".join(_resident_roadmap_section().split())

    for requirement in (
        ARTIFACT_PATH,
        ".artifacts/benchmarks/gpu_benchmark_results.json",
        "coagulation-only",
        "not resident evidence",
        "absent in this revision",
        COMMAND,
        "CUDA/native-capture-only",
        "no CPU or Warp-CPU fallback",
        "1, 10, 100, and 1000 boxes",
        "`(B, 16, 2)` shape",
        "100% activity",
        "gas communication",
        "gas/saturation diagnostics",
        "seed 1582",
        "two warmups",
        "three measured timesteps",
        "2 GiB budget",
        "64 MiB, 256 MiB, 1 GiB, and 4 GiB",
        "planning inputs, not measured allocator consumption",
        "unavailable",
        "not measured and is not zero",
        "`skipped_budget`",
    ):
        assert requirement in normalized


def test_current_evidence_table_publishes_only_unavailable_rows() -> None:
    """Validate current evidence has only unavailable, nonnumeric case rows."""
    report = REPORT_PATH.read_text(encoding="utf-8")
    rows = _current_evidence_rows(report)

    assert len(rows) == 4
    assert [row[0] for row in rows] == ["1", "10", "100", "1000"]
    for row in rows:
        assert len(row) == 5
        assert row[1] == "unavailable"
        assert row[2] == "not measured"
        assert row[3] == "not measured"
        assert row[4] == ABSENT_REASON
        assert not re.search(r"\d", " ".join(row[2:4]))
    assert not re.search(r"\|\s*skipped_budget\s*\|", report)


def test_report_distinguishes_accounting_terms_and_unimplemented_tape() -> None:
    """Validate the report separates accounting and projected tape vocabulary."""
    section = _section(
        REPORT_PATH.read_text(encoding="utf-8"),
        "## Timing and memory evidence schema",
    )
    normalized = " ".join(section.split())

    for requirement in (
        "analytical logical steady-state categories",
        "primary state, registry manifest, selected diagnostics, and selected",
        "communication metadata",
        "Inactive capacity attribution is non-additive",
        "checkpoint host-copy scenario",
        "Allocator-observed CUDA default-pool high-water delta",
        "signed observed-minus-analytical difference",
        "method, version, coverage, and machine context",
        "`timesteps × state_bytes`",
        "`ceil(timesteps / interval) × checkpoint_bytes + interval ×",
        "state_bytes`",
        "Autodiff tape is not implemented or measured",
    ):
        assert requirement in normalized


def test_report_preserves_paired_timing_modes_and_provenance_vocabulary() -> (
    None
):
    """Validate future benchmark records use the fixed evidence vocabulary."""
    section = _section(
        REPORT_PATH.read_text(encoding="utf-8"),
        "## Timing and memory evidence schema",
    )
    normalized = " ".join(section.split())

    for requirement in (
        "alternating, device-synchronized modes",
        "`prepared_uncaptured_device_synchronized`",
        "`captured_replay_device_synchronized`",
        "count, minimum, median, mean, and p95",
        "Setup and capture provenance",
        "excluded from timing samples",
        "UTC timestamp, Python/platform, Warp, device, synchronization, "
        "signature, seed, warmups, and sample count",
        "timing and allocator values are unavailable in this revision",
        "unavailable readings are not zero",
        "unknown Epic I overhead is excluded",
    ):
        assert requirement in normalized


def test_report_states_all_documentation_scope_limitations() -> None:
    """Validate the report states documentation-only scope and limitations."""
    section = _section(
        REPORT_PATH.read_text(encoding="utf-8"),
        "## Supported limitations",
    ).lower()
    normalized = " ".join(section.split())

    for limitation in (
        "documentation-only scope",
        "does not change collection code",
        "artifacts, apis, ci policy, lifecycle behavior, or examples",
        "no cpu fallback",
        "no warp-cpu capture emulation",
        "no inferred measurements",
        "universal speedups",
        "hard performance ci gates",
        "allocator guarantees",
        "implemented autodiff storage",
    ):
        assert limitation in normalized


def test_module_imports_are_stdlib_only() -> None:
    """Keep this hardware-free documentation contract test dependency-free."""
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))

    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(
                alias.name.split(".")[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.add(node.module.split(".")[0])

    assert imported_modules <= STDLIB_IMPORTS
