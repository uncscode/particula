"""Regression tests for resident benchmark evidence publication."""

from __future__ import annotations

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


def _section(content: str, heading: str) -> str:
    """Return an exact level-two section through the next level-one/two heading."""
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
    """Return trimmed cells from a Markdown table row."""
    assert line.rstrip().endswith("|"), f"Malformed table row: {line}"
    return [cell.strip() for cell in line.strip().split("|")[1:-1]]


def _current_evidence_rows(content: str) -> list[list[str]]:
    """Return the canonical rows in the current-evidence table."""
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
    """Return normalized local Markdown-link destinations."""
    destinations = []
    for destination in re.findall(r"(?<!!)\[[^]]*\]\(([^)]+)\)", content):
        destination = destination.split(maxsplit=1)[0].strip("<>")
        if "://" in destination or destination.startswith("#"):
            continue
        destinations.append(
            (source_path.parent / destination.split("#")[0]).resolve()
        )
    return destinations


def test_roadmap_links_once_to_resident_benchmark_report() -> None:
    """Roadmap names the artifact and links once to the local report."""
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
    """Report records the fixed benchmark configuration and planning inputs."""
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


def test_current_evidence_table_publishes_only_unavailable_rows() -> None:
    """Current evidence has four unavailable, nonnumeric rows only."""
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
    """Report separates accounting vocabulary from projected tape scenarios."""
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
        "`timesteps × state_bytes`",
        "`ceil(timesteps / interval) × checkpoint_bytes + interval ×",
        "state_bytes`",
        "Autodiff tape is not implemented or measured",
    ):
        assert requirement in normalized


def test_report_states_all_documentation_scope_limitations() -> None:
    """Report limits publication to documentation without unsupported claims."""
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
