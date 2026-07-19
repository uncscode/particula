"""Regression tests for the condensation parity walkthrough ownership record."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECORD_PATH = ROOT / "docs/Features/Roadmap/condensation-parity-walkthrough.md"
STIFFNESS_PATH = ROOT / "docs/Features/Roadmap/condensation-stiffness-study.md"
ROADMAP_PATH = ROOT / "docs/Features/Roadmap/data-oriented-gpu.md"
RECORD_TEST_PATH = (
    ROOT / "particula/tests/condensation_parity_walkthrough_docs_test.py"
)
WALKTHROUGH_PATH = ROOT / "docs/Examples/gpu_condensation_parity_walkthrough.py"
CONDENSATION_GUIDE_PATH = ROOT / "docs/Features/condensation_strategy_system.md"
FOUNDATIONS_GUIDE_PATH = (
    ROOT / "docs/Features/data-containers-and-gpu-foundations.md"
)
EXAMPLES_INDEX_PATH = ROOT / "docs/Examples/index.md"
ROADMAP_INDEX_PATH = ROOT / "docs/Features/Roadmap/index.md"
CANONICAL_P4_PATHS = (
    CONDENSATION_GUIDE_PATH,
    FOUNDATIONS_GUIDE_PATH,
    EXAMPLES_INDEX_PATH,
    ROADMAP_INDEX_PATH,
    ROADMAP_PATH,
)
P4_LINKS = {
    CONDENSATION_GUIDE_PATH: (
        "[GPU condensation parity walkthrough](../Examples/"
        "gpu_condensation_parity_walkthrough.py)",
        "[condensation parity walkthrough ownership record](Roadmap/"
        "condensation-parity-walkthrough.md)",
    ),
    FOUNDATIONS_GUIDE_PATH: (
        "[GPU condensation parity walkthrough](../Examples/"
        "gpu_condensation_parity_walkthrough.py)",
        "[condensation parity walkthrough ownership record](Roadmap/"
        "condensation-parity-walkthrough.md)",
    ),
    EXAMPLES_INDEX_PATH: (
        "[GPU condensation parity walkthrough]("
        "gpu_condensation_parity_walkthrough.py)",
        "[condensation parity walkthrough ownership record](../Features/"
        "Roadmap/condensation-parity-walkthrough.md)",
    ),
    ROADMAP_INDEX_PATH: (
        "[GPU condensation parity walkthrough](../../Examples/"
        "gpu_condensation_parity_walkthrough.py)",
        "[condensation parity walkthrough ownership record]("
        "condensation-parity-walkthrough.md)",
    ),
    ROADMAP_PATH: (
        "[GPU condensation parity walkthrough](../../Examples/"
        "gpu_condensation_parity_walkthrough.py)",
        "[condensation parity walkthrough ownership record]("
        "condensation-parity-walkthrough.md)",
    ),
}
P4_COMMANDS = (
    "python docs/Examples/gpu_condensation_parity_walkthrough.py",
    "pytest particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py "
    "-q -Werror",
    "pytest particula/tests/condensation_parity_walkthrough_docs_test.py -q "
    "-Werror",
)

LABELS = (
    "thermal_work consumption",
    "temperature feedback",
    "adaptive stepping",
    "backend selection",
    "high-level Aerosol/Runnable integration",
    "GPU-resident/full-workflow coupling",
    "general CPU workflow/strategy parity",
    "graph capture/replay",
    "host-validation/capture separation",
    "performance/benchmarking",
    "memory-budget work",
    "broad state/multi-step autodiff",
    "phase-aware surface tension",
    "BAT activity",
)

NUMERICAL_OWNER = "Future approved condensation numerical-method work"
PHYSICS_OWNER = "Approved condensation-physics expansion"
EXPECTED_OWNERS = {
    **{label: NUMERICAL_OWNER for label in LABELS[:3]},
    **{label: "Epic G" for label in LABELS[3:7]},
    **{label: "Epic H" for label in LABELS[7:11]},
    LABELS[11]: "Epic I",
    **{label: PHYSICS_OWNER for label in LABELS[12:]},
}
GATE_SNIPPETS = {
    NUMERICAL_OWNER: "approved numerical-method plan and numerical/validation",
    "Epic G": "Backend-selection and GPU-resident integration work",
    "Epic I": "Differentiability and gradient-validation work",
    PHYSICS_OWNER: "approved plan and physics-validation contract",
}
EPIC_H_GATE_SNIPPETS = {
    "graph capture/replay": "Graph-capturable stable-shape loop validation",
    "host-validation/capture separation": (
        "Graph-capturable stable-shape loop validation"
    ),
    "performance/benchmarking": "Benchmark/memory-budget exit work",
    "memory-budget work": "Benchmark/memory-budget exit work",
}
EXPECTED_HEADERS = (
    "Deferred capability",
    "Downstream owner",
    "Entry gate",
    "E5-F8 non-claim",
)
LINKS = (
    (
        "[walkthrough source](../../Examples/gpu_condensation_parity_walkthrough.py)",
        "../../Examples/gpu_condensation_parity_walkthrough.py",
        None,
    ),
    (
        "[walkthrough regression test](../../../particula/gpu/tests/"
        "gpu_condensation_parity_walkthrough_test.py)",
        "../../../particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py",
        None,
    ),
    (
        "[fixed-four direct-kernel evidence boundary](condensation-stiffness-"
        "study.md#p1--p4-production-hook-evidence-boundary)",
        "condensation-stiffness-study.md",
        "### P1--P4 production-hook evidence boundary",
    ),
    (
        "[Epic G](data-oriented-gpu.md#epic-g-backend-selection-and-gpu-"
        "resident-simulation)",
        "data-oriented-gpu.md",
        "## Epic G: Backend Selection and GPU-Resident Simulation",
    ),
    (
        "[Epic H](data-oriented-gpu.md#epic-h-graph-capture-and-performance)",
        "data-oriented-gpu.md",
        "## Epic H: Graph Capture and Performance",
    ),
    (
        "[Epic I](data-oriented-gpu.md#epic-i-differentiability-and-global-"
        "optimization)",
        "data-oriented-gpu.md",
        "## Epic I: Differentiability and Global Optimization",
    ),
)
COMMANDS = (
    "python docs/Examples/gpu_condensation_parity_walkthrough.py",
    "pytest particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py "
    "-q -Werror",
    "pytest particula/gpu/kernels/tests/condensation_test.py -q -Werror",
    "pytest particula/gpu/kernels/tests/condensation_stiffness_test.py -q "
    "-Werror",
    "pytest particula/gpu/kernels/tests/condensation_graph_capture_test.py -q "
    "-Werror",
    "pytest particula/gpu/kernels/tests/condensation_autodiff_test.py -q "
    "-Werror",
    "pytest particula/tests/condensation_parity_walkthrough_docs_test.py -q "
    "-Werror",
)
COMMAND_TARGETS = (
    ROOT / "docs/Examples/gpu_condensation_parity_walkthrough.py",
    ROOT / "particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py",
    ROOT / "particula/gpu/kernels/tests/condensation_test.py",
    ROOT / "particula/gpu/kernels/tests/condensation_stiffness_test.py",
    ROOT / "particula/gpu/kernels/tests/condensation_graph_capture_test.py",
    ROOT / "particula/gpu/kernels/tests/condensation_autodiff_test.py",
)
BOUNDARY_SNIPPETS = (
    "independent fp64, fixed-four-substep NumPy oracle",
    "low-level\ndirect GPU kernel",
    "Physics, conservation, and energy are separately evaluated evidence.",
    "Warp CPU\nis the supported baseline when installed; CUDA is optional, additive evidence.",
    "does not establish high-level CPU strategy parity, `Runnable`\nparity, or general CPU workflow parity.",
)


def _section(content: str, heading: str) -> str:
    """Return an exact level-two section through the next level-one/two heading."""
    lines = content.splitlines()
    matches = [index for index, line in enumerate(lines) if line == heading]
    assert matches, f"Missing exact heading: {heading}"
    assert len(matches) == 1, f"Heading must occur once: {heading}"
    start = matches[0]
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if re.fullmatch(r"#{1,2} .+", lines[index]):
            end = index
            break
    return "\n".join(lines[start:end])


def _table_pairs(content: str) -> list[int]:
    """Return line indexes for every Markdown table header/separator pair."""
    lines = content.splitlines()
    return [
        index
        for index in range(len(lines) - 1)
        if lines[index].lstrip().startswith("|")
        and re.fullmatch(r"\s*\|(?:\s*:?-+:?\s*\|)+\s*", lines[index + 1])
    ]


def _parse_ownership_table(content: str) -> list[tuple[str, str, str, str]]:
    """Parse the ownership table immediately following its level-two heading."""
    section = _section(content, "## Deferred capability ownership")
    lines = section.splitlines()[1:]
    while lines and not lines[0].strip():
        lines.pop(0)
    assert len(lines) >= 2, "Ownership heading must be followed by a table"
    assert lines[0].startswith("|"), "Ownership table must follow heading"
    assert re.fullmatch(r"\s*\|(?:\s*:?-+:?\s*\|)+\s*", lines[1]), (
        "Ownership table has a malformed separator"
    )
    header = _split_row(lines[0])
    assert tuple(header) == EXPECTED_HEADERS, "Ownership table header is wrong"

    rows: list[tuple[str, str, str, str]] = []
    for line in lines[2:]:
        if not line.strip():
            break
        assert line.startswith("|"), f"Malformed ownership row: {line}"
        cells = _split_row(line)
        assert len(cells) == 4, f"Ownership row must have four cells: {line}"
        assert all(cells), f"Ownership row has a blank cell: {line}"
        rows.append((cells[0], cells[1], cells[2], cells[3]))
    return rows


def _split_row(line: str) -> list[str]:
    """Split a pipe-delimited Markdown row and trim its outer whitespace."""
    assert line.rstrip().endswith("|"), f"Malformed Markdown table row: {line}"
    return [cell.strip() for cell in line.strip().split("|")[1:-1]]


def test_ownership_table_is_unique_and_routes_every_deferred_capability() -> (
    None
):
    """Ownership table has one valid row for each approved downstream route."""
    content = RECORD_PATH.read_text(encoding="utf-8")
    rows = _parse_ownership_table(content)

    assert len(_table_pairs(content)) == 1, (
        "Record must contain one Markdown table"
    )
    assert len(rows) == 14, "Ownership table must have exactly 14 body rows"
    labels = [row[0] for row in rows]
    assert set(labels) == set(LABELS), "Ownership labels do not match contract"
    assert all(labels.count(label) == 1 for label in LABELS)

    for label, owner, gate, non_claim in rows:
        assert owner == EXPECTED_OWNERS[label]
        if owner == "Epic H":
            gate_snippet = EPIC_H_GATE_SNIPPETS[label]
        else:
            gate_snippet = GATE_SNIPPETS[owner]
        assert gate_snippet in gate
        assert "outside the walkthrough/E5-F8 scope" in non_claim


def test_record_links_resolve_once_to_existing_artifacts_and_headings() -> None:
    """Every required source link resolves once and its Markdown anchor exists."""
    content = RECORD_PATH.read_text(encoding="utf-8")
    stiffness = STIFFNESS_PATH.read_text(encoding="utf-8")
    roadmap = ROADMAP_PATH.read_text(encoding="utf-8")

    for link, destination, heading in LINKS:
        assert content.count(link) == 1, f"Link must occur once: {link}"
        target = RECORD_PATH.parent / destination
        assert target.exists(), f"Missing linked target: {target}"
        if heading is not None:
            target_content = stiffness if target == STIFFNESS_PATH else roadmap
            assert heading in target_content, (
                f"Missing linked heading: {heading}"
            )


def test_focused_commands_are_complete_warning_clean_and_resolvable() -> None:
    """Focused command section contains only the required executable commands."""
    content = RECORD_PATH.read_text(encoding="utf-8")
    section = _section(content, "## Focused reproduction commands")
    fences = re.findall(
        r"^```bash\n(.*?)^```$", section, flags=re.MULTILINE | re.DOTALL
    )

    assert len(fences) == 1, "Focused commands require exactly one bash fence"
    commands = tuple(line for line in fences[0].splitlines() if line.strip())
    assert commands == COMMANDS, "Focused commands must match the contract"
    for command in commands[1:]:
        assert "-q -Werror" in command, (
            f"Pytest command lacks -q -Werror: {command}"
        )

    for command, target in zip(commands[:-1], COMMAND_TARGETS, strict=True):
        assert command.split()[1] == str(target.relative_to(ROOT))
        assert target.exists(), f"Missing command target: {target}"
    assert RECORD_TEST_PATH.exists()


def test_record_preserves_supported_evidence_boundary_and_deferred_limits() -> (
    None
):
    """Record distinguishes direct evidence from unsupported future capabilities."""
    content = RECORD_PATH.read_text(encoding="utf-8")
    boundary = _section(content, "## Supported evidence boundary")
    ownership = _section(content, "## Deferred capability ownership")

    for snippet in BOUNDARY_SNIPPETS:
        assert " ".join(snippet.split()) in " ".join(boundary.split())
    plan_ids = set(re.findall(r"E\d+-[A-Z]\d+", ownership))
    assert plan_ids <= {"E5-F8"}, "Deferred owners must not claim plan IDs"
    assert "Epic F" not in ownership
    for _label, owner, gate, non_claim in _parse_ownership_table(content):
        assert owner and gate and non_claim
        if owner in {NUMERICAL_OWNER, PHYSICS_OWNER}:
            assert not re.search(r"E\d+-[A-Z]\d+", f"{owner} {gate}")

    deferred = (
        "high-level Aerosol/Runnable integration",
        "adaptive stepping",
        "graph capture/replay",
        "performance/benchmarking",
        "broad state/multi-step autodiff",
        "phase-aware surface tension",
        "BAT activity",
    )
    normalized = " ".join((boundary + "\n" + ownership).split())
    for capability in deferred:
        positive_claim = re.compile(
            rf"(?:{re.escape(capability)}\s+(?:is\s+)?"
            rf"(?:supported|delivered|implemented)|"
            rf"(?:supports|delivers|implements)\s+(?:the\s+)?"
            rf"{re.escape(capability)})",
            re.IGNORECASE,
        )
        assert not positive_claim.search(normalized), (
            f"Deferred capability has a positive support/delivery claim: "
            f"{capability}"
        )


def test_canonical_pages_link_once_to_walkthrough_and_ownership_record() -> (
    None
):
    """Canonical P4 pages link once to the local walkthrough artifacts."""
    assert WALKTHROUGH_PATH.exists(), f"Missing walkthrough: {WALKTHROUGH_PATH}"
    assert RECORD_PATH.exists(), f"Missing ownership record: {RECORD_PATH}"
    for source_path, links in P4_LINKS.items():
        assert source_path.exists(), f"Missing canonical source: {source_path}"
        content = source_path.read_text(encoding="utf-8")
        for link in links:
            assert content.count(link) == 1, (
                f"Link must occur once in {source_path}: {link}"
            )
            relative_target = link.removesuffix(")").rsplit("(", maxsplit=1)[1]
            target = source_path.parent / relative_target
            assert target.exists(), f"Missing linked target: {target}"


def test_canonical_pages_preserve_p4_evidence_boundary() -> None:
    """Canonical P4 pages state evidence categories and retained limits."""
    contents = [path.read_text(encoding="utf-8") for path in CANONICAL_P4_PATHS]
    normalized = " ".join("\n".join(contents).split())

    for command in P4_COMMANDS:
        assert command in normalized
    for label in ("physics", "conservation", "energy"):
        assert label in normalized
    assert "fixed-four-substep" in normalized
    assert "low-level direct-kernel" in normalized
    assert "Warp CPU" in normalized
    assert "optional additive CUDA" in normalized
    assert "kg * J/kg = J" in normalized
    assert "energy_transfer" in normalized
    assert "caller-owned" in normalized
    assert "write-only" in normalized
    assert "not a return value" in normalized
    assert "not temperature feedback" in normalized

    unsupported = (
        "temperature feedback",
        "strategy/`Runnable` parity",
        "graph capture/replay",
        "broad autodiff",
        "adaptive stepping",
        "performance",
        "required CUDA",
    )
    for capability in unsupported:
        positive_claim = re.compile(
            rf"(?:{re.escape(capability)}\s+(?:is\s+)?"
            rf"(?:supported|delivered|implemented)|"
            rf"(?:supports|delivers|implements)\s+(?:the\s+)?"
            rf"{re.escape(capability)})",
            re.IGNORECASE,
        )
        assert not positive_claim.search(normalized), (
            f"Unsupported capability has a positive claim: {capability}"
        )
