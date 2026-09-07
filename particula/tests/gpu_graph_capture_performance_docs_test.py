"""Validate the hardware-free graph-capture profiling publication boundary."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AGENTS_PATH = ROOT / "AGENTS.md"
ROADMAP_PATH = ROOT / "docs/Features/Roadmap/data-oriented-gpu.md"
RECORD_PATH = ROOT / "docs/Features/gpu_graph_capture_performance.md"
DOCUMENT_PATHS = (AGENTS_PATH, ROADMAP_PATH, RECORD_PATH)
COMMANDS = (
    "pytest particula/gpu/tests/profiling_support_test.py -q --no-cov",
    "pytest particula/gpu/tests/benchmark_helpers_test.py -q --no-cov",
    "pytest particula/gpu/tests/benchmark_test.py --benchmark -k resident -v "
    "-s --no-cov",
    "pytest particula/gpu/tests/profiling_smoke_test.py --benchmark -q --no-cov",
    "pytest particula/tests/gpu_graph_capture_performance_docs_test.py -q "
    "--no-cov",
    ".opencode/tools/run_pytest.py",
    "mkdocs build --strict",
)
STDLIB_IMPORTS = {"__future__", "ast", "pathlib"}


def _normalized(content: str) -> str:
    """Return content with whitespace and case normalized."""
    return " ".join(content.lower().split())


def _documents() -> dict[Path, str]:
    """Read only the three scoped documentation inputs."""
    return {path: path.read_text(encoding="utf-8") for path in DOCUMENT_PATHS}


def _ownership_clauses(content: str) -> list[str]:
    """Return sentences that make an E8-F8 ownership statement."""
    return [
        clause.strip()
        for clause in _normalized(content).replace(";", ".").split(".")
        if any(
            marker in clause
            for marker in ("e8-f8 is ", "e8-f8 only", "e8-f8 restricted")
        )
    ]


def test_documents_assign_profiling_to_e8_f7_and_restrict_e8_f8() -> None:
    """Keep profiling ownership distinct from the E8-F8 closeout scope."""
    documents = _documents()
    for content in documents.values():
        normalized = _normalized(content)
        assert "e8-f7/t7" in normalized
        assert "profiling" in normalized
        clauses = _ownership_clauses(content)
        assert clauses
        for clause in clauses:
            e8_f8_scope = clause.split("e8-f8", maxsplit=1)[1]
            assert all(
                word in e8_f8_scope
                for word in ("example", "limitation", "closeout")
            )
            assert "profiling" not in e8_f8_scope
            assert "recommendation" not in e8_f8_scope

    combined = _normalized(" ".join(documents.values()))
    assert "e8-f5--e8-f8" not in combined
    assert "e8-f5--e8-f7" not in combined


def test_canonical_record_freezes_matrix_commands_and_provenance() -> None:
    """Require the canonical record to retain reproducible future-row rules."""
    record = RECORD_PATH.read_text(encoding="utf-8")
    normalized = _normalized(record)

    for requirement in (
        "small `(1, 16, 2)` and medium `(1000, 16, 2)`",
        "100% activity",
        "gas communication",
        "communication/environment/gas/condensation/coagulation/dilution/wall-loss/ nucleation/diagnostics processes",
        "gas/saturation diagnostics",
        "two warmups",
        "three samples",
        "seed 1582",
        "duration of 0.5 s",
        "replay counts 1/10/100/1000",
        "`prepared_uncaptured` / `host_launch`",
        "`prepared_uncaptured` / `synchronized_elapsed`",
        "`captured_replay` / `host_launch`",
        "`captured_replay` / `synchronized_elapsed`",
        "time dispatch submission only; exclude in-interval synchronization",
        "time dispatch through one post-dispatch completion boundary",
        "time replay submission only; exclude in-interval synchronization",
        "time replay dispatch through one post-dispatch completion boundary",
        "the captured-replay rows use the specified replay counts",
        "those counts do not alter any excluded interval",
        "native-cuda qualification",
        "no reviewed normalized artifact or manifest is checked in",
        "kernel profiling is limited to compatible complete captured-replay small evidence",
        "nsys 2026.1.3.425-1",
        "ncu 2026.2.1.5-1",
        ".artifacts/benchmarks/profiling/",
        ".artifacts/benchmarks/profiling/raw/",
        "machine identifier/platform",
        "cuda device/architecture",
        "driver/runtime",
        "python, warp, source revision",
        "selected `nsys`/`ncu` versions",
        "command, timestamp, workload, mode, method",
        "raw artifact reference",
        "relative raw filename, byte size, and lowercase sha-256 reference",
        "unavailable and unshipped",
    ):
        assert requirement in normalized

    normalized_commands = _normalized(record)
    for command in COMMANDS:
        assert _normalized(command) in normalized_commands


def test_canonical_record_preserves_evidence_and_safety_limits() -> None:
    """Prevent unqualified timing claims or unsafe profiler publication."""
    normalized = _normalized(RECORD_PATH.read_text(encoding="utf-8"))

    for requirement in (
        "host launch has no in-interval synchronization",
        "one post-dispatch completion boundary",
        "graph launch is not device or kernel duration",
        "nsight perturbs timing",
        "not unprofiled throughput",
        "never publish raw reports, absolute paths, credentials, usernames, or device pointers",
        "no cpu or warp-cpu fallback",
        "portability or cross-machine/ workload inference",
        "speed thresholds",
        "public runtime profiler api",
        "auto-tuning",
        "separately referenced correctness plan",
        "missing prerequisites or counters are unavailable evidence",
        "never favorable evidence",
        "not measured and is not zero",
        "no fabricated result, speedup, ranking, or recommendation",
    ):
        assert requirement in normalized


def test_short_documents_link_to_and_preserve_record_boundary() -> None:
    """Keep abbreviated documents linked to the canonical publication record."""
    agents = _normalized(AGENTS_PATH.read_text(encoding="utf-8"))
    roadmap = _normalized(ROADMAP_PATH.read_text(encoding="utf-8"))

    assert "docs/features/gpu_graph_capture_performance.md" in agents
    assert "../gpu_graph_capture_performance.md" in roadmap
    assert (
        ROADMAP_PATH.parent / "../gpu_graph_capture_performance.md"
    ).resolve() == RECORD_PATH

    for content in (agents, roadmap):
        for requirement in (
            "e8-f7/t7",
            "unavailable and unshipped",
            "native-cuda-only",
            "no cpu or warp-cpu fallback",
        ):
            assert requirement in content


def test_contract_test_uses_only_approved_stdlib_imports() -> None:
    """Keep this contract independent of test, GPU, and artifact tooling."""
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
