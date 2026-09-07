"""Validate the hardware-free resident graph-capture operator runbook."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNBOOK_PATH = ROOT / "docs/Features/gpu_graph_capture.md"
COMMANDS = (
    "pytest particula/tests/gpu_graph_capture_runbook_docs_test.py "
    "particula/tests/gpu_resident_graph_capture_docs_test.py -q --no-cov",
    "pytest particula/execution/tests/graph_capture_test.py -q --no-cov",
    "pytest particula/execution/tests/captured_full_loop_test.py -q --no-cov",
    "pytest particula/execution/tests/captured_full_loop_test.py -q "
    '-m "warp and cuda" --no-cov',
    ".opencode/tools/run_pytest.py",
    "mkdocs build --strict",
)
STDLIB_IMPORTS = {"__future__", "ast", "pathlib"}


def _normalized(content: str) -> str:
    """Return content with whitespace and case normalized."""
    return " ".join(content.replace("\\", "").lower().split())


def _markdown_links(content: str) -> list[str]:
    """Return local Markdown link targets without anchors."""
    targets: list[str] = []
    cursor = 0
    while (start := content.find("](", cursor)) != -1:
        end = content.find(")", start)
        if end == -1:
            break
        target = content[start + 2 : end].split("#", maxsplit=1)[0]
        if target and not target.startswith(("http://", "https://")):
            targets.append(target)
        cursor = end + 1
    return targets


def test_runbook_covers_direct_cuda_setup_replay_and_recovery() -> None:
    """Require the concrete setup, replay, synchronization, and renewal flow."""
    content = _normalized(RUNBOOK_PATH.read_text(encoding="utf-8"))
    for requirement in (
        "concrete, direct-import-only machinery",
        "non-cpu native warp cuda device",
        "lazily qualify native cuda before cpu fixture construction",
        "pinned resource registry",
        "prepare_capture_resources()",
        "explicitly initialize the `coagulation` and `wall_loss` resident streams",
        "ready binding",
        "explicitly capture it to reach captured",
        "authentic issued record",
        "matching duration",
        "one call is one replay",
        "warp.synchronize_device(...)",
        "retire stale or invalidated metadata",
        "renew only a retired binding",
        "re-prepare, re-qualify, and explicitly capture",
    ):
        assert requirement in content


def test_runbook_lists_canonical_drift_order_and_nontriggers() -> None:
    """Require ordered fail-closed structural recapture triggers."""
    content = _normalized(RUNBOOK_PATH.read_text(encoding="utf-8"))
    identifiers = (
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
    positions = [content.index(f"`{identifier}`") for identifier in identifiers]
    assert positions == sorted(positions)
    for requirement in (
        "replacing any listed item fails closed",
        "invalidates replay",
        "same-object payload updates",
        "active/free-slot changes",
        "advancing resident rng words are non-triggers",
    ):
        assert requirement in content


def test_runbook_covers_lifecycle_and_failure_actions() -> None:
    """Keep skips, drift, faulting, and release handling explicit."""
    content = _normalized(RUNBOOK_PATH.read_text(encoding="utf-8"))
    for state in (
        "ready",
        "captured",
        "invalidated",
        "faulted",
        "retired",
        "closed",
    ):
        assert f"`{state}`" in content
    for requirement in (
        "clean skip",
        "no setup or capture and do not fall back",
        "structural drift invalidates capture",
        "no rollback or retry",
        "release failure does not restore provenance",
        "close graph capture before closing the session",
    ):
        assert requirement in content


def test_runbook_preserves_handle_and_restart_limits() -> None:
    """Prevent unsupported recovery, transfer, and portability claims."""
    content = _normalized(RUNBOOK_PATH.read_text(encoding="utf-8"))
    for requirement in (
        "opaque, nonserializable, and uncheckpointed",
        "no stale reuse, migration, serialization, cross-device replay",
        "fresh session, resource and array identities, qualification, setup, and capture",
        "restart restores resident bytes and rng continuation only",
        "normal replay never initializes or resets streams",
        "explicit resident-stream lifecycle operation is the only reset boundary",
        "automatic recapture/retry/rollback",
        "hidden transfer/readback/synchronization",
        "resizing, compaction",
        "cpu or warp-cpu capture/emulation/fallback",
        "portable performance claim",
    ):
        assert requirement in content


def test_runbook_commands_and_relative_links_resolve() -> None:
    """Require literal reproduction commands and valid local documentation links."""
    content = RUNBOOK_PATH.read_text(encoding="utf-8")
    normalized = _normalized(content)
    for command in COMMANDS:
        assert _normalized(command) in normalized
    targets = _markdown_links(content)
    assert targets
    for target in targets:
        assert (RUNBOOK_PATH.parent / target).resolve().exists()


def test_contract_test_uses_only_approved_stdlib_imports() -> None:
    """Keep the documentation contract independent of runtime and hardware."""
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
