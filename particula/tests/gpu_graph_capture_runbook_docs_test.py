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
SOURCE_REFERENCES = (
    "particula/execution/tests/graph_capture_test.py",
    "particula/execution/tests/captured_full_loop_test.py",
    "particula/tests/gpu_graph_capture_runbook_docs_test.py",
    "particula/tests/gpu_resident_graph_capture_docs_test.py",
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
        target_start = start + 2
        depth = 1
        end = target_start
        while end < len(content) and depth:
            if content[end] == "(":
                depth += 1
            elif content[end] == ")":
                depth -= 1
            end += 1
        if depth:
            break
        target = content[target_start : end - 1].split("#", maxsplit=1)[0]
        if target and not target.startswith(("http://", "https://")):
            targets.append(target)
        cursor = end
    return targets


def _assert_local_links_resolve(
    content: str, source: Path, docs_root: Path
) -> None:
    """Require local links to name regular files within the docs root."""
    for target in _markdown_links(content):
        resolved = (source.parent / target).resolve()
        assert resolved.is_relative_to(docs_root.resolve())
        assert resolved.is_file()


def test_runbook_covers_direct_cuda_setup_replay_and_recovery() -> None:
    """Require the concrete setup, replay, synchronization, and renewal flow."""
    content = _normalized(RUNBOOK_PATH.read_text(encoding="utf-8"))
    for requirement in (
        "concrete, direct-import-only machinery",
        "non-cpu native warp cuda device",
        "resolve native-cuda capability and probes before cpu fixture construction",
        "only an unavailable capability or probe outcome is a clean skip",
        "qualification rejection raises `valueerror`; it is not a clean skip",
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
    for row in (
        "| `captured` | yes, after every replay precondition | replay, or replace an identity to invalidate before retirement. |",
        "| `invalidated` | no | retire stale metadata, renew it to `ready`, then prepare, qualify, and explicitly capture. |",
        "| `retired` | no | renew the retired binding to `ready`, then prepare, qualify, and explicitly capture. |",
    ):
        assert row in content


def test_runbook_names_concrete_lifecycle_operations() -> None:
    """Require all direct-import graph-capture operations named by the runbook."""
    content = _normalized(RUNBOOK_PATH.read_text(encoding="utf-8"))
    for operation in (
        "resolve_graph_capture_capability()",
        "qualify_prepared_resident_graph_capture()",
        "capture_prepared_resident_graph()",
        "replay_captured_resident_graph()",
        "retire_resident_graph_capture()",
        "renew_resident_graph_capture()",
        "close_resident_graph_capture()",
    ):
        assert operation in content


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


def test_runbook_commands_references_and_relative_links_resolve() -> None:
    """Require literal commands, source references, and valid documentation links."""
    content = RUNBOOK_PATH.read_text(encoding="utf-8")
    normalized = _normalized(content)
    for command in COMMANDS:
        assert _normalized(command) in normalized
    for source_reference in SOURCE_REFERENCES:
        assert source_reference in content
    assert _markdown_links(content)
    _assert_local_links_resolve(content, RUNBOOK_PATH, ROOT / "docs")


def test_local_links_reject_external_paths_and_nonfiles(tmp_path: Path) -> None:
    """Keep local runbook links confined to regular documentation files."""
    docs_root = tmp_path / "docs"
    source = docs_root / "Features" / "runbook.md"
    source.parent.mkdir(parents=True)
    source.touch()
    target = docs_root / "Examples" / "example.md"
    target.parent.mkdir()
    target.touch()

    _assert_local_links_resolve(
        "[example](../Examples/example.md)", source, docs_root
    )

    for content in (
        "[outside](../../outside.md)",
        "[directory](../Examples)",
    ):
        try:
            _assert_local_links_resolve(content, source, docs_root)
        except AssertionError:
            pass
        else:
            raise AssertionError("invalid local Markdown link was accepted")


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
