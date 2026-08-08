"""Focused transport tests for the read-only Git diagnostics adapter."""

from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch


def _load_adapter():
    path = Path(__file__).parents[1] / "read_only_git_diagnostics.py"
    spec = importlib.util.spec_from_file_location("read_only_git_diagnostics", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _registered_worktree_topology(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create one primary checkout and one registered linked worktree fixture."""
    primary = tmp_path / "primary"
    common = primary / ".git"
    linked = tmp_path / "linked"
    metadata = common / "worktrees" / "linked"
    metadata.mkdir(parents=True)
    linked.mkdir()
    (common / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (common / "config").write_text("[core]\n", encoding="utf-8")
    (metadata / "HEAD").write_text("ref: refs/heads/linked\n", encoding="utf-8")
    (metadata / "commondir").write_text("../..\n", encoding="utf-8")
    (metadata / "gitdir").write_text(f"{linked / '.git'}\n", encoding="utf-8")
    (linked / ".git").write_text(f"gitdir: {metadata}\n", encoding="utf-8")
    return primary, common, linked


def test_adapter_rejects_legacy_fields_without_core_dispatch(tmp_path: Path) -> None:
    """Legacy presentation fields are never translated into core requests."""
    adapter = _load_adapter()
    with patch.object(adapter, "execute_git_tool") as core:
        result = adapter.handle(
            {"command": "status", "worktree_path": str(tmp_path), "porcelain": True}
        )
    assert result["error"]["type"] == "invalid_request"
    assert "evidence_identity" not in result
    core.assert_not_called()


def test_adapter_constructs_context_and_dispatches_once() -> None:
    """An admitted request uses only explicit worktree-derived authority."""
    adapter = _load_adapter()
    expected = {
        "ok": True,
        "operation": "git_status",
        "data": {"status": "clean"},
        "evidence_identity": {"contract": "e37-m2-validation-git", "version": 1},
    }
    with (
        patch.object(adapter.shutil, "which", return_value="/usr/bin/git"),
        patch.object(adapter, "execute_git_tool", return_value=expected) as core,
    ):
        result = adapter.handle(
            {"command": "status", "worktree_path": str(adapter._TRUSTED_WORKTREE)}
        )
    assert result == expected
    context, operation, arguments = core.call_args.args
    assert context.root == adapter._TRUSTED_WORKTREE
    assert operation == "git_status"
    assert arguments == {"worktree_path": str(adapter._TRUSTED_WORKTREE)}


def test_adapter_admits_registered_linked_worktree_and_dispatches_once(tmp_path: Path) -> None:
    """A matching registry pointer, commondir, and backlink establish authority."""
    primary, common, linked = _registered_worktree_topology(tmp_path)
    adapter = _load_adapter()
    expected = {"ok": True, "operation": "git_status", "data": {"status": "clean"}}

    with (
        patch.object(adapter, "_TRUSTED_WORKTREE", primary),
        patch.object(adapter, "_COMMON_GIT_DIR", common),
        patch.object(adapter.shutil, "which", return_value="/usr/bin/git"),
        patch.object(adapter, "execute_git_tool", return_value=expected) as core,
    ):
        result = adapter.handle({"command": "status", "worktree_path": str(linked)})

    assert result == expected
    context, operation, arguments = core.call_args.args
    assert context.root == linked
    assert operation == "git_status"
    assert arguments == {"worktree_path": str(linked)}


def test_adapter_derives_common_directory_when_loaded_from_linked_worktree(
    tmp_path: Path,
) -> None:
    """Adapter checkout topology may itself be a registered linked worktree."""
    _, common, linked = _registered_worktree_topology(tmp_path)
    adapter = _load_adapter()

    assert adapter._derive_common_git_dir(linked) == common

    with (
        patch.object(adapter, "_TRUSTED_WORKTREE", linked),
        patch.object(adapter, "_COMMON_GIT_DIR", common),
        patch.object(adapter.shutil, "which", return_value="/usr/bin/git"),
        patch.object(adapter, "execute_git_tool", return_value={"ok": True}) as core,
    ):
        result = adapter.handle({"command": "status", "worktree_path": str(linked)})

    assert result == {"ok": True}
    core.assert_called_once()


def test_adapter_rejects_oversized_metadata_pointer_without_unbounded_read(
    tmp_path: Path,
) -> None:
    """Pointer admission rejects oversized files before reading their contents."""
    pointer = tmp_path / "pointer"
    pointer.write_bytes(b"x" * 4097)
    adapter = _load_adapter()

    with patch.object(adapter.os, "read", wraps=adapter.os.read) as read_call:
        assert adapter._read_pointer(pointer) is None

    read_call.assert_not_called()


def test_adapter_rejects_foreign_or_forged_linked_worktree(tmp_path: Path) -> None:
    """Foreign metadata and mismatched backlinks fail before core dispatch."""
    primary, common, linked = _registered_worktree_topology(tmp_path)
    foreign = tmp_path / "foreign"
    foreign_metadata = tmp_path / "foreign.git" / "worktrees" / "foreign"
    foreign_metadata.mkdir(parents=True)
    foreign.mkdir()
    (foreign_metadata / "HEAD").write_text("ref: refs/heads/foreign\n", encoding="utf-8")
    (foreign_metadata / "commondir").write_text("../..\n", encoding="utf-8")
    (foreign_metadata / "gitdir").write_text(f"{foreign / '.git'}\n", encoding="utf-8")
    (foreign / ".git").write_text(f"gitdir: {foreign_metadata}\n", encoding="utf-8")
    adapter = _load_adapter()

    with (
        patch.object(adapter, "_TRUSTED_WORKTREE", primary),
        patch.object(adapter, "_COMMON_GIT_DIR", common),
        patch.object(adapter, "execute_git_tool") as core,
    ):
        foreign_result = adapter.handle({"command": "status", "worktree_path": str(foreign)})
        (common / "worktrees" / "linked" / "gitdir").write_text(
            f"{foreign / '.git'}\n", encoding="utf-8"
        )
        forged_result = adapter.handle({"command": "status", "worktree_path": str(linked)})

    assert foreign_result["error"]["message"] == "worktree_path is not a registered linked worktree"
    assert forged_result["error"]["message"] == "worktree_path is not a registered linked worktree"
    core.assert_not_called()


def test_adapter_rejects_symlinked_linked_worktree_metadata(tmp_path: Path) -> None:
    """Symlinked registration pointers never establish repository authority."""
    primary, common, linked = _registered_worktree_topology(tmp_path)
    git_entry = linked / ".git"
    pointer = linked / "git-pointer"
    pointer.write_bytes(git_entry.read_bytes())
    git_entry.unlink()
    git_entry.symlink_to(pointer)
    adapter = _load_adapter()

    with (
        patch.object(adapter, "_TRUSTED_WORKTREE", primary),
        patch.object(adapter, "_COMMON_GIT_DIR", common),
        patch.object(adapter, "execute_git_tool") as core,
    ):
        result = adapter.handle({"command": "status", "worktree_path": str(linked)})

    assert result["error"]["message"] == "worktree_path is not a registered linked worktree"
    core.assert_not_called()


def test_adapter_rejects_unapproved_git_worktree_without_core_dispatch(tmp_path: Path) -> None:
    """An arbitrary sibling Git directory never becomes repository authority."""
    (tmp_path / ".git").mkdir()
    adapter = _load_adapter()
    with patch.object(adapter, "execute_git_tool") as core:
        result = adapter.handle({"command": "status", "worktree_path": str(tmp_path)})
    assert result["error"]["message"] == "worktree_path is not a registered linked worktree"
    core.assert_not_called()


def test_adapter_rejects_worktree_admission_failure_without_core_dispatch(tmp_path: Path) -> None:
    """A non-Git worktree is rejected before availability checks or core dispatch."""
    adapter = _load_adapter()

    with (
        patch.object(adapter, "execute_git_tool") as core,
        patch.object(adapter.shutil, "which") as which,
    ):
        result = adapter.handle({"command": "status", "worktree_path": str(tmp_path)})

    assert result == {
        "ok": False,
        "error": {
            "type": "invalid_request",
            "message": "worktree_path is not a registered linked worktree",
        },
    }
    assert "evidence_identity" not in result
    core.assert_not_called()
    which.assert_not_called()


def test_adapter_main_rejects_trailing_json_without_core_dispatch() -> None:
    """The JSON-lines transport accepts exactly one object and emits one response line."""
    adapter = _load_adapter()
    stdout = io.StringIO()

    with (
        patch.object(sys, "stdin", io.TextIOWrapper(io.BytesIO(b"{} {}"), encoding="utf-8")),
        patch.object(sys, "stdout", stdout),
        patch.object(adapter, "execute_git_tool") as core,
    ):
        adapter.main()

    assert stdout.getvalue() == (
        '{"error":{"message":"input must be one bounded UTF-8 JSON object",'
        '"type":"invalid_request"},"ok":false}\n'
    )
    core.assert_not_called()


def test_adapter_rejects_invalid_utf8_without_core_dispatch() -> None:
    """Invalid transport bytes produce one bounded response before dispatch."""
    adapter = _load_adapter()
    stdout = io.StringIO()

    with (
        patch.object(sys, "stdin", io.TextIOWrapper(io.BytesIO(b"\xff"), encoding="utf-8")),
        patch.object(sys, "stdout", stdout),
        patch.object(adapter, "execute_git_tool") as core,
    ):
        adapter.main()

    assert json.loads(stdout.getvalue()) == {
        "error": {
            "message": "input must be one bounded UTF-8 JSON object",
            "type": "invalid_request",
        },
        "ok": False,
    }
    core.assert_not_called()


def test_adapter_transport_rejects_empty_oversized_and_non_object_requests() -> None:
    """Transport boundaries reject invalid bodies before request handling."""
    adapter = _load_adapter()
    cases = (
        (b"", "request body is required"),
        (b'["not-an-object"]', "input must be one bounded UTF-8 JSON object"),
        (b"x" * (adapter._MAX_REQUEST_BYTES + 1), "request body is too large"),
    )

    for payload, message in cases:
        with patch.object(
            sys,
            "stdin",
            io.TextIOWrapper(io.BytesIO(payload), encoding="utf-8"),
        ):
            assert adapter._read_request() == {
                "ok": False,
                "error": {"type": "invalid_request", "message": message},
            }


def test_adapter_rejects_invalid_command_and_unknown_fields_without_dispatch(
    tmp_path: Path,
) -> None:
    """Command and operation field admission happens before core dispatch."""
    (tmp_path / ".git").mkdir()
    adapter = _load_adapter()

    with patch.object(adapter, "execute_git_tool") as core:
        invalid_command = adapter.handle({"command": "commit", "worktree_path": str(tmp_path)})
        unknown_field = adapter.handle(
            {"command": "status", "worktree_path": str(tmp_path), "base": "HEAD"}
        )

    assert (
        invalid_command["error"]["message"] == "command must be one of status, diff, log, or show"
    )
    assert unknown_field == {
        "ok": False,
        "operation": "git_status",
        "error": {"type": "invalid_request", "message": "request contains unsupported fields"},
    }
    assert "evidence_identity" not in invalid_command
    assert "evidence_identity" not in unknown_field
    core.assert_not_called()


def test_adapter_returns_core_error_envelope_unchanged() -> None:
    """An admitted core failure remains the sole diagnostic authority."""
    adapter = _load_adapter()
    expected = {
        "ok": False,
        "operation": "git_diff",
        "error": {"type": "invalid_request", "message": "target requires base"},
        "evidence_identity": {"contract": "e37-m2-validation-git", "version": 1},
    }

    with (
        patch.object(adapter.shutil, "which", return_value="/usr/bin/git"),
        patch.object(adapter, "execute_git_tool", return_value=expected) as core,
    ):
        result = adapter.handle(
            {"command": "diff", "worktree_path": str(adapter._TRUSTED_WORKTREE)}
        )

    assert result == expected
    core.assert_called_once()


def test_adapter_reports_missing_git_after_worktree_admission() -> None:
    """A valid worktree with no Git executable does not reach the core seam."""
    adapter = _load_adapter()

    with (
        patch.object(adapter.shutil, "which", return_value=None),
        patch.object(adapter, "execute_git_tool") as core,
    ):
        result = adapter.handle({"command": "log", "worktree_path": str(adapter._TRUSTED_WORKTREE)})

    assert result == {
        "ok": False,
        "operation": "git_log",
        "error": {"type": "unavailable", "message": "git executable not available"},
    }
    assert "evidence_identity" not in result
    core.assert_not_called()


def test_adapter_context_rejects_non_absolute_or_missing_worktree() -> None:
    """Context admission never selects authority from relative or missing paths."""
    adapter = _load_adapter()

    assert adapter._context_for_worktree(None)["error"]["message"] == (
        "worktree_path must be an admitted absolute directory"
    )
    assert adapter._context_for_worktree("relative")["error"]["message"] == (
        "worktree_path must be an admitted absolute directory"
    )
    assert adapter._context_for_worktree("/definitely/missing")["error"]["message"] == (
        "worktree_path must be an admitted existing directory"
    )


def test_adapter_main_hides_unexpected_exception_details() -> None:
    """Unexpected adapter failures produce a stable response without a traceback."""
    adapter = _load_adapter()
    stdout = io.StringIO()

    with (
        patch.object(adapter, "handle_request", side_effect=RuntimeError("secret detail")),
        patch.object(sys, "stdout", stdout),
    ):
        adapter.main()

    assert json.loads(stdout.getvalue()) == {
        "ok": False,
        "error": {"type": "adapter_failed", "message": "git diagnostics adapter failed"},
    }


def test_adapter_import_does_not_load_cli_roots() -> None:
    """A fresh adapter import stays independent of the ADW and ADforge CLI roots."""
    adapter_path = Path(__file__).parents[1] / "read_only_git_diagnostics.py"
    script = """
import importlib.util
import json
import sys

spec = importlib.util.spec_from_file_location("adapter_probe", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
forbidden = (
    "adw",
    "adforge_core.cli",
)
loaded = sorted(
    name
    for name in sys.modules
    if any(name == root or name.startswith(f"{root}.") for root in forbidden)
)
print(json.dumps(loaded))
"""

    result = subprocess.run(
        [sys.executable, "-c", script, str(adapter_path)],
        capture_output=True,
        check=True,
        text=True,
    )

    assert json.loads(result.stdout) == []
