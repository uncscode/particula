import importlib.util
import json
from pathlib import Path

TOOL_PATH = Path(__file__).resolve().parents[1] / "run_ctest.py"
SPEC = importlib.util.spec_from_file_location("run_ctest_tool", TOOL_PATH)
assert SPEC is not None
assert SPEC.loader is not None
run_ctest_tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_ctest_tool)


def test_resolve_within_repo_root_rejects_out_of_root_build_dir():
    outside = Path("/tmp/adw-outside-build")

    try:
        run_ctest_tool._resolve_within_repo_root(outside)
    except ValueError as exc:
        assert "outside repository root" in str(exc)
    else:
        raise AssertionError("expected out-of-root build directory to be rejected")


def test_run_ctest_json_reports_out_of_root_build_dir():
    exit_code, output = run_ctest_tool.run_ctest(Path("/tmp/adw-outside-build"), output_mode="json")

    assert exit_code == 1
    payload = json.loads(output)
    assert payload["success"] is False
    assert any("outside repository root" in item for item in payload["validation_errors"])
