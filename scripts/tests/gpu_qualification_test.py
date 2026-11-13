"""Strict qualification must distinguish passing CUDA work from skipped work."""

import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "qualify_gpu", Path(__file__).resolve().parents[1] / "qualify_gpu.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize(
    ("xml", "qualified", "passed", "problems"),
    [
        ("<testsuites/>", False, 0, 0),
        ("<testsuite><testcase name='cuda'/></testsuite>", True, 1, 0),
        (
            "<testsuite><testcase><skipped message='CUDA unavailable'/>"
            "</testcase></testsuite>",
            False,
            0,
            1,
        ),
        (
            "<testsuite><testcase/><testcase><failure message='mass drift'/>"
            "</testcase></testsuite>",
            False,
            1,
            1,
        ),
        (
            "<testsuites><testsuite><testcase><error message='collection'/>"
            "</testcase></testsuite></testsuites>",
            False,
            0,
            1,
        ),
    ],
)
def test_qualification_requires_nonempty_unskipped_success(
    xml, qualified, passed, problems
):
    """Neither empty collection, partial passes nor skips establish readiness."""
    result = MODULE.summarize_results(ET.fromstring(xml))  # noqa: S314 -- literal test fixtures
    assert result["qualified"] is qualified
    assert result["passed"] == passed
    assert len(result["problems"]) == problems
