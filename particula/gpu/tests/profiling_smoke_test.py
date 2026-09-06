"""Opt-in native-CUDA Nsight profiling smoke evidence."""

from __future__ import annotations

from pathlib import Path

import pytest

from particula.gpu.tests.profiling_support import (
    NsightEvidence,
    NsightUnavailable,
    collect_nsight_evidence,
)

pytestmark = [pytest.mark.benchmark, pytest.mark.warp, pytest.mark.cuda]


@pytest.mark.parametrize(
    "tool, filename", (("nsys", "systems.csv"), ("ncu", "compute.csv"))
)
def test_native_cuda_nsight_small_worker_smoke(
    tool: str, filename: str
) -> None:
    """Collect exactly one closed small-worker report when Nsight is available."""
    root = Path(".artifacts")
    if not root.is_dir():
        pytest.skip("profiling artifact root is unavailable")
    result = collect_nsight_evidence(
        tool=tool,
        artifact_root=root,
        report_filename=filename,
        process_ids={},
    )
    if isinstance(result, NsightUnavailable):
        pytest.skip(result.reason)
    if not isinstance(result, NsightEvidence):
        pytest.fail(f"Nsight {tool} collection failed: {result}")
    assert any(row.attribution == "attributed" for row in result.rows)
