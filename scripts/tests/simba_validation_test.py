"""Extended qualification for Particula's coupled reaction discovery study."""

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

STUDY = Path(__file__).resolve().parents[2] / "studies" / "simba"


@pytest.mark.slow
def test_simba_reference_cases(monkeypatch):
    """Regenerate the eleven paper candidates and native process comparison."""
    monkeypatch.syspath_prepend(str(STUDY))
    enumeration = importlib.import_module("enumeration")
    for expected in json.loads((STUDY / "enumeration.json").read_text()):
        row = enumeration.enumerate_case(
            expected["case"], expected["iteration"]
        )
        assert row["table_finalist_found"]
        assert row["examined"] == expected["examined"]
        assert row["stopped_at"] is None
    native = importlib.import_module("native_parity")
    row = native.compare(0.025)
    np.testing.assert_array_less(row["channel_peak_normalized_errors"], 0.002)


@pytest.mark.slow
@pytest.mark.warp
@pytest.mark.cuda
def test_simba_coupled_cuda_qualification(monkeypatch):
    """Require the published trajectory and held-out recovery on a real GPU."""
    wp = pytest.importorskip("warp")
    if not wp.is_cuda_available():
        pytest.skip("Actual CUDA required for coupled inverse qualification")
    monkeypatch.syspath_prepend(str(STUDY))
    trajectory = importlib.import_module("qualify_trajectory")
    for vapor in (0.0, 1e-7):
        trajectory.qualify("cuda:0", vapor)
    fits = importlib.import_module("heldout_recovery").recover("cuda:0")
    assert all(fits["AD"]["converged"])
