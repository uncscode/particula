"""Numerical controls for the published lab-extension and replay example."""

import numpy as np
import pytest


def test_dated_calibration_and_replayed_concentration():
    """Two effective certificates change flow, while zero and missing stay distinct."""
    pytest.importorskip("polars")
    from docs.studies.examples.calibrated_particle_count import run_example

    result = run_example()
    assert result["relative_s"] == [0, 1, 2, 3, 4]
    np.testing.assert_allclose(result["sample_flow"], [0.6, 0.6, 1.2, 0])
    np.testing.assert_allclose(result["concentration"][:3], [10.0, 0.0, 5.0])
    assert result["concentration"][3:] == [None, None]
    assert "nonpositive:sample_flow" in result["flags"][3]
    assert "unmatched:sample_flow" in result["flags"][4]
    assert "bench-A" in result["calibrations"][0]
    assert "bench-B" in result["calibrations"][2]
