"""Exercise both published readers on the original instrument excerpts."""

import numpy as np
import pytest


def test_measurement_import_example():
    """Check the repair and scan integral against the physical source lines."""
    pytest.importorskip("polars")
    from docs.studies.examples.measurement_import import run_example

    result = run_example()
    counter = result["counter"]
    np.testing.assert_array_equal(counter["elapsed_s"], np.arange(0, 60, 3))
    assert counter["rate_counts_s"][0] == 32660.0
    assert np.isclose(counter["temperature_K"][0], 290.25)
    assert counter["repair_lines"] == [[1, 2]]
    assert counter["repaired"] == [True] + [False] * 19

    spectrum = result["spectrum"]
    np.testing.assert_allclose(spectrum["vendor_total_cm3"], 2169.0, rtol=1e-12)
    np.testing.assert_allclose(spectrum["per_decade_cm3"][0], 6103.186)
    assert (
        len(spectrum["diameter_nm"]) == len(spectrum["per_decade_cm3"]) == 203
    )
    np.testing.assert_allclose(
        [spectrum["diameter_nm"][0], spectrum["diameter_nm"][-1]],
        [20.72, 784.39],
    )
    np.testing.assert_allclose(
        spectrum["integral_cm3"], 2168.9981640625, rtol=1e-12
    )
    assert abs(spectrum["integral_cm3"] - spectrum["vendor_total_cm3"]) < 0.002
    assert spectrum["flags"] == [
        "Detector aerosol flow rate error",
        "Incomplete Scan",
    ]
