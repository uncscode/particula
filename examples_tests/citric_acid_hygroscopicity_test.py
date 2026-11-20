"""Scientific regression for the 12-point CK-EDB citric acid comparison."""

import numpy as np
from scripts.docs_examples import import_study


def test_hygroscopicity_inference_and_kinetic_endpoints():
    """Preserve signed bias and mass exchange direction at every activity."""
    result = import_study("citric-acid-hygroscopicity").simulate()
    np.testing.assert_allclose(result["rms"], 0.06160186603, rtol=1e-6)
    assert result["covered"] == 0
    assert result["max_mfs_root_error"] < 1e-7
    exchange = np.asarray(result["reservoir_exchange_kg"])
    assert exchange.shape == (2, 12)
    assert np.all(exchange[0] < 0) and np.all(exchange[1] > 0)
    native = result["native_restart"]
    np.testing.assert_allclose(
        native["kinetic_mfs"], native["root_mfs"], atol=1e-6
    )
    np.testing.assert_equal(
        native["observation_times_s"], [0.0, 1.0, 10.0, 2000.0]
    )
