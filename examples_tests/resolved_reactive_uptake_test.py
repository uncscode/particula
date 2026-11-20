"""Check the worked chamber's extensive histories on each available backend."""

import numpy as np
import pytest
from docs.examples.processes.customization.resolved_reactive_uptake import (
    radial_reference,
    run_backend,
)


@pytest.mark.parametrize("device", ["numpy", "cuda:0"])
def test_aging_and_uptake(device):
    """Verify every population and its chemical state against radial growth."""
    if device == "cuda:0":
        import warp as wp

        if not wp.is_cuda_available():
            pytest.skip("CUDA device unavailable")
    history, _ = run_backend(device)
    np.testing.assert_allclose(
        history, radial_reference(), rtol=2e-7, atol=2e-10
    )
    np.testing.assert_allclose(history.sum(axis=1), 1.0, rtol=1e-12, atol=0.0)
    assert history[-1, 0] < 0.003
    assert history[-1, 4] > history[-1, 3]
