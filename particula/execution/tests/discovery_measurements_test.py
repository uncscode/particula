"""Conservation and analytical gradients at a size instrument's bin edge."""

import numpy as np
import pytest
from scipy.special import ndtr

wp = pytest.importorskip("warp")
from particula.execution._discovery_measurements import (  # noqa: E402
    observe_number_spectrum,
)


@pytest.mark.warp
@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)]
)
def test_instrument_edge(device):
    """Two seed classes contribute to all bins; edge slopes are finite."""
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    sizes = np.array([[80e-9, 120e-9]])
    weights = np.array([[2e8, 1e8]])
    edges = np.array([80e-9, 140e-9])
    width = 0.12
    opts = dict(dtype=wp.float64, device=device)
    diam = wp.array(sizes, requires_grad=True, **opts)
    num = wp.array(weights, requires_grad=True, **opts)
    bins = wp.empty((1, 3), requires_grad=True, **opts)
    cutoffs = wp.array(edges, **opts)
    with wp.Tape() as tape:
        wp.launch(
            observe_number_spectrum,
            (1, 3),
            [diam, num, cutoffs, width, bins],
            device=device,
        )
    z = np.log(edges[:, None] / sizes) / width
    fractions = np.diff(np.vstack((np.zeros(2), ndtr(z), np.ones(2))), axis=0)
    np.testing.assert_allclose(
        bins.numpy()[0], fractions @ weights[0], rtol=1e-12
    )
    np.testing.assert_allclose(bins.numpy().sum(), weights.sum(), rtol=1e-14)
    tape.backward(grads={bins: wp.array([[0.0, 1.0, 0.0]], **opts)})
    np.testing.assert_allclose(num.grad.numpy()[0], fractions[1], rtol=1e-12)
    density = np.exp(-z * z / 2) / np.sqrt(2 * np.pi)
    expected = (density[0] - density[1]) * weights[0] / (width * sizes[0])
    np.testing.assert_allclose(diam.grad.numpy()[0], expected, rtol=1e-12)
