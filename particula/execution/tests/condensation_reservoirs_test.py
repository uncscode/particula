"""Independent high-accuracy sums for large shared gas reservoirs."""

import math

import numpy as np
import pytest

wp = pytest.importorskip("warp")
pytestmark = pytest.mark.warp


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("slots", [7, 65, 10_000])
@pytest.mark.parametrize("resolved", [0, 1])
def test_reservoir_sum_matches_compensated_scalar_oracle(
    device, slots, resolved
):
    """Mix signed flux, holes, weights, unequal volumes and an unmapped gas."""
    if device != "cpu" and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    from particula.execution import _condensation_reservoirs as kernels

    rng = np.random.default_rng(19)
    rate = rng.normal(size=(2, slots, 3)) * 1e-20
    count = rng.uniform(0.1, 3.0, (2, slots))
    count[:, ::7] = 0
    volume = np.array([1e-8, 3e-6])
    mapping = np.array([2, -1, 0, 1], dtype=np.int32)
    mass_size = rate.size
    scale = np.concatenate(
        (np.full(mass_size, 1e-15), np.geomspace(1e-8, 1e-3, 8))
    )

    def array(value):
        return wp.array(value, dtype=wp.float64, device=device)

    parts = wp.zeros((2, 4, min(slots, 64)), dtype=wp.float64, device=device)
    derivative = wp.zeros(scale.size, dtype=wp.float64, device=device)
    invalid = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(
        kernels.gas_parts,
        dim=parts.shape,
        inputs=[
            array(rate),
            array(count),
            array(volume),
            wp.array(mapping, dtype=wp.int32, device=device),
            resolved,
            parts,
        ],
        device=device,
    )
    wp.launch(
        kernels.finish_gas,
        dim=(2, 4),
        inputs=[parts, mass_size, array(scale), derivative, invalid],
        device=device,
    )
    actual = derivative.numpy()[mass_size:].reshape(2, 4)
    expected = np.zeros_like(actual)
    for box in range(2):
        for gas, material in enumerate(mapping):
            if material < 0:
                continue
            terms = [
                -float(count[box, slot] / (volume[box] if resolved else 1))
                * float(rate[box, slot, material])
                for slot in range(slots)
            ]
            expected[box, gas] = (
                math.fsum(terms) / scale[mass_size + box * 4 + gas]
            )
            # Absolute summation error is scaled by total unsigned flux,
            # avoiding an ill-conditioned relative comparison near cancellation.
            bound = (
                5e-15
                * math.fsum(abs(value) for value in terms)
                / scale[mass_size + box * 4 + gas]
            )
            assert abs(actual[box, gas] - expected[box, gas]) <= bound
    np.testing.assert_array_equal(actual[:, 1], 0)
    np.testing.assert_array_equal(invalid.numpy(), 0)
