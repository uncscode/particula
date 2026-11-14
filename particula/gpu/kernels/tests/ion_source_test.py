"""Paired ion injection statistics, charge balance and capacity admission."""

import numpy as np
import pytest

wp = pytest.importorskip("warp")

from particula.gpu.kernels.ion_source import inject_ion_pairs  # noqa: E402


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)]
)
def test_pair_source_poisson_statistics_and_inventory(device):
    """Independent boxes recover a Poisson mean and variance with paired signs."""
    if device != "cpu" and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    boxes, capacity = 8192, 64
    masses = wp.zeros((boxes, capacity, 2), dtype=wp.float64, device=device)
    counts = wp.zeros((boxes, capacity), dtype=wp.float64, device=device)
    charge = wp.zeros_like(counts)
    volume = wp.array(np.full(boxes, 1e-6), dtype=wp.float64, device=device)
    rng = wp.array(
        np.arange(1, boxes + 1, dtype=np.uint32) * np.uint32(2654435761),
        dtype=wp.uint32,
        device=device,
    )
    injected = wp.zeros(boxes, dtype=wp.int64, device=device)
    failure = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(
        inject_ion_pairs,
        dim=boxes,
        inputs=[
            masses,
            counts,
            charge,
            volume,
            wp.float64(1e7),
            wp.float64(0.5),
            0,
            1,
            wp.float64(2e-25),
            wp.float64(1e-25),
            rng,
            injected,
            failure,
        ],
        device=device,
    )
    assert failure.numpy()[0] == 0
    events = injected.numpy()
    assert abs(events.mean() - 5) < 5 * np.sqrt(5 / boxes)
    assert abs(events.var() - 5) < 5 * np.sqrt((5 + 2 * 5**2) / boxes)
    np.testing.assert_array_equal(charge.numpy().sum(axis=1), 0)
    np.testing.assert_array_equal(counts.numpy().sum(axis=1), 2 * events)
    np.testing.assert_allclose(
        masses.numpy().sum(axis=1),
        events[:, None] * np.array([2e-25, 1e-25]),
        rtol=1e-14,
        atol=0,
    )


def test_capacity_failure_preserves_source_rng_and_particles():
    """An impossible pair never partially writes one polarity."""
    device = "cpu"
    masses = wp.zeros((1, 1, 2), dtype=wp.float64, device=device)
    counts = wp.zeros((1, 1), dtype=wp.float64, device=device)
    charge = wp.zeros_like(counts)
    volume = wp.ones(1, dtype=wp.float64, device=device)
    rng = wp.array([12345], dtype=wp.uint32, device=device)
    injected = wp.zeros(1, dtype=wp.int64, device=device)
    failure = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(
        inject_ion_pairs,
        dim=1,
        inputs=[
            masses,
            counts,
            charge,
            volume,
            wp.float64(100),
            wp.float64(1),
            0,
            1,
            wp.float64(1e-25),
            wp.float64(1e-25),
            rng,
            injected,
            failure,
        ],
        device=device,
    )
    assert failure.numpy()[0] == 3
    np.testing.assert_array_equal(rng.numpy(), [12345])
    assert not masses.numpy().any()
    assert not counts.numpy().any()
    assert not charge.numpy().any()
    assert not injected.numpy().any()
