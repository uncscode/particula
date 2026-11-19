"""Full-trajectory sensitivities for the closed, smooth condensation profile."""

import numpy as np
import pytest
import warp as wp

from particula.execution.inverse_condensation import terminal_radius


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_shared_reservoir_loss_and_radius_gradients(device):
    """Each terminal observation senses other components only in its box."""
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("requires actual CUDA")
    initial = np.array([[0.84, 1.07], [0.96, 1.16]])
    numbers = np.array([[3e8, 7e8], [6e8, 4e8]])
    options = dict(
        density=1000.0,
        mean_path=1e-7,
        diffusivity=2e-5,
        duration=1.4,
        steps=24,
        device=device,
    )

    def forward(x, seed=None):
        return terminal_radius(
            x, numbers, [1e-5, 8e-6], adjoint=seed, **options
        )

    radius, _ = forward(initial)
    target = initial + 0.1
    loss_seed = radius - target
    for seed in (np.array([[1.0, 0.0], [0.0, 0.0]]), loss_seed):
        _, gradient = forward(initial, seed)
        segmented, segmented_gradient = terminal_radius(
            initial,
            numbers,
            [1e-5, 8e-6],
            adjoint=seed,
            segment_steps=5,
            **options,
        )
        np.testing.assert_array_equal(segmented, radius)
        np.testing.assert_allclose(
            segmented_gradient, gradient, rtol=1e-12, atol=1e-13
        )
        if seed[0, 1] == 0:
            assert gradient[0, 1] < 0
            np.testing.assert_array_equal(gradient[1], [0, 0])
        for index in np.ndindex(initial.shape):
            for h in (1e-3, 1e-4):
                delta = np.zeros_like(initial)
                delta[index] = h
                plus, _ = forward(initial + delta)
                minus, _ = forward(initial - delta)
                if seed is loss_seed:
                    difference = (
                        np.sum((plus - target) ** 2)
                        - np.sum((minus - target) ** 2)
                    ) / 2
                else:
                    difference = np.sum(seed * (plus - minus))
                np.testing.assert_allclose(
                    gradient[index], difference / (2 * h), rtol=1e-4, atol=1e-9
                )


@pytest.mark.parametrize(
    "override",
    [
        {"steps": 0},
        {"segment_steps": 33},
        {"duration": 1e6},
        {"initial_um": [[0.0]]},
        {"number_density": [[1e8, 2e8]]},
        {"vapor": [np.nan]},
        {"adjoint": [[np.inf]]},
    ],
)
def test_reject_invalid_growth_domain(override):
    """Check dimensions, smooth interior margins and finite reverse seeds."""
    arguments = dict(
        initial_um=[[1.0]],
        number_density=[[1e9]],
        vapor=[1e-5],
        density=1000.0,
        mean_path=1e-7,
        diffusivity=2e-5,
        duration=1.4,
        steps=32,
        device="cpu",
    )
    with pytest.raises(ValueError):
        terminal_radius(**(arguments | override))
