"""Real CUDA replay of a condensation loss with per-box observation noise."""

import numpy as np
import pytest
import warp as wp

from particula.execution._condensation_ode_kernels import (
    FloatMatrix,
    FloatVector,
    IntVector,
)
from particula.execution.inverse_condensation import (
    _advance,
    _mass_from_radius,
    _observe,
    _reservoir,
)


@wp.kernel
def _draw_noise(
    counter: IntVector,
    draws: FloatVector,
):
    b = wp.tid()
    state = wp.rand_init(wp.int32(279) + wp.int32(b), wp.int32(counter[b]))  # type: ignore[arg-type]
    noise = wp.float64(wp.randn(state)) / wp.float64(100.0)
    draws[b] = noise
    counter[b] += 1


@wp.kernel
def _noisy_loss(
    radius: FloatMatrix,
    draws: FloatVector,
    losses: FloatVector,
):
    b = wp.tid()
    residual = radius[b, 0] - wp.float64(1.25) - draws[b]
    losses[b] = residual * residual / wp.float64(2.0)


def _experiment(x, captured, compute_gradient):
    device = "cuda:0"

    def array(x, grad=False):
        return wp.array(x, dtype=wp.float64, device=device, requires_grad=grad)

    radii = array(x, compute_gradient)
    m = [array(np.zeros_like(x), compute_gradient) for _ in range(9)]
    gases = [array([0.0, 0.0], compute_gradient) for _ in range(8)]
    terminal = array(np.zeros_like(x), compute_gradient)
    losses = array([0.0, 0.0], compute_gradient)
    counter = wp.zeros(2, dtype=int, device=device)
    draws = array([0.0, 0.0])
    weights = array([[1e9], [1e9]])
    g0 = array([1e-5, 8e-6])
    seeds = array([1.0, 1.0])
    factor = 4 * np.pi * 1000 / 3

    def record():
        wp.launch(_draw_noise, 2, [counter, draws], device=device)
        tape = wp.Tape()
        with tape:
            wp.launch(
                _mass_from_radius,
                x.shape,
                [radii, factor, m[0]],
                device=device,
            )
            for j in range(8):
                wp.launch(
                    _reservoir,
                    2,
                    [m[0], m[j], weights, g0, gases[j]],
                    device=device,
                )
                wp.launch(
                    _advance,
                    x.shape,
                    [m[j], gases[j], factor, 1e-7, 2e-5, 1.4 / 8, m[j + 1]],
                    device=device,
                )
            wp.launch(
                _observe, x.shape, [m[-1], factor, terminal], device=device
            )
            wp.launch(
                _noisy_loss,
                2,
                [terminal, draws, losses],
                device=device,
            )
        if compute_gradient:
            tape.backward(grads={losses: seeds})

    if captured:
        with wp.ScopedCapture(device=device) as scope:
            record()
    outputs = []
    for _ in range(3):
        for a in [radii, *m, *gases, terminal, losses]:
            if a.grad is not None:
                a.grad.zero_()
        if captured:
            wp.capture_launch(scope.graph)
        else:
            record()
        outputs.append(
            [
                draws.numpy().copy(),
                losses.numpy().copy(),
                radii.grad.numpy().ravel().copy()
                if compute_gradient
                else np.zeros(2),
            ]
        )
    return np.array(outputs)


def test_captured_condensation_gradient_advances_per_box_rng():
    """Match replay with ordinary launches, then check loss derivatives."""
    if not wp.is_cuda_available():
        pytest.skip("native graph capture requires CUDA")
    device = "cuda:0"
    wp.load_module(module=_advance.module, device=device)
    wp.load_module(module=_noisy_loss.module, device=device)
    initial = np.array([[0.86], [1.13]])

    actual = _experiment(initial, True, True)
    expected = _experiment(initial, False, True)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)
    assert len(np.unique(actual[:, 0, :])) == 6
    h = 1e-5
    fd = (
        _experiment(initial + h, False, False)[:, 1]
        - _experiment(initial - h, False, False)[:, 1]
    ) / (2 * h)
    np.testing.assert_allclose(actual[:, 2], fd, rtol=1e-6, atol=1e-9)
