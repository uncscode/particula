"""Automatic differentiation of paired perturbations to the simulator. Includes finite-difference verification."""

import numpy as np
import warp as wp


@wp.kernel
def importance(
    theta: wp.array(dtype=wp.float64),
    samples: wp.array2d(dtype=wp.float64),
    origin: wp.array(dtype=wp.float64),
    losses: wp.array(dtype=wp.float64),
    sigma: wp.float64,
    normalizer: wp.float64,
    objective: wp.array(dtype=wp.float64),
):
    i = wp.tid()
    exponent = wp.float64(0.0)
    for j in range(9):
        a = samples[i, j] - origin[j]
        b = samples[i, j] - theta[j]
        exponent += (a * a - b * b) / (wp.float64(2.0) * sigma * sigma)
    wp.atomic_add(objective, 0, wp.exp(exponent) * losses[i] / normalizer)


def gradient(center, sample, loss, sigma):
    """Differentiate the sampling density of the real simulator evaluations."""

    def array(a, grad=False):
        return wp.array(
            np.array(a, dtype=np.float64),
            dtype=wp.float64,
            device="cpu",
            requires_grad=grad,
        )

    # Antithetic pairs cancel any common baseline exactly at the sampling center.
    val = loss - loss.mean()
    theta = array(center, True)
    out = wp.zeros(1, dtype=wp.float64, device="cpu", requires_grad=True)
    args = [
        theta,
        array(sample),
        array(center),
        array(val),
        sigma,
        float(len(val)),
        out,
    ]
    with wp.Tape() as tape:
        wp.launch(importance, len(val), args, device="cpu")
    tape.backward(loss=out)
    g = theta.grad.numpy().copy()
    check = val @ (sample - center) / (len(val) * sigma * sigma)
    np.testing.assert_allclose(g, check, rtol=1e-10, atol=1e-10)
    # Finite difference the same importance-weighted empirical objective at center.
    direction = np.cos(np.arange(9))
    direction /= np.linalg.norm(direction)

    def reweight(step):
        exponent = (
            (sample - center) @ (step * direction) - 0.5 * step**2
        ) / sigma**2
        return np.mean(val * np.exp(exponent))

    fd = (reweight(1.0e-5) - reweight(-1.0e-5)) / 2.0e-5
    np.testing.assert_allclose(fd, g @ direction, atol=1e-8)
    return g
