"""Optical time-series inversion across checkpoint segments."""

import numpy as np
import numpy.testing as npt
import pytest
import warp as wp

from particula.execution.inverse_condensation import terminal_radius
from particula.execution.inverse_optics import condensation_optics
from particula.optics import get_mie_efficiencies


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_optical_trajectory(device):
    """Compare scheduled spectra and a complete loss direction on each backend."""
    if device == "cuda:0" and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    x0 = np.array([[0.31, 0.65], [0.28, 0.75]])
    density = np.array([[1.2e9, 5e8], [1.35e9, 7.5e8]])
    vapor = np.array([1e-5, 0.75e-5])
    wave = np.array([450e-9, 700e-9])
    index = 1.55 + 0.013j
    steps = (0, 3, 19, 24)
    options = dict(
        density=1000,
        mean_path=70e-9,
        diffusivity=1e-5,
        duration=0.15,
        device=device,
    )

    def solve(x, **params):
        return condensation_optics(
            x,
            density,
            vapor,
            wavelength=wave,
            refractive_index=index,
            observation_steps=steps,
            steps=steps[-1],
            **options,
            **params,
        )

    trajectory, _ = solve(x0)
    for observed, step in zip(trajectory, steps, strict=True):
        radius = (
            x0
            if step == 0
            else terminal_radius(
                x0,
                density,
                vapor,
                steps=step,
                **(
                    options
                    | {"duration": options["duration"] * step / steps[-1]}
                ),
            )[0]
        )
        r = 1e-6 * radius[:, None, :]
        q = get_mie_efficiencies(r, wave[None, :, None], index)
        reference = np.sum(
            np.stack([q.extinction, q.scattering, q.absorption], axis=-1)
            * (np.pi * r**2 * density[:, None, :])[..., None],
            axis=-2,
        )
        npt.assert_allclose(observed, reference, rtol=1e-10)
    direction = np.array([[0.13, -0.4], [0.12, 0.2]])
    adjoint = np.random.default_rng(1342).normal(size=trajectory.shape)
    for interval in (4, 11, 24):
        values, derivative = solve(x0, adjoint=adjoint, segment_steps=interval)
        npt.assert_allclose(values, trajectory, rtol=2e-14)
        for h in (3e-6, 1e-6):
            high = solve(x0 + h * direction)[0]
            low = solve(x0 - h * direction)[0]
            difference = np.sum((high - low) * adjoint) / (2 * h)
            npt.assert_allclose(
                np.sum(derivative * direction), difference, rtol=8e-6
            )
