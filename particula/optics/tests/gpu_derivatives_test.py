"""The local Mie adjoint agrees with differences of the forward CPU solve."""

import numpy as np
import numpy.testing as npt
import pytest
import warp as wp
from particula.optics import get_mie_efficiencies
from particula.optics.gpu.derivatives import mie_cross_section_sums_jacobian
from particula.optics.gpu.inverse import BulkOpticsAdjoint
from particula.optics.gpu.mie import mie_coefficients_kernel


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_mie_series_derivatives(device):
    """Check both sides of the small-x branch and absorbing resonances."""
    if device == "cuda:0" and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    cases = np.array(
        [
            [1.33, 0.004, 0.03],
            [1.7, 0.08, 0.4],
            [1.5, 0.03, 1.8],
            [1.8, 0.001, 7.5],
            [1.6, 0.2, 22.0],
        ],
    )
    count, capacity = len(cases), 64

    def array(values, dtype):
        return wp.array(values, dtype=dtype, device=device)

    def scratch(shape, dtype):
        return wp.empty(shape, dtype=dtype, device=device)

    index = array(cases[:, :2].copy(), wp.vec2d)
    size = array(cases[:, 2].copy(), wp.float64)
    work = (
        index,
        size,
        scratch((count, capacity + 1), wp.vec2d),
        scratch((count, capacity), wp.vec2d),
        scratch((count, capacity), wp.vec2d),
        scratch(count, wp.int32),
        scratch(count, wp.int32),
    )
    ext = scratch(count, wp.vec3d)
    sca = scratch(count, wp.vec3d)
    wp.launch(mie_coefficients_kernel, count, inputs=work, device=device)
    wp.launch(
        mie_cross_section_sums_jacobian,
        count,
        inputs=[*work, ext, sca],
        device=device,
    )
    npt.assert_array_equal(work[-1].numpy(), 0)

    def reference(params):
        n, k, x = params.T
        q = get_mie_efficiencies(x / (2 * np.pi), 1.0, n + 1j * k)
        return (
            np.stack([q.extinction, q.scattering], axis=-1)
            * x[:, None] ** 2
            / 2
        )

    obtained = np.stack([ext.numpy(), sca.numpy()], axis=1)
    for h in (2e-6, 1e-6):
        for axis, basis in enumerate(np.eye(3)):
            slope = (
                reference(cases + h * basis) - reference(cases - h * basis)
            ) / (2 * h)
            npt.assert_allclose(
                obtained[..., axis], slope, rtol=4e-5, atol=1e-9
            )


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_bulk_optics_tape(device):
    """Seed the actual bulk loss and check its full directional derivative."""
    if device == "cuda:0" and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    wave = np.array([450e-9, 550e-9])
    radii = np.array([[50e-9, 152e-9, 245e-9], [33e-9, 57e-9, 112e-9]])
    optical_index = np.array(
        [
            [1.41 + 0.03j, 1.52 + 0.018j, 1.62 + 0.09j],
            [1.33 + 0.02j, 1.6 + 0.025j, 1.65 + 0.07j],
        ]
    )
    weights = np.array([[1e8, 4e8, 2e9], [7e7, 1e9, 3e9]])
    seed = np.array([[1.0, 2.0, 3.0], [1.0, -0.5, 0.5]])
    medium = 1.0003
    workspace = BulkOpticsAdjoint(
        radii.shape, wave, device=device, particle_tile=4, medium_index=medium
    )
    options = {"dtype": wp.float64, "device": device}
    radius_gpu = wp.array(radii, **options, requires_grad=True)
    weight_gpu = wp.array(weights, **options, requires_grad=True)
    index_gpu = wp.array(
        np.stack([optical_index.real, optical_index.imag], axis=-1),
        dtype=wp.vec2d,
        device=device,
        requires_grad=True,
    )
    with wp.Tape() as tape:
        bulk = workspace.evaluate(radius_gpu, index_gpu, weight_gpu, tape=tape)
    tape.backward(grads={bulk: wp.array(seed, **options)})

    def reference(radius, index, weight):
        q = get_mie_efficiencies(radius, wave[:, None], index, medium)
        q = np.stack([q.extinction, q.scattering, q.absorption], axis=-1)
        area_density = np.pi * radius**2 * weight
        return (q * area_density[..., None]).sum(axis=1)

    npt.assert_allclose(
        bulk.numpy(), reference(radii, optical_index, weights), rtol=4e-11
    )
    perturb_index = np.array([[0.4 + 0.1j, 0.2j, -0.2 - 0.25j]] * 2)
    perturb_radius = radii * np.array([[0.14, -0.3, 0.2]])
    perturb_weight = weights * np.array([[-0.4, 0.5, 0.6]])
    directions = [
        (
            perturb_radius,
            0,
            0,
            np.sum(radius_gpu.grad.numpy() * perturb_radius),
        ),
        (
            0,
            perturb_index,
            0,
            np.sum(
                index_gpu.grad.numpy()
                * np.stack([perturb_index.real, perturb_index.imag], axis=-1)
            ),
        ),
        (
            0,
            0,
            perturb_weight,
            np.sum(weight_gpu.grad.numpy() * perturb_weight),
        ),
    ]
    for dr, dm, dw, slope in directions:
        for h in (1e-5, 5e-6):
            high = reference(
                radii + h * dr, optical_index + h * dm, weights + h * dw
            )
            low = reference(
                radii - h * dr, optical_index - h * dm, weights - h * dw
            )
            difference = np.sum(seed * (high - low)) / (2 * h)
            npt.assert_allclose(slope, difference, rtol=2e-6, atol=1e-12)
