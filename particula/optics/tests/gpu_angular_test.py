"""Native CPU/CUDA angular and Stokes parity with coefficient reuse."""

import numpy as np
import pytest
from particula.optics import (
    apply_mueller_matrix,
    get_mie_coefficients,
    get_mueller_matrix,
    get_scattering_amplitudes,
)

wp = pytest.importorskip("warp")
from particula.optics.gpu.angular import (  # noqa: E402
    apply_mueller_kernel,
    mueller_kernel,
    scattering_amplitudes_kernel,
)
from particula.optics.gpu.mie import mie_coefficients_kernel  # noqa: E402


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_device_angular_stokes(device):
    """Compare full pipeline at endpoints and fine forward angles to x=10000."""
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("Native CUDA is unavailable")
    x = np.tile([0.0, 1e-6, 0.05, 0.5, 1.0, 10.0, 100.0, 1000.0, 10000.0], 3)
    m = np.repeat([1.5 + 0.1j, 1.33 + 1e-8j, 0.2 + 3j], 9)
    angles = np.array(
        [0.0, 1e-6, 1e-4, 0.01, 0.5, np.pi / 2, np.pi - 1e-4, np.pi]
    )
    coefficients = get_mie_coefficients(m, x)
    s1, s2 = get_scattering_amplitudes(coefficients, angles)
    scale = np.linspace(0.1, 2.0, len(x))
    expected_matrix = get_mueller_matrix(s1, s2) * scale[:, None, None, None]
    shape = (len(x), len(angles))
    width = coefficients.electric.shape[-1]
    index = wp.array(
        np.stack([m.real, m.imag], axis=-1), dtype=wp.vec2d, device=device
    )
    sizes = wp.array(x, dtype=wp.float64, device=device)
    d = wp.empty((len(x), width + 1), dtype=wp.vec2d, device=device)
    a = wp.empty((len(x), width), dtype=wp.vec2d, device=device)
    b = wp.empty_like(a)
    count = wp.empty(len(x), dtype=wp.int32, device=device)
    status = wp.empty_like(count)
    theta = wp.array(angles, dtype=wp.float64, device=device)
    s1_device = wp.empty(shape, dtype=wp.vec2d, device=device)
    s2_device = wp.empty_like(s1_device)
    angular_status = wp.empty(shape, dtype=wp.int32, device=device)
    matrix = wp.empty(shape + (4, 4), dtype=wp.float64, device=device)
    matrix_status = wp.empty_like(angular_status)
    scaling = wp.array(scale, dtype=wp.float64, device=device)
    incident = np.broadcast_to([1.0, 0.3, -0.4, 0.2], shape + (4,)).copy()
    angles_in = np.broadcast_to(angles[None, :], shape).copy()
    angles_out = -angles_in * 0.3
    incoming = wp.array(incident, dtype=wp.vec4d, device=device)
    rotation_in = wp.array(angles_in, dtype=wp.float64, device=device)
    rotation_out = wp.array(angles_out, dtype=wp.float64, device=device)
    outgoing = wp.empty(shape, dtype=wp.vec4d, device=device)
    stokes_status = wp.empty_like(angular_status)
    wp.launch(
        mie_coefficients_kernel,
        dim=len(x),
        inputs=[index, sizes, d, a, b, count, status],
        device=device,
    )
    for _ in range(2):
        wp.launch(
            scattering_amplitudes_kernel,
            dim=shape,
            inputs=[
                a,
                b,
                count,
                status,
                theta,
                s1_device,
                s2_device,
                angular_status,
            ],
            device=device,
        )
        wp.launch(
            mueller_kernel,
            dim=shape,
            inputs=[
                s1_device,
                s2_device,
                angular_status,
                scaling,
                matrix,
                matrix_status,
            ],
            device=device,
        )
        wp.launch(
            apply_mueller_kernel,
            dim=shape,
            inputs=[
                matrix,
                matrix_status,
                incoming,
                rotation_in,
                rotation_out,
                outgoing,
                stokes_status,
            ],
            device=device,
        )
        for codes in (status, angular_status, matrix_status, stokes_status):
            np.testing.assert_array_equal(codes.numpy(), 0)
        for actual, expected in [(s1_device, s1), (s2_device, s2)]:
            parts = actual.numpy()
            complex_values = parts[..., 0] + 1j * parts[..., 1]
            norm = np.maximum(
                np.max(abs(expected), axis=-1, keepdims=True), 1e-20
            )
            np.testing.assert_allclose(
                complex_values / norm, expected / norm, rtol=0, atol=1e-8
            )
        norm = np.maximum(np.max(abs(expected_matrix), axis=(1, 2, 3)), 1e-40)
        np.testing.assert_allclose(
            matrix.numpy() / norm[:, None, None, None],
            expected_matrix / norm[:, None, None, None],
            rtol=0,
            atol=1e-8,
        )
        expected = apply_mueller_matrix(
            expected_matrix, incident, angles_in, angles_out
        )
        np.testing.assert_allclose(
            outgoing.numpy() / norm[:, None, None],
            expected / norm[:, None, None],
            rtol=0,
            atol=1e-8,
        )


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_angular_failure_propagation(device):
    """Invalid angles and failed coefficients suppress downstream matrices."""
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("Native CUDA is unavailable")
    a = wp.zeros((2, 1), dtype=wp.vec2d, device=device)
    b = wp.zeros_like(a)
    counts = wp.array(np.array([1, 1]), dtype=wp.int32, device=device)
    code = wp.array(np.array([0, 2]), dtype=wp.int32, device=device)
    angles = wp.array(
        np.array([0.0, -1.0, np.nan, np.pi]), dtype=wp.float64, device=device
    )
    s1 = wp.empty((2, 4), dtype=wp.vec2d, device=device)
    s2 = wp.empty_like(s1)
    status = wp.empty((2, 4), dtype=wp.int32, device=device)
    matrix = wp.ones((2, 4, 4, 4), dtype=wp.float64, device=device)
    matrix_status = wp.empty_like(status)
    scale = wp.ones(2, dtype=wp.float64, device=device)
    wp.launch(
        scattering_amplitudes_kernel,
        dim=(2, 4),
        inputs=[a, b, counts, code, angles, s1, s2, status],
        device=device,
    )
    wp.launch(
        mueller_kernel,
        dim=(2, 4),
        inputs=[s1, s2, status, scale, matrix, matrix_status],
        device=device,
    )
    np.testing.assert_array_equal(status.numpy(), [[0, 1, 1, 0], [2, 2, 2, 2]])
    np.testing.assert_array_equal(
        matrix_status.numpy(), [[0, 2, 2, 0], [2, 2, 2, 2]]
    )
    np.testing.assert_array_equal(matrix.numpy(), 0.0)
