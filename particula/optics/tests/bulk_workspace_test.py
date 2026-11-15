"""Streaming population parity, tail chunks, reuse and failure isolation."""

import numpy as np
import pytest
from particula.optics import (
    get_cross_sections,
    get_differential_mueller_matrix,
    get_mie_coefficients,
    get_mie_efficiencies,
    get_mueller_matrix,
    get_population_mueller_matrix,
    get_population_optics,
    get_scattering_amplitudes,
)

wp = pytest.importorskip("warp")
from particula.optics.gpu import prepare_population_mie  # noqa: E402


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("angular", [False, True])
def test_streamed_population(device, angular):
    """Chunked totals agree with particle CPU results for distinct groups."""
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("Native CUDA is unavailable")
    x = np.array([[0.0, 0.1, 1.0, 10.0, 100.0], [2, 3, 4, 5, 6]])
    radius = x * 500e-9 / (2 * np.pi)
    weights = np.array([[0, 1, 2, 3, 4], [5, 4, 3, 2, 1]]) * 1e8
    indices = np.full(x.shape, 1.5 + 0.1j)
    indices[1] = 1.33
    theta = np.array([0, 0.01, 0.3, np.pi / 2, np.pi])
    m = wp.array(
        np.stack((indices.real, indices.imag), axis=-1),
        dtype=wp.vec2d,
        device=device,
    )
    sizes = wp.array(x, dtype=wp.float64, device=device)
    radii = wp.array(radius, dtype=wp.float64, device=device)
    density = wp.array(weights, dtype=wp.float64, device=device)
    angles = (
        wp.array(theta, dtype=wp.float64, device=device) if angular else None
    )
    q = get_mie_efficiencies(radius, 500e-9, indices)
    cross = get_cross_sections(q, radius)
    expected = get_population_optics(cross, q.asymmetry, weights)
    if angular:
        amplitudes = get_scattering_amplitudes(
            get_mie_coefficients(indices, x), theta
        )
        matrix = get_differential_mueller_matrix(
            get_mueller_matrix(*amplitudes), 500e-9
        )
        expected_matrix = get_population_mueller_matrix(matrix, weights)
    for tile in (2, 5):
        work = prepare_population_mie(
            m,
            sizes,
            radii,
            density,
            angles=angles,
            particle_tile=tile,
            angle_tile=2,
            coefficient_capacity=130,
        )
        pointer = work.optics.ptr
        for _ in range(2):
            work.enqueue()
            work.check()
            assert work.optics.ptr == pointer
            for j, value in enumerate(expected):
                np.testing.assert_allclose(
                    work.optics.numpy()[:, j], value, rtol=1e-8, atol=1e-12
                )
            if angular:
                np.testing.assert_allclose(
                    work.mueller.numpy(),
                    expected_matrix,
                    rtol=1e-8,
                    atol=1e-12,
                )
            else:
                assert work.mueller.shape == (2, 0, 4, 4)
    bad = weights.copy()
    bad[0, -1] = -1
    density.assign(bad)
    work.enqueue()
    with pytest.raises(RuntimeError, match="Population optics"):
        work.check()
    np.testing.assert_array_equal(work.status.numpy(), [2, 0])
    density.assign(np.zeros_like(weights))
    work.enqueue()
    work.check()
    np.testing.assert_array_equal(work.optics.numpy(), 0)
    np.testing.assert_array_equal(work.mueller.numpy(), 0)


def test_population_metadata_rejection():
    """Input metadata must agree before execution is prepared."""
    m = wp.zeros((2, 3), dtype=wp.vec2d, device="cpu")
    x = wp.zeros((2, 3), dtype=wp.float64, device="cpu")
    with pytest.raises(ValueError):
        prepare_population_mie(m, x, x, x, particle_tile=0)
    with pytest.raises(ValueError):
        prepare_population_mie(m, x, x, wp.zeros(6, device="cpu"))
