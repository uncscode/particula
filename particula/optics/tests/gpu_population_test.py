"""CPU/CUDA population measures, optical reductions, and column parity."""

import numpy as np
import pytest
from particula.optics import (
    OpticalCrossSections,
    get_column_optics,
    get_mass_specific_opacity,
    get_population_mueller_matrix,
    get_population_optics,
    get_population_weights,
)

wp = pytest.importorskip("warp")
from particula.optics.gpu.column import (  # noqa: E402
    column_optics_kernel,
    mass_opacity_kernel,
)
from particula.optics.gpu.population import (  # noqa: E402
    population_mueller_kernel,
    population_optics_kernel,
    population_weights_kernel,
)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("measure", [0, 1, 2])
def test_device_population_pipeline(device, measure):
    """Compare all three measures, signed Mueller sums, and column outputs."""
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("Native CUDA is unavailable")
    concentration = np.array([[2.0, 4.0, 6.0], [3.0, 8.0, 1.0]])
    volume = np.array([2.0, 4.0])
    grid = np.array([1.0, 2.0, 5.0]) * 1e-6
    names = ["discrete", "particle_resolved", "continuous_pdf"]
    kwargs = [{}, {"volume": volume}, {"radius_grid": grid}][measure]
    weights_cpu = get_population_weights(
        concentration, names[measure], **kwargs
    )

    def array(values, dtype=wp.float64):
        return wp.array(np.asarray(values), dtype=dtype, device=device)

    def empty(shape, dtype=wp.float64):
        return wp.empty(shape, dtype=dtype, device=device)

    weights = empty((2, 3))
    weight_status = empty((2, 3), wp.int32)
    wp.launch(
        population_weights_kernel,
        dim=(2, 3),
        inputs=[
            array(concentration),
            array(volume),
            array(grid),
            measure,
            weights,
            weight_status,
        ],
        device=device,
    )
    np.testing.assert_array_equal(weight_status.numpy(), 0)
    np.testing.assert_allclose(weights.numpy(), weights_cpu, rtol=1e-14)
    cross = np.broadcast_to(
        np.array(
            [
                [2.0, 1.0, 1.0, 0.2, 1.5],
                [4.0, 3.0, 1.0, 0.5, 2.5],
                [8.0, 6.0, 2.0, 0.8, 5.0],
            ]
        ),
        (2, 3, 5),
    ).copy()
    g = np.array([[0.5, 0.5, 0.5], [-0.2, 0.1, 0.7]])
    # Radiation pressure is an independently supplied cross section here.
    expected = get_population_optics(
        OpticalCrossSections(*(cross[..., i] for i in range(5))), g, weights_cpu
    )
    output = empty((2, 9))
    status = empty(2, wp.int32)
    wp.launch(
        population_optics_kernel,
        dim=2,
        inputs=[array(cross), array(g), weights, weight_status, output, status],
        device=device,
    )
    np.testing.assert_array_equal(status.numpy(), 0)
    for j, value in enumerate(expected):
        np.testing.assert_allclose(output.numpy()[:, j], value, rtol=1e-13)
    matrices = np.arange(2 * 3 * 4 * 16, dtype=float).reshape(2, 3, 4, 16) - 100
    matrix_codes = wp.zeros((2, 3, 4), dtype=wp.int32, device=device)
    bulk = empty((2, 4, 16))
    bulk_status = empty((2, 4, 16), wp.int32)
    wp.launch(
        population_mueller_kernel,
        dim=(2, 4, 16),
        inputs=[array(matrices), weights, matrix_codes, bulk, bulk_status],
        device=device,
    )
    expected_matrix = get_population_mueller_matrix(
        matrices.reshape(2, 3, 4, 4, 4), weights_cpu
    )
    np.testing.assert_array_equal(bulk_status.numpy(), 0)
    np.testing.assert_allclose(
        bulk.numpy().reshape(2, 4, 4, 4),
        expected_matrix,
        rtol=1e-13,
        atol=1e-15,
    )
    extinction = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]])
    lengths = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    codes = wp.zeros((2, 3), dtype=wp.int32, device=device)
    depth, total, transmission = empty((2, 3)), empty(2), empty(2)
    wp.launch(
        column_optics_kernel,
        dim=2,
        inputs=[
            array(extinction),
            array(lengths),
            codes,
            depth,
            total,
            transmission,
            status,
        ],
        device=device,
    )
    np.testing.assert_array_equal(status.numpy(), 0)
    reference = get_column_optics(extinction, lengths)
    for actual, value in zip(
        (depth, total, transmission), reference, strict=True
    ):
        np.testing.assert_allclose(actual.numpy(), value, rtol=1e-13)
    opacity, valid = empty(2), empty(2, wp.int32)
    codes = wp.zeros(2, dtype=wp.int32, device=device)
    wp.launch(
        mass_opacity_kernel,
        dim=2,
        inputs=[
            array([0.0, 0.3]),
            array([0.0, 0.01]),
            codes,
            opacity,
            valid,
            status,
        ],
        device=device,
    )
    expected_opacity, expected_valid = get_mass_specific_opacity(
        [0.0, 0.3], [0.0, 0.01]
    )
    np.testing.assert_array_equal(status.numpy(), 0)
    np.testing.assert_allclose(opacity.numpy(), expected_opacity)
    np.testing.assert_array_equal(valid.numpy(), expected_valid)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_device_population_failures_and_empty_columns(device):
    """Reject invalid PDF grids and preserve clear upstream error status."""
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("Native CUDA is unavailable")
    values = wp.ones((1, 3), dtype=wp.float64, device=device)
    weights = wp.empty_like(values)
    status = wp.empty((1, 3), dtype=wp.int32, device=device)
    grid = wp.array(np.array([1.0, 1.0, 2.0]), dtype=wp.float64, device=device)
    volume = wp.ones(1, dtype=wp.float64, device=device)
    wp.launch(
        population_weights_kernel,
        dim=(1, 3),
        inputs=[values, volume, grid, 2, weights, status],
        device=device,
    )
    np.testing.assert_array_equal(status.numpy(), [[1, 1, 0]])
    cross = wp.ones((1, 3, 5), dtype=wp.float64, device=device)
    g = wp.zeros_like(values)
    output = wp.empty((1, 9), dtype=wp.float64, device=device)
    bulk_status = wp.empty(1, dtype=wp.int32, device=device)
    wp.launch(
        population_optics_kernel,
        dim=1,
        inputs=[cross, g, weights, status, output, bulk_status],
        device=device,
    )
    np.testing.assert_array_equal(bulk_status.numpy(), [2])
    empty = wp.empty((2, 0), dtype=wp.float64, device=device)
    empty_codes = wp.empty((2, 0), dtype=wp.int32, device=device)
    depth = wp.empty_like(empty)
    total = wp.empty(2, dtype=wp.float64, device=device)
    transmission = wp.empty_like(total)
    codes = wp.empty(2, dtype=wp.int32, device=device)
    wp.launch(
        column_optics_kernel,
        dim=2,
        inputs=[empty, empty, empty_codes, depth, total, transmission, codes],
        device=device,
    )
    np.testing.assert_array_equal(codes.numpy(), 0)
    np.testing.assert_array_equal(total.numpy(), 0.0)
    np.testing.assert_array_equal(transmission.numpy(), 1.0)
