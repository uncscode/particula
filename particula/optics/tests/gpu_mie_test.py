"""CPU and native CUDA coefficient/efficiency parity and buffer reuse."""

import json
from pathlib import Path

import numpy as np
import pytest
from particula.optics import get_mie_coefficients, get_mie_efficiencies

wp = pytest.importorskip("warp")
from particula.optics.gpu.mie import (  # noqa: E402
    mie_coefficients_kernel,
    mie_efficiencies_kernel,
)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_mie_device_parity(device):
    """Compare the complete reference input matrix through x=10000."""
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("Native CUDA is unavailable")
    data = json.loads(
        (Path(__file__).parent / "data/pymiescatt_spheres.json").read_text()
    )
    m = np.array([complex(*c["index"]) for c in data["cases"]] + [1.0, 1.5])
    x = np.array([c["size_parameter"] for c in data["cases"]] + [1.0, 0.0])
    cpu = get_mie_coefficients(m, x)
    q = get_mie_efficiencies(x / (2 * np.pi), 1.0, m)
    width = cpu.electric.shape[1]
    index = wp.array(
        np.stack([m.real, m.imag], axis=-1), dtype=wp.vec2d, device=device
    )
    sizes = wp.array(x, dtype=wp.float64, device=device)
    d = wp.empty((len(x), width + 1), dtype=wp.vec2d, device=device)
    a = wp.empty((len(x), width), dtype=wp.vec2d, device=device)
    b = wp.empty_like(a)
    counts = wp.empty(len(x), dtype=wp.int32, device=device)
    status = wp.empty_like(counts)
    outputs = wp.empty((len(x), 7), dtype=wp.float64, device=device)
    pointers = (d.ptr, a.ptr, b.ptr, outputs.ptr)
    for _ in range(2):
        wp.launch(
            mie_coefficients_kernel,
            dim=len(x),
            inputs=[index, sizes, d, a, b, counts, status],
            device=device,
        )
        wp.launch(
            mie_efficiencies_kernel,
            dim=len(x),
            inputs=[index, sizes, a, b, counts, status, outputs],
            device=device,
        )
        np.testing.assert_array_equal(status.numpy(), 0)
        np.testing.assert_array_equal(counts.numpy(), cpu.term_count)
        for actual, expected in [(a, cpu.electric), (b, cpu.magnetic)]:
            values = actual.numpy()
            np.testing.assert_allclose(
                values[..., 0] + 1j * values[..., 1],
                expected,
                rtol=1e-8,
                atol=1e-12,
            )
        actual = outputs.numpy()
        for j, expected in enumerate(q):
            np.testing.assert_allclose(
                actual[:, j], expected, rtol=1e-8, atol=1e-12
            )
        assert pointers == (d.ptr, a.ptr, b.ptr, outputs.ptr)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_invalid_device_cases(device):
    """Invalid/capacity cases suppress dependent efficiency calculations."""
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("Native CUDA is unavailable")
    indices = np.array(
        [
            [1.5, 0.0],
            [1.5, 0.0],
            [1.5, -0.1],
            [0.0, 0.0],
            [1.0, 0.0],
            [1.5, 0.0],
        ]
    )
    index = wp.array(indices, dtype=wp.vec2d, device=device)
    sizes = wp.array(
        np.array([-1.0, 10001.0, 1.0, 1.0, 1.0, 100.0]),
        dtype=wp.float64,
        device=device,
    )
    d = wp.zeros((6, 11), dtype=wp.vec2d, device=device)
    a = wp.zeros((6, 10), dtype=wp.vec2d, device=device)
    b = wp.zeros_like(a)
    counts = wp.zeros(6, dtype=wp.int32, device=device)
    status = wp.zeros_like(counts)
    output = wp.ones((6, 7), dtype=wp.float64, device=device)
    wp.launch(
        mie_coefficients_kernel,
        dim=6,
        inputs=[index, sizes, d, a, b, counts, status],
        device=device,
    )
    wp.launch(
        mie_efficiencies_kernel,
        dim=6,
        inputs=[index, sizes, a, b, counts, status, output],
        device=device,
    )
    np.testing.assert_array_equal(status.numpy(), [1, 1, 1, 1, 0, 1])
    np.testing.assert_array_equal(output.numpy(), 0.0)
    np.testing.assert_array_equal(counts.numpy(), 0)
