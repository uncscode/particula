"""Prepared GPU metadata admission, tile invariance and explicit observation."""

import numpy as np
import pytest
from particula.optics import get_mie_efficiencies

wp = pytest.importorskip("warp")
from particula.optics.gpu import (  # noqa: E402
    prepare_mie,
    prepare_population_mie,
)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_workspace_tiles_and_reuse(device):
    """Preparation is inert and repeated launches preserve output pointers."""
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("Native CUDA is unavailable")
    indices = np.tile([1.5, 0.1], (5, 1))
    x = np.array([0.0, 0.1, 1.0, 10.0, 100.0])
    m = wp.array(indices, dtype=wp.vec2d, device=device)
    sizes = wp.array(x, dtype=wp.float64, device=device)
    angles = wp.array(
        np.array([0.0, 0.01, 0.3, np.pi / 2, np.pi]),
        dtype=wp.float64,
        device=device,
    )
    full = prepare_mie(
        m,
        sizes,
        angles=angles,
        coefficient_capacity=130,
        case_tile=5,
        angle_tile=5,
    )
    tiled = prepare_mie(
        m,
        sizes,
        angles=angles,
        coefficient_capacity=130,
        case_tile=2,
        angle_tile=2,
    )
    full.efficiencies.fill_(123.0)
    np.testing.assert_array_equal(full.efficiencies.numpy(), 123.0)
    pointer = tiled.efficiencies.ptr
    for _ in range(2):
        full.enqueue()
        tiled.enqueue()
        full.check()
        tiled.check()
        assert tiled.efficiencies.ptr == pointer
        for name in ("efficiencies", "perpendicular", "parallel", "mueller"):
            np.testing.assert_allclose(
                getattr(full, name).numpy(),
                getattr(tiled, name).numpy(),
                rtol=1e-12,
                atol=1e-12,
            )
    expected = get_mie_efficiencies(x / (2 * np.pi), 1.0, 1.5 + 0.1j)
    for j, value in enumerate(expected):
        np.testing.assert_allclose(
            tiled.efficiencies.numpy()[:, j], value, rtol=1e-8, atol=1e-12
        )
    sizes.assign(np.array([0.0, 0.1, -1.0, 10.0, 100.0]))
    tiled.enqueue()
    with pytest.raises(RuntimeError, match="coefficients"):
        tiled.check()
    np.testing.assert_array_equal(tiled.status.numpy(), [0, 0, 1, 0, 0])


def test_workspace_metadata_rejection():
    """Reject incompatible metadata before compiling scientific commands."""
    m = wp.zeros(2, dtype=wp.vec2d, device="cpu")
    x = wp.zeros(2, dtype=wp.float64, device="cpu")
    with pytest.raises(ValueError):
        prepare_mie(m, x, case_tile=0)
    with pytest.raises(ValueError):
        prepare_mie(m, x, coefficient_capacity=10091)
    with pytest.raises(ValueError):
        prepare_mie(m, wp.zeros(2, dtype=wp.float32, device="cpu"))
    with pytest.raises(ValueError):
        prepare_mie(m, x, matrix_scale=x)


@pytest.mark.parametrize("population", [False, True])
@pytest.mark.parametrize("explicit_stream", [False, True])
def test_workspace_replay_uses_prepared_cuda_stream(
    population, explicit_stream
):
    """Capture on the prepared stream even when another stream is current."""
    if not wp.is_cuda_available():
        pytest.skip("Native CUDA is unavailable")
    device = "cuda:0"
    prepared_stream = wp.Stream(device)
    other_stream = wp.Stream(device)
    shape = (1, 1) if population else (1,)
    m = wp.array(
        np.full(shape + (2,), [1.5, 0.1]), dtype=wp.vec2d, device=device
    )
    x = wp.array(np.ones(shape), dtype=wp.float64, device=device)
    radius = wp.array(np.full(shape, 1e-6), dtype=wp.float64, device=device)
    weights = wp.array(np.full(shape, 1e8), dtype=wp.float64, device=device)
    angles = wp.array([0.0, 1.0, np.pi], dtype=wp.float64, device=device)
    with wp.ScopedStream(other_stream if explicit_stream else prepared_stream):
        kwargs = dict(
            coefficient_capacity=20,
            angles=angles,
            stream=prepared_stream if explicit_stream else None,
        )
        work = (
            prepare_population_mie(m, x, radius, weights, **kwargs)
            if population
            else prepare_mie(m, x, **kwargs)
        )
    wp.synchronize_device(device)
    with wp.ScopedStream(other_stream):
        wp.capture_begin(stream=prepared_stream)
        work.enqueue()
        graph = wp.capture_end(stream=prepared_stream)
    # Erase any accidental execution outside the captured stream. Only graph
    # replay may now produce the expected scalar and angular optical outputs.
    wp.synchronize_device(device)
    output = work.optics if population else work.efficiencies
    output.fill_(-123.0)
    work.mueller.fill_(-123.0)
    wp.synchronize_device(device)
    wp.capture_launch(graph, stream=prepared_stream)
    work.check()
    expected = get_mie_efficiencies(1 / (2 * np.pi), 1.0, 1.5 + 0.1j)
    scale = np.pi * 1e-12 * 1e8 if population else 1.0
    np.testing.assert_allclose(
        output.numpy()[0, 0], expected.extinction * scale
    )
    assert np.all(np.isfinite(work.mueller.numpy()))
    assert np.all(work.mueller.numpy()[..., 0, 0] >= 0)


@pytest.mark.parametrize("population", [False, True])
@pytest.mark.parametrize("invalid", [False, True])
def test_workspace_initializes_on_prepared_stream(
    monkeypatch, population, invalid
):
    """Initialization precedes first execution without cross-stream barriers."""
    if not wp.is_cuda_available():
        pytest.skip("Native CUDA is unavailable")
    device = "cuda:0"
    prepared = wp.Stream(device)
    other = wp.Stream(device)
    shape = (1, 1) if population else (1,)
    with wp.ScopedStream(prepared, sync_enter=False):
        m = wp.array(
            np.full(shape + (2,), [1.5, 0.1]), dtype=wp.vec2d, device=device
        )
        x = wp.array(
            np.full(shape, -1.0 if invalid else 1.0),
            dtype=wp.float64,
            device=device,
        )
        radius = wp.array(np.full(shape, 1e-6), dtype=wp.float64, device=device)
        weights = wp.array(np.full(shape, 1e8), dtype=wp.float64, device=device)
        angles = wp.array([0.0, 1.0, np.pi], dtype=wp.float64, device=device)

    fills = []

    def observe_fill(original):
        def wrapped(*args, **kwargs):
            fills.append(wp.get_stream(device))
            return original(*args, **kwargs)

        return wrapped

    # Observe actual fill scheduling as well as numerical results, making the
    # regression deterministic even when a wrong-stream fill finishes quickly.
    for name in ("zeros", "zeros_like", "ones"):
        monkeypatch.setattr(wp, name, observe_fill(getattr(wp, name)))
    with wp.ScopedStream(other, sync_enter=False):
        kwargs = dict(angles=angles, coefficient_capacity=20, stream=prepared)
        work = (
            prepare_population_mie(m, x, radius, weights, **kwargs)
            if population
            else prepare_mie(m, x, **kwargs)
        )
        assert wp.get_stream(device) == other
        work.enqueue()
        assert fills and all(stream == prepared for stream in fills)
        if invalid:
            with pytest.raises(RuntimeError):
                work.check()
        else:
            work.check()
            expected = get_mie_efficiencies(1 / (2 * np.pi), 1.0, 1.5 + 0.1j)
            output = work.optics if population else work.efficiencies
            scale = np.pi * 1e-12 * 1e8 if population else 1.0
            np.testing.assert_allclose(
                output.numpy()[0, 0], expected.extinction * scale
            )
            assert np.all(np.isfinite(work.mueller.numpy()))
            assert np.all(work.mueller.numpy()[..., 0, 0] > 0)
