"""Resolved-count admission and invariant checks for GPU exhaustion."""

import numpy as np
import pytest

from particula.gpu.kernels.tests.exhaustion_test import _state

pytestmark = [pytest.mark.warp, pytest.mark.gpu_parity]


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    """Require Warp CPU and run optional CUDA when available."""
    wp = pytest.importorskip("warp")
    if request.param == "cuda" and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    return request.param


def native_state(device):
    """Use the transferable native carrier with existing fixed-shape scratch."""
    from particula.gpu import to_warp_particle_data
    from particula.particles import ParticleData

    fields, counts, buffers = _state(device)
    cpu = ParticleData(
        **{
            name: getattr(fields, name).numpy()
            for name in (
                "masses",
                "concentration",
                "charge",
                "density",
                "volume",
            )
        },
        distribution_type="particle_resolved",
    )
    cpu.volume[:] = 0.25
    return to_warp_particle_data(cpu, device=device), counts, buffers


def moments(particles):
    """Compute independent represented-count species mass and charge."""
    counts = particles.concentration.numpy()
    return (
        np.sum(particles.masses.numpy() * counts[..., None], axis=1),
        np.sum(particles.charge.numpy() * counts, axis=1),
        counts.sum(axis=1),
    )


def test_native_resampling_preserves_count_moments(device):
    """Releasing a slot preserves count-weighted mass and charge at V!=1."""
    from particula.gpu.kernels.exhaustion import resampling_step_gpu

    particles, counts, buffers = native_state(device)
    before = moments(particles)
    assert resampling_step_gpu(particles, counts, buffers) is particles
    for actual, expected in zip(moments(particles), before, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-30)
    np.testing.assert_array_equal(particles.volume.numpy(), [0.25])
    assert np.count_nonzero(particles.concentration.numpy()) == 2


def test_native_scaling_preserves_density_and_repeated_enqueue(
    device, monkeypatch
):
    """Prepared count/volume scaling remains setup-free across repeated calls."""
    import warp as wp

    from particula.gpu.kernels import exhaustion as module

    particles, _, _ = native_state(device)
    before = moments(particles)
    demand = wp.array([8.0], dtype=wp.float64, device=device)
    flag = wp.array([1], dtype=wp.int32, device=device)
    factor = wp.array([0.5], dtype=wp.float64, device=device)
    minimum = wp.array([0.25], dtype=wp.float64, device=device)
    minimum_volume = wp.array([0.01], dtype=wp.float64, device=device)
    resolved = wp.zeros(1, dtype=wp.float64, device=device)
    prepared = module._prepare_representative_volume_scaling_step_gpu(
        particles,
        demand,
        flag,
        factor,
        minimum,
        minimum_volume,
        resolved,
    )
    module._enqueue_prepared_representative_volume_scaling_call(prepared)
    module._observe_prepared_representative_volume_scaling_call(prepared)

    def forbidden(*args, **kwargs):
        pytest.fail("prepared native scaling performed host setup")

    with monkeypatch.context() as patch:
        for name in ("zeros", "empty", "array", "full", "synchronize"):
            patch.setattr(wp, name, forbidden)
        patch.setattr(prepared.status, "numpy", forbidden)
        module._enqueue_prepared_representative_volume_scaling_call(prepared)
    module._observe_prepared_representative_volume_scaling_call(prepared)
    for actual, expected in zip(moments(particles), before, strict=True):
        np.testing.assert_allclose(
            actual, expected * 0.25, rtol=1e-12, atol=1e-30
        )
    np.testing.assert_array_equal(particles.volume.numpy(), [0.0625])
    np.testing.assert_array_equal(demand.numpy(), [2.0])


def test_native_resampling_live_density_overflow_rejects_without_commit(device):
    """Prepared state retains live physical validation of count density."""
    import warp as wp

    from particula.gpu.kernels import exhaustion as module

    particles, counts, buffers = native_state(device)
    prepared = module._prepare_resampling_step_gpu(particles, counts, buffers)
    wp.copy(
        particles.volume, wp.array([1e-309], dtype=wp.float64, device=device)
    )
    before = moments(particles)
    module._enqueue_prepared_resampling_call(prepared)
    with pytest.raises(ValueError, match="invalid values"):
        module._observe_prepared_resampling_call(prepared)
    for actual, expected in zip(moments(particles), before, strict=True):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("code", [1, 2])
@pytest.mark.parametrize("operation", ["resampling", "scaling"])
def test_unsupported_native_populations_reject_before_payload(code, operation):
    """Fixed-slot exhaustion never treats PMF or PDF as resolved counts."""
    from particula.gpu.kernels import exhaustion as module

    class Tagged:
        distribution_kind = code

        def __getattr__(self, name):
            pytest.fail(f"unsupported population accessed {name}")

    with pytest.raises(ValueError, match="only particle_resolved"):
        if operation == "resampling":
            module._prepare_resampling_step_gpu(Tagged(), None, None)
        else:
            module._prepare_representative_volume_scaling_step_gpu(
                Tagged(),
                None,
                None,
                None,
                None,
                None,
                None,
            )
