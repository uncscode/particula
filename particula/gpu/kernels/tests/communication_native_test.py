"""Native count transport and physical-volume inventory contracts."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from particula.gpu.kernels.tests.communication_test import (
    _containers,
    _particle_configuration,
)

wp = pytest.importorskip("warp")
pytestmark = pytest.mark.warp


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    """Exercise real device backends with no fallback."""
    if request.param == "cuda" and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    return request.param


def population(device):
    """Two unequal-volume boxes, two mass lanes and three gas lanes."""
    from particula.gpu import to_warp_particle_data
    from particula.particles import ParticleData

    particles, gas = _containers(
        np.array([0.25, 4.0]),
        np.array([[8.0, 0.0], [0.0, 0.0]]),
        np.array([[2.0, 3.0, 5.0], [7.0, 11.0, 13.0]]),
        device=device,
    )
    masses = np.zeros((2, 2, 2), dtype=np.float64)
    masses[0, 0] = [2e-18, 3e-18]
    charge = np.zeros((2, 2), dtype=np.float64)
    charge[0, 0] = -2.0
    particles = to_warp_particle_data(
        ParticleData(
            masses=masses,
            concentration=particles.concentration.numpy(),
            charge=charge,
            density=np.array([1000.0, 1500.0]),
            volume=np.array([0.25, 4.0]),
            distribution_type="particle_resolved",
        ),
        device=device,
    )
    return particles, gas


def configuration(device):
    """One edge moving half of source counts at unit duration."""
    config = _particle_configuration(
        np.array([0]), np.array([1]), np.array([0.5])
    )
    old = config.communication_map
    return replace(
        config,
        communication_map=replace(
            old,
            **{
                name: wp.array(
                    getattr(old, name).numpy(),
                    dtype=getattr(old, name).dtype,
                    device=device,
                )
                for name in (
                    "source_boxes",
                    "destination_boxes",
                    "enabled",
                    "rates",
                )
            },
        ),
    )


@pytest.mark.parametrize("prepared", [False, True])
def test_physical_volume_keeps_counts_and_extensive_inventories(
    device, prepared, monkeypatch
):
    """Volume changes gas density, never native particle counts."""
    from particula.gpu.kernels import communication as module

    particles, gas = population(device)
    protected = (particles.masses, particles.concentration, particles.charge)
    before = [array.numpy().copy() for array in protected]
    gas_amount = gas.concentration.numpy() * particles.volume.numpy()[:, None]
    final = wp.array([0.5, 2.0], dtype=wp.float64, device=device)
    invalid = wp.zeros(1, dtype=wp.int32, device=device)
    changed = wp.zeros(1, dtype=wp.int32, device=device)
    if prepared:

        def forbidden(*args, **kwargs):
            pytest.fail("prepared dispatch performed setup or synchronization")

        with monkeypatch.context() as patch:
            for name in ("zeros", "empty", "synchronize"):
                patch.setattr(wp, name, forbidden)
            for _ in range(2):
                module._enqueue_prepared_resident_volume_evolution(
                    particles.volume,
                    particles.concentration,
                    gas.concentration,
                    2,
                    final,
                    invalid,
                    changed,
                    particles.volume.device,
                    True,
                )
    else:
        assert module.volume_evolution_step_gpu(particles, gas, final) == (
            particles,
            gas,
        )
    wp.synchronize()
    for array, expected in zip(protected, before, strict=True):
        np.testing.assert_array_equal(array.numpy(), expected)
    np.testing.assert_array_equal(particles.volume.numpy(), [0.5, 2.0])
    np.testing.assert_allclose(
        gas.concentration.numpy() * final.numpy()[:, None],
        gas_amount,
        rtol=1e-12,
        atol=1e-30,
    )


@pytest.mark.parametrize("prepared", [False, True])
def test_transport_moves_counts_without_volume_ratio(
    device, prepared, monkeypatch
):
    """Closed transport preserves count, species mass and signed charge."""
    from particula.gpu.kernels import communication as module

    particles, _ = population(device)
    config = configuration(device)
    buffers = module.ParticleCommunicationBuffers(
        wp.empty((2, 2), dtype=wp.float64, device=device),
        wp.empty((2, 2), dtype=wp.float64, device=device),
        wp.empty((1, 2), dtype=wp.int32, device=device),
        wp.empty((1, 2), dtype=wp.float64, device=device),
    )
    if prepared:
        invalid = wp.zeros(1, dtype=wp.int32, device=device)
        demand = wp.zeros(1, dtype=wp.int32, device=device)
        snapshots = [
            wp.empty_like(a)
            for a in (
                particles.masses,
                particles.concentration,
                particles.charge,
            )
        ]
        map_data = config.communication_map

        def forbidden(*args, **kwargs):
            pytest.fail("prepared dispatch performed setup or synchronization")

        with monkeypatch.context() as patch:
            for name in ("zeros", "empty", "synchronize"):
                patch.setattr(wp, name, forbidden)
            for _ in range(2):
                module._enqueue_prepared_resident_particle_communication(
                    particles.masses,
                    particles.concentration,
                    particles.charge,
                    particles.density,
                    particles.volume,
                    2,
                    2,
                    2,
                    map_data.source_boxes,
                    map_data.destination_boxes,
                    map_data.enabled,
                    map_data.rates,
                    1,
                    0,
                    1.0,
                    particles.masses.device,
                    buffers,
                    invalid,
                    demand,
                    *snapshots,
                    True,
                )
        expected = [[2.0, 0.0], [6.0, 0.0]]
    else:
        assert (
            module.particle_communication_step_gpu(
                particles, config, 1.0, buffers
            )
            is particles
        )
        expected = [[4.0, 0.0], [4.0, 0.0]]
    wp.synchronize()
    counts = particles.concentration.numpy()
    np.testing.assert_array_equal(counts, expected)
    np.testing.assert_allclose(
        np.sum(particles.masses.numpy() * counts[..., None], axis=(0, 1)),
        [16e-18, 24e-18],
        rtol=1e-12,
        atol=1e-30,
    )
    assert np.sum(particles.charge.numpy() * counts) == -16.0


@pytest.mark.parametrize("final", [[0.0, 2.0], [1e-310, 2.0], [np.nan, 2.0]])
def test_invalid_physical_volume_gates_all_writers(device, final):
    """Invalid target density or volume preserves primary bytes."""
    from particula.gpu.kernels.communication import volume_evolution_step_gpu

    particles, gas = population(device)
    fields = (particles.volume, particles.concentration, gas.concentration)
    before = [array.numpy().copy() for array in fields]
    volume_evolution_step_gpu(
        particles, gas, wp.array(final, dtype=wp.float64, device=device)
    )
    wp.synchronize()
    for array, expected in zip(fields, before, strict=True):
        np.testing.assert_array_equal(array.numpy(), expected)


@pytest.mark.parametrize("code", [1, 2])
def test_unsupported_population_rejects_before_payload(code):
    """Binned population transport is not silently inferred."""
    from particula.gpu.kernels.communication import (
        particle_communication_step_gpu,
        volume_evolution_step_gpu,
    )

    particles = SimpleNamespace(distribution_kind=code)
    with pytest.raises(ValueError, match="particle_resolved"):
        particle_communication_step_gpu(particles, None, 0.0, None)
    with pytest.raises(ValueError, match="particle_resolved"):
        volume_evolution_step_gpu(particles, None, None)


@pytest.mark.parametrize("failure", ["overdraw", "density", "mass", "alias"])
def test_native_transport_rejects_invalid_state_without_primary_writes(
    device, failure
):
    """A bad plan cannot modify source or destination primary state."""
    from particula.gpu.kernels import communication as module

    particles, _ = population(device)
    config = configuration(device)
    if failure == "overdraw":
        config.communication_map.rates.assign(np.array([2.0]))
    elif failure == "density":
        particles.volume.assign(np.array([0.25, 1e-310]))
    elif failure == "mass":
        values = particles.masses.numpy()
        values[0, 0, 0] = np.nan
        particles.masses.assign(values)
    else:
        particles.charge = particles.concentration
    buffers = module.ParticleCommunicationBuffers(
        wp.empty((2, 2), dtype=wp.float64, device=device),
        wp.empty((2, 2), dtype=wp.float64, device=device),
        wp.empty((1, 2), dtype=wp.int32, device=device),
        wp.empty((1, 2), dtype=wp.float64, device=device),
    )
    fields = (
        particles.masses,
        particles.concentration,
        particles.charge,
        particles.volume,
    )
    before = [array.numpy().copy() for array in fields]
    if failure == "alias":
        with pytest.raises(ValueError, match="alias"):
            module.particle_communication_step_gpu(
                particles, config, 1.0, buffers
            )
    else:
        module.particle_communication_step_gpu(particles, config, 1.0, buffers)
    wp.synchronize()
    for array, expected in zip(fields, before, strict=True):
        np.testing.assert_array_equal(array.numpy(), expected)
