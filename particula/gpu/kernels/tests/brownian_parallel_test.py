"""Exact continuation, physical maxima and memory bounds for parallel Brownian."""

from dataclasses import replace

import numpy as np
import pytest

from particula.dynamics.coagulation.brownian_kernel import get_brownian_kernel
from particula.particles import ParticleData

wp = pytest.importorskip("warp")
pytestmark = pytest.mark.warp


@pytest.fixture(params=["cpu", "cuda:0"])
def device(request):
    """Keep actual CUDA evidence distinct from the Warp CPU regression."""
    if request.param != "cpu" and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    return request.param


def population(slots, occupancy):
    """Use three temperatures/volumes, mixed compositions and scattered holes."""
    size = np.geomspace(1e-23, 1e-15, slots)
    composition = np.stack(
        (
            np.linspace(0.1, 0.7, slots),
            np.linspace(0.7, 0.1, slots),
            np.full(slots, 0.2),
        ),
        axis=1,
    )
    masses = np.tile((size[:, None] * composition)[None, ...], (3, 1, 1))
    counts = np.broadcast_to(
        np.array([1.0, 7.0, 0.3])[:, None], (3, slots)
    ).copy()
    active = np.random.default_rng(19).permutation(slots)[
        : int(slots * occupancy)
    ]
    mask = np.ones(slots, dtype=bool)
    mask[active] = False
    counts[:, mask] = 0
    masses[:, mask] = 0
    charge = np.broadcast_to(
        np.linspace(-3, 3, slots)[None, :], (3, slots)
    ).copy()
    charge[:, mask] = 0
    return ParticleData(
        masses=masses,
        concentration=counts,
        charge=charge,
        density=np.array([1000.0, 1400.0, 1900.0]),
        volume=np.array([1e-12, 2e-12, 4e-12]),
        distribution_type="particle_resolved",
    )


def prepare(cpu, device, seed):
    """Make independently owned direct preparation with fixed RNG provenance."""
    from particula.gpu.conversion import to_warp_particle_data
    from particula.gpu.kernels.coagulation import _prepare_coagulation_step_gpu

    particles = to_warp_particle_data(cpu, device=device)
    temperature = wp.array(
        [260.0, 298.15, 330.0], dtype=wp.float64, device=device
    )
    return _prepare_coagulation_step_gpu(
        particles,
        temperature,
        101325.0,
        0.1,
        max_collisions=max(cpu.masses.shape[1] // 2, 1),
        rng_seed=seed,
        initialize_rng=True,
    )


@pytest.mark.parametrize("slots", [17, 65, 257])
@pytest.mark.parametrize("occupancy", [0.25, 0.75, 1.0])
@pytest.mark.parametrize("seed", [0, 37])
def test_parallel_brownian_preserves_legacy_bytes_and_rng(
    device, slots, occupancy, seed
):
    """Only the work distribution changes; accepted events and continuation do not."""
    from particula.gpu.kernels.coagulation import (
        _enqueue_prepared_coagulation_call,
    )

    cpu = population(slots, occupancy)
    optimized = prepare(cpu, device, seed)
    legacy = replace(prepare(cpu, device, seed), brownian_parallel=None)
    initial_mass = np.sum(cpu.masses * cpu.concentration[..., None], axis=1)
    initial_charge = np.sum(cpu.charge * cpu.concentration, axis=1)
    events = 0
    for _ in range(3):
        _enqueue_prepared_coagulation_call(legacy)
        _enqueue_prepared_coagulation_call(optimized)
        for name in (
            "masses",
            "concentration",
            "charge",
            "collision_pairs",
            "n_collisions",
        ):
            np.testing.assert_array_equal(
                getattr(optimized, name).numpy(), getattr(legacy, name).numpy()
            )
        for index in (15, 16, 17, 18, 22, 27):
            np.testing.assert_array_equal(
                optimized.selector_inputs[index].numpy(),
                legacy.selector_inputs[index].numpy(),
            )
        masses, counts, charge = (
            array.numpy()
            for array in (
                optimized.masses,
                optimized.concentration,
                optimized.charge,
            )
        )
        np.testing.assert_allclose(
            np.sum(masses * counts[..., None], axis=1),
            initial_mass,
            rtol=1e-14,
            atol=1e-30,
        )
        np.testing.assert_allclose(
            np.sum(charge * counts, axis=1),
            initial_charge,
            rtol=1e-13,
            atol=1e-12,
        )
        events += optimized.n_collisions.numpy().sum()
    if slots == 257:
        assert events > 0
    work = optimized.brownian_parallel
    assert work.active.shape == (3, slots)
    assert work.partials.shape[1] <= 4096


@pytest.mark.parametrize("active", [0, 1, 2, 33])
def test_parallel_majorant_matches_independent_full_pair_matrix(device, active):
    """The largest upper-triangle rate is exact, including empty/singletons."""
    from particula.gpu.kernels.coagulation import (
        _enqueue_prepared_coagulation_call,
    )

    cpu = population(33, active / 33)
    prepared = prepare(cpu, device, 4)
    _enqueue_prepared_coagulation_call(prepared)
    work = prepared.brownian_parallel
    actual = work.majorants.numpy()
    properties = [
        prepared.selector_inputs[index].numpy() for index in (15, 16, 17, 18)
    ]
    for box in range(3):
        mask = cpu.concentration[box] > 0
        expected = 0.0
        if active > 1:
            rates = get_brownian_kernel(
                *(values[box, mask] for values in properties)
            )
            expected = np.max(rates[np.triu_indices(active, 1)])
        np.testing.assert_allclose(actual[box], expected, rtol=2e-14, atol=0)
    if active < 2:
        np.testing.assert_array_equal(prepared.n_collisions.numpy(), 0)


def test_parallel_workspace_is_linear_at_large_capacity(device):
    """Preparation never allocates a dense pair matrix, even beyond int32 N²."""
    prepared = prepare(population(65537, 0.0), device, 4)
    work = prepared.brownian_parallel
    assert work.partials.shape == (3, 4096)
    assert work.active.shape == (3, 65537)
    assert work.majorants.shape == (3,)
