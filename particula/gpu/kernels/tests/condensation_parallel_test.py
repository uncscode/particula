"""Conservation, roundoff bounds and capture for chunked condensation sums."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from particula.gpu.kernels.tests.condensation_native_test import _oracle, _state

wp = pytest.importorskip("warp")
pytestmark = pytest.mark.warp


@pytest.fixture(params=["cpu", "cuda:0"])
def device(request):
    """Keep CPU regression separate from actual CUDA execution."""
    if request.param != "cpu" and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    return request.param


def population(device, slots, kind, scenario):
    """Extend independent multicomponent fixtures across partial chunks."""
    from particula.gpu.conversion import to_warp_particle_data
    from particula.gpu.kernels.condensation import CondensationScratchBuffers
    from particula.gpu.species_mapping import prepare_condensation_species_map

    state = _state(device, kind=kind, volume=1.0, scenario=scenario)
    indices = np.arange(slots) % 3
    scale = np.geomspace(0.5, 2.0, slots)
    state.cpu_p = replace(
        state.cpu_p,
        masses=state.cpu_p.masses[:, indices] * scale[None, :, None],
        concentration=state.cpu_p.concentration[:, indices] * (3.0 / slots),
        charge=state.cpu_p.charge[:, indices].copy(),
    )
    state.p = to_warp_particle_data(state.cpu_p, device=device)
    previous = state.kwargs["species_binding"]
    state.kwargs["species_binding"] = prepare_condensation_species_map(
        state.p,
        state.g,
        gas_names=state.cpu_g.name,
        species_map=previous.species_map,
    )
    state.total = wp.zeros(
        state.p.masses.shape, dtype=wp.float64, device=device
    )
    state.kwargs["scratch_buffers"] = CondensationScratchBuffers(
        total_mass_transfer=state.total,
    )
    state.kwargs["time_step"] = 0.001
    return state


@pytest.mark.parametrize("slots", [257, 4097])
@pytest.mark.parametrize("kind", ["discrete", "particle_resolved"])
@pytest.mark.parametrize("scenario", ["mixed", "uptake", "evaporation"])
def test_chunk_reduction_matches_independent_physics_and_legacy(
    device, slots, kind, scenario
):
    """Different summation grouping retains mapped mass and latent energy."""
    from particula.gpu.kernels import _condensation_parallel as parallel
    from particula.gpu.kernels import condensation as module

    state = population(device, slots, kind, scenario)
    legacy = population(device, slots, kind, scenario)
    oracle = _oracle(state)
    prepared = module._prepare_condensation_step_gpu(
        state.p, state.g, **state.kwargs
    )
    if slots >= parallel.MIN_SLOTS:
        assert prepared.parallel_reductions is not None
    else:
        assert prepared.parallel_reductions is None
        prepared = replace(
            prepared,
            parallel_reductions=parallel.prepare(2, slots, 4, device),
        )
    old = replace(
        module._prepare_condensation_step_gpu(
            legacy.p, legacy.g, **legacy.kwargs
        ),
        parallel_reductions=None,
    )
    module._enqueue_prepared_condensation_call(prepared)
    module._enqueue_prepared_condensation_call(old)
    masses = state.p.masses.numpy()
    gas = state.g.concentration.numpy()
    for observed, expected in (
        (masses, oracle[0]),
        (gas, oracle[1]),
        (masses, legacy.p.masses.numpy()),
        (gas, legacy.g.concentration.numpy()),
        (state.energy.numpy(), legacy.energy.numpy()),
    ):
        np.testing.assert_allclose(observed, expected, rtol=2e-12, atol=1e-30)
    weights = state.cpu_p.concentration
    if kind == "particle_resolved":
        weights = weights / state.cpu_p.volume[:, None]
    for source, material in state.pairs:
        before = np.sum(state.cpu_p.masses[..., material] * weights, axis=1)
        after = np.sum(masses[..., material] * weights, axis=1)
        np.testing.assert_allclose(
            after + gas[:, source],
            before + state.cpu_g.concentration[:, source],
            rtol=2e-12,
            atol=1e-30,
        )
    assert np.all(masses >= 0) and np.all(gas >= 0)


@pytest.mark.parametrize("kind,volume", [(1, 1.0), (3, 0.25)])
@pytest.mark.parametrize("fault", ["sub_ulp", "overdraw", "nonfinite"])
def test_reconciliation_never_spends_unrepresentable_inventory(
    device, kind, volume, fault
):
    """Large cross-chunk demand cannot bypass rounded-transfer safety."""
    from particula.gpu.kernels import _condensation_native as native
    from particula.gpu.kernels import _condensation_parallel as parallel
    from particula.gpu.kernels import condensation as module

    slots = 513

    def array(values, dtype=wp.float64):
        return wp.array(np.asarray(values), dtype=dtype, device=device)

    before = np.full((1, slots, 1), 1e-18)
    weights = np.full((1, slots), 1e12 * (volume if kind == 3 else 1.0))
    gas = array([[1e-22]])
    p = SimpleNamespace(
        candidate=array(np.full(before.shape, 1e-20)),
        particle_masses=array(before),
        particle_concentration=array(weights),
        particle_volume=array([volume]),
        particle_to_gas=array([0], wp.int32),
        gas_to_particle=array([0], wp.int32),
        gas_concentration=gas,
        demand=array([[0.0]]),
        release=array([[0.0]]),
        gas_delta=array([[0.0]]),
        distribution_kind=kind,
        finalized=array(np.zeros_like(before)),
        dimensions=before.shape,
        device=device,
        parallel_reductions=parallel.prepare(1, slots, 1, device),
    )
    parallel.candidates(p)
    scale = array([[0.0]])
    wp.launch(
        module._scale_inventory_uptake_kernel,
        dim=(1, 1),
        inputs=[gas, p.demand, p.release, scale],
        device=device,
    )
    wp.launch(
        native.finalize_transfer,
        dim=(1, slots),
        inputs=[
            p.particle_masses,
            p.candidate,
            scale,
            p.finalized,
            p.particle_to_gas,
        ],
        device=device,
    )
    if fault == "overdraw":
        p.finalized.fill_(1e-20)
    elif fault == "nonfinite":
        p.finalized.fill_(np.inf)
    parallel.finalized(p)
    np.testing.assert_array_equal(p.finalized.numpy(), 0)
    np.testing.assert_array_equal(p.gas_delta.numpy(), 0)


def test_parallel_enqueue_has_no_host_boundary(device, monkeypatch):
    """Capture-safe work allocates only during preparation and uses fixed arrays."""
    from particula.gpu.kernels import condensation as module

    state = population(device, 4097, "particle_resolved", "mixed")
    p = module._prepare_condensation_step_gpu(state.p, state.g, **state.kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("host allocation or observation inside enqueue")

    with monkeypatch.context() as patch:
        for name in (
            "zeros",
            "empty",
            "full",
            "synchronize",
            "synchronize_device",
        ):
            patch.setattr(wp, name, forbidden)
        patch.setattr(wp.array, "numpy", forbidden)
        module._enqueue_prepared_condensation_call(p)
    assert np.all(np.isfinite(state.p.masses.numpy()))
