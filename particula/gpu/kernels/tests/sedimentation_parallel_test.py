"""Exact pair maxima and continuation for parallel settling collisions."""

import numpy as np
import pytest

from particula.gpu.kernels.tests.brownian_parallel_test import population

wp = pytest.importorskip("warp")
pytestmark = pytest.mark.warp


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("brownian", [False, True])
def test_parallel_settling_preserves_events_and_rng(device, brownian):
    """A mixed-density, sparse population retains the original exact sampler."""
    if device != "cpu" and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    from particula.gpu import to_warp_particle_data
    from particula.gpu.kernels.coagulation import (
        CoagulationMechanismConfig,
        _enqueue_prepared_coagulation_call,
        _native_coagulation_kernel,
        _prepare_coagulation_step_gpu,
        apply_coagulation_kernel,
        coagulation_preflight_gpu,
        initialize_coagulation_rng_states,
    )

    initial = population(67, 0.8)
    mechanisms = ("sedimentation_sp2016",)
    if brownian:
        mechanisms = ("brownian", *mechanisms)
    inspection = coagulation_preflight_gpu(
        to_warp_particle_data(initial, device=device),
        298.15,
        101325.0,
        1.0,
        mechanism_config=CoagulationMechanismConfig(mechanisms),
    )
    step = 0.5 * float(np.min(inspection["max_time_step"]))
    assert step > 0
    owners = [
        _prepare_coagulation_step_gpu(
            to_warp_particle_data(initial, device=device),
            298.15,
            101325.0,
            step,
            mechanism_config=CoagulationMechanismConfig(mechanisms),
            rng_seed=44,
            initialize_rng=True,
        )
        for _ in range(2)
    ]
    legacy = owners[1]
    initialize_coagulation_rng_states(44, legacy.selector_inputs[27])
    events = 0
    for _ in range(4):
        _enqueue_prepared_coagulation_call(owners[0])
        # The unchanged serial pair scan remains an independent execution
        # oracle, not a fallback selected by the production wrapper.
        wp.launch(
            _native_coagulation_kernel,
            dim=legacy.n_boxes,
            inputs=legacy.selector_inputs,
            device=device,
        )
        wp.launch(
            apply_coagulation_kernel,
            dim=(legacy.n_boxes, legacy.max_collisions),
            inputs=[
                legacy.masses,
                legacy.concentration,
                legacy.charge,
                legacy.collision_pairs,
                legacy.n_collisions,
            ],
            device=device,
        )
        events += int(owners[0].n_collisions.numpy().sum())
        for name in (
            "masses",
            "concentration",
            "charge",
            "collision_pairs",
            "n_collisions",
        ):
            np.testing.assert_array_equal(
                getattr(owners[0], name).numpy(),
                getattr(owners[1], name).numpy(),
            )
        for index in (15, 19, 27):
            left = owners[0].selector_inputs[index].numpy()
            right = owners[1].selector_inputs[index].numpy()
            if index != 27:
                # Inactive transport scratch is intentionally unspecified.
                active = owners[0].concentration.numpy() > 0
                left, right = left[active], right[active]
            np.testing.assert_array_equal(left, right)

    assert events > 0


def test_full_biomass_capacity_settling_conserves_on_cuda():
    """Ten thousand active particles execute without a CPU stage or dense matrix."""
    if not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    from particula.gpu import to_warp_particle_data
    from particula.gpu.kernels.coagulation import (
        CoagulationMechanismConfig,
        _enqueue_prepared_coagulation_call,
        _prepare_coagulation_step_gpu,
    )

    initial = population(10_000, 1.0)
    initial.volume[:] = 1e-6
    owner = _prepare_coagulation_step_gpu(
        to_warp_particle_data(initial, device="cuda:0"),
        298.15,
        101325.0,
        0.36,
        mechanism_config=CoagulationMechanismConfig(
            ("brownian", "sedimentation_sp2016")
        ),
        rng_seed=100,
        initialize_rng=True,
    )
    _enqueue_prepared_coagulation_call(owner)
    masses = owner.masses.numpy()
    counts = owner.concentration.numpy()
    np.testing.assert_allclose(
        np.sum(masses * counts[..., None], axis=1),
        np.sum(initial.masses * initial.concentration[..., None], axis=1),
        rtol=3e-14,
        atol=1e-30,
    )
    assert owner.brownian_parallel.partials.shape == (3, 4096)
    assert np.all(np.isfinite(masses))
    assert np.all(masses >= 0)
