"""Population-independent admission and additive tiled reduction regressions."""

import numpy as np
import pytest

from particula.gpu.kernels.tests.brownian_parallel_test import population

wp = pytest.importorskip("warp")


@pytest.fixture(params=["cpu", "cuda:0"])
def device(request):
    """Require native CUDA when selecting the CUDA variant."""
    if request.param != "cpu" and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    return request.param


def prepared_case(device, mask, slots=33):
    """Prepare independent populations with unequal volumes and equal weights."""
    from particula.gpu import to_warp_particle_data
    from particula.gpu.kernels.coagulation import (
        CoagulationMechanismConfig,
        _prepare_coagulation_step_gpu,
    )

    names = (
        "brownian",
        "charged_hard_sphere",
        "sedimentation_sp2016",
        "turbulent_shear_st1956",
    )
    cpu = population(slots, 0.75)
    prepared = _prepare_coagulation_step_gpu(
        to_warp_particle_data(cpu, device=device),
        283.15,
        101325.0,
        1e-5,
        max_collisions=max(1, slots // 2),
        mechanism_config=CoagulationMechanismConfig(
            tuple(name for bit, name in enumerate(names) if mask & (1 << bit))
        ),
        turbulent_dissipation=1e-4,
        fluid_density=1.2,
    )
    return cpu, prepared


@pytest.mark.parametrize("mask", range(1, 16))
def test_all_additive_masks_conserve_inventory(device, mask):
    """All subsets share the finite-time path and conserve each inventory."""
    from particula.gpu.kernels.coagulation import (
        _enqueue_prepared_coagulation_call,
    )

    cpu, prepared = prepared_case(device, mask)
    _enqueue_prepared_coagulation_call(prepared)
    mass = prepared.masses.numpy()
    count = prepared.concentration.numpy()
    charge = prepared.charge.numpy()
    np.testing.assert_allclose(
        np.sum(mass * count[..., None], axis=1),
        np.sum(cpu.masses * cpu.concentration[..., None], axis=1),
        rtol=1e-14,
        atol=1e-30,
    )
    np.testing.assert_allclose(
        np.sum(charge * count, axis=1),
        np.sum(cpu.charge * cpu.concentration, axis=1),
        rtol=1e-13,
        atol=1e-12,
    )
    assert np.all(prepared.brownian_parallel.majorants.numpy() > 0)


@pytest.mark.parametrize("mask", [2, 4, 7, 11, 13, 14])
@pytest.mark.parametrize("slots", [257, 16385, 32768])
def test_preparation_has_no_physical_population_cap(device, mask, slots):
    """Admission exceeds both former caps without allocating a pair matrix."""
    _, prepared = prepared_case(device, mask, slots)
    assert prepared.brownian_parallel.active.shape == (3, slots)
    assert prepared.brownian_parallel.partials.shape[1] <= 4096


def test_budget_admission_reports_all_boxes_without_rng_mutation(device):
    """A rejecting box leaves RNG and caller outputs untouched across the batch."""
    from particula.gpu.kernels._coagulation_budget import (
        CoagulationBudgetError,
        admit_coagulation_work,
        raise_budget_error,
    )

    _, prepared = prepared_case(device, 13)
    work = prepared.brownian_parallel
    work.prepare_rates(None)
    watched = [
        prepared.selector_inputs[27],
        prepared.collision_pairs,
        prepared.n_collisions,
        prepared.masses,
        prepared.concentration,
        prepared.charge,
    ]
    before = [value.numpy().copy() for value in watched]
    report = wp.zeros((3, 4), dtype=wp.float64, device=device)
    failure = wp.zeros(1, dtype=wp.int32, device=device)
    # A deterministic majorant fixture isolates normalization from kernel fits.
    majorants = wp.array([1e-15, 1e-10, 1e-15], dtype=wp.float64, device=device)
    wp.launch(
        admit_coagulation_work,
        dim=3,
        inputs=[
            work.counts,
            prepared.selector_inputs[24],
            prepared.concentration,
            prepared.selector_inputs[3],
            majorants,
            1.0,
            65536,
            prepared.max_collisions,
            report,
            failure,
        ],
        device=device,
    )
    rows = report.numpy()
    active = work.counts.numpy().astype(np.float64)
    expected = (
        majorants.numpy()
        * active
        * (active - 1)
        / 2
        * np.array([1.0, 7.0, 0.3])
        / np.array([1e-12, 2e-12, 4e-12])
    )
    np.testing.assert_allclose(rows[:, 0], expected, rtol=1e-14)
    np.testing.assert_array_equal(rows[:, 1], np.ceil(expected))
    assert failure.numpy()[0] == 9
    with pytest.raises(CoagulationBudgetError, match="box 1"):
        raise_budget_error(report)
    work.select(failure)
    for value, saved in zip(watched, before, strict=True):
        np.testing.assert_array_equal(value.numpy(), saved)


@pytest.mark.parametrize("initialize_rng", [False, True])
def test_direct_budget_rejection_preserves_all_caller_state(
    device, initialize_rng
):
    """Even an explicitly requested RNG reset must wait for budget admission."""
    from particula.gpu.kernels._coagulation_budget import CoagulationBudgetError
    from particula.gpu.kernels.coagulation import (
        CoagulationMechanismConfig,
        CoagulationWorkConfig,
        coagulation_step_gpu,
    )

    _, prepared = prepared_case(device, 13)
    rng = prepared.selector_inputs[27]
    rng.fill_(123)
    prepared.collision_pairs.fill_(17)
    prepared.n_collisions.fill_(2)
    prepared.work_counts.fill_(19)
    watched = [
        prepared.masses,
        prepared.concentration,
        prepared.charge,
        rng,
        prepared.collision_pairs,
        prepared.n_collisions,
        prepared.work_counts,
    ]
    before = [value.numpy().copy() for value in watched]
    with pytest.raises(CoagulationBudgetError, match="requires.*trials"):
        coagulation_step_gpu(
            prepared.particles,
            283.15,
            101325.0,
            1e9,
            max_collisions=prepared.max_collisions,
            collision_pairs=prepared.collision_pairs,
            n_collisions=prepared.n_collisions,
            work_counts=prepared.work_counts,
            rng_states=rng,
            initialize_rng=initialize_rng,
            mechanism_config=CoagulationMechanismConfig(
                (
                    "brownian",
                    "sedimentation_sp2016",
                    "turbulent_shear_st1956",
                )
            ),
            turbulent_dissipation=1e-4,
            fluid_density=1.2,
            work_config=CoagulationWorkConfig(max_trials_per_box=1),
        )
    for value, saved in zip(watched, before, strict=True):
        np.testing.assert_array_equal(value.numpy(), saved)


def test_prepared_work_config_survives_restart(device):
    """Canonical plans preserve custom budgets and exact RNG continuation."""
    from particula.dynamics import BrownianCoagulationStrategy, Coagulation
    from particula.dynamics.tests.native_runnable_test import state
    from particula.gpu.kernels.coagulation import CoagulationWorkConfig

    process = Coagulation(BrownianCoagulationStrategy("particle_resolved"))
    config = CoagulationWorkConfig(max_trials_per_box=31)
    with process.prepare(
        state(volume=1e-13),
        device=device,
        time_step=1e-3,
        coagulation_work_config=config,
    ) as original:
        assert not original.coagulation_diagnostics()["attempted_trials"].any()
        if device != "cpu":
            original.capture()
            assert not original.coagulation_diagnostics()[
                "attempted_trials"
            ].any()
        original.step()
        saved = original.checkpoint()
        assert saved.processes[0].values["max_trials_per_box"] == 31
        with process.resume(saved, device=device) as restored:
            diagnostics = restored.coagulation_diagnostics()
            assert not diagnostics["attempted_trials"].any()
            assert not diagnostics["accepted_collisions"].any()
            original.step()
            restored.step()
            for name, values in original.coagulation_diagnostics().items():
                np.testing.assert_array_equal(
                    values, restored.coagulation_diagnostics()[name]
                )
            left, right = original.checkpoint(), restored.checkpoint()
            assert left.processes == right.processes
            assert left.resident.payloads == right.resident.payloads
            assert (
                left.resident.rng_continuation
                == right.resident.rng_continuation
            )


def test_prepared_budget_rejection_preserves_particles_and_rng(device):
    """An overloaded resident step guards all writers before observation faults."""
    from particula.dynamics import BrownianCoagulationStrategy, Coagulation
    from particula.dynamics.tests.native_runnable_test import state

    process = Coagulation(BrownianCoagulationStrategy("particle_resolved"))
    with process.prepare(
        state(volume=1e-20),
        device=device,
        time_step=1,
    ) as owner:
        binding = owner._bindings["coagulation"]
        watched = [
            binding.masses,
            binding.concentration,
            binding.charge,
            binding.collision_pairs,
            binding.n_collisions,
            binding.selector_inputs[27],
        ]
        before = [value.numpy().copy() for value in watched]
        owner.step()
        with pytest.raises(
            ValueError, match="Coagulation box 0 requires.*trials"
        ):
            owner.synchronize()
        for value, saved in zip(watched, before, strict=True):
            np.testing.assert_array_equal(value.numpy(), saved)


def test_admission_skips_uninitialized_scratch_after_prior_failure(device):
    """A previous process failure must prevent reading invalid rate scratch."""
    from particula.gpu.kernels._coagulation_budget import admit_coagulation_work

    _, prepared = prepared_case(device, 13)
    work = prepared.brownian_parallel
    # Deliberately invalid indices model scratch not populated by guarded rates.
    work.counts.fill_(33)
    prepared.selector_inputs[24].fill_(2**30)
    work.majorants.fill_(1.0)
    prepared.budget_report.fill_(-17.0)
    prepared.budget_failure.fill_(4)
    wp.launch(
        admit_coagulation_work,
        dim=prepared.n_boxes,
        inputs=[
            work.counts,
            prepared.selector_inputs[24],
            prepared.concentration,
            prepared.selector_inputs[3],
            work.majorants,
            1.0,
            65536,
            prepared.max_collisions,
            prepared.budget_report,
            prepared.budget_failure,
        ],
        device=device,
    )
    np.testing.assert_array_equal(prepared.budget_report.numpy(), -17.0)
    np.testing.assert_array_equal(prepared.budget_failure.numpy(), [4])


@pytest.mark.parametrize(
    "shape,match",
    [
        ((1, 2**31, 1), "slots.*int32"),
        ((2**31, 1, 1), "boxes.*int32"),
        ((1, 1, 2**31), "species.*int32"),
        ((2**30, 2**30, 2), "byte size"),
    ],
)
def test_layout_overflow_rejects_without_device_allocation(shape, match):
    """Metadata-only checks reject index/byte overflow without huge buffers."""
    from particula.gpu.kernels._coagulation_budget import (
        validate_coagulation_layout,
    )

    with pytest.raises(ValueError, match=match):
        validate_coagulation_layout(*shape)


def test_layout_keeps_large_addressable_populations():
    """Index guards impose no former 256 or 16384 population ceiling."""
    from particula.gpu.kernels._coagulation_budget import (
        validate_coagulation_layout,
    )

    validate_coagulation_layout(3, 32768, 2)
    validate_coagulation_layout(1, 2**31 - 1, 1)


def test_preflight_counts_all_bound_workspace_bytes(device):
    """Independent shape arithmetic includes RNG and normalized environment."""
    from particula.gpu.kernels.coagulation import (
        CoagulationMechanismConfig,
        coagulation_preflight_gpu,
    )

    _, prepared = prepared_case(device, 13, slots=33)
    report = coagulation_preflight_gpu(
        prepared.particles,
        283.15,
        101325.0,
        1e-5,
        mechanism_config=CoagulationMechanismConfig(
            ("brownian", "sedimentation_sp2016", "turbulent_shear_st1956")
        ),
        turbulent_dissipation=1e-4,
        fluid_density=1.2,
        max_collisions=16,
    )
    boxes, slots, capacity = 3, 33, 16
    lanes = (slots * slots + 31) // 32
    # Six fp64 properties + two int32 active arrays per slot, pair output,
    # partial maxima, and per-box scalar storage, plus one failure word.
    expected_bytes = boxes * (56 * slots + 8 * capacity + 8 * lanes + 96) + 4
    assert report["workspace_and_collision_bytes"] == expected_bytes


def test_executed_work_counters_preserve_rng_and_results(device):
    """Optional counters observe actual work without changing random draws."""
    from particula.gpu import to_warp_particle_data
    from particula.gpu.kernels.coagulation import (
        coagulation_preflight_gpu,
        coagulation_step_gpu,
    )

    cpu = population(33, 0.75)
    left = to_warp_particle_data(cpu, device=device)
    right = to_warp_particle_data(cpu, device=device)
    report = coagulation_preflight_gpu(left, 283.15, 101325.0, 1.0)
    dt = 0.9 * float(np.min(report["max_time_step"]))
    counts = wp.full((3, 2), -1, dtype=wp.int32, device=device)
    rngs = [wp.zeros(3, dtype=wp.uint32, device=device) for _ in range(2)]
    accepted = wp.zeros(3, dtype=wp.int32, device=device)
    for index, particle in enumerate((left, right)):
        coagulation_step_gpu(
            particle,
            283.15,
            101325.0,
            dt,
            rng_states=rngs[index],
            initialize_rng=True,
            rng_seed=52,
            work_counts=counts if index == 0 else None,
            n_collisions=accepted if index == 0 else None,
        )
    observed = counts.numpy()
    np.testing.assert_array_equal(observed[:, 0], observed[:, 1])
    assert observed[:, 1].sum() > 0
    assert np.all(accepted.numpy() <= observed[:, 1])
    for field in ("masses", "concentration", "charge"):
        np.testing.assert_array_equal(
            getattr(left, field).numpy(), getattr(right, field).numpy()
        )
    np.testing.assert_array_equal(rngs[0].numpy(), rngs[1].numpy())


def test_work_counter_validation(device):
    """Counter shape and ownership are checked before any particle writes."""
    from particula.gpu.kernels.coagulation import coagulation_step_gpu

    _, prepared = prepared_case(device, 1)
    for shape, dtype in (((3, 1), wp.int32), ((3, 2), wp.float64)):
        with pytest.raises(ValueError, match="work_counts must be int32"):
            coagulation_step_gpu(
                prepared.particles,
                283.15,
                101325.0,
                1e-5,
                work_counts=wp.zeros(shape, dtype=dtype, device=device),
            )
    aliased = wp.array(
        ptr=prepared.masses.ptr,
        shape=(3, 2),
        dtype=wp.int32,
        device=device,
        copy=False,
    )
    before = prepared.masses.numpy().copy()
    with pytest.raises(ValueError, match="overlap|alias"):
        coagulation_step_gpu(
            prepared.particles,
            283.15,
            101325.0,
            1e-5,
            work_counts=aliased,
        )
    np.testing.assert_array_equal(prepared.masses.numpy(), before)
