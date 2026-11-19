"""Live work admission, stochastic continuation and atomic outer intervals."""

from dataclasses import replace

import numpy as np
import pytest

from particula.dynamics import Dilution, DilutionStrategy
from particula.execution.tests.particle_fields_test import add_fields
from particula.execution.tests.prepared_cloud_collisions_test import (
    collision,
    material_case,
)
from particula.execution.tests.prepared_restart_capture_test import (
    canonical_equal,
)
from particula.gpu import to_warp_particle_data
from particula.gpu.kernels._coagulation_budget import CoagulationBudgetError
from particula.gpu.kernels.coagulation import (
    CoagulationMechanismConfig,
    CoagulationSubcyclingConfig,
    CoagulationSubcyclingError,
    CoagulationWorkConfig,
    adaptive_coagulation_step_gpu,
    coagulation_step_gpu,
)

wp = pytest.importorskip("warp")
pytestmark = pytest.mark.warp
POLICY = CoagulationSubcyclingConfig(1e-6, 16, 2**24)


@pytest.fixture(params=["cpu", "cuda:0"])
def device(request):
    """Exercise ordinary Warp CPU and available native CUDA."""
    if request.param != "cpu" and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    return request.param


def evolving():
    """Six active slots per box. Seed zero merges a particle in each box."""
    initial, _ = material_case(3)
    initial.particles.volume[:] = 1e-8
    return initial


def test_evolved_state_requires_new_admission(device):
    """An evolved box requires more trials than its disjoint pairs permit."""
    initial = evolving()
    process = collision("combined")
    options = dict(device=device, root_seed=0, time_step=0.1)
    with process.prepare(initial, **options) as fixed:
        fixed.step()
        fixed.synchronize()
        fixed.step()
        with pytest.raises(CoagulationBudgetError, match="permits 2"):
            fixed.synchronize()
    with process.prepare(
        initial, **options, coagulation_subcycling=POLICY
    ) as sim:
        sim.step()
        assert sim.coagulation_diagnostics()["substep_count"] == 1
        sim.step()
        rows = sim.coagulation_diagnostics()
        assert rows["outer_remaining_required_trials"][0, 0] == 3
        assert rows["active_counts"][0, 0] == 5
        assert rows["disjoint_pair_limited"][0, 0]
        assert not rows["buffer_limited"][0, 0]
        assert rows["substep_count"] >= 2
        np.testing.assert_allclose(
            sum(rows["substep_sizes"]), 0.1, rtol=0, atol=8 * np.spacing(0.1)
        )
        assert np.all(rows["required_trials"] <= rows["available_trials"])


def test_composed_rollback_resume_and_graph(device):
    """Restore a complete sequence and compare replay and checkpoint paths."""
    initial = evolving()
    add_fields(initial)
    process = Dilution(DilutionStrategy(0.05)) | collision("combined")
    options = dict(device=device, time_step=0.1, root_seed=0)
    limited = replace(POLICY, max_subcycles=1)
    with process.prepare(
        initial, **options, coagulation_subcycling=limited
    ) as owner:
        if device != "cpu":
            owner.capture()
        owner.step()
        committed = owner.checkpoint()
        last = owner.coagulation_diagnostics()
        with pytest.raises(
            CoagulationSubcyclingError, match="max_subcycles"
        ) as exc:
            owner.run(2)
        assert exc.value.diagnostics["substep_count"] == 1
        assert exc.value.diagnostics["rolled_back"]
        saved = owner.checkpoint()
        canonical_equal(saved, committed)
        assert saved.field_payloads == committed.field_payloads
        np.testing.assert_array_equal(
            owner.coagulation_diagnostics()["scheduled_trials"],
            last["scheduled_trials"],
        )
        if device != "cpu":
            assert owner.captured
    with process.resume(saved, retry_subcycling=POLICY) as retry:
        assert not retry.captured
        if device != "cpu":
            retry.capture()
        retry.step()
        assert (
            retry.coagulation_diagnostics()["policy_change"]["previous"][1] == 1
        )
        midway = retry.checkpoint()
        retry.step()
        final = retry.checkpoint()
    with process.resume(midway) as resumed:
        resumed.step()
        canonical_equal(final, resumed.checkpoint())
    with process.prepare(
        initial, **options, coagulation_subcycling=POLICY
    ) as ordinary:
        ordinary.run(3)
        baseline = ordinary.checkpoint()
        canonical_equal(final, baseline)
        assert final.field_payloads == baseline.field_payloads


@pytest.mark.parametrize(
    "trials,capacity,controlling",
    [
        (1, 3, "trial_limited"),
        (32, 1, "buffer_limited"),
        (1, 1, "trial_limited"),
    ],
)
def test_direct_replays_ordered_pairs_and_rng(
    device, trials, capacity, controlling
):
    """Compare each pair boundary with manually replayed fixed calls."""
    initial = evolving().particles
    initial.concentration[1] = 0
    initial.masses[1] = 0
    initial.charge[1] = 0
    direct = to_warp_particle_data(initial, device=device)
    reference = to_warp_particle_data(initial, device=device)
    rng = wp.array([513, 107], dtype=wp.uint32, device=device)
    other_rng = wp.clone(rng)
    work = CoagulationWorkConfig(trials, 16 * capacity)
    options = dict(
        temperature=283.15,
        pressure=101325.0,
        work_config=work,
        mechanism_config=CoagulationMechanismConfig(
            ("brownian", "sedimentation_sp2016")
        ),
    )
    result = adaptive_coagulation_step_gpu(
        direct, time_step=0.2, rng_states=rng, subcycling=POLICY, **options
    )
    diag = result.diagnostics
    assert diag[controlling][0, 0]
    assert diag["buffer_limited"][0, 0] == (capacity == 1)
    assert diag["trial_limited"][0, 0] == (trials == 1)
    assert diag["substep_count"] > 1
    begin = 0
    for k, dt in enumerate(diag["substep_sizes"]):
        _, pairs, count = coagulation_step_gpu(
            reference,
            time_step=dt,
            max_collisions=capacity,
            rng_states=other_rng,
            **options,
        )
        end = diag["pair_end_offsets"][k, 0]
        assert count.numpy()[0] == end - begin
        np.testing.assert_array_equal(
            result.collision_pairs.numpy()[0, begin:end],
            pairs.numpy()[0, : end - begin],
        )
        begin = end
    for name in ("masses", "charge", "concentration"):
        np.testing.assert_array_equal(
            getattr(reference, name).numpy(), getattr(direct, name).numpy()
        )
    np.testing.assert_array_equal(rng.numpy(), other_rng.numpy())
    assert rng.numpy()[1] == 107
    assert (
        np.count_nonzero(initial.concentration)
        - np.count_nonzero(direct.concentration.numpy())
        == begin
    )
    counts = direct.concentration.numpy()
    np.testing.assert_allclose(
        (direct.masses.numpy() * counts[..., None]).sum(axis=1),
        (initial.masses * initial.concentration[..., None]).sum(axis=1),
        rtol=2e-15,
        atol=0,
    )
    np.testing.assert_array_equal(
        (direct.charge.numpy() * counts).sum(axis=1),
        (initial.charge * initial.concentration).sum(axis=1),
    )


def test_reject_before_allocation_and_minimum(device, monkeypatch):
    """Reject a short reservation before upload and preserve rejected steps."""
    from particula.execution.prepared import PreparedSimulation

    initial = evolving()
    process = collision("combined")
    with monkeypatch.context() as patch:
        patch.setattr(
            PreparedSimulation,
            "_upload",
            lambda *a, **k: pytest.fail("uploaded before admission"),
        )
        with pytest.raises(MemoryError, match="max_device_bytes"):
            process.prepare(
                initial,
                device=device,
                time_step=0.1,
                coagulation_subcycling=replace(POLICY, max_device_bytes=10),
            )
    limited = replace(POLICY, min_substep=0.09)
    with process.prepare(
        initial, device=device, time_step=0.1, coagulation_subcycling=limited
    ) as sim:
        sim.step()
        saved = sim.checkpoint()
        with pytest.raises(CoagulationSubcyclingError, match="min_substep"):
            sim.step()
        canonical_equal(saved, sim.checkpoint())
        with pytest.raises(ValueError, match="relax"):
            process.resume(
                saved, retry_subcycling=replace(limited, min_substep=0.2)
            )
        (plan,) = saved.processes
        values = plan.values
        values["subcycling"] = (2, *values["subcycling"][1:])
        corrupt = replace(
            saved, processes=(replace(plan, parameters=tuple(values.items())),)
        )
        with pytest.raises(ValueError, match="incompatible"):
            process.resume(corrupt)


def test_fixed_refinement_ensemble():
    """Across independent seeds compare mean merger count and radius moment.

    A predeclared four-standard-error band combines the independent samples.
    The third moment is already constrained by mass conservation. Use the
    number-weighted second radius moment as a separate distribution statistic.
    """
    initial = evolving()
    outcomes = []
    for dt, adaptive, seed_offset in ((0.1, POLICY, 0), (0.0125, None, 1000)):
        observations = []
        for seed in range(32):
            with collision("combined").prepare(
                initial,
                device="cpu",
                time_step=dt,
                root_seed=seed_offset + seed,
                coagulation_subcycling=adaptive,
            ) as sim:
                sim.run(round(0.2 / dt))
                final = sim.snapshot().particles
                live = final.concentration > 0
                observations.append(
                    [12 - live.sum(), np.mean(final.radii[live] ** 2)]
                )
        outcomes.append(np.asarray(observations))
    first, second = outcomes
    se = np.sqrt(
        (first.var(axis=0, ddof=1) + second.var(axis=0, ddof=1)) / len(first)
    )
    assert np.all(np.abs(first.mean(axis=0) - second.mean(axis=0)) <= 4 * se)


@pytest.mark.parametrize(
    "rate,expected,code",
    [
        (1.0, 1.0, 0),
        (np.nextafter(1.0, 2.0), 0.5, 0),
        (np.inf, 0.0, 12),
    ],
)
def test_exact_admission_boundary(device, rate, expected, code):
    """An expected demand one ULP above capacity must subdivide."""
    from particula.gpu.kernels._coagulation_subcycling import _choose

    def array(values, dtype=wp.float64):
        return wp.array(values, dtype=dtype, device=device)

    dt = array([0])
    failure = array([0], wp.int32)
    args = [
        array([2], wp.int32),
        array([[0, 1]], wp.int32),
        array([[1, 1]]),
        array([1]),
        array([rate]),
        1,
        1,
        wp.float64(0.01),
        array([1]),
        dt,
        array([0], wp.int32),
        wp.zeros((1, 1, 10), dtype=wp.float64, device=device),
        wp.zeros((1, 4), dtype=wp.float64, device=device),
        failure,
    ]
    wp.launch(_choose, dim=1, inputs=args, device=device)
    assert failure.numpy()[0] == code
    assert dt.numpy()[0] == expected


def test_direct_partial_rejection_restores_supplied_outputs(device):
    """One tentative pass may merge and consume draws before outer rollback."""
    initial = evolving().particles
    particles = to_warp_particle_data(initial, device=device)
    rng = wp.array([19, 123], dtype=wp.uint32, device=device)
    pairs = wp.full((2, 3, 2), -7, dtype=wp.int32, device=device)
    counts = wp.full(2, 77, dtype=wp.int32, device=device)
    work = wp.full((2, 2), -8, dtype=wp.int32, device=device)
    arrays = [
        particles.masses,
        particles.charge,
        particles.concentration,
        rng,
        pairs,
        counts,
        work,
    ]
    before = [a.numpy().copy() for a in arrays]
    with pytest.raises(CoagulationSubcyclingError) as caught:
        adaptive_coagulation_step_gpu(
            particles,
            283.15,
            101325.0,
            0.4,
            subcycling=replace(POLICY, max_subcycles=1),
            rng_states=rng,
            initialize_rng=True,
            collision_pairs=pairs,
            n_collisions=counts,
            work_counts=work,
            mechanism_config=CoagulationMechanismConfig(
                ("brownian", "sedimentation_sp2016")
            ),
        )
    assert caught.value.diagnostics["substep_count"] == 1
    for a, old in zip(arrays, before, strict=True):
        np.testing.assert_array_equal(a.numpy(), old)


def test_cohort_observes_each_accepted_pass(device):
    """A first-pass merger precedes a zero-demand pass in the same interval."""
    from particula.execution.cohort import SeedCohort
    from particula.execution.tests.prepared_cohort_test import cohort_case

    initial = cohort_case(background=True)
    process = collision("brownian")
    kernel = process.coagulation_strategy.kernel(
        initial.particles,
        float(initial.environment.temperature[0]),
        float(initial.environment.pressure[0]),
    )[0, 1]
    dt = 2 * (1 - 1e-8) * initial.particles.volume[0] / kernel
    process = process | SeedCohort(0, 1, 50e-9)
    with process.prepare(
        initial,
        device=device,
        root_seed=17,
        time_step=dt,
        coagulation_subcycling=replace(POLICY, min_substep=dt / 4),
    ) as sim:
        sim.step()
        rows = sim.coagulation_diagnostics(include_pairs=True)
        np.testing.assert_array_equal(rows["accepted_collisions"][:, 0], [1, 0])
        np.testing.assert_array_equal(sim.cohort_diagnostics()["captured"], [1])
