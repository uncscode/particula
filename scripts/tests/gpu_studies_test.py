"""Scientific controls for the reproducible GPU studies and their CPU baseline."""

import numpy as np
import pytest
from particula._cpu_batch import detached
from scripts.gpu_studies.cases import StudyCase, build_case
from scripts.gpu_studies.validation import native_cpu_steps, validate_evolution


@pytest.mark.cuda
@pytest.mark.parametrize(
    "family", ["dilution", "condensation", "brownian", "full"]
)
@pytest.mark.parametrize(
    "materials,gases,occupancy", [(2, 3, 0.75), (4, 5, 0.25)]
)
def test_study_has_real_physics_and_independent_inventories(
    family, materials, gases, occupancy
):
    """Exercise mixed widths, sparse slots, heterogeneous boxes and source pulse."""
    wp = pytest.importorskip("warp")
    if not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    case = StudyCase(
        "smoke",
        family,
        4,
        128,
        materials,
        gases,
        occupancy,
        time_step=1.0,
        steps=2,
        heterogeneous=True,
    )
    initial, process, processes = build_case(case)
    with process.prepare(
        initial, device="cuda:0", time_step=case.time_step, root_seed=case.seed
    ) as simulation:
        simulation.step()
        if family == "full":
            simulation.synchronize()
            np.testing.assert_array_equal(
                simulation._registry.acquire_nucleation().finalized_demand.accepted_counts.numpy(),
                2,
            )
            gas = simulation.diagnostics().gas_concentration
            gas[:, 1] = 0
            simulation.update_gas(gas)
        simulation.step()
        result = simulation.snapshot()
    validate_evolution(case, initial, result)
    if family in ("dilution", "condensation"):
        reference = native_cpu_steps(case, detached(initial), processes)
        validate_evolution(case, initial, reference)
        for name in ("masses", "concentration", "charge"):
            np.testing.assert_allclose(
                getattr(result.particles, name),
                getattr(reference.particles, name),
                rtol=2e-12,
                atol=1e-30,
            )
        np.testing.assert_allclose(
            result.gas.concentration,
            reference.gas.concentration,
            rtol=2e-12,
            atol=1e-30,
        )


@pytest.mark.cuda
@pytest.mark.parametrize(
    "experiment", ["heterogeneous", "forcing", "stochastic", "lifecycle"]
)
def test_scientific_history_controls_and_exact_lifecycle(experiment):
    """Validate source/sink ledgers and scientific controls before scaling up."""
    wp = pytest.importorskip("warp")
    if not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    from scripts.gpu_studies.trajectories import run_trajectory

    family = (
        "full"
        if experiment == "lifecycle"
        else "brownian"
        if experiment == "stochastic"
        else "condensation"
    )
    case = StudyCase(
        "history-smoke",
        family,
        4,
        128,
        occupancy=0.75,
        time_step=1.0 if family != "condensation" else 0.1,
        steps=2 if family != "condensation" else 20,
        experiment=experiment,
    )
    initial, process, processes = build_case(case)
    result = run_trajectory(case, initial, process, processes)
    if experiment == "lifecycle":
        assert result["capture_exact"] and result["restart_exact"]
        assert result["snapshot_isolated"]
        assert len(result["process_ledger"]) == 5
    elif experiment == "stochastic":
        assert len(result["result_hashes"]) == 16
    else:
        assert len(result["frames"]) == 3
        assert result["max_box_species_relative_error"] < 2e-12


def test_campaign_retains_the_full_required_matrix():
    """Protect workload sizes and each scientific study from accidental pruning."""
    from scripts.gpu_studies.campaign import all_jobs

    jobs = all_jobs()
    names = {case.name for case, _ in jobs}
    assert {f"brownian-{n}" for n in (128, 512, 2048, 8192, 16384)} <= names
    for family in ("condensation", "dilution"):
        assert {
            f"particles-{family}-{n}" for n in (1024, 16384, 131072, 1048576)
        } <= names
    assert {f"boxes-{b}" for b in (1, 8, 64, 512, 1024)} <= names
    assert {
        "time-varying-16",
        "stochastic-16-seeds",
        "million-lifecycle",
    } <= names
    for case, mode in jobs:
        if case.family == "full":
            assert case.boxes * int(case.slots * case.occupancy) > 1_000_000
        if mode == "numpy":
            assert case.family in ("condensation", "dilution")
