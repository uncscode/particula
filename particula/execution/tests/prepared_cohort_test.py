"""Original seed lineage outcomes, merge precedence and exact continuation."""

from dataclasses import replace

import numpy as np
import pytest

from particula import Aerosol
from particula.dynamics import BrownianCoagulationStrategy, Coagulation
from particula.dynamics.tests.native_runnable_test import state
from particula.execution.cohort import SeedCohort
from particula.particles import ParticleData

wp = pytest.importorskip("warp")


def cohort_case(background=False, seed_diameter=3e-9):
    """Two collision partners with separately marked nonvolatile materials."""
    base = state(volume=1e-25)
    masses = np.zeros((1, 2, 3))
    masses[0, 0, 0] = 1000 * np.pi / 6 * seed_diameter**3
    if background:
        masses[0, 1, 1] = 1000 * np.pi / 6 * (100e-9) ** 3
    else:
        masses[0, 1, 0] = masses[0, 0, 0]
    return Aerosol(
        particles=ParticleData(
            masses,
            np.ones((1, 2)),
            np.zeros((1, 2)),
            np.full(3, 1000.0),
            np.array([1e-25]),
            distribution_type="particle_resolved",
        ),
        gas=base.gas,
        environment=base.environment,
    )


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)]
)
@pytest.mark.parametrize(
    "background,seed_diameter,target,expected",
    [
        (False, 3e-9, 3.5e-9, [2, 2, 0, 0]),
        (False, 3e-9, 50e-9, [2, 0, 0, 2]),
        (True, 3e-9, 50e-9, [1, 0, 1, 0]),
        (True, 60e-9, 50e-9, [1, 1, 0, 0]),
    ],
)
def test_original_lineage_partition_and_restart(
    device, background, seed_diameter, target, expected
):
    """Capture takes priority over merger-induced size; prior success persists."""
    if device != "cpu" and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    process = Coagulation(
        BrownianCoagulationStrategy("particle_resolved")
    ) | SeedCohort(0, 1, target)
    initial = cohort_case(background, seed_diameter)
    # One pair: choose a near-unit event probability from the CPU kernel.
    # Keep the required merge and lineage assertions without clipped demand.
    kernel = BrownianCoagulationStrategy("particle_resolved").kernel(
        initial.particles,
        float(initial.environment.temperature[0]),
        float(initial.environment.pressure[0]),
    )[0, 1]
    step = (1 - 1e-8) * initial.particles.volume[0] / kernel
    with process.prepare(
        initial, device=device, time_step=float(step), root_seed=17
    ) as simulation:
        before = simulation.checkpoint()
        if device != "cpu":
            simulation.capture()
        simulation.step()
        result = simulation.snapshot()
        observed = simulation.cohort_diagnostics()
        after = simulation.checkpoint()
        assert np.count_nonzero(result.particles.concentration) == 1
        for key, value in zip(
            ("initial", "reached", "captured", "unresolved"),
            expected,
            strict=True,
        ):
            np.testing.assert_array_equal(observed[key], [value])
        simulation.run(3)
        for key in observed:
            np.testing.assert_array_equal(
                simulation.cohort_diagnostics()[key], observed[key]
            )
    for checkpoint in (before, after):
        with process.resume(checkpoint) as restarted:
            restarted.step()
            for key in observed:
                np.testing.assert_array_equal(
                    restarted.cohort_diagnostics()[key], observed[key]
                )


def test_corrupt_checkpoint_lineage_partition_rejects():
    """A changed outcome count cannot silently redefine the original cohort."""
    process = SeedCohort(0, 1)
    with process.prepare(
        cohort_case(), device="cpu", time_step=1
    ) as simulation:
        checkpoint = simulation.checkpoint()
    auxiliary = []
    for item in checkpoint.auxiliary:
        if item.role == "cohort_statistics":
            values = np.frombuffer(item.data, dtype=item.dtype).copy()
            values[1] += 1
            item = replace(item, data=values.tobytes())
        auxiliary.append(item)
    with pytest.raises(ValueError, match="cohort lineage partition"):
        process.resume(replace(checkpoint, auxiliary=tuple(auxiliary)))


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)]
)
def test_condensation_updates_first_passage_after_process(device):
    """Growth is observed before the next process; labels describe initial state."""
    from particula.execution.tests.prepared_test import condensation_case

    if device != "cpu" and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    initial, condensation = condensation_case()
    initial.particles.concentration[:] = 1
    initial.particles.masses[..., 0] = 0
    volume = (initial.particles.masses / initial.particles.density).sum(axis=2)
    target = float(np.max((6 * volume / np.pi) ** (1 / 3)) * 1.05)
    process = SeedCohort(1, 0, target) | condensation
    with process.prepare(initial, device=device, time_step=1e-4) as simulation:
        np.testing.assert_array_equal(
            simulation.cohort_diagnostics()["unresolved"], [2]
        )
        simulation.run(3)
        result = simulation.snapshot()
        volume = (result.particles.masses / result.particles.density).sum(
            axis=2
        )
        assert np.all((6 * volume / np.pi) ** (1 / 3) > target)
        np.testing.assert_array_equal(
            simulation.cohort_diagnostics()["reached"], [2]
        )
        np.testing.assert_array_equal(
            simulation.cohort_diagnostics()["captured"], [0]
        )


def test_ion_source_cannot_reuse_cohort_markers():
    """Distinct ion lanes prevent labels leaking across slot reuse."""
    from particula.dynamics.ions.source import IonPairSource

    process = IonPairSource(1, 0, 1, 2e-25, 1e-25) | SeedCohort(0, 1)
    with pytest.raises(ValueError, match="distinct from cohort markers"):
        process.prepare(cohort_case(), device="cpu", time_step=1)


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)]
)
def test_evaporated_background_does_not_capture_after_slot_reuse(device):
    """Retired background labels cannot turn a later ion merger into capture."""
    from particula.execution.tests.prepared_ion_condensation_test import (
        ion_condensation_case,
    )
    from particula.gpu.kernels.cohort import merge_cohort, observe_cohort

    if device != "cpu" and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    initial, condensation = ion_condensation_case()
    initial.particles.masses[:] = 0
    initial.particles.masses[0, 0, 0] = 1e-24  # Volatile background.
    initial.particles.masses[0, 1, 1] = 1e-21  # Nonvolatile seed.
    initial.particles.concentration[:] = [[1, 1, 0, 0]]
    initial.particles.charge[:] = 0
    initial.gas.concentration[:, 2] = 0
    process = condensation | SeedCohort(1, 0, target_diameter=1e-6)
    with process.prepare(initial, device=device, time_step=1e-3) as owner:
        owner.step()
        after = owner.snapshot()
        assert after.particles.concentration[0, 0] == 0
        p = owner._session.particles
        pending = owner._auxiliary["cohort_pending"]
        background = owner._auxiliary["cohort_background"]
        statistics = owner._auxiliary["cohort_statistics"]
        # Model a source reusing slot 0 for an ion, then an applied pair with
        # seed recipient 1 and ion donor 0. Physical post-merge state is known.
        after.particles.masses[0, 1, 2] += 2e-25
        wp.copy(
            p.masses,
            wp.array(after.particles.masses, dtype=wp.float64, device=device),
        )
        wp.launch(
            merge_cohort,
            dim=1,
            device=device,
            inputs=[
                wp.array([[[1, 0]]], dtype=wp.int32, device=device),
                wp.array([1], dtype=wp.int32, device=device),
                pending,
                background,
                owner._device_failure,
            ],
        )
        wp.launch(
            observe_cohort,
            dim=1,
            device=device,
            inputs=[
                p.masses,
                p.concentration,
                p.density,
                wp.float64(np.pi / 6 * 1e-18),
                pending,
                background,
                statistics,
                owner._device_failure,
            ],
        )
        np.testing.assert_array_equal(statistics.numpy(), [[1, 0, 0, 1]])
        np.testing.assert_array_equal(background.numpy(), 0)
        np.testing.assert_array_equal(owner._device_failure.numpy(), 0)


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)]
)
def test_evaporated_unresolved_seed_still_faults(device):
    """Clearing retired background metadata must not discard seed lineages."""
    from particula.execution.tests.prepared_test import condensation_case

    if device != "cpu" and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    initial, condensation = condensation_case()
    initial.particles.masses[:] = 0
    initial.particles.masses[..., 0] = 1e-24
    initial.particles.concentration[:] = 1
    initial.gas.concentration[:, 2] = 0
    process = condensation | SeedCohort(0, 1, target_diameter=1e-6)
    with process.prepare(initial, device=device, time_step=1e-3) as owner:
        owner.step()
        with pytest.raises(ValueError, match="cohort"):
            owner.cohort_diagnostics()
