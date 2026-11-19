"""Exercise live collision work admission and restart in a coupled cycle."""

import numpy as np
import pytest
from particula.execution.gpu_coupled import GPUCoupledSimulation
from particula.execution.gpu_coupled_restart import resume_coupled
from particula.execution.tests.gpu_coupled_test import equal_native
from particula.execution.tests.prepared_cloud_collisions_test import (
    collision,
    material_case,
)
from particula.gpu.kernels.coagulation import CoagulationSubcyclingConfig

wp = pytest.importorskip("warp")


@pytest.mark.warp
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_coupled_coagulation_splits_interval_and_resumes(device):
    """Retain full cycle time, collision counts and RNG across a restart."""
    resident = device != "cpu"
    if resident and (
        not wp.is_cuda_available() or not wp.is_conditional_graph_supported()
    ):
        pytest.skip("native conditional CUDA unavailable")
    initial, strategy = material_case(3)
    initial.particles.volume[:] = 1e-8
    strategy.diffusion_coefficient = 1e-14
    events = (collision("combined"),)
    with GPUCoupledSimulation(
        initial,
        events,
        strategy,
        device=device,
        resident=resident,
        time_step=0.1,
        root_seed=0,
        coagulation_subcycling=CoagulationSubcyclingConfig(1e-6, 16, 2**24),
    ) as owner:
        if resident:
            owner.capture_events()
        owner.step()
        before = np.count_nonzero(
            owner.snapshot().particles.concentration, axis=1
        )
        checkpoint = owner.checkpoint()
        owner.step()
        work = owner.coagulation_diagnostics()
        assert work["substep_count"] >= 2
        np.testing.assert_allclose(
            sum(work["substep_sizes"]), 0.1, rtol=0, atol=8 * np.spacing(0.1)
        )
        assert owner.completed_steps == 2 and owner.time == pytest.approx(0.2)
        final = owner.snapshot()
        np.testing.assert_array_equal(
            before - np.count_nonzero(final.particles.concentration, axis=1),
            work["accepted_collisions"].sum(axis=0),
        )
    with resume_coupled(checkpoint, events, strategy, device=device) as resumed:
        if resident:
            resumed.capture_events()
        resumed.step()
        equal_native(resumed.snapshot(), final)
        np.testing.assert_array_equal(
            resumed.coagulation_diagnostics()["substep_sizes"],
            work["substep_sizes"],
        )
