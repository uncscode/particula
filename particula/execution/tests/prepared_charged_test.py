"""Prepared charged model translation, merge invariants and continuation."""

import numpy as np
import pytest

from particula.dynamics import ChargedCoagulationStrategy, Coagulation
from particula.dynamics.coagulation import charged_kernel_strategy as models
from particula.dynamics.tests.native_runnable_test import state
from particula.execution.prepared import prepare_simulation

wp = pytest.importorskip("warp")


@pytest.mark.parametrize(
    "model",
    [
        models.HardSphereKernelStrategy,
        models.CoulombDyachkov2007KernelStrategy,
        models.CoulombGatti2008KernelStrategy,
        models.CoulombGopalakrishnan2012KernelStrategy,
        models.CoulumbChahl2019KernelStrategy,
    ],
)
@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)]
)
def test_prepared_charged_merge_and_restart(model, device):
    """Every translated model conserves inventories and survives a checkpoint."""
    if device != "cpu" and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    initial = state(volume=1e-20)
    initial.particles.charge[:] = [[1, -1]]
    process = Coagulation(
        ChargedCoagulationStrategy("particle_resolved", model())
    )
    with prepare_simulation(
        (process,), initial, device=device, time_step=1, root_seed=17
    ) as simulation:
        expected = process.coagulation_strategy.kernel(
            initial.particles,
            float(initial.environment.temperature[0]),
            float(initial.environment.pressure[0]),
        )[0, 1]
        checkpoint = simulation.checkpoint()
        if device != "cpu":
            simulation.capture()
            assert simulation.completed_steps == 0
        simulation.step()
        result = simulation.snapshot()
        observed = simulation._bindings[
            "coagulation"
        ].brownian_parallel.majorants.numpy()[0]
        np.testing.assert_allclose(observed, expected, rtol=1e-10, atol=0)
    assert np.count_nonzero(result.particles.concentration) == 1
    np.testing.assert_allclose(
        result.particles.masses.sum(axis=1),
        initial.particles.masses.sum(axis=1),
        rtol=1e-14,
    )
    np.testing.assert_array_equal(result.particles.charge.sum(axis=1), 0)
    with process.resume(checkpoint) as resumed:
        resumed.step()
        np.testing.assert_array_equal(
            resumed.snapshot().particles.masses, result.particles.masses
        )
