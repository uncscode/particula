"""Native CUDA pair-rate, graph replay, and checkpoint qualification."""

import numpy as np
import pytest

from particula.dynamics import Coagulation, VanDerWaalsCoagulationStrategy
from particula.dynamics.tests.native_runnable_test import state
from particula.execution.prepared import prepare_simulation

wp = pytest.importorskip("warp")


@pytest.mark.cuda
@pytest.mark.parametrize("charge", [(1, -1), (0, 1), (1, 1), (0, 0)])
def test_dielectric_cuda_capture_and_restart(charge):
    """An actual unequal-size pair sets admission and merges identically on replay."""
    if not wp.is_cuda_available():
        pytest.skip("requires native CUDA")
    initial = state(volume=1e-20)
    initial.particles.charge[:] = [charge]
    strategy = VanDerWaalsCoagulationStrategy("particle_resolved", 6.4e-20, 5.9)
    process = Coagulation(strategy)
    rate = strategy.kernel(
        initial.particles,
        float(initial.environment.temperature[0]),
        float(initial.environment.pressure[0]),
    )[0, 1]
    dt = (
        0.999999
        * initial.particles.volume[0]
        / (rate * initial.particles.concentration[0, 0])
    )
    with prepare_simulation(
        (process,), initial, device="cuda:0", time_step=float(dt), root_seed=17
    ) as sim:
        checkpoint = sim.checkpoint()
        sim.capture()
        assert sim.completed_steps == 0
        sim.step()
        result = sim.snapshot().particles
        bound = sim._bindings[
            "coagulation"
        ].brownian_parallel.majorants.numpy()[0]
        np.testing.assert_allclose(bound, rate, rtol=1e-9, atol=0)
        after_merge = sim.checkpoint()
        sim.step()
        continued = sim.snapshot().particles
    assert np.count_nonzero(result.concentration) == 1
    np.testing.assert_allclose(
        result.masses.sum(axis=1),
        initial.particles.masses.sum(axis=1),
        rtol=1e-14,
        atol=0,
    )
    assert np.sum(result.charge) == sum(charge)
    for saved, expected in [(checkpoint, result), (after_merge, continued)]:
        with process.resume(saved) as resumed:
            resumed.step()
            actual = resumed.snapshot().particles
            for name in ("masses", "charge", "concentration"):
                np.testing.assert_array_equal(
                    getattr(actual, name), getattr(expected, name)
                )


@pytest.mark.cuda
def test_direct_domain_rejection_preserves_state_and_rng():
    """A strong unlike pair fails admission before RNG or state changes."""
    from particula.gpu.conversion import to_warp_particle_data
    from particula.gpu.kernels.coagulation import (
        CoagulationMechanismConfig,
        coagulation_step_gpu,
    )

    if not wp.is_cuda_available():
        pytest.skip("requires native CUDA")
    initial = state()
    initial.particles.charge[:] = [[3, -3]]
    device = to_warp_particle_data(initial.particles, device="cuda:0")
    config = CoagulationMechanismConfig(
        mechanisms=("van_der_waals",),
        hamaker_constant=6.4e-20,
        relative_permittivity=5.9,
    )
    rng = wp.array([173], dtype=wp.uint32, device="cuda:0")
    before = [
        field.numpy()
        for field in (device.masses, device.concentration, device.charge, rng)
    ]
    with pytest.raises(ValueError, match="capture table domain"):
        coagulation_step_gpu(
            device,
            298.0,
            101325.0,
            1e-6,
            mechanism_config=config,
            rng_states=rng,
            initialize_rng=True,
            max_collisions=1,
        )
    for field, expected in zip(
        (device.masses, device.concentration, device.charge, rng),
        before,
        strict=True,
    ):
        np.testing.assert_array_equal(field.numpy(), expected)


@pytest.mark.cuda
def test_dispersion_settling_majorant_and_material_restart():
    """The summed pair hazard drives admission and freezes material identity."""
    from particula.dynamics import (
        CombineCoagulationStrategy,
        SedimentationCoagulationStrategy,
    )

    if not wp.is_cuda_available():
        pytest.skip("requires native CUDA")
    initial = state()
    kind = "particle_resolved"
    material = VanDerWaalsCoagulationStrategy(kind, 6.4e-20, 5.9)
    strategy = CombineCoagulationStrategy(
        [material, SedimentationCoagulationStrategy(kind)]
    )
    process = Coagulation(strategy)
    expected = strategy.kernel(
        initial.particles,
        float(initial.environment.temperature[0]),
        float(initial.environment.pressure[0]),
    )[0, 1]
    with process.prepare(
        initial, device="cuda:0", time_step=1e-9, root_seed=771
    ) as simulation:
        simulation.step()
        simulation.synchronize()
        actual = simulation._bindings[
            "coagulation"
        ].brownian_parallel.majorants.numpy()[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=0)
        checkpoint = simulation.checkpoint()
    material.relative_permittivity = 6.0
    with pytest.raises(ValueError):
        process.resume(checkpoint)
