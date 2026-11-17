"""Native collision-law admission and resident cloud continuation."""

import numpy as np
import pytest

from particula.dynamics import (
    BrownianCoagulationStrategy,
    Coagulation,
    CombineCoagulationStrategy,
    SedimentationCoagulationStrategy,
    TurbulentShearCoagulationStrategy,
)
from particula.execution.gpu_coupled import GPUCoupledSimulation
from particula.execution.gpu_coupled_restart import resume_coupled
from particula.execution.prepared import prepare_simulation
from particula.execution.tests.gpu_coupled_test import equal_native
from particula.execution.tests.gpu_multimaterial_condensation_test import (
    material_case,
)
from particula.execution.tests.prepared_restart_capture_test import (
    canonical_equal,
)

wp = pytest.importorskip("warp")
pytestmark = pytest.mark.warp


def collision(kind):
    """Native laws used by the biomass cloud branches."""
    brownian = BrownianCoagulationStrategy("particle_resolved")
    sedimentation = SedimentationCoagulationStrategy("particle_resolved")
    shear = TurbulentShearCoagulationStrategy("particle_resolved", 0.1, 1.225)
    return Coagulation(
        {
            "brownian": brownian,
            "sedimentation": sedimentation,
            "combined": CombineCoagulationStrategy([brownian, sedimentation]),
            "shear": shear,
        }[kind]
    )


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("kind", ["sedimentation", "combined", "shear"])
def test_prepared_cloud_kernel_matches_native_material_law(device, kind):
    """Changing material composition retains mean-density settling and shear."""
    if device != "cpu" and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    from particula.dynamics.coagulation.sedimentation_kernel import (
        get_sedimentation_kernel_sp2016,
    )
    from particula.particles.properties.settling_velocity import (
        get_particle_settling_velocity_via_system_state,
    )

    initial, _ = material_case(3)
    process = collision(kind)
    with prepare_simulation(
        (process,), initial, device=device, time_step=1e-6
    ) as prepared:
        prepared.step()
        prepared.synchronize()
        binding = prepared._bindings["coagulation"]
        if kind == "shear":
            # A direct kernel comparison in the existing GPU suite owns ST1956.
            assert int(binding.selector_inputs[28]) == 8
            assert np.all(binding.selector_inputs[20].numpy() == 0.1)
            return
        work = binding.brownian_parallel
        for box in range(2):
            radii = initial.particles.radii[box]
            active = initial.particles.concentration[box] > 0
            mean_density = initial.particles.effective_density[
                box, active
            ].mean()
            expected_velocity = get_particle_settling_velocity_via_system_state(
                particle_radius=radii[active],
                particle_density=mean_density,
                temperature=initial.environment.temperature[box],
                pressure=initial.environment.pressure[box],
            )
            np.testing.assert_allclose(
                binding.selector_inputs[19].numpy()[box, active],
                expected_velocity,
                rtol=5e-14,
            )
            if kind == "sedimentation":
                expected = get_sedimentation_kernel_sp2016(
                    radii[active],
                    expected_velocity,
                    calculate_collision_efficiency=False,
                )
                np.testing.assert_allclose(
                    work.majorants.numpy()[box], np.max(expected), rtol=5e-14
                )


@pytest.mark.parametrize("kind", ["brownian", "combined", "shear"])
def test_material_coupling_restarts_persistent_collision_rng(kind):
    """A checkpoint retains collision stream bytes and CUDA adaptive policy."""
    if not wp.is_cuda_available() or not wp.is_conditional_graph_supported():
        pytest.skip("native conditional CUDA unavailable")
    initial, strategy = material_case(3)
    initial.particles.volume[:] = 1e-15
    # Slow exchange isolates stochastic restart from an extreme reservoir transient.
    strategy.diffusion_coefficient = 1e-12
    events = (collision(kind),)
    with GPUCoupledSimulation(
        initial,
        events,
        strategy,
        device="cuda:0",
        resident=True,
        time_step=1e-5,
        root_seed=317,
    ) as owner:
        owner.capture_events()
        owner.step()
        checkpoint = owner.checkpoint()
        owner.run(3)
        final = owner.snapshot()
        saved_final = owner._events.checkpoint()
        with resume_coupled(
            checkpoint, events, strategy, device="cuda:0"
        ) as restored:
            restored.capture_events()
            restored.run(3)
            equal_native(final, restored.snapshot())
            assert restored.statistics == owner.statistics
            assert restored.coupling_downloads == restored.coupling_uploads == 0
            # Full restart includes the prepared checkpoint's persisted streams.
            canonical_equal(saved_final, restored._events.checkpoint())
