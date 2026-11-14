"""Free-ion condensation exclusion without disturbing particle/gas inventories."""

import numpy as np
import pytest

from particula import Aerosol
from particula.dynamics.ions.condensation import IonCondensationExclusion
from particula.execution.tests.prepared_test import condensation_case
from particula.particles import ParticleData
from particula.particles.activity_strategies import ActivityIdealMolar

wp = pytest.importorskip("warp")


def ion_condensation_case(*, latent=False):
    """Three ion-only particles and one aerosol carrying attached ion mass."""
    base, condensation = condensation_case(latent=latent)
    masses = np.zeros((1, 4, 4))
    masses[0, 0, 2] = 2e-25
    masses[0, 1, 3] = 1e-25
    masses[0, 2, 2:] = [2e-25, 1e-25]
    masses[0, 3, 1:3] = [1e-21, 2e-25]
    ion_volume = np.pi / 6 * (0.5e-9) ** 3
    initial = Aerosol(
        particles=ParticleData(
            masses,
            np.ones((1, 4)),
            np.array([[1.0, -1.0, 0.0, 0.0]]),
            np.array([1000.0, 1000.0, 2e-25 / ion_volume, 1e-25 / ion_volume]),
            np.array([1e-12]),
            distribution_type="particle_resolved",
        ),
        gas=base.gas,
        environment=base.environment,
    )
    strategy = condensation.condensation_strategy
    strategy.activity_strategy = ActivityIdealMolar(
        np.array([0.018, 0.046, 0.150, 0.090])
    )
    strategy.surface_strategy.surface_tension = np.array(
        [0.06, 0.08, 0.08, 0.08]
    )
    return initial, condensation


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)]
)
@pytest.mark.parametrize("latent", [False, True])
def test_ion_exclusion_conservation_capture_restart(device, latent):
    """Free ions and products remain exact while aerosol vapor uptake proceeds."""
    if device != "cpu" and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    initial, condensation = ion_condensation_case(latent=latent)
    process = condensation | IonCondensationExclusion(2, 3)
    with process.prepare(initial, device=device, time_step=1e-4) as simulation:
        checkpoint = simulation.checkpoint()
        if device != "cpu":
            simulation.capture()
        simulation.run(3)
        result = simulation.snapshot()
        transfer = simulation._bindings[
            "condensation"
        ].prepared.total_mass_transfer.numpy()
        np.testing.assert_array_equal(transfer[:, :3], 0)
    np.testing.assert_array_equal(
        result.particles.masses[:, :3], initial.particles.masses[:, :3]
    )
    np.testing.assert_array_equal(
        result.particles.masses[..., 1:], initial.particles.masses[..., 1:]
    )
    np.testing.assert_array_equal(
        result.particles.charge, initial.particles.charge
    )
    np.testing.assert_array_equal(
        result.particles.concentration, initial.particles.concentration
    )
    uptake = (
        result.particles.masses[0, 3, 0] - initial.particles.masses[0, 3, 0]
    )
    assert uptake > 0
    gas_loss = (
        initial.gas.concentration[0, 2] - result.gas.concentration[0, 2]
    ) * initial.particles.volume[0]
    np.testing.assert_allclose(uptake, gas_loss, rtol=1e-10, atol=0)
    with process.resume(checkpoint) as restarted:
        restarted.run(3)
        np.testing.assert_array_equal(
            restarted.snapshot().particles.masses, result.particles.masses
        )
        np.testing.assert_array_equal(
            restarted.snapshot().gas.concentration, result.gas.concentration
        )


def test_ion_exclusion_requires_condensation_and_nonvolatile_materials():
    """Misconfigured companion controls reject before resident upload."""
    initial, condensation = ion_condensation_case()
    with pytest.raises(ValueError, match="requires condensation"):
        IonCondensationExclusion(2, 3).prepare(
            initial, device="cpu", time_step=1
        )
    with pytest.raises(ValueError, match="must be nonvolatile"):
        (condensation | IonCondensationExclusion(0, 2)).prepare(
            initial, device="cpu", time_step=1
        )


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)]
)
def test_mixed_material_density_matches_native_cpu_kelvin(device):
    """Physical mixture volume and CPU surface density remain distinct."""
    from particula._cpu_batch import detached

    if device != "cpu" and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    initial, process = condensation_case()
    initial.particles.density[1] = 2000.0
    cpu = detached(initial)
    process.execute(cpu, time_step=1e-6, sub_steps=4)
    with process.prepare(initial, device=device, time_step=1e-6) as simulation:
        simulation.step()
        result = simulation.snapshot()
    np.testing.assert_allclose(
        result.particles.masses, cpu.particles.masses, rtol=2e-12, atol=0
    )
    np.testing.assert_allclose(
        result.gas.concentration, cpu.gas.concentration, rtol=2e-12, atol=0
    )
