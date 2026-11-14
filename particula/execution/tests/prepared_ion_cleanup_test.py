"""Selective neutral product export, inventory accounting and lifecycle checks."""

import numpy as np
import pytest

from particula import Aerosol
from particula.dynamics.ions.cleanup import IonNeutralCleanup
from particula.dynamics.tests.native_runnable_test import state
from particula.particles import ParticleData

wp = pytest.importorskip("warp")


def cleanup_case():
    """Mix products, neutral aerosols, charged ions and free slots."""
    base = state(volume=1e-6)
    masses = np.array(
        [
            [
                [2e-25, 1e-25, 0],  # removable neutral ion product
                [2e-25, 1e-25, 1e-22],  # neutralized aerosol: must remain
                [2e-25, 0, 0],  # positive free ion
                [0, 1e-25, 0],  # negative free ion
                [2e-25, 1e-25, 0],  # charged ion-only cluster: must remain
                [0, 0, 0],  # already free
            ]
        ]
    )
    return Aerosol(
        particles=ParticleData(
            masses,
            np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 0.0]]),
            np.array([[0.0, 0.0, 1.0, -1.0, 1.0, 0.0]]),
            np.full(3, 1000.0),
            np.array([1e-6]),
            distribution_type="particle_resolved",
        ),
        gas=base.gas,
        environment=base.environment,
    )


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)]
)
def test_cleanup_inventory_capture_and_restart(device):
    """Only the neutral product leaves; ledger and state continue exactly."""
    if device != "cpu" and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    initial = cleanup_case()
    process = IonNeutralCleanup(0, 1)
    with process.prepare(initial, device=device, time_step=1) as simulation:
        if device != "cpu":
            simulation.capture()
        simulation.step()
        result = simulation.snapshot()
        ledger = simulation.ion_cleanup_diagnostics()
        checkpoint = simulation.checkpoint()
        simulation.run(2)
        np.testing.assert_array_equal(
            simulation.ion_cleanup_diagnostics()["exported_particles"], [1]
        )
    np.testing.assert_array_equal(
        result.particles.masses[:, 1:], initial.particles.masses[:, 1:]
    )
    np.testing.assert_array_equal(
        result.particles.charge, initial.particles.charge
    )
    assert not result.particles.masses[:, 0].any()
    assert result.particles.concentration[0, 0] == 0
    np.testing.assert_array_equal(ledger["exported_particles"], [1])
    np.testing.assert_array_equal(ledger["exported_mass"], [[2e-25, 1e-25, 0]])
    np.testing.assert_allclose(
        result.particles.masses.sum(axis=1) + ledger["exported_mass"],
        initial.particles.masses.sum(axis=1),
        rtol=1e-14,
        atol=0,
    )
    with process.resume(checkpoint) as restarted:
        restarted.run(2)
        np.testing.assert_array_equal(
            restarted.snapshot().particles.masses, result.particles.masses
        )
        np.testing.assert_array_equal(
            restarted.ion_cleanup_diagnostics()["exported_mass"],
            ledger["exported_mass"],
        )


@pytest.mark.parametrize("indices", [(0, 0), (-1, 1), (0, 3), (True, 1)])
def test_cleanup_rejects_invalid_material_map(indices):
    """Reserved material lanes must be distinct valid integer indices."""
    with pytest.raises(ValueError, match="species|distinct"):
        IonNeutralCleanup(*indices).prepare(
            cleanup_case(), device="cpu", time_step=1
        )


def test_cleanup_rejects_evolving_fractional_counts():
    """A fractional-count box faults without exporting its neutral product."""
    from particula.gpu.kernels.ion_cleanup import remove_neutral_ion_products

    initial = cleanup_case().particles
    counts = initial.concentration.copy()
    counts[0, 2] = 0.5
    masses_gpu = wp.array(initial.masses, dtype=wp.float64, device="cpu")
    counts_gpu = wp.array(counts, dtype=wp.float64, device="cpu")
    charges_gpu = wp.array(initial.charge, dtype=wp.float64, device="cpu")
    exported_mass = wp.zeros((1, 3), dtype=wp.float64, device="cpu")
    exported_particles = wp.zeros(1, dtype=wp.int64, device="cpu")
    failure = wp.zeros(1, dtype=wp.int32, device="cpu")
    wp.launch(
        remove_neutral_ion_products,
        dim=1,
        inputs=[
            masses_gpu,
            counts_gpu,
            charges_gpu,
            0,
            1,
            exported_mass,
            exported_particles,
            failure,
            wp.int64(1),
            wp.zeros(1, dtype=wp.int64, device="cpu"),
        ],
        device="cpu",
    )
    assert failure.numpy()[0] == 5
    np.testing.assert_array_equal(masses_gpu.numpy(), initial.masses)
    np.testing.assert_array_equal(counts_gpu.numpy(), counts)
    assert not exported_mass.numpy().any()
    assert not exported_particles.numpy().any()


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)]
)
def test_periodic_cleanup_preserves_phase_across_capture_restart(device):
    """Products remain until the third call, including after a midperiod restart."""
    if device != "cpu" and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    initial = cleanup_case()
    process = IonNeutralCleanup(0, 1, interval_steps=3)
    with process.prepare(initial, device=device, time_step=0.5) as simulation:
        simulation.step()
        checkpoint = simulation.checkpoint()
        if device != "cpu":
            simulation.capture()
        simulation.step()
        np.testing.assert_array_equal(
            simulation.snapshot().particles.masses, initial.particles.masses
        )
        np.testing.assert_array_equal(
            simulation.ion_cleanup_diagnostics()["exported_particles"], [0]
        )
        simulation.step()
        result = simulation.snapshot()
        ledger = simulation.ion_cleanup_diagnostics()
        np.testing.assert_array_equal(ledger["exported_particles"], [1])
        assert not result.particles.masses[:, 0].any()
        simulation.run(3)
        np.testing.assert_array_equal(
            simulation.ion_cleanup_diagnostics()["exported_mass"],
            ledger["exported_mass"],
        )
    with process.resume(checkpoint) as simulation:
        simulation.step()
        np.testing.assert_array_equal(
            simulation.snapshot().particles.masses, initial.particles.masses
        )
        simulation.step()
        np.testing.assert_array_equal(
            simulation.snapshot().particles.masses, result.particles.masses
        )
        np.testing.assert_array_equal(
            simulation.ion_cleanup_diagnostics()["exported_mass"],
            ledger["exported_mass"],
        )
    with pytest.raises(ValueError, match="configuration/order"):
        IonNeutralCleanup(0, 1, interval_steps=2).resume(checkpoint)


@pytest.mark.parametrize("interval", [0, -1, True, 1.5, 2**63])
def test_cleanup_interval_rejects_invalid_values(interval):
    """The periodic clock requires a positive exactly representable integer."""
    with pytest.raises(ValueError, match="interval_steps"):
        IonNeutralCleanup(0, 1, interval).prepare(
            cleanup_case(), device="cpu", time_step=1
        )


@pytest.mark.parametrize("phase", [-1, 3])
def test_cleanup_checkpoint_rejects_out_of_range_phase(phase):
    """Restart cannot admit a negative or overdue cleanup phase."""
    from dataclasses import replace

    process = IonNeutralCleanup(0, 1, interval_steps=3)
    with process.prepare(
        cleanup_case(), device="cpu", time_step=1
    ) as simulation:
        checkpoint = simulation.checkpoint()
    auxiliary = tuple(
        replace(item, data=np.array([phase], dtype="<i8").tobytes())
        if item.role == "ion_cleanup_phase"
        else item
        for item in checkpoint.auxiliary
    )
    with pytest.raises(ValueError, match="cleanup phase"):
        process.resume(replace(checkpoint, auxiliary=auxiliary))
