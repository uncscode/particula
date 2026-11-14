"""Prepared ion production with exact restart, capture and source ledgers."""

import numpy as np
import pytest

from particula import Aerosol
from particula.dynamics.ions.source import IonPairSource
from particula.dynamics.tests.native_runnable_test import state
from particula.particles import ParticleData

wp = pytest.importorskip("warp")


def source_case(capacity=128):
    """Construct a closed empty population receiving external ion mass."""
    base = state(volume=1e-6)
    aerosol = Aerosol(
        particles=ParticleData(
            np.zeros((1, capacity, 2)),
            np.zeros((1, capacity)),
            np.zeros((1, capacity)),
            np.array([1000.0, 1000.0]),
            np.array([1e-6]),
            distribution_type="particle_resolved",
        ),
        gas=base.gas,
        environment=base.environment,
    )
    return aerosol, IonPairSource(1e7, 0, 1, 2e-25, 1e-25)


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)]
)
def test_prepared_source_restart_and_ledger(device):
    """Source draws continue exactly and ledger accounts for all added mass."""
    if device != "cpu" and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    initial, source = source_case()
    with source.prepare(
        initial, device=device, time_step=0.1, root_seed=7
    ) as simulation:
        simulation.run(3)
        checkpoint = simulation.checkpoint()
        if device != "cpu":
            simulation.capture()
        simulation.run(4)
        final = simulation.snapshot()
        ledger = simulation.ion_source_diagnostics()
    with source.resume(checkpoint) as restarted:
        restarted.run(4)
        np.testing.assert_array_equal(
            restarted.snapshot().particles.masses, final.particles.masses
        )
        np.testing.assert_array_equal(
            restarted.ion_source_diagnostics()["injected_pairs"],
            ledger["injected_pairs"],
        )
    np.testing.assert_allclose(
        final.particles.masses.sum(axis=1),
        ledger["injected_mass"],
        rtol=1e-14,
        atol=0,
    )
    np.testing.assert_array_equal(final.particles.charge.sum(axis=1), 0)
    np.testing.assert_array_equal(
        final.particles.concentration.sum(axis=1), 2 * ledger["injected_pairs"]
    )
    assert ledger["injected_pairs"].sum() > 0


def test_prepared_source_exhaustion_faults_owner():
    """Capacity failures surface on observation and prevent further steps."""
    initial, source = source_case(capacity=1)
    with source.prepare(initial, device="cpu", time_step=10) as simulation:
        simulation.step()
        with pytest.raises(ValueError, match="capacity"):
            simulation.snapshot()
        with pytest.raises(RuntimeError, match="faulted"):
            simulation.step()


def test_source_composes_with_charged_coagulation():
    """Unit-weight source and charged merges retain the external mass budget."""
    from particula.dynamics import ChargedCoagulationStrategy, Coagulation
    from particula.dynamics.coagulation.charged_kernel_strategy import (
        HardSphereKernelStrategy,
    )

    initial, source = source_case()
    process = source | Coagulation(
        ChargedCoagulationStrategy(
            "particle_resolved", HardSphereKernelStrategy()
        )
    )
    with process.prepare(initial, device="cpu", time_step=0.1) as simulation:
        simulation.run(10)
        result = simulation.snapshot()
        ledger = simulation.ion_source_diagnostics()
    np.testing.assert_allclose(
        result.particles.masses.sum(axis=1),
        ledger["injected_mass"],
        rtol=1e-14,
        atol=0,
    )
    np.testing.assert_array_equal(result.particles.charge.sum(axis=1), 0)


def test_source_zero_rate_preserves_random_state():
    """Zero source work does not advance its checkpointed stream."""
    from dataclasses import replace

    initial, source = source_case()
    source = replace(source, production_rate=0)
    with source.prepare(initial, device="cpu", time_step=1) as simulation:
        before = simulation.checkpoint().auxiliary
        simulation.run(3)
        after = simulation.checkpoint().auxiliary
        assert before == after
        assert not simulation.snapshot().particles.concentration.any()
