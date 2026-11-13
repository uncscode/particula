"""Exercise the shared runnable transaction independently of physics."""

import numpy as np
import pytest
from particula import Aerosol
from particula.gas import EnvironmentData, GasData
from particula.particles import create_resolved_particle_data
from particula.runnable import RunnableABC


def state():
    """Construct two distinct boxes with unequal represented volumes."""
    return Aerosol(
        particles=create_resolved_particle_data(
            masses=np.full((2, 3, 1), 1e-18),
            density=1000.0,
            volume=np.array([0.25, 4.0]),
        ),
        gas=GasData(["water"], np.array([0.018]), np.ones((2, 1)), [True]),
        environment=EnvironmentData([280, 300], [101325, 101325], [[1], [1]]),
    )


class Process(RunnableABC):
    """Simple hook process with an optional late-box failure."""

    fail = False

    def _writer_fields(self, aerosol):
        return ("particles.concentration",)

    def _rate_box(self, aerosol):
        return aerosol.particles.concentration[0]

    def _execute_box(self, aerosol, time_step, sub_steps=1):
        aerosol.particles.concentration *= np.exp(-time_step)
        if self.fail and aerosol.environment.temperature[0] > 290:
            raise ValueError("late box failure")
        return aerosol


def test_shared_driver_identity_shapes_and_box_independence():
    """Rates keep B and execution publishes into original arrays."""
    aerosol = state()
    target = aerosol.particles.concentration
    process = Process()
    assert process.rate(aerosol).shape == (2, 3)
    assert process.execute(aerosol, 1, sub_steps=2) is aerosol
    assert aerosol.particles.concentration is target
    np.testing.assert_allclose(target, np.exp(-1))


def test_later_box_failure_preserves_whole_batch():
    """An earlier staged mutation cannot escape a rejected substep."""
    aerosol = state()
    process = Process()
    process.fail = True
    with pytest.raises(ValueError, match="late box"):
        process.execute(aerosol, 1)
    np.testing.assert_array_equal(aerosol.particles.concentration, 1)


def test_original_readonly_storage_rejects_before_staging():
    """Copies cannot conceal unwritable original destinations."""
    aerosol = state()
    aerosol.particles.concentration.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        Process().execute(aerosol, 1)


def test_zero_step_never_publishes_hook_writes():
    """A successful zero-duration call retains the original values."""
    aerosol = state()
    Process().execute(aerosol, 0)
    np.testing.assert_array_equal(aerosol.particles.concentration, 1)
