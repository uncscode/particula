"""Test the Process class.

Build tests when we get default setups for Aerosol, Gas, Particle,
and Process
"""

from unittest.mock import MagicMock

from particula.aerosol import Aerosol
from particula.runnable import RunnableABC, RunnableSequence


class MockRunnable(RunnableABC):
    """A mock Runnable for testing that tracks how many times it is executed."""

    def __init__(self):
        """Initialize the MockRunnable.

        Sets up a counter to track execution times for testing.
        """
        self.times_executed = 0

    def rate(self, aerosol):
        """Return zero rate for testing."""
        return 0

    def execute(self, aerosol, time_step, sub_steps=1):
        """Execute the runnable and increment counter."""
        self.times_executed += 1
        return aerosol


def test_runnable_sequence_execute():
    """Verify that each Runnable in the sequence executes the expected
    number of times.
    """
    seq = RunnableSequence()
    r1 = MockRunnable()
    r2 = MockRunnable()

    seq.add_process(r1)
    seq.add_process(r2)

    mock_aerosol = MagicMock(spec=Aerosol)
    final_aerosol = seq.execute(mock_aerosol, time_step=2.0, sub_steps=2)

    assert r1.times_executed == 2
    assert r2.times_executed == 2
    assert final_aerosol == mock_aerosol
