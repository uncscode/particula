"""RunnableProcess classes for modifying aerosol instances.
"""

from abc import ABC, abstractmethod
from typing import Any

from particula.aerosol import Aerosol


class Runnable(ABC):
    """Runnable process that can modify an aerosol instance.

    Parameters: None

    Methods:
    - rate: Return the rate of the process.
    - execute: Execute the process and modify the aerosol instance.
    - __or__: Chain this process with another process using the | operator.
    """

    @abstractmethod
    def rate(self, aerosol: Aerosol) -> Any:
        """Return the rate of the process.

        Parameters:
        - aerosol (Aerosol): The aerosol instance to modify."""

    @abstractmethod
    def execute(
        self,
        aerosol: Aerosol,
        time_step: float,
        sub_steps: int = 1,
    ) -> Aerosol:
        """Execute the process and modify the aerosol instance.

        Parameters:
            aerosol (Aerosol): The aerosol instance to modify.
            time_step (float): The time step for the process in seconds.
            sub_steps (int): The number of sub-steps to use for the process,
                default is 1. Which means the full time step is used. A value
                of 2 would mean the time step is divided into two sub-steps.
        """

    def __or__(self, other: "Runnable"):
        """Chain this process with another process using the | operator."""

        sequence = RunnableSequence()
        sequence.add_process(self)
        sequence.add_process(other)
        return sequence


class RunnableSequence:
    """A sequence of processes to be executed in order.

    Attributes:
    - processes (List[Runnable]): A list of RunnableProcess objects.

    Methods:
    - add_process: Add a process to the sequence.
    - execute: Execute the sequence of processes on an aerosol instance.
    - __or__: Add a process to the sequence using the | operator.
    """

    def __init__(self):
        self.processes: list[Runnable] = []

    def add_process(self, process: Runnable):
        """Add a process to the sequence."""
        self.processes.append(process)

    def execute(self, aerosol: Aerosol, time_step: float) -> Aerosol:
        """Execute the sequence of runnables on an aerosol instance."""
        result = aerosol
        for process in self.processes:
            result = process.execute(result, time_step)
        return result

    def __or__(self, process: Runnable):
        """Add a runnable to the sequence using the | operator."""
        self.add_process(process)
        return self
