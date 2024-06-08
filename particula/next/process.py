"""RunnableProcess classes for modifying aerosol instances."""

from abc import ABC, abstractmethod
from typing import Any

from particula.next.aerosol import Aerosol


class RunnableProcess(ABC):
    """Runnable process that can modify an aerosol instance."""

    @abstractmethod
    def execute(self, aerosol: Aerosol) -> Aerosol:
        """Execute the process and modify the aerosol instance.

        Parameters:
        - aerosol (Aerosol): The aerosol instance to modify."""

    @abstractmethod
    def rate(self, aerosol: Aerosol) -> float:
        """Return the rate of the process.

        Parameters:
        - aerosol (Aerosol): The aerosol instance to modify."""

    def __or__(self, other):
        """Chain this process with another process using the | operator."""
        if not isinstance(other, RunnableProcess):
            raise TypeError(f"Cannot chain {type(self)} with {type(other)}")

        sequence = ProcessSequence()
        sequence.add_process(self)
        sequence.add_process(other)
        return sequence


class MassCondensation(RunnableProcess):
    """MOCK-UP: Runnable process that modifies an aerosol instance by
    mass condensation."""
    def __init__(self, other_settings: Any):
        self.other_settings = other_settings

    def execute(self, aerosol: Aerosol) -> Aerosol:
        # Perform mass condensation calculations
        # Modify the aerosol instance or return a new one
        aerosol.particle.distribution *= 1.5
        return aerosol  # Placeholder

    def rate(self, aerosol: Aerosol) -> float:
        return 0.5


class MassCoagulation(RunnableProcess):
    """MOCK-UP Runnable process that modifies an aerosol instance by
    mass coagulation.
    """
    def __init__(self, other_setting2: Any):
        self.other_setting2 = other_setting2

    def execute(self, aerosol: Aerosol) -> Aerosol:
        # Perform mass coagulation calculations
        # Modify the aerosol instance or return a new one
        aerosol.particle.distribution *= 0.5
        return aerosol  # Placeholder

    def rate(self, aerosol: Aerosol) -> float:
        return 0.5


class ProcessSequence:
    """A sequence of processes to be executed in order.

    Attributes:
    - processes (List[RunnableProcess]): A list of RunnableProcess objects.

    Methods:
    - add_process: Add a process to the sequence.
    - execute: Execute the sequence of processes on an aerosol instance.
    - __or__: Add a process to the sequence using the | operator.
    """
    def __init__(self):
        self.processes: list[RunnableProcess] = []

    def add_process(self, process: RunnableProcess):
        """Add a process to the sequence."""
        self.processes.append(process)

    def execute(self, aerosol: Aerosol) -> Aerosol:
        """Execute the sequence of processes on an aerosol instance."""
        result = aerosol
        for process in self.processes:
            result = process.execute(result)
        return result

    def __or__(self, process: RunnableProcess):
        """Add a process to the sequence using the | operator."""
        self.add_process(process)
        return self
