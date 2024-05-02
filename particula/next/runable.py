"""RunnableProcess classes for modifying aerosol instances."""

from abc import ABC, abstractmethod

from particula.next.aerosol import Aerosol


class Runnable(ABC):
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

    def execute(self, aerosol: Aerosol) -> Aerosol:
        """Execute the sequence of runnables on an aerosol instance."""
        result = aerosol
        for process in self.processes:
            result = process.execute(result)
        return result

    def __or__(self, process: Runnable):
        """Add a runnable to the sequence using the | operator."""
        self.add_process(process)
        return self
