"""
Runnable process utilities for modifying Aerosol instances.

This module provides abstract and concrete classes for processes
that can modify or transform an Aerosol object over a simulation
time step.

Classes:
    - Runnable: Abstract base class requiring 'rate' and 'execute' methods.
    - RunnableSequence: Manages a sequence of Runnable processes.

Examples:
    ```py title="Chaining processes"
    process1 = SomeRunnableProcess()
    process2 = AnotherRunnableProcess()
    chained = process1 | process2
    final_aerosol = chained.execute(aerosol, time_step=1.0)
    ```
"""

from abc import ABC, abstractmethod
from typing import Any

from particula.aerosol import Aerosol


class Runnable(ABC):
    """
    Abstract base class for processes modifying an Aerosol instance.

    This class enforces the implementation of a process rate calculation
    and an execution method that modifies the Aerosol in-place. Subclasses
    must implement both 'rate' and 'execute', which define how the process
    affects the Aerosol over a time step.

    Methods:
        - rate: Calculate and return the rate of the process.
        - execute: Apply the process logic to the Aerosol over a specified
            time.
        - __or__: Chain two processes using the '|' operator.

    Examples:
        ```py title="Defining a Custom Process"
        class CustomProcess(Runnable):
            def rate(self, aerosol):
                return 42

            def execute(self, aerosol, time_step, sub_steps=1):
                # Modify aerosol here
                return aerosol
        ```

    References:
        - No references available yet.
    """

    @abstractmethod
    def rate(self, aerosol: Aerosol) -> Any:
        """
        Calculate and return the rate of this process for the given Aerosol.

        Arguments:
            - aerosol : The Aerosol instance on which to calculate the rate.

        Returns:
            - Any : The computed rate of this process.

        Examples:
            ```py title="Using the rate method"
            process = CustomProcess()
            process_rate = process.rate(my_aerosol)
            print(process_rate)
            ```
        """

    @abstractmethod
    def execute(
        self,
        aerosol: Aerosol,
        time_step: float,
        sub_steps: int = 1,
    ) -> Aerosol:
        """
        Execute the process, modifying the Aerosol in-place over a time step.

        Arguments:
            - aerosol : The Aerosol instance to be updated.
            - time_step : The time step size in seconds.
            - sub_steps : Number of sub-steps to subdivide the time step,
                default 1.

        Returns:
            - The updated Aerosol after this process runs.

        Examples:
            ```py title="Executing the process"
            process = CustomProcess()
            updated_aerosol = process.execute(my_aerosol, time_step=1.0)
            ```
        """

    def __or__(self, other: "Runnable"):
        """
        Chain this Runnable with another using the '|' operator.

        This method enables an easy way to sequence processes, so they
        can be executed in a defined order.

        Arguments:
            - other : Another Runnable to append after this one.

        Returns:
            - RunnableSequence : A sequence containing both processes.

        Examples:
            ```py title="Chaining two processes"
            combined_process = process1 | process2
            final_aerosol = combined_process.execute(aerosol, time_step=1.0)
            ```
        """

        sequence = RunnableSequence()
        sequence.add_process(self)
        sequence.add_process(other)
        return sequence


class RunnableSequence:
    """
    A sequence of Runnable processes executed in order.

    This class maintains a list of processes to be applied sequentially
    to an Aerosol. Each process modifies the Aerosol and passes it along
    to the next in the sequence.

    Attributes:
        - processes : A list of Runnable objects forming the sequence.

    Methods:
        - add_process: Add a Runnable to the sequence.
        - execute: Apply each Runnable in the sequence to an Aerosol.
        - __or__: Chain a new Runnable into this sequence.

    Examples:
        ```py title="Building and running a RunnableSequence"
        sequence = RunnableSequence()
        sequence.add_process(CustomProcess())
        sequence.add_process(AnotherProcess())
        final_aerosol = sequence.execute(aerosol, time_step=2.0)
        ```
    """

    def __init__(self):
        self.processes: list[Runnable] = []

    def add_process(self, process: Runnable):
        """
        Add a Runnable to the sequence.

        Arguments:
            - process : The Runnable to add.

        Examples:
            ```py
            sequence = RunnableSequence()
            sequence.add_process(CustomProcess())
            ```
        """
        self.processes.append(process)

    def execute(self, aerosol: Aerosol, time_step: float) -> Aerosol:
        """
        Execute all processes in the sequence on the given Aerosol.

        Each Runnable in the sequence modifies the Aerosol and passes
        it to the next Runnable until all have been executed.

        Arguments:
            - aerosol : The Aerosol instance to be updated.
            - time_step : The time step size in seconds for each process.

        Returns:
            - Aerosol : The resulting Aerosol after all processes run.

        Examples:
            ```py title="Executing a RunnableSequence"
            sequence = RunnableSequence()
            final_aerosol = sequence.execute(aerosol, time_step=1.0)
            ```
        """
        result = aerosol
        for process in self.processes:
            result = process.execute(result, time_step)
        return result

    def __or__(self, process: Runnable):
        """
        Chain another Runnable into this sequence using the '|' operator.

        Arguments:
            - process : The Runnable to add.

        Returns:
            - RunnableSequence : This sequence with the new Runnable appended.

        Examples:
            ```py
            sequence = RunnableSequence()
            sequence |= CustomProcess()
            # or
            sequence = sequence | AnotherProcess()
            ```
        """
        self.add_process(process)
        return self
