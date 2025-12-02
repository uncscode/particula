"""Runnable process utilities for modifying Aerosol instances.

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

# Check if tqdm is available for progress bars
try:
    from tqdm import tqdm
except ImportError:
    AVAILABLE_TQDM = False
else:
    AVAILABLE_TQDM = True


from particula.aerosol import Aerosol


class RunnableABC(ABC):
    """Abstract base class for processes modifying an Aerosol instance.

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
        """Calculate and return the rate of this process for the given Aerosol.

        Args:
            aerosol: The Aerosol instance on which to calculate the rate.

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
        """Execute the process, modifying the Aerosol in-place over a time step.

        Args:
            aerosol: The Aerosol instance to be updated.
            time_step: The time step size in seconds.
            sub_steps: Number of sub-steps to subdivide the time step,
                default 1.

        Returns:
            - The updated Aerosol after this process runs.

        Examples:
            ```py title="Executing the process"
            process = CustomProcess()
            updated_aerosol = process.execute(my_aerosol, time_step=1.0)
            ```
        """

    def __or__(self, other: "RunnableABC") -> "RunnableSequence":
        """Chain this Runnable with another using the '|' operator.

        This method enables an easy way to sequence processes, so they
        can be executed in a defined order.

        Args:
            other: Another Runnable to append after this one.

        Returns:
            A sequence containing both processes.

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
    """A sequence of Runnable processes executed in order.

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
        """Initialize the RunnableSequence.

        Sets up an empty list to hold the sequence of Runnable processes.
        """
        self.processes: list[RunnableABC] = []

    def add_process(self, process: RunnableABC):
        """Add a Runnable to the sequence.

        Args:
            process: The Runnable to add.

        Examples:
            ```py
            sequence = RunnableSequence()
            sequence.add_process(CustomProcess())
            ```
        """
        self.processes.append(process)

    def execute(
        self, aerosol: Aerosol, time_step: float, sub_steps: int = 1
    ) -> Aerosol:
        """Execute all processes in the sequence on the given Aerosol.

        Each Runnable in the sequence modifies the Aerosol and passes
        it to the next Runnable until all have been executed. A full cycle is
        performed over each sub-step of the time step.

        Args:
            aerosol: The Aerosol instance to be updated.
            time_step: The time step size in seconds for each process.
            sub_steps: Number of sub-steps to subdivide the time step,
                default 1.

        Returns:
            - Aerosol : The resulting Aerosol after all processes run.

        Examples:
            ```py title="Executing a RunnableSequence"
            sequence = RunnableSequence()
            final_aerosol = sequence.execute(
                aerosol, time_step=1.0, sub_steps=4
            )
            ```
        """
        sub_step_time_step = time_step / sub_steps
        # If tqdm is available, wrap the process loop in a progress bar
        loop_iterator = (
            range(sub_steps)
            if not AVAILABLE_TQDM
            else tqdm(
                range(sub_steps),
                desc="Executing Runnable",
                mininterval=0.5,
            )
        )
        for _ in loop_iterator:
            # loop over each process in the sequence
            for process in self.processes:
                aerosol = process.execute(
                    aerosol, sub_step_time_step, sub_steps=1
                )
        return aerosol

    def __or__(self, process: RunnableABC) -> "RunnableSequence":
        """Chain another Runnable into this sequence using the '|' operator.

        Args:
            process: The Runnable to add.

        Returns:
            This sequence with the new Runnable appended.

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
