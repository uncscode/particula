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

from copy import copy
from typing import Any

import numpy as np

# Check if tqdm is available for progress bars
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[assignment]
    AVAILABLE_TQDM = False
else:
    AVAILABLE_TQDM = True


from particula.aerosol import Aerosol


class RunnableABC:
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

    def rate(self, aerosol: Aerosol) -> Any:
        """Evaluate detached box hooks and retain a leading box axis."""
        from particula import _cpu_batch

        if not isinstance(aerosol, Aerosol):
            raise TypeError(f"{type(self).__name__} requires a native Aerosol")
        _cpu_batch.validate(aerosol)
        self._validate_batch(aerosol)
        candidate = _cpu_batch.detached(aerosol)
        results = []
        for box in range(aerosol.particles.n_boxes):
            worker = self._box_worker(box, aerosol.particles.n_boxes, None)
            results.append(
                worker._rate_box(_cpu_batch.box_view(candidate, box))
            )
        return np.stack(results, axis=0)

    def execute(  # noqa: C901
        self,
        aerosol: Aerosol,
        time_step: float,
        sub_steps: int = 1,
        *,
        context=None,
    ) -> Aerosol:
        """Advance every box, committing atomically after each process substep.

        Contexts own independent logical-box random streams. Completed substeps
        remain committed if a later one fails. Custom subclasses overriding
        this method retain responsibility for their own execution semantics.
        """
        from particula import _cpu_batch
        from particula.execution import CPUExecutionContext

        step = _cpu_batch.duration(time_step, sub_steps)
        if not isinstance(aerosol, Aerosol):
            raise TypeError(f"{type(self).__name__} requires a native Aerosol")
        maximum = getattr(self, "_maximum_sub_steps", None)
        if maximum is not None and sub_steps > maximum:
            raise ValueError(f"sub_steps must not exceed {maximum}")
        writers = self._writer_fields(aerosol)
        _cpu_batch.validate(aerosol, writers)
        self._validate_batch(aerosol)
        boxes = aerosol.particles.n_boxes
        if context is not None:
            if not isinstance(context, CPUExecutionContext):
                raise TypeError("context must be CPUExecutionContext")
            context._begin(boxes)
        if boxes > 1 and self._is_stochastic(aerosol) and context is None:
            raise ValueError("stochastic multi-box execution requires context")
        for _ in range(sub_steps):
            _cpu_batch.validate(aerosol, writers)
            transaction = None if context is None else context._begin(boxes)
            candidate = _cpu_batch.detached(aerosol)
            workers = []
            for box in range(boxes):
                worker = self._box_worker(box, boxes, transaction)
                view = _cpu_batch.box_view(candidate, box)
                result = worker._execute_box(view, step, sub_steps=1)
                if result is not view:
                    raise ValueError(
                        "CPU box hooks must retain aerosol identity"
                    )
                workers.append(worker)
            if step == 0:
                continue
            _cpu_batch.publish(aerosol, candidate, writers)
            if transaction is not None:
                transaction.commit()
            self._commit_workers(workers, context is not None)
        return aerosol

    def _validate_batch(self, aerosol):
        """Validate process-wide controls before staging boxes."""

    def _writer_fields(self, aerosol):
        """Declare numeric destinations for transactional publication."""
        raise NotImplementedError

    def _is_stochastic(self, aerosol):
        """Report whether this process requires independent box streams."""
        return False

    def _box_worker(self, box, boxes, transaction):
        """Create isolated hook state for a single box."""
        return copy(self)

    def _commit_workers(self, workers, uses_context):
        """Publish staged process diagnostics after numeric success."""

    def _rate_box(self, aerosol):
        """Evaluate the single-box numerical building block."""
        raise NotImplementedError

    def _execute_box(self, aerosol, time_step, sub_steps=1):
        """Execute the single-box numerical building block."""
        raise NotImplementedError

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

    def __init__(self) -> None:
        """Initialize the RunnableSequence.

        Sets up an empty list to hold the sequence of Runnable processes.
        """
        self.processes: list[RunnableABC] = []

    def add_process(self, process: RunnableABC) -> None:
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
        self,
        aerosol: Aerosol,
        time_step: float,
        sub_steps: int = 1,
        *,
        context=None,
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
        from particula._cpu_batch import duration

        sub_step_time_step = duration(time_step, sub_steps)
        # If tqdm is available, wrap the process loop in a progress bar
        loop_iterator = range(sub_steps)
        if AVAILABLE_TQDM and tqdm is not None:
            loop_iterator = tqdm(
                range(sub_steps),
                desc="Executing Runnable",
                mininterval=0.5,
            )
        for _ in loop_iterator:
            # loop over each process in the sequence
            for process in self.processes:
                aerosol = process.execute(
                    aerosol,
                    sub_step_time_step,
                    sub_steps=1,
                    **({"context": context} if context is not None else {}),
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
