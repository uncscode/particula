"""Explicit host execution of a box process inside a prepared GPU sequence."""

from copy import deepcopy

from particula.box_process import BoxProcess
from particula.runnable import RunnableABC


class CPUTransfer(RunnableABC):
    """Opt into a validated host round trip for a public BoxProcess.

    Python callbacks may call arbitrary scientific libraries. Their declared
    arrays publish only after all boxes validate. Prepared sequences containing
    this wrapper cannot use CUDA capture. External callback side effects and
    hidden mutable closure state are not part of checkpoint continuation.
    """

    def __init__(self, process: BoxProcess):
        """Retain the explicitly selected CPU process until preparation."""
        if not isinstance(process, BoxProcess):
            raise TypeError("CPUTransfer requires a BoxProcess")
        self.process = process

    def execute(
        self, aerosol, time_step, sub_steps=1, *, context=None, time=0.0
    ):
        """Delegate ordinary CPU execution without introducing any transfer."""
        return self.process.execute(
            aerosol, time_step, sub_steps, context=context, time=time
        )


class _CPUTransferAdapter:
    """Freeze an explicitly wrapped CPU implementation at preparation."""

    def plan(self, wrapper, aerosol):
        """Check all callback requirements without advancing or drawing RNG."""
        from particula.execution.cpu_context import CPUExecutionContext
        from particula.execution.extension_registry import ProcessCapabilities
        from particula.execution.process_extensions import GPUProcessPlan

        self.process = deepcopy(wrapper.process)
        process = self.process
        process.execute(
            aerosol,
            0,
            context=CPUExecutionContext(
                0, tuple(str(i) for i in range(aerosol.particles.n_boxes))
            ),
        )
        return GPUProcessPlan(
            name=process.name,
            extension_id=process.extension_id,
            version=process.version,
            reads=process.reads,
            writes=process.writes,
            parameters=(
                ("parameters", process.parameters),
                ("representations", process.representations),
                ("stochastic", process.stochastic),
            ),
            capabilities=ProcessCapabilities(checkpoint=True),
            execution_mode="cpu_transfer",
        )
