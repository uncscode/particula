"""Resident adapter for resolved bulk chemistry and reactive uptake."""

from dataclasses import astuple, dataclass

import numpy as np
import warp as wp

from particula.dynamics.chemistry import resolved_device as kernels
from particula.execution.extension_registry import (
    PreparedOperation,
    ProcessCapabilities,
)
from particula.execution.process_extensions import GPUProcessPlan, GPUStateSpec


@dataclass(frozen=True)
class ResolvedChemistryAdapter:
    """Preallocate all trial vectors and share a failure latch across boxes."""

    workspace_bytes: int = 256 * 1024**2

    def plan(self, process, aerosol):
        """Validate metadata, index limits and scratch demand before upload."""
        process.validate(aerosol)
        b, solver = process.bound, process.solver
        boxes, slots, ns = aerosol.particles.masses.shape
        width = len(b.gas_names) + slots * ns
        required = 8 * boxes * (6 * width + 2 * slots + len(b.kinds))
        if (
            type(self.workspace_bytes) is not int
            or self.workspace_bytes < required
        ):
            raise ValueError(
                f"resolved chemistry requires {required} scratch bytes"
            )
        limit = np.iinfo(np.int32).max
        if max(width, solver.max_trials, boxes * width * 6) > limit or any(
            abs(n) > limit
            for table in (b.net, b.orders)
            for row in table
            for n in row
        ):
            raise ValueError("resolved chemistry exceeds device index limits")
        return GPUProcessPlan(
            name=process.name,
            extension_id=process.extension_id,
            version=process.version,
            reads=process.reads,
            writes=process.writes,
            parameters=(
                ("identity", process.configuration()),
                ("gm", b.gas_masses),
                ("pm", b.particle_masses),
                ("density", b.density),
                ("net", b.net),
                ("orders", b.orders),
                ("params", b.parameters),
                ("kinds", b.kinds),
                ("sources", b.sources),
                ("solver", astuple(solver)),
            ),
            capabilities=ProcessCapabilities(
                resident=True, capture=True, checkpoint=True
            ),
            state=(GPUStateSpec("work_counts", (boxes, 2), "int64"),),
        )

    def prepare(self, plan, context):
        """Bind native storage and allocate per-box trial workspace."""
        data = plan.values
        target = context.device
        m = kernels.Network()
        for name in ("gm", "pm", "density", "params"):
            setattr(
                m, name, wp.array(data[name], dtype=wp.float64, device=target)
            )
        for name in ("net", "orders", "kinds", "sources"):
            setattr(m, name, wp.array(data[name], dtype=int, device=target))
        _, m.rtol, m.atol, m.first, m.minimum, m.maximum, m.limit = data[
            "solver"
        ]
        mass = context.arrays["particles.masses"]
        gas = context.arrays["gas.concentration"]
        boxes, slots, ns = mass.shape
        width = gas.shape[1] + slots * ns
        w = kernels.Stages()
        w.v = wp.zeros((boxes, 6, width), dtype=wp.float64, device=target)
        w.reference = wp.zeros((boxes, slots), dtype=wp.float64, device=target)
        w.weights = wp.zeros_like(w.reference)
        w.rates = wp.zeros(
            (boxes, len(data["kinds"])), dtype=wp.float64, device=target
        )
        w.counts = context.state["work_counts"]
        inputs = (
            m,
            w,
            gas,
            mass,
            context.arrays["particles.concentration"],
            context.arrays["particles.volume"],
            context.arrays["environment.temperature"],
            wp.float64(context.time_step),
            context.failure,
        )
        publish = (m, w, gas, mass, context.failure)

        def enqueue():
            wp.launch(kernels.stage, dim=boxes, inputs=inputs, device=target)
            wp.launch(
                kernels.publish,
                dim=(boxes, width),
                inputs=publish,
                device=target,
            )

        return PreparedOperation(plan.name, enqueue, inputs, plan.capabilities)
