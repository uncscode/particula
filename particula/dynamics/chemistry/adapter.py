"""Prepare and budget the resident gas mechanism on an explicit device."""

from dataclasses import astuple, dataclass

import numpy as np
import warp as wp

from particula.dynamics.chemistry import device
from particula.dynamics.chemistry.binding import _sparse_stoichiometry
from particula.execution.extension_registry import (
    PreparedOperation,
    ProcessCapabilities,
)
from particula.execution.process_extensions import GPUProcessPlan, GPUStateSpec


@dataclass(frozen=True)
class GasChemistryAdapter:
    """Resident implementation for an explicitly registered GasChemistry.

    workspace_bytes bounds chemistry scratch in bytes across all boxes.
    RK23 uses linear species workspace. ROS3 also owns two dense matrices
    and pivot indices per box. Mechanism inputs and counters are separate.
    """

    workspace_bytes: int = 256 * 1024**2

    def plan(self, process, aerosol):
        """Check the species layout and all scratch sizes before allocation."""
        process.validate(aerosol)
        bound, solver = process.bound, process.solver
        limit = np.iinfo(np.int32).max
        if solver.max_trials > limit or any(
            abs(value) > limit
            for rows in (bound.stoichiometry, bound.orders)
            for row in rows
            for value in row
        ):
            raise ValueError(
                "chemistry configuration exceeds device int32 range"
            )
        boxes = aerosol.gas.n_boxes
        n, nr = len(bound.indices), len(bound.orders)
        matrix_size = n if solver.method == "ros3" else 1
        per_box = 8 * (9 * n + nr + 2 * matrix_size**2) + 4 * matrix_size
        if (
            type(self.workspace_bytes) is not int
            or self.workspace_bytes < boxes * per_box
        ):
            raise ValueError(
                f"gas chemistry requires {boxes * per_box} scratch bytes"
            )
        return GPUProcessPlan(
            name=process.name,
            extension_id=process.extension_id,
            version=process.version,
            reads=process.reads,
            writes=process.writes,
            parameters=(
                ("identity", process.configuration()),
                ("indices", bound.indices),
                ("masses", bound.molar_masses),
                ("orders", bound.orders),
                ("net", bound.stoichiometry),
                ("coefficients", bound.parameters),
                ("solver", astuple(solver)),
            ),
            capabilities=ProcessCapabilities(resident=True, checkpoint=True),
            state=(GPUStateSpec("work_counts", (boxes, 4), "int64"),),
        )

    def prepare(self, plan, context):
        """Upload fixed chemistry data once and retain all work on device."""
        values = plan.values
        target = context.device
        m = device.DeviceMechanism()
        m.lanes = wp.array(values["indices"], dtype=int, device=target)
        m.masses = wp.array(values["masses"], dtype=wp.float64, device=target)
        m.offsets, m.columns, m.coefficients_net = (
            wp.array(array, dtype=int, device=target)
            for array in _sparse_stoichiometry(values["net"])
        )
        m.active = wp.array(
            np.any(values["net"], axis=0), dtype=int, device=target
        )
        m.orders = wp.array(values["orders"], dtype=int, device=target)
        m.coefficients = wp.array(
            values["coefficients"], dtype=wp.float64, device=target
        )
        method, rtol, atol, first, minimum, maximum, limit = values["solver"]
        n = m.lanes.shape[0]
        m.atol = wp.array(
            np.broadcast_to(atol, (n,)).copy(), dtype=wp.float64, device=target
        )
        m.rtol, m.first_step, m.min_step, m.max_step = (
            rtol,
            first,
            minimum,
            maximum,
        )
        m.trials, m.stiff = limit, int(method == "ros3")
        boxes = len(context.logical_box_ids)
        size = n if method == "ros3" else 1
        w = device.Work()
        w.vectors = wp.zeros((boxes, 9, n), dtype=wp.float64, device=target)
        w.matrix = wp.zeros(
            (boxes, size, size), dtype=wp.float64, device=target
        )
        w.lu = wp.zeros_like(w.matrix)
        w.pivots = wp.zeros((boxes, size), dtype=int, device=target)
        w.rates = wp.zeros(
            (boxes, m.orders.shape[0]), dtype=wp.float64, device=target
        )
        w.counts = context.state["work_counts"]
        gas = context.arrays["gas.concentration"]
        integrate = (
            m,
            w,
            gas,
            context.arrays["environment.temperature"],
            wp.float64(context.time_step),
            context.failure,
        )
        publish = (m, w, gas, context.failure)

        def enqueue():
            wp.launch(
                device.integrate_boxes,
                dim=boxes,
                inputs=integrate,
                device=target,
            )
            wp.launch(
                device.publish_boxes,
                dim=(boxes, n),
                inputs=publish,
                device=target,
            )

        return PreparedOperation(
            plan.name, enqueue, integrate, plan.capabilities
        )
