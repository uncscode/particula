"""Retained analytical-material gather and neutral-activity scatter for Warp."""

import numpy as np
import warp as wp

from ._cuda import CudaModel


@wp.kernel(enable_backward=False)
def _gather(
    masses: wp.array3d(dtype=wp.float64),
    counts: wp.array2d(dtype=wp.float64),
    temperatures: wp.array(dtype=wp.float64),
    order: wp.array(dtype=int),
    offset: int,
    dissolved: wp.array2d(dtype=wp.float64),
    temp: wp.array(dtype=wp.float64),
    active: wp.array(dtype=int),
):
    j = wp.tid()
    row = offset + j
    box = row // masses.shape[1]
    slot = row % masses.shape[1]
    active[j] = int(counts[box, slot] > wp.float64(0.0))
    temp[j] = temperatures[box]
    for c in range(order.shape[0]):
        dissolved[j, c] = masses[box, slot, order[c]]


@wp.kernel(enable_backward=False)
def _scatter(
    out: wp.array2d(dtype=wp.float64),
    indices: wp.array(dtype=int),
    lanes: wp.array(dtype=int),
    active: wp.array(dtype=int),
    flags: wp.array(dtype=int),
    offset: int,
    activity: wp.array3d(dtype=wp.float64),
    status: wp.array2d(dtype=int),
    invalid: wp.array(dtype=int),
    detail: wp.array(dtype=int),
    stage: int,
):
    j = wp.tid()
    box = (j + offset) // activity.shape[1]
    slot = (j + offset) % activity.shape[1]
    status[box, slot] = flags[j]
    if flags[j] < 0:
        wp.atomic_max(invalid, 0, 1)
        if wp.atomic_cas(detail, 0, 0, 1) == 0:
            detail[1] = box
            detail[2] = slot
            detail[3] = stage
            detail[4] = flags[j]
    for k in range(lanes.shape[0]):
        a = wp.float64(0.0)
        if active[j] != 0 and flags[j] >= 0:
            a = out[j, indices[k]]
        activity[box, slot, lanes[k]] = a


class DeviceActivities:
    """One owner and launch stream per binding; chunks share bounded scratch."""

    def __init__(self, provider, shape, targets, device):
        rows = shape[0] * shape[1]
        self.model = CudaModel(
            provider.prepared, device, capacity=max(1, min(4096, rows))
        )
        self.device = self.model.device
        self.values = wp.zeros(shape, dtype=wp.float64, device=device)
        self.detail = wp.zeros(5, dtype=int, device=device)
        self.status = wp.zeros(shape[:2], dtype=int, device=device)
        self.order = wp.array(
            np.array(provider.material_order), dtype=int, device=device
        )
        self.lanes = wp.array(
            np.array(targets, dtype=int), dtype=int, device=device
        )
        start = self.model.output_slices["activity"].start
        self.indices = wp.array(
            np.array(
                [start + provider.neutral_lanes[i] for i in targets], dtype=int
            ),
            dtype=int,
            device=device,
        )
        cap = self.model.capacity
        dissolved = wp.empty(
            (cap, len(provider.material_order)), dtype=wp.float64, device=device
        )
        temp = wp.empty(cap, dtype=wp.float64, device=device)
        active = wp.empty(cap, dtype=int, device=device)
        self.chunks = []
        for offset in range(0, rows, cap):
            n = min(rows - offset, cap)
            self.chunks.append(
                (
                    offset,
                    n,
                    dissolved[:n],
                    temp[:n],
                    active[:n],
                    self.model.output[:n],
                    self.model.status[:n],
                    self.model.scratch[:n],
                )
            )

    def failure_message(self):
        failed, box, slot, stage, code = self.detail.numpy()
        return (
            f"AIOMFAC failed at box {box}, slot {slot}, "
            f"rate stage {stage}, backend status {code}"
            if failed
            else ""
        )

    def evaluate(self, masses, counts, temperature, invalid, stage=0):
        stream = wp.get_stream(self.device)
        for offset, n, mass, temp, active, out, flags, work in self.chunks:
            wp.launch(
                _gather,
                n,
                inputs=[
                    masses,
                    counts,
                    temperature,
                    self.order,
                    offset,
                    mass,
                    temp,
                    active,
                ],
                device=self.device,
                stream=stream,
            )
            self.model.evaluate_into(
                mass, temp, out, flags, work, stream=stream, active=active
            )
            wp.launch(
                _scatter,
                n,
                inputs=[
                    out,
                    self.indices,
                    self.lanes,
                    active,
                    flags,
                    offset,
                    self.values,
                    self.status,
                    invalid,
                    self.detail,
                    stage,
                ],
                device=self.device,
                stream=stream,
            )
