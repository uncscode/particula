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
    counts: wp.array(dtype=wp.int64),
):
    j = wp.tid()
    box = (j + offset) // activity.shape[1]
    slot = (j + offset) % activity.shape[1]
    status[box, slot] = flags[j]
    if flags[j] < 0:
        wp.atomic_add(counts, 6, wp.int64(1))
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


@wp.kernel(enable_backward=False)
def _record(
    active: wp.array(dtype=int),
    reason: wp.array(dtype=int),
    estimates: wp.array2d(dtype=wp.float64),
    lanes: wp.array(dtype=int),
    offset: int,
    errors: wp.array3d(dtype=wp.float64),
    counts: wp.array(dtype=wp.int64),
):
    i = wp.tid()
    box = (offset + i) // errors.shape[1]
    slot = (offset + i) % errors.shape[1]
    if active[i] != 0:
        wp.atomic_add(counts, reason[i], wp.int64(1))
    for k in range(lanes.shape[0]):
        errors[box, slot, lanes[k]] = wp.where(
            active[i] != 0, estimates[i, k], wp.float64(0.0)
        )


@wp.kernel(enable_backward=False)
def _mask(
    active: wp.array(dtype=int), mask: wp.array2d(dtype=int), offset: int
):
    i = wp.tid()
    active[i] = mask[
        (i + offset) // mask.shape[1], (i + offset) % mask.shape[1]
    ]


@wp.kernel(enable_backward=False)
def _refined(
    out: wp.array2d(dtype=wp.float64),
    indices: wp.array(dtype=int),
    lanes: wp.array(dtype=int),
    active: wp.array(dtype=int),
    flags: wp.array(dtype=int),
    offset: int,
    values: wp.array3d(dtype=wp.float64),
    errors: wp.array3d(dtype=wp.float64),
    status: wp.array2d(dtype=int),
    invalid: wp.array(dtype=int),
    detail: wp.array(dtype=int),
    stage: int,
    counts: wp.array(dtype=wp.int64),
):
    i = wp.tid()
    if active[i] == 0:
        return
    box = (offset + i) // values.shape[1]
    slot = (offset + i) % values.shape[1]
    wp.atomic_sub(counts, 0, wp.int64(1))
    wp.atomic_add(counts, 5, wp.int64(1))
    status[box, slot] = flags[i]
    if flags[i] < 0:
        wp.atomic_add(counts, 6, wp.int64(1))
        wp.atomic_max(invalid, 0, 1)
        if wp.atomic_cas(detail, 0, 0, 1) == 0:
            detail[1] = box
            detail[2] = slot
            detail[3] = stage
            detail[4] = flags[i]
    for j in range(lanes.shape[0]):
        values[box, slot, lanes[j]] = out[i, indices[j]]
        errors[box, slot, lanes[j]] = wp.float64(0.0)


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
        names = [
            provider.prepared.neutrals[provider.neutral_lanes[i]]
            for i in targets
        ]
        self.hybrid = (
            bool(names)
            and provider.table is not None
            and set(names) <= set(provider.table.recipe.targets)
        )
        self.counts = wp.zeros(7, dtype=wp.int64, device=device)
        self.extra = []
        if self.hybrid:
            from ._sparse_cuda import DeviceTable

            self.table = DeviceTable(provider.table, self.model, names)
            self.errors = wp.zeros(shape, dtype=wp.float64, device=device)
            self.uncertain = wp.zeros(shape[:2], dtype=int, device=device)
            err = wp.empty((cap, len(names)), dtype=wp.float64, device=device)
            reason = wp.empty(cap, dtype=int, device=device)
            miss = wp.empty(cap, dtype=int, device=device)
        self.chunks = []
        for offset in range(0, rows, cap):
            n = min(rows - offset, cap)
            if self.hybrid:
                self.extra.append((err[:n], reason[:n], miss[:n]))
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

    def evaluate_direct(self, masses, counts, temperature, invalid, stage=0):
        """Use the existing device workspace for a physical Jacobian sample."""
        self.evaluate(masses, counts, temperature, invalid, stage, exact=True)

    def evaluate(
        self,
        masses,
        counts,
        temperature,
        invalid,
        stage=0,
        *,
        refine=False,
        exact=False,
    ):
        stream = wp.get_stream(self.device)
        for j, (offset, n, mass, temp, active, out, flags, work) in enumerate(
            self.chunks
        ):
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
            direct = active
            if refine:
                wp.launch(
                    _mask,
                    n,
                    inputs=[active, self.uncertain, offset],
                    device=self.device,
                    stream=stream,
                )
            elif self.hybrid and not exact:
                err, reason, direct = self.extra[j]
                self.table.lookup_into(
                    mass, temp, active, out, err, reason, direct, stream
                )
            self.model.evaluate_into(
                mass, temp, out, flags, work, stream=stream, active=direct
            )
            if refine:
                wp.launch(
                    _refined,
                    n,
                    inputs=[
                        out,
                        self.indices,
                        self.lanes,
                        active,
                        flags,
                        offset,
                        self.values,
                        self.errors,
                        self.status,
                        invalid,
                        self.detail,
                        stage,
                        self.counts,
                    ],
                    device=self.device,
                    stream=stream,
                )
                continue
            if self.hybrid and not exact:
                wp.launch(
                    _record,
                    n,
                    inputs=[
                        active,
                        reason,
                        err,
                        self.lanes,
                        offset,
                        self.errors,
                        self.counts,
                    ],
                    device=self.device,
                    stream=stream,
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
                    self.counts,
                ],
                device=self.device,
                stream=stream,
            )
