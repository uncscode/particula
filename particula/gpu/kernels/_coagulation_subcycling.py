"""Bounded, live-state work admission for native coagulation intervals."""

from dataclasses import dataclass
from functools import partial
from math import ulp
from numbers import Real
from typing import Any, no_type_check

import numpy as np
import warp as wp

from particula.gpu.kernels._coagulation_budget import CoagulationBudgetError


@dataclass(frozen=True)
class CoagulationSubcyclingConfig:
    """Set the minimum duration [s], maximum passes and memory ceiling [B]."""

    min_substep: float
    max_subcycles: int
    max_device_bytes: int

    def __post_init__(self):
        for name in ("max_subcycles", "max_device_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_subcycles > np.iinfo(np.int32).max:
            raise ValueError("max_subcycles exceeds signed int32 indexing")
        if (
            isinstance(self.min_substep, bool)
            or not isinstance(self.min_substep, Real)
            or not (np.isfinite(self.min_substep) and self.min_substep > 0)
        ):
            raise ValueError("min_substep must be finite and positive")


@no_type_check
@wp.kernel
def _choose(
    counts: wp.array(dtype=wp.int32),
    indices: Any,
    weights: Any,
    volumes: Any,
    majorants: Any,
    trial_limit: int,
    capacity: int,
    minimum: wp.float64,
    remaining: Any,
    dt: Any,
    iteration: Any,
    history: Any,
    report: Any,
    failure: Any,
):
    if failure[0] != 0:
        return
    k = iteration[0]
    h = remaining[0]
    for b in range(counts.shape[0]):
        n = counts[b]
        numerator = wp.float64(0.0)
        rate = wp.float64(0.0)
        if n > 1:
            pair_count = wp.float64(n) * wp.float64(n - 1) / wp.float64(2.0)
            numerator = majorants[b] * pair_count * weights[b, indices[b, 0]]
            rate = numerator / volumes[b]
        limit = wp.min(wp.min(trial_limit, capacity), n // 2)
        history[k, b, 8] = majorants[b]
        history[k, b, 9] = wp.float64(n)
        history[k, b, 0] = numerator
        history[k, b, 1] = wp.ceil(numerator * remaining[0] / volumes[b])
        history[k, b, 2] = wp.float64(limit)
        history[k, b, 3] = wp.float64(n // 2)
        if not wp.isfinite(rate) or rate < wp.float64(0.0):
            failure[0] = 12
            return
        while (
            wp.ceil(numerator * h / volumes[b]) > wp.float64(limit)
            and h >= minimum
        ):
            h *= wp.float64(0.5)
    dt[0] = h
    if (
        remaining[0] > wp.float64(0.0)
        and h >= minimum
        and remaining[0] - h == remaining[0]
    ):
        failure[0] = 13
        return
    if h < minimum and remaining[0] != wp.float64(0.0):
        failure[0] = 11
        return
    for b in range(counts.shape[0]):
        numerator = history[k, b, 0]
        rate = numerator / volumes[b]
        available = history[k, b, 2]
        report[b, 0] = numerator * h / volumes[b]
        report[b, 1] = wp.ceil(report[b, 0])
        history[k, b, 0] = report[b, 1]
        report[b, 2] = available
        report[b, 3] = wp.float64(0.0)
        if rate > wp.float64(0.0):
            report[b, 3] = available / rate


@no_type_check
@wp.kernel
def _finish(
    remaining: Any,
    dt: Any,
    iteration: Any,
    history: Any,
    pairs: Any,
    accepted: Any,
    work: Any,
    log: Any,
    offsets: Any,
    failure: Any,
    tolerance: wp.float64,
    duration: wp.float64,
    elapsed: Any,
):
    if failure[0] != 0:
        return
    k = iteration[0]
    for b in range(accepted.shape[0]):
        start = offsets[b]
        for j in range(accepted[b]):
            log[b, start + j, 0] = pairs[b, j, 0]
            log[b, start + j, 1] = pairs[b, j, 1]
        offsets[b] = start + accepted[b]
        history[k, b, 4] = dt[0]
        history[k, b, 5] = wp.float64(work[b, 0])
        history[k, b, 6] = wp.float64(work[b, 1])
        history[k, b, 7] = wp.float64(accepted[b])
    # Compensated elapsed time bounds cumulative round-off across many passes.
    increment = dt[0] - elapsed[1]
    total = elapsed[0] + increment
    elapsed[1] = (total - elapsed[0]) - increment
    elapsed[0] = total
    remaining[0] = duration - total
    iteration[0] += 1
    if remaining[0] <= tolerance:
        failure[0] = -1
    elif iteration[0] == history.shape[0]:
        failure[0] = 11


@no_type_check
@wp.kernel
def _end(failure: Any):
    if failure[0] == -1:
        failure[0] = 0


@dataclass(init=False)
class AdaptiveInterval:
    """Own the duration, admission history and ordered collision log."""

    config: CoagulationSubcyclingConfig
    duration: float
    remaining: Any
    elapsed: Any
    dt: Any
    iteration: Any
    history: Any
    pairs: Any
    offsets: Any

    def __init__(self, call, config, duration):
        self.config = config
        self.duration = float(duration)
        b, n = call.concentration.shape
        k = config.max_subcycles
        required = interval_bytes(b, n, k)
        if required > config.max_device_bytes:
            raise MemoryError(f"adaptive buffers require {required} bytes")
        dev = call.device
        self.remaining = wp.zeros(1, dtype=wp.float64, device=dev)
        self.elapsed = wp.zeros(2, dtype=wp.float64, device=dev)
        self.dt = wp.zeros_like(self.remaining)
        self.iteration = wp.zeros(1, dtype=wp.int32, device=dev)
        self.history = wp.zeros((k, b, 10), dtype=wp.float64, device=dev)
        self.pairs = wp.zeros((b, max(1, n - 1), 2), dtype=wp.int32, device=dev)
        self.offsets = wp.zeros(b, dtype=wp.int32, device=dev)

    def begin(self):
        """Reset progress on the current stream before the first pass."""
        self.remaining.fill_(self.duration)
        self.elapsed.zero_()
        self.iteration.zero_()
        self.offsets.zero_()

    def choose(self, call, failure):
        """Admit one shared duration against all live box majorants."""
        work = call.brownian_parallel
        args = call.selector_inputs
        wp.launch(
            _choose,
            1,
            [
                work.counts,
                args[24],
                args[1],
                args[3],
                work.majorants,
                call.work_config.max_trials_per_box,
                call.max_collisions,
                wp.float64(self.config.min_substep),
                self.remaining,
                self.dt,
                self.iteration,
                self.history,
                call.budget_report,
                failure,
            ],
            device=call.device,
        )

    def finish(self, call, failure):
        """Append the accepted work and consume its physical duration."""
        wp.launch(
            _finish,
            1,
            [
                self.remaining,
                self.dt,
                self.iteration,
                self.history,
                call.collision_pairs,
                call.n_collisions,
                call.work_counts,
                self.pairs,
                self.offsets,
                failure,
                wp.float64(8 * ulp(self.duration)),
                wp.float64(self.duration),
                self.elapsed,
            ],
            device=call.device,
        )

    def enqueue(self, call, failure, operation):
        """Record the bounded pass loop, including per-merge dependents."""
        self.begin()
        for _ in range(self.config.max_subcycles):
            operation()
            self.finish(call, failure)
        wp.launch(_end, 1, [failure], device=call.device)

    def diagnostics(self, call):
        """Download per-pass rows delimiting the ordered pair log."""
        k = int(self.iteration.numpy()[0])
        rows = self.history.numpy()[:k].copy()
        available = rows[:, :, 2]
        accepted = rows[:, :, 7].astype(np.int64)
        ends = np.cumsum(accepted, axis=0)
        return {
            "requested_time_step": self.duration,
            "substep_count": k,
            "substep_sizes": rows[:, 0, 4],
            "majorants": rows[:, :, 8],
            "active_counts": rows[:, :, 9].astype(np.int64),
            "required_trials": rows[:, :, 0],
            "outer_remaining_required_trials": rows[:, :, 1],
            "available_trials": available,
            "disjoint_pair_capacity": rows[:, :, 3],
            "trial_limit": call.work_config.max_trials_per_box,
            "collision_capacity": call.max_collisions,
            "trial_limited": available == call.work_config.max_trials_per_box,
            "buffer_limited": available == call.max_collisions,
            "disjoint_pair_limited": available == rows[:, :, 3],
            "scheduled_trials": rows[:, :, 5].astype(np.int64),
            "attempted_trials": rows[:, :, 6].astype(np.int64),
            "accepted_collisions": accepted,
            "pair_end_offsets": ends,
            "resource_growth": False,
        }

    def rejection(self, call, code):
        """Keep tentative diagnostics independent of the restore point."""
        k = int(self.iteration.numpy()[0])
        remaining = float(self.remaining.numpy()[0])
        exhausted = k == self.config.max_subcycles
        reason = "max_subcycles" if exhausted else "min_substep"
        if code == 13:
            reason = "floating-point progress"
        row = self.history.numpy()[k - 1 if exhausted else k]
        available = row[:, 2].copy()
        # Zero-capacity boxes have fewer than two live slots and zero demand.
        load = np.divide(
            row[:, 1],
            available,
            out=np.zeros_like(available),
            where=available > 0,
        )
        box = int(np.argmax(load))
        remedy = (
            "Increase the controlling capacity or use a shorter outer interval."
            if code == 13
            else f"Relax {reason} and retry."
        )
        error = CoagulationSubcyclingError(
            f"{reason} exhausted after {k} substeps. "
            f"{remaining:g} s remain. "
            f"Last admission in box {box} requested {row[box, 1]:g} trials "
            f"with {available[box]:g} available. {remedy}"
        )
        error.diagnostics = dict(
            self.diagnostics(call),
            reason=reason,
            remaining_seconds=remaining,
            rolled_back=True,
            limiting_box=box,
            last_required_trials=row[:, 1].copy(),
            last_available_trials=available,
            last_trial_limited=available == call.work_config.max_trials_per_box,
            last_buffer_limited=available == call.max_collisions,
            last_disjoint_pair_limited=available == row[:, 3],
            last_active_counts=row[:, 9].astype(np.int64),
            last_majorants=row[:, 8].copy(),
            last_candidate_seconds=float(self.dt.numpy()[0]),
        )
        return error


class CoagulationSubcyclingError(CoagulationBudgetError):
    """A recoverable outer interval rejection carrying tentative diagnostics."""

    diagnostics: dict[str, Any]


def adaptive_capacity(boxes, slots, byte_limit):
    """Choose the largest physical single-pass capacity permitted by bytes."""
    capacity = min(max(1, slots // 2), byte_limit // (8 * boxes))
    if capacity < 1:
        raise MemoryError("collision byte budget cannot hold one pair per box")
    return capacity


def interval_bytes(boxes, slots, passes):
    """Return bytes for controller scalars, per-box logs and diagnostic rows."""
    return 36 + boxes * (4 + 8 * max(1, slots - 1) + 80 * passes)


def wrap_subcycles(operations, plans, call, failure):
    """Retain per-merge field/cohort operations inside the guarded loop."""
    from dataclasses import replace

    return tuple(
        replace(
            op,
            handler=partial(call.adaptive.enqueue, call, failure, op),
            resources=(*op.resources, op),
        )
        if plan.family == "coagulation"
        else op
        for plan, op in zip(plans, operations, strict=True)
    )


def workspace_bytes(boxes, slots, capacity, epsilon=None, polarizable=False):
    """Upper bound for the direct sampler with all optional transport arrays.

    Six fp64 slot properties, two int32 slot indices, fp64 reduction lanes,
    five environment inputs, mean density, rate, trial report, output count,
    active count, RNG word/init flag and two work counters per box. Include
    validation flags and both the persistent and setup failure latch.
    """
    lanes = max(1, min(4096, (slots * slots + 31) // 32))
    total = boxes * (56 * slots + 8 * (lanes + capacity) + 112) + 64
    if polarizable:
        from particula.dynamics.coagulation.van_der_waals_kernel import (
            DISTANCES,
            RATIO_INTERVALS,
        )

        total += DISTANCES.nbytes * (
            1 + (3 * (RATIO_INTERVALS + 1) if epsilon != 1 else 0)
        )
    return total


def check_memory(config, categories):
    """Reject the complete reservation before creating adaptive allocations."""
    total = sum(categories.values())
    if total > config.max_device_bytes:
        raise MemoryError(
            f"Adaptive coagulation requires a reservation of {total} bytes "
            f"({categories}), max_device_bytes={config.max_device_bytes}."
        )
