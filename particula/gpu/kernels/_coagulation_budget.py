"""Explicit work admission for equal-count particle-resolved coagulation."""

from dataclasses import dataclass
from typing import no_type_check

import numpy as np
import warp as wp


@dataclass(frozen=True)
class CoagulationWorkConfig:
    """Bound execution work independently of the physical population size.

    Attributes:
        max_trials_per_box: Maximum scheduled candidate trials per step.
        max_collision_buffer_bytes: Maximum collision-output allocation [byte].
    """

    max_trials_per_box: int = 65_536
    max_collision_buffer_bytes: int = 256 * 1024 * 1024

    def __post_init__(self):
        for name in ("max_trials_per_box", "max_collision_buffer_bytes"):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
        if self.max_trials_per_box > np.iinfo(np.int32).max:
            raise ValueError("max_trials_per_box exceeds signed int32 indexing")


def validate_coagulation_layout(boxes: int, slots: int, species: int) -> None:
    """Reject unaddressable native layouts before allocating device scratch.

    Warp dimensions and slot indices are signed int32. Pair ranks use int64;
    host byte calculations use Python integers so multiplication cannot wrap.
    Physical allocation failures remain explicit device allocation errors.
    """
    int32_max = np.iinfo(np.int32).max
    address_max = np.iinfo(np.intp).max
    for name, size in (
        ("boxes", boxes),
        ("slots", slots),
        ("species", species),
    ):
        if size < 0 or size > int32_max:
            raise ValueError(
                f"coagulation {name} exceeds signed int32 indexing"
            )
    largest_elements = max(boxes * slots * species, boxes * slots, boxes * 4096)
    if largest_elements * 8 > address_max:
        raise ValueError(
            "coagulation array byte size exceeds host address space"
        )


class CoagulationBudgetError(ValueError):
    """An entire coagulation operation was rejected before mutable work."""


@no_type_check
@wp.kernel
def admit_coagulation_work(
    counts: wp.array(dtype=wp.int32),
    indices: wp.array2d(dtype=wp.int32),
    concentration: wp.array2d(dtype=wp.float64),
    volume: wp.array(dtype=wp.float64),
    majorants: wp.array(dtype=wp.float64),
    time_step: wp.float64,
    trial_budget: wp.int32,
    collision_capacity: wp.int32,
    report: wp.array2d(dtype=wp.float64),
    failure: wp.array(dtype=wp.int32),
):
    """Report demand per box and latch rejection before any selector launch.

    Report columns are expected trials, conservative trial bound, available
    trials, and maximum budget-admissible timestep [s].
    """
    # One admission owner avoids inter-box races on the failure latch and
    # never reads rate scratch after an earlier resident operation failed.
    if wp.tid() != 0 or failure[0] != 0:
        return
    for box in range(counts.shape[0]):
        active = counts[box]
        available = wp.min(trial_budget, collision_capacity)
        available = wp.min(available, active // 2)
        expected = wp.float64(0.0)
        rate = wp.float64(0.0)
        if active > 1 and majorants[box] > wp.float64(0.0):
            rate = (
                majorants[box]
                * wp.float64(active)
                * wp.float64(active - 1)
                * wp.float64(0.5)
                * concentration[box, indices[box, 0]]
                / volume[box]
            )
            if time_step > wp.float64(0.0):
                expected = rate * time_step
        required = wp.ceil(expected)
        allowed_dt = wp.float64(0.0)
        if wp.isfinite(rate) and rate > wp.float64(0.0):
            allowed_dt = wp.float64(available) / rate
        report[box, 0] = expected
        report[box, 1] = required
        report[box, 2] = wp.float64(available)
        report[box, 3] = allowed_dt
        if not wp.isfinite(expected) or required > wp.float64(available):
            wp.atomic_cas(failure, 0, 0, 9)


def raise_budget_error(report):
    """Observe only compact admission diagnostics, never particle state."""
    rows = report.numpy()
    for box, (expected, required, available, time_step) in enumerate(rows):
        if not np.isfinite(expected) or required > available:
            raise CoagulationBudgetError(
                f"Coagulation box {box} requires {required:g} trials; "
                f"budget/collision capacity permits {available:g}. "
                f"Use time_step <= {time_step:.17g} s or increase resources."
            )


@no_type_check
@wp.kernel
def initialize_admitted_rng(
    rng: wp.array(dtype=wp.uint32),
    pending: wp.array(dtype=wp.int32),
    seed: wp.int32,
    failure: wp.array(dtype=wp.int32),
):
    """Initialize each stream once after successful admission."""
    box = wp.tid()
    if failure[0] != 0 or pending[box] == 0:
        return
    rng[box] = wp.rand_init(seed, wp.int32(box))
    pending[box] = 0
