"""Caller-owned, transactional random streams for CPU box execution.

This module does not import Warp or share the resident GPU stream format.
Context views share continuation state, while transactions stage independent
generators until a successful runnable substep publishes them.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from numbers import Integral

import numpy as np


def _box_ids(values: tuple[str, ...]) -> tuple[str, ...]:
    """Validate stable, unique logical identities without normalization."""
    if not isinstance(values, tuple):
        raise TypeError("box_ids must be a tuple")
    if not values or any(not isinstance(v, str) or not v for v in values):
        raise ValueError("box_ids must contain nonempty strings")
    if len(set(values)) != len(values):
        raise ValueError("box_ids must be unique")
    return values


class CPUExecutionContext:
    """Retain persistent process-specific streams keyed by logical box ID.

    Args:
        root_seed: Nonnegative integer seed; booleans are not accepted.
        box_ids: Unique nonempty strings in current numeric-array box order.

    Reordering numeric storage requires a matching ``with_box_ids`` view.
    Reusing a context and process namespace deliberately continues its streams.
    Create a new context to reset them. Contexts are not thread-safe and must
    not be used concurrently. No CPU/GPU bitwise equivalence is implied.
    """

    def __init__(self, root_seed: int, box_ids: tuple[str, ...]) -> None:
        """Validate metadata without creating or advancing random streams."""
        if (
            isinstance(root_seed, (bool, np.bool_))
            or not isinstance(root_seed, Integral)
            or root_seed < 0
        ):
            raise ValueError("root_seed must be a nonnegative integer")
        self._root_seed: int = int(root_seed)
        self._box_ids: tuple[str, ...] = _box_ids(box_ids)
        self._streams: dict[tuple[str, str], np.random.Generator] = {}

    @property
    def root_seed(self) -> int:
        """Return the immutable root seed."""
        return self._root_seed

    @property
    def box_ids(self) -> tuple[str, ...]:
        """Return logical IDs in this view's numeric-array order."""
        return self._box_ids

    def with_box_ids(self, box_ids: tuple[str, ...]) -> CPUExecutionContext:
        """Create a reordered, reduced, or extended continuation-sharing view.

        New IDs acquire streams lazily; existing IDs retain continuation even
        when absent from a view. The caller must reorder arrays consistently.
        """
        result = CPUExecutionContext(self.root_seed, box_ids)
        result._streams = self._streams
        return result

    def _begin(self, n_boxes: int) -> _CPUStreamTransaction:
        """Validate layout and begin a private substep transaction."""
        if n_boxes != len(self.box_ids):
            raise ValueError("context box_ids must match the aerosol box count")
        return _CPUStreamTransaction(self)

    def _new_generator(
        self, namespace: str, box_id: str
    ) -> np.random.Generator:
        """Derive a versioned PCG64 stream independently of lane/order."""
        payload = json.dumps(
            ["particula.cpu_rng", 1, self.root_seed, namespace, box_id],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("ascii")
        seed = int.from_bytes(hashlib.sha256(payload).digest(), "big")
        return np.random.Generator(np.random.PCG64(seed))


class _CPUStreamTransaction:
    """Stage stream continuation without touching caller-owned state."""

    def __init__(self, context: CPUExecutionContext) -> None:
        self._context = context
        self._staged: dict[tuple[str, str], np.random.Generator] = {}

    def generator(self, namespace: str, box: int) -> np.random.Generator:
        """Return a staged generator for one process and logical box."""
        if not isinstance(namespace, str) or not namespace:
            raise ValueError("process namespace must be a nonempty string")
        key = namespace, self._context.box_ids[box]
        if key not in self._staged:
            current = self._context._streams.get(key)
            self._staged[key] = (
                deepcopy(current)
                if current is not None
                else self._context._new_generator(*key)
            )
        return self._staged[key]

    def commit(self) -> None:
        """Publish staged streams after successful numeric publication."""
        self._context._streams.update(self._staged)
