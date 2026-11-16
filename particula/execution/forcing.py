"""Immutable environmental forcing with explicit one-sided event semantics.

A table is right-continuous at knots. An integration interval freezes the
selected segment, so endpoint stages before a jump use the left-hand physics.
An accepted snapshot at the knot uses the right-hand environment. Event position
is derived from absolute time and needs no mutable restart cursor.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import cast

import numpy as np


@dataclass(frozen=True)
class EnvironmentForcing:
    """Declare temperature [K] and pressure [Pa] at absolute times [s].

    times starts at zero and increases strictly. Each value input may be a
    scalar, a vector with one value per time, or a time-by-box matrix. A width
    of one broadcasts across boxes. mode='step' holds each row to the next knot;
    mode='linear' interpolates. The last row is held indefinitely. Inputs are
    copied into immutable tuples; no user callbacks or file-named code is used.
    """

    times: tuple | list | np.ndarray
    temperature: float | tuple | list | np.ndarray
    pressure: float | tuple | list | np.ndarray
    mode: str = "linear"

    def __post_init__(self):
        """Validate finite positive environmental rows and freeze their data."""
        times = np.asarray(self.times, dtype=float)
        if (
            times.ndim != 1
            or not len(times)
            or times[0] != 0
            or not np.all(np.isfinite(times))
            or np.any(np.diff(times) <= 0)
            or self.mode not in ("step", "linear")
        ):
            raise ValueError(
                "forcing requires ordered times from zero and step/linear mode"
            )
        object.__setattr__(
            self, "times", tuple(float(value) for value in times)
        )
        for name in ("temperature", "pressure"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.ndim == 0:
                value = np.full((len(times), 1), value)
            elif value.ndim == 1:
                value = value[:, None]
            if (
                value.ndim != 2
                or value.shape[0] != len(times)
                or value.shape[1] < 1
                or not np.all(np.isfinite(value))
                or np.any(value <= 0)
            ):
                raise ValueError(
                    "forcing values must be positive finite time-by-box data"
                )
            object.__setattr__(
                self, name, tuple(tuple(float(x) for x in row) for row in value)
            )

    def _index(self, time):
        if not np.isfinite(time) or time < 0:
            raise ValueError("forcing time must be nonnegative and finite")
        return int(np.searchsorted(self.times, time, side="right") - 1)

    def _values(self, index, time):
        fraction = 0.0
        if self.mode == "linear" and index + 1 < len(self.times):
            fraction = min(
                1.0,
                max(
                    0.0,
                    (time - self.times[index])
                    / (self.times[index + 1] - self.times[index]),
                ),
            )
        result = []
        for raw in (self.temperature, self.pressure):
            rows = cast(
                tuple, raw
            )  # __post_init__ freezes every accepted input.
            value = np.asarray(rows[index])
            if fraction:
                value = value + fraction * (np.asarray(rows[index + 1]) - value)
            result.append(float(value[0]) if len(value) == 1 else value.copy())
        return tuple(result)

    def __call__(self, time):
        """Return right-continuous environmental values at absolute time."""
        return self._values(self._index(time), time)

    def next_event(self, time):
        """Return the next knot strictly after time, or infinity."""
        index = self._index(time)
        return self.times[index + 1] if index + 1 < len(self.times) else np.inf

    def segment(self, time):
        """Freeze one segment through its endpoint for all trial stages."""
        return _EnvironmentSegment(
            self, self._index(time), self.next_event(time)
        )


@dataclass(frozen=True)
class _EnvironmentSegment:
    table: EnvironmentForcing
    index: int
    end: float

    def __call__(self, time):
        return self.table._values(self.index, time)


@contextmanager
def forcing_interval(owner, bound):
    """Bind interval stages separately from accepted observation forcing."""
    original = owner._forcing
    if type(original) is not EnvironmentForcing:
        yield bound
        return
    owner._forcing = original.segment(owner.time)
    try:
        yield min(bound, owner._forcing.end)
    finally:
        owner._forcing = original


def checkpoint_forcing(forcing):
    """Admit only immutable tables as exact restart data."""
    if forcing is not None and type(forcing) is not EnvironmentForcing:
        raise ValueError("callable forcing has no exact disk identity")
    return forcing
