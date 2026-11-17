"""Explicit deterministic process grouping and accepted complete-cycle clocks.

Each component advances its own physical interval. In a symmetric group,
outer components advance [t,t+h/2] then [t+h/2,t+h], around the inner full step.
Groups are never flattened. Observers belong outside this physics scheduler.
Stochastic updates and equilibrium projections require separate contracts and
are rejected here rather than being silently passed through symmetric stages.
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable

import numpy as np

from particula import _cpu_batch
from particula.aerosol import Aerosol


@dataclass(frozen=True)
class Flow:
    """Declare an exact map or deterministic continuous component.

    advance(state, start_time, duration) receives detached native state and may
    mutate/return it. It must not mutate external resources or consume RNG.
    order declares the component's convergence order, not a scheduler guarantee.
    Exact finite-step maps use kind='exact'; continuous methods use
    kind='continuous'. Neither runnable.rate nor execute is inferred here.
    """

    name: str
    advance: Callable[[Aerosol, float, float], Aerosol]
    kind: str = "continuous"
    order: int = 2

    def __post_init__(self):
        """Reject ambiguous names and unsupported process contracts early."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("flow requires a nonempty instance name")
        if self.kind not in ("exact", "continuous"):
            raise ValueError(
                "only deterministic continuous/exact flows supported"
            )
        if not callable(self.advance):
            raise TypeError("advance must be callable")
        if type(self.order) is not int or self.order < 1:
            raise ValueError("order must be a positive integer")


@dataclass(frozen=True)
class SplitGroup:
    """Retain nesting and sequential or symmetric composition."""

    children: tuple
    method: str = "strang"

    def __post_init__(self):
        """Freeze groups and validate component accuracy declarations."""
        object.__setattr__(self, "children", tuple(self.children))
        if not self.children or self.method not in ("sequential", "strang"):
            raise ValueError(
                "nonempty group and sequential/strang method required"
            )
        if any(
            not isinstance(child, (Flow, SplitGroup)) for child in self.children
        ):
            raise TypeError("groups contain only declared flows or groups")
        if self.method == "strang" and any(
            _order(child) < 2 for child in self.children
        ):
            raise ValueError("Strang requires at least second-order components")
        names = [flow.name for flow in _leaves(self)]
        if len(names) != len(set(names)):
            raise ValueError("flow instance names must be unique across groups")


def _order(node):
    if isinstance(node, Flow):
        return np.inf if node.kind == "exact" else node.order
    return min(
        1 if node.method == "sequential" else 2,
        *(_order(child) for child in node.children),
    )


def _leaves(node):
    if isinstance(node, Flow):
        yield node
    else:
        for child in node.children:
            yield from _leaves(child)


def _advance(node, state, time, duration):
    if isinstance(node, Flow):
        result = node.advance(state, time, duration)
        _cpu_batch.validate(result)
        return result
    if node.method == "sequential":
        for child in node.children:
            state = _advance(child, state, time, duration)
        return state
    for child in node.children[:-1]:
        state = _advance(child, state, time, duration / 2)
    state = _advance(node.children[-1], state, time, duration)
    for child in reversed(node.children[:-1]):
        state = _advance(child, state, time + duration / 2, duration / 2)
    return state


class SplitSimulation:
    """Own native accepted state and publish only complete deterministic cycles.

    A component failure discards the full current cycle; earlier accepted cycles
    remain. Callbacks must obey Flow's pure-resource contract. Fixed outer steps
    limit splitting error independently of adaptive component error control.
    """

    def __init__(self, aerosol, group, *, time_step, time=0.0):
        """Validate schedule and detach the initial accepted state."""
        if not isinstance(group, (Flow, SplitGroup)):
            raise TypeError("group must be an explicit Flow or SplitGroup")
        _cpu_batch.validate(aerosol)
        self._state = _cpu_batch.detached(aerosol)
        self.group = group
        self.time_step = _cpu_batch.duration(time_step, 1)
        self._time = _cpu_batch.duration(time, 1)
        if self.time_step <= 0:
            raise ValueError("time_step must be positive")
        self.completed_steps = 0

    @property
    def time(self):
        """Return accepted absolute time [s]."""
        return self._time

    def step(self, bound):
        """Advance one complete cycle, shortened to land on bound [s]."""
        bound = _cpu_batch.duration(bound, 1)
        if bound < self.time:
            raise ValueError("bound precedes accepted time")
        if bound == self.time:
            return self.time
        from particula.execution.forcing import EnvironmentForcing

        for flow in _leaves(self.group):
            if type(flow.advance) is _CondensationAdvance:
                forcing = flow.advance._controls.get("forcing")
                if type(forcing) is EnvironmentForcing:
                    bound = min(bound, forcing.next_event(self.time))
        duration = min(self.time_step, bound - self.time)
        if self.time + duration == self.time:
            raise RuntimeError(
                "cycle duration cannot advance floating-point time"
            )
        candidate = _advance(
            self.group, _cpu_batch.detached(self._state), self.time, duration
        )
        # Detach callback-owned results before they become accepted state.
        accepted = _cpu_batch.detached(candidate)
        self._state = accepted
        self._time = (
            bound if duration == bound - self.time else self.time + duration
        )
        self.completed_steps += 1
        return self.time

    def snapshot(self):
        """Return detached state after a complete accepted cycle."""
        return _cpu_batch.detached(self._state)


class _CondensationAdvance:
    """Retain a detached component for explicit restart identification."""

    def __init__(self, strategy, controls):
        self._strategy = deepcopy(strategy)
        self._controls = deepcopy(controls)

    def __call__(self, state, time, duration):
        from particula.execution.condensation_ode import CondensationODE

        owner = CondensationODE(
            state, self._strategy, time=time, **self._controls
        )
        bound = time + duration
        while owner.time < bound:
            owner.step(bound)
        return owner.snapshot()


class _DilutionAdvance:
    """Retain exact dilution configuration independently of caller objects."""

    def __init__(self, coefficient):
        from particula.dynamics.dilution import DilutionStrategy

        self._strategy = DilutionStrategy(coefficient)

    def __call__(self, state, time, duration):
        for box in range(state.particles.n_boxes):
            view = _cpu_batch.box_view(state, box)
            self._strategy.step(view, duration)
        return state


def condensation_flow(name, strategy, **controls):
    """Construct continuous native condensation with explicit controls.

    Component controllers restart at split boundaries. The outer step and
    component tolerances must both be refined to establish trajectory accuracy.
    """
    return Flow(name, _CondensationAdvance(strategy, controls), order=2)


def dilution_flow(name, coefficient):
    """Construct exact native gas/particle dilution with hazard [s^-1]."""
    return Flow(name, _DilutionAdvance(coefficient), kind="exact")
