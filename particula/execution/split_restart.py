"""Complete-cycle CPU split restart with caller-supplied compatible flows.

Only the known condensation/dilution flow factories have a disk identity.
Neither callbacks nor live strategies are serialized or imported from a file.
Component controllers restart at every split boundary, so a complete-cycle
checkpoint needs no hidden half-step or component-controller state.
"""

from dataclasses import dataclass

from particula.execution.cpu_configuration import _freeze
from particula.execution.cpu_simulation import pack_native, unpack_native
from particula.execution.splitting import (
    Flow,
    SplitGroup,
    SplitSimulation,
    _CondensationAdvance,
    _DilutionAdvance,
)


@dataclass(frozen=True)
class SplitCheckpoint:
    """Own native accepted state, nested flow identity and the outer clock."""

    schema_version: int
    state: tuple
    configuration: tuple
    time: float
    time_step: float
    completed_steps: int


def _identity(node):
    if type(node) is SplitGroup:
        return (
            "group",
            node.method,
            tuple(_identity(child) for child in node.children),
        )
    if type(node) is not Flow:
        raise TypeError(
            "split restart requires known groups and flow factories"
        )
    advance = node.advance
    configuration: tuple
    if type(advance) is _DilutionAdvance:
        configuration = ("dilution", _freeze(advance._strategy))
    elif type(advance) is _CondensationAdvance:
        configuration = (
            "condensation",
            _freeze(advance._strategy),
            tuple(
                (name, _freeze(value))
                for name, value in sorted(advance._controls.items())
            ),
        )
    else:
        raise TypeError("custom flow callback has no exact disk identity")
    return ("flow", node.name, node.kind, node.order, configuration)


def checkpoint_split(owner):
    """Capture a detached complete accepted cycle and its declared ordering."""
    if type(owner) is not SplitSimulation:
        raise TypeError("requires SplitSimulation")
    return SplitCheckpoint(
        1,
        pack_native(owner.snapshot()),
        _identity(owner.group),
        owner.time,
        owner.time_step,
        owner.completed_steps,
    )


def resume_split(saved, group):
    """Resume with an explicitly supplied matching nested group and controls."""
    if (
        type(saved) is not SplitCheckpoint
        or type(saved.schema_version) is not int
        or saved.schema_version != 1
    ):
        raise ValueError("unsupported split checkpoint schema")
    if _identity(group) != saved.configuration:
        raise ValueError("split group physics, controls or ordering differs")
    if type(saved.completed_steps) is not int or saved.completed_steps < 0:
        raise ValueError("invalid completed split-cycle count")
    owner = SplitSimulation(
        unpack_native(saved.state),
        group,
        time_step=saved.time_step,
        time=saved.time,
    )
    owner.completed_steps = saved.completed_steps
    return owner
