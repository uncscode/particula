"""Native CPU/adaptive disk restart with data records and caller physics."""

from dataclasses import dataclass, replace

import numpy as np

from particula import __version__
from particula.execution.archive import read_archive, write_archive
from particula.execution.checkpoint_io import _decode, _Encoder, _types
from particula.execution.condensation_ode import CondensationODE
from particula.execution.cpu_configuration import condensation_identity
from particula.execution.cpu_simulation import (
    CPUCheckpoint,
    pack_native,
    unpack_native,
)
from particula.execution.ode import AdaptiveSolver, StepStatistics


@dataclass(frozen=True)
class CondensationCheckpoint:
    """Keep scaled state, controller history and native configuration."""

    schema_version: int
    base: tuple
    configuration: object
    state: np.ndarray
    scale: np.ndarray
    time: float
    method: str
    rtol: np.ndarray
    atol: np.ndarray
    next_step: float
    min_step: float
    max_step: float
    max_trials: int
    last_error: float | None
    statistics: StepStatistics
    viscosity: np.ndarray | None


def checkpoint_condensation(owner):
    """Capture adaptive state without advancing or resetting its controller."""
    if type(owner) is not CondensationODE:
        raise TypeError("requires CondensationODE")
    if owner._forcing is not None:
        raise ValueError("callable forcing has no exact disk identity")
    solver = owner.solver
    plan = condensation_identity(owner._strategy)
    return CondensationCheckpoint(
        1,
        pack_native(owner._base),
        plan,
        solver.state,
        owner._scale.copy(),
        solver.time,
        solver.method,
        solver.rtol.copy(),
        solver.atol.copy(),
        solver.next_step,
        solver.min_step,
        solver.max_step,
        solver.max_trials,
        solver.last_error,
        replace(solver.statistics),
        None if owner._viscosity is None else owner._viscosity.copy(),
    )


def resume_condensation(saved, strategy):
    """Restore the same scaled coordinates and adaptive continuation policy."""
    if (
        type(saved) is not CondensationCheckpoint
        or type(saved.schema_version) is not int
        or saved.schema_version != 1
    ):
        raise ValueError("unsupported adaptive condensation checkpoint")
    base = unpack_native(saved.base)
    plan = condensation_identity(strategy)
    if plan != saved.configuration:
        raise ValueError("condensation physics differs from checkpoint")
    owner = CondensationODE(base, strategy, dynamic_viscosity=saved.viscosity)
    if not np.array_equal(saved.scale, owner._scale):
        raise ValueError("adaptive coordinates differ from checkpoint")
    if saved.last_error is not None and (
        not np.isfinite(saved.last_error) or not 0 <= saved.last_error <= 1
    ):
        raise ValueError("invalid accepted error estimate")
    if type(saved.statistics) is not StepStatistics or any(
        type(value) is not int or value < 0
        for value in vars(saved.statistics).values()
    ):
        raise ValueError("invalid adaptive work counters")
    owner.solver = AdaptiveSolver(
        owner.problem,
        saved.state,
        time=saved.time,
        method=saved.method,
        rtol=saved.rtol,
        atol=saved.atol,
        first_step=saved.next_step,
        min_step=saved.min_step,
        max_step=saved.max_step,
        max_trials=saved.max_trials,
    )
    owner.solver.last_error = saved.last_error
    owner.solver.statistics = replace(saved.statistics)
    return owner


def save_cpu_checkpoint(checkpoint, path, *, compressed=True):
    """Atomically save supported native CPU or adaptive controller data only."""
    if type(checkpoint) not in (CPUCheckpoint, CondensationCheckpoint):
        raise TypeError("unsupported native CPU checkpoint")
    encoder = _Encoder()
    tree = encoder.encode(checkpoint)
    metadata = {
        "schema": "particula-cpu-restart",
        "version": 1,
        "package_version": __version__,
        "numpy_version": np.__version__,
        "checkpoint": tree,
    }
    write_archive(
        path,
        kind="restart",
        metadata=metadata,
        arrays=encoder.arrays,
        blobs=encoder.blobs,
        compressed=compressed,
    )


def load_cpu_checkpoint(path, *, max_bytes=512 * 1024 * 1024):
    """Read CPU restart data; resume separately with matching physics."""
    metadata, arrays, blobs = read_archive(
        path, kind="restart", max_bytes=max_bytes
    )
    try:
        if (
            metadata["schema"] != "particula-cpu-restart"
            or type(metadata["version"]) is not int
            or metadata["version"] != 1
            or metadata["package_version"] != __version__
            or metadata["numpy_version"] != np.__version__
        ):
            raise ValueError(
                "CPU restart requires matching schema/package/runtime"
            )
        records, enums = _types()
        saved = _decode(metadata["checkpoint"], arrays, blobs, records, enums)
        if type(saved) not in (CPUCheckpoint, CondensationCheckpoint):
            raise ValueError("unsupported CPU restart record")
        return saved
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("malformed CPU restart record") from error
