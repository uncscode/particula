"""Exact-device disk continuation for native adaptive GPU condensation.

Checkpointing explicitly downloads accepted scaled coordinates and tolerances.
Restart uploads those exact coordinates, not a multiply/divide reconstruction
from a physical snapshot. Trial scratch is recomputed on the next step.
"""

from dataclasses import dataclass, replace

import numpy as np
import warp as wp

from particula import __version__, _cpu_batch
from particula.execution._adaptive_condensation_plan import (
    adaptive_condensation_identity,
)
from particula.execution.archive import read_archive, write_archive
from particula.execution.checkpoint_io import _decode, _Encoder, _types
from particula.execution.cpu_simulation import pack_native, unpack_native
from particula.execution.forcing import EnvironmentForcing, checkpoint_forcing
from particula.execution.gpu_condensation_ode import GPUCondensationODE
from particula.execution.ode import AdaptiveSolver, StepStatistics


@dataclass(frozen=True)
class GPUCondensationCheckpoint:
    """Own accepted arrays, exact device identity and controller data."""

    schema_version: int
    runtime: tuple
    device: tuple
    base: tuple
    configuration: object
    state: np.ndarray
    scale: np.ndarray
    rtol: np.ndarray
    atol: np.ndarray
    time: float
    method: str
    next_step: float
    min_step: float
    max_step: float
    max_trials: int
    last_error: float | None
    statistics: StepStatistics
    control_readbacks: int
    forcing: EnvironmentForcing | None


def _runtime():
    return (__version__, np.__version__, wp.__version__)


def _device_identity(device):
    return (
        device.alias,
        device.name,
        device.is_cuda,
        device.arch,
        device.uuid if device.is_cuda else None,
    )


def checkpoint_gpu_condensation(owner):
    """Explicitly download an accepted adaptive checkpoint outside capture."""
    if type(owner) is not GPUCondensationODE:
        raise TypeError("requires GPUCondensationODE")
    owner._check_ready()
    forcing = checkpoint_forcing(owner._forcing)
    return GPUCondensationCheckpoint(
        5,
        _runtime(),
        _device_identity(owner.device),
        pack_native(owner._base),
        adaptive_condensation_identity(
            owner._strategy, owner._plan.values["bat_table"]
        ),
        owner._state.numpy().copy(),
        owner._host_scale.copy(),
        owner._rtol.numpy().copy(),
        owner._atol.numpy().copy(),
        owner.time,
        owner.method,
        owner.next_step,
        owner.min_step,
        owner.max_step,
        owner.max_trials,
        owner.last_error,
        replace(owner.statistics),
        owner.control_readbacks,
        forcing,
    )


def _validate(saved):
    """Validate all host state before allocation or archive replacement."""
    if (
        type(saved) is not GPUCondensationCheckpoint
        or type(saved.schema_version) is not int
        or saved.schema_version != 5
        or saved.runtime != _runtime()
    ):
        raise ValueError(
            "adaptive GPU restart requires matching schema/runtime"
        )
    if (
        type(saved.device) is not tuple
        or len(saved.device) != 5
        or type(saved.device[0]) is not str
        or type(saved.device[1]) is not str
        or type(saved.device[2]) is not bool
        or type(saved.device[3]) is not int
        or (saved.device[2] and type(saved.device[4]) is not str)
        or (not saved.device[2] and saved.device[4] is not None)
    ):
        raise ValueError("invalid adaptive GPU device identity")
    base = unpack_native(saved.base)
    checkpoint_forcing(saved.forcing)
    _cpu_batch.validate(base)
    _validate_controls(saved)
    return base, _validate_arrays(saved, base)


def _validate_arrays(saved, base):
    """Check scaled arrays and native inventory before any device upload."""
    shape = base.particles.masses.shape
    size = base.particles.masses.size + base.gas.concentration.size
    expected_scale = np.concatenate(
        (
            np.broadcast_to(
                np.maximum(
                    base.particles.masses.sum(axis=2, keepdims=True), 1e-30
                ),
                shape,
            ).ravel(),
            np.maximum(base.gas.concentration, 1e-12).ravel(),
        )
    )
    for name in ("state", "scale", "rtol", "atol"):
        value = getattr(saved, name)
        if (
            type(value) is not np.ndarray
            or value.dtype != np.dtype("float64")
            or value.shape != (size,)
            or not np.all(np.isfinite(value))
            or np.any(value < 0)
        ):
            raise ValueError("invalid adaptive GPU coordinates/tolerances")
    if not np.array_equal(saved.scale, expected_scale):
        raise ValueError("adaptive coordinates differ from checkpoint base")
    if np.any(saved.atol <= 0) or np.any(saved.rtol != saved.rtol[0]):
        raise ValueError("invalid adaptive GPU tolerance policy")
    with np.errstate(over="ignore", invalid="ignore"):
        physical = saved.state * saved.scale
        absolute = saved.atol * saved.scale
    if not np.all(np.isfinite(physical)) or (
        not np.all(np.isfinite(absolute)) or np.any(absolute <= 0)
    ):
        raise ValueError(
            "adaptive physical state/tolerances are unrepresentable"
        )
    accepted = _cpu_batch.detached(base)
    mass_size = base.particles.masses.size
    accepted.particles.masses[:] = physical[:mass_size].reshape(shape)
    accepted.gas.concentration[:] = physical[mass_size:].reshape(
        base.gas.concentration.shape
    )
    _cpu_batch.validate(accepted)
    return absolute


def _validate_controls(saved):
    """Check the accepted clock, controller limits and work history."""
    if saved.method not in ("rk23", "ros3"):
        raise ValueError("adaptive GPU restart supports rk23 or ros3")
    AdaptiveSolver._scalar(saved.time, "time", positive=False)
    for name in ("min_step", "max_step", "next_step"):
        AdaptiveSolver._scalar(getattr(saved, name), name)
    if not saved.min_step <= saved.next_step <= saved.max_step:
        raise ValueError("invalid adaptive timestep bounds")
    if type(saved.max_trials) is not int or saved.max_trials < 1:
        raise ValueError("invalid adaptive trial budget")
    if saved.last_error is not None and (
        not np.isfinite(saved.last_error) or not 0 <= saved.last_error <= 1
    ):
        raise ValueError("invalid accepted error estimate")
    if (
        type(saved.statistics) is not StepStatistics
        or any(
            type(value) is not int or value < 0
            for value in vars(saved.statistics).values()
        )
        or type(saved.control_readbacks) is not int
        or saved.control_readbacks < 0
        or saved.statistics.domain_rejected > saved.statistics.rejected
    ):
        raise ValueError("invalid adaptive work counters")


def resume_gpu_condensation(saved, strategy, *, device):
    """Restore exact coordinates and controller on the same explicit device."""
    base, absolute = _validate(saved)
    if adaptive_condensation_identity(strategy) != saved.configuration:
        raise ValueError("condensation physics differs from checkpoint")
    target = wp.get_device(device)
    if _device_identity(target) != saved.device:
        raise ValueError("adaptive GPU restart requires the exact saved device")
    if target.is_capturing:
        raise RuntimeError("adaptive restart cannot run during CUDA capture")
    mass_size = base.particles.masses.size
    owner = GPUCondensationODE(
        base,
        strategy,
        device=target,
        method=saved.method,
        time=saved.time,
        particle_atol=absolute[:mass_size].reshape(base.particles.masses.shape),
        gas_atol=absolute[mass_size:].reshape(base.gas.concentration.shape),
        rtol=float(saved.rtol[0]),
        first_step=saved.next_step,
        min_step=saved.min_step,
        max_step=saved.max_step,
        max_trials=saved.max_trials,
        forcing=saved.forcing,
    )
    # Constructor arithmetic is only preflight/setup. These exact saved values
    # remain the continuation authority, including the tolerance roundoff bits.
    owner._state.assign(saved.state.copy())
    owner._atol.assign(saved.atol.copy())
    owner._rtol.assign(saved.rtol.copy())
    owner.last_error = saved.last_error
    owner.statistics = replace(saved.statistics)
    owner.control_readbacks = saved.control_readbacks
    return owner


def save_gpu_checkpoint(checkpoint, path, *, compressed=True):
    """Atomically store downloaded adaptive state as non-executable data."""
    _validate(checkpoint)
    encoder = _Encoder()
    tree = encoder.encode(checkpoint)
    write_archive(
        path,
        kind="restart",
        metadata={
            "schema": "particula-adaptive-gpu-restart",
            "version": 1,
            "checkpoint": tree,
        },
        arrays=encoder.arrays,
        blobs=encoder.blobs,
        compressed=compressed,
    )


def load_gpu_checkpoint(path, *, max_bytes=512 * 1024 * 1024):
    """Load validated adaptive host data without creating a device owner."""
    metadata, arrays, blobs = read_archive(
        path, kind="restart", max_bytes=max_bytes
    )
    try:
        if (
            metadata["schema"] != "particula-adaptive-gpu-restart"
            or type(metadata["version"]) is not int
            or metadata["version"] != 1
        ):
            raise ValueError("unsupported adaptive GPU restart schema")
        records, enums = _types()
        saved = _decode(metadata["checkpoint"], arrays, blobs, records, enums)
        _validate(saved)
        return saved
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("malformed adaptive GPU restart record") from error
