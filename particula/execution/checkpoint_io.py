"""Safe disk transport for existing exact-device prepared checkpoints.

Only a fixed allowlist of data records/enums is decoded. No class name from a
file is imported and no pickle is accepted. Canonical bytes, RNG continuation,
process plans and auxiliary fields retain the existing restart authority; CPU
inspection arrays never replace canonical device payloads. Resuming still
requires the caller's compatible processes and explicit execution registry.
"""

from dataclasses import fields, replace
from enum import Enum

import numpy as np

from particula import __version__
from particula.execution.archive import read_archive, write_archive
from particula.execution.prepared_types import SimulationCheckpoint


def _types():
    """List permitted data constructors explicitly, without dynamic imports."""
    from particula.execution import Backend, Device
    from particula.execution.checkpoint import (
        CheckpointPayload,
        CommunicationCheckpointMetadata,
        ResidentCheckpoint,
        RNGContinuationCheckpoint,
        RNGContinuationPayload,
    )
    from particula.execution.cpu_restart import CondensationCheckpoint
    from particula.execution.cpu_simulation import CPUCheckpoint
    from particula.execution.extension_registry import ProcessCapabilities
    from particula.execution.forcing import EnvironmentForcing
    from particula.execution.gpu_restart import GPUCondensationCheckpoint
    from particula.execution.gpu_session import (
        ResidentDimensions,
        ResidentLifecycle,
    )
    from particula.execution.ode import StepStatistics
    from particula.execution.prepared_processes import _ProcessPlan
    from particula.execution.process_extensions import (
        GPUProcessPlan,
        GPUStateSpec,
    )
    from particula.execution.rng import StreamDescriptor, StreamKey
    from particula.execution.split_restart import SplitCheckpoint
    from particula.extension_fields import FieldSpec
    from particula.gas import EnvironmentData, GasData
    from particula.particles import ParticleData

    records = (
        CPUCheckpoint,
        CondensationCheckpoint,
        SplitCheckpoint,
        GPUCondensationCheckpoint,
        GPUProcessPlan,
        GPUStateSpec,
        ProcessCapabilities,
        EnvironmentForcing,
        StepStatistics,
        FieldSpec,
        SimulationCheckpoint,
        ResidentCheckpoint,
        CheckpointPayload,
        CommunicationCheckpointMetadata,
        RNGContinuationCheckpoint,
        RNGContinuationPayload,
        ResidentDimensions,
        Device,
        _ProcessPlan,
        StreamDescriptor,
        StreamKey,
        ParticleData,
        GasData,
        EnvironmentData,
    )
    enums = (Backend, ResidentLifecycle)
    return {cls.__name__: cls for cls in records}, {
        cls.__name__: cls for cls in enums
    }


class _Encoder:
    """Encode only known records and immutable/numeric primitive payloads."""

    def __init__(self):
        self.records, self.enums = _types()
        self.arrays = {}
        self.blobs = {}

    def encode(self, value):
        """Produce a JSON data tree and detached payload tables."""
        if isinstance(value, Enum):
            if type(value) not in self.enums.values():
                raise TypeError("unsupported checkpoint enum")
            return {"enum": type(value).__name__, "value": value.value}
        if value is None or type(value) in (str, bool, int, float):
            return value
        if isinstance(value, np.generic):
            return self.encode(value.item())
        if type(value) is np.ndarray:
            key = str(len(self.arrays))
            self.arrays[key] = value.copy()
            return {"array": key}
        if type(value) is bytes:
            key = str(len(self.blobs))
            self.blobs[key] = value
            return {"bytes": key}
        if type(value) in (tuple, list):
            return {
                "sequence": "tuple" if type(value) is tuple else "list",
                "items": [self.encode(item) for item in value],
            }
        if type(value) in self.records.values():
            return {
                "record": type(value).__name__,
                "fields": {
                    field.name: self.encode(getattr(value, field.name))
                    for field in fields(value)
                    if field.init
                },
            }
        raise TypeError(
            f"checkpoint disk codec does not support {type(value).__name__}"
        )


def _decode(value, arrays, blobs, records, enums, depth=0):
    if depth > 64:
        raise ValueError("checkpoint metadata nesting exceeds supported depth")
    if value is None or type(value) in (str, bool, int, float):
        return value
    if type(value) is not dict:
        raise ValueError("invalid checkpoint metadata node")
    if set(value) == {"array"}:
        return arrays[value["array"]].copy()
    if set(value) == {"bytes"}:
        return blobs[value["bytes"]]
    if set(value) == {"enum", "value"}:
        return enums[value["enum"]](value["value"])
    if set(value) == {"sequence", "items"}:
        if (
            value["sequence"] not in ("tuple", "list")
            or type(value["items"]) is not list
        ):
            raise ValueError("invalid checkpoint sequence")
        items = [
            _decode(item, arrays, blobs, records, enums, depth + 1)
            for item in value["items"]
        ]
        return tuple(items) if value["sequence"] == "tuple" else items
    if set(value) == {"record", "fields"}:
        cls = _record_class(value, records)
        arguments = {
            name: _decode(item, arrays, blobs, records, enums, depth + 1)
            for name, item in value["fields"].items()
        }
        return cls(**arguments)
    raise ValueError("unsupported checkpoint metadata node")


def _validate(checkpoint):
    from particula.execution.checkpoint import _preflight_restart
    from particula.execution.checkpoint_extensions import validate_envelope

    if type(checkpoint) is not SimulationCheckpoint:
        raise TypeError("disk restart requires an exact SimulationCheckpoint")
    validate_envelope(checkpoint)
    _preflight_restart(checkpoint.resident, checkpoint.resident.device)


def save_checkpoint(checkpoint, path, *, compressed=True):
    """Atomically store an already synchronized prepared checkpoint on host.

    Call owner.checkpoint() explicitly first; this function performs no device
    transfers and cannot run in CUDA capture. Unregistered data record types
    fail before touching the destination. Installed package/NumPy/Warp versions
    must match at load; restart also enforces device and process compatibility.
    """
    import warp as wp

    _validate(checkpoint)
    encoder = _Encoder()
    from particula.execution.checkpoint import _preflight_restart
    from particula.execution.prepared_checkpoint import _host_state

    restart = _preflight_restart(
        checkpoint.resident, checkpoint.resident.device
    )
    host = _host_state(checkpoint.resident, restart.primary)
    canonical = replace(
        checkpoint,
        resident=replace(
            checkpoint.resident,
            particles=host.particles,
            gas=host.gas,
            environment=host.environment,
        ),
    )
    tree = encoder.encode(canonical)
    metadata = {
        "schema": "particula-prepared-restart",
        "version": 1,
        "package_version": __version__,
        "numpy_version": np.__version__,
        "warp_version": wp.__version__,
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


def load_checkpoint(path, *, max_bytes=512 * 1024 * 1024):
    """Load exact checkpoint bytes without launching kernels or restoring state.

    Pass the result to the original compatible runnable.resume(). Never load an
    analysis file as restart, silently migrate schema, or recreate RNG words.
    """
    import warp as wp

    metadata, arrays, blobs = read_archive(
        path, kind="restart", max_bytes=max_bytes
    )
    try:
        if (
            metadata["schema"] != "particula-prepared-restart"
            or type(metadata["version"]) is not int
            or metadata["version"] != 1
        ):
            raise ValueError("unsupported prepared restart schema")
        if (
            metadata["package_version"] != __version__
            or metadata["numpy_version"] != np.__version__
            or metadata["warp_version"] != wp.__version__
        ):
            raise ValueError(
                "exact restart requires matching package/runtime versions"
            )
        records, enums = _types()
        checkpoint = _decode(
            metadata["checkpoint"], arrays, blobs, records, enums
        )
        _validate(checkpoint)
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(
            "malformed or unsupported prepared checkpoint"
        ) from error
    return checkpoint


def _record_class(value, records):
    cls = records[value["record"]]
    if set(value["fields"]) != {f.name for f in fields(cls) if f.init}:
        raise ValueError("checkpoint fields do not match the installed schema")
    return cls
