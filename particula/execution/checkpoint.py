"""Provide explicit in-memory checkpoints for concrete resident Warp sessions.

Checkpoints are immutable host snapshots for same-device recovery only.  They
need roughly one additional host copy of every resident primary and acquired
sidecar, in addition to detached CPU inspection containers.  They are not a
serializer, device migration mechanism, CPU fallback, or rollback facility.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, cast

import numpy as np

from particula.execution import Device, _isfinite_real
from particula.execution.gpu_session import (
    ResidentDimensions,
    ResidentLifecycle,
    ResidentSession,
    ResidentStepGuard,
    setup_resident_session,
)

_SCHEMA_VERSION = 1
_PRIMARY_ROLES = (
    ("particles", "masses"),
    ("particles", "concentration"),
    ("particles", "charge"),
    ("particles", "density"),
    ("particles", "volume"),
    ("gas", "molar_mass"),
    ("gas", "concentration"),
    ("gas", "vapor_pressure"),
    ("gas", "partitioning"),
    ("environment", "temperature"),
    ("environment", "pressure"),
    ("environment", "saturation_ratio"),
)


@dataclass(frozen=True)
class CheckpointPayload:
    """Store an immutable exact host representation of one Warp array.

    Attributes:
        family: Primary carrier family or acquired resource family.
        role: Array role within its family.
        dtype: Exact NumPy-compatible dtype spelling.
        shape: Exact array shape.
        data: Immutable contiguous array bytes.
        capacity: Optional collision capacity metadata.
    """

    family: str
    role: str
    dtype: str
    shape: tuple[int, ...]
    data: bytes
    capacity: int | None = None


@dataclass(frozen=True)
class ResidentCheckpoint:
    """Retain a versioned immutable inspection and recovery snapshot.

    Inspection carriers are detached and ``gas`` intentionally has no vapor
    pressure field.  Canonical payload bytes, not inspection carriers, are
    authoritative for restart.
    """

    schema_version: int
    carrier_type: str
    dimensions: object
    device: object
    gas_names: tuple[str, ...]
    completed_steps: int
    simulated_time: Real
    lifecycle: ResidentLifecycle
    particles: object
    gas: object
    environment: object
    payloads: tuple[CheckpointPayload, ...]


def _payload(
    family: str, role: str, value: Any, capacity: int | None = None
) -> CheckpointPayload:
    """Copy one synchronized Warp array into immutable canonical bytes."""
    array = np.ascontiguousarray(value.numpy())
    return CheckpointPayload(
        family,
        role,
        array.dtype.str,
        tuple(array.shape),
        array.tobytes(),
        capacity,
    )


def _validate_payload(item: object) -> CheckpointPayload:
    """Validate one immutable payload descriptor without materializing it."""
    if type(item) is not CheckpointPayload:
        raise TypeError(
            "checkpoint payloads must be exact CheckpointPayload values."
        )
    if type(item.family) is not str or type(item.role) is not str:
        raise TypeError("checkpoint payload family and role must be str.")
    if type(item.dtype) is not str or type(item.shape) is not tuple:
        raise TypeError("checkpoint payload dtype and shape are invalid.")
    if any(type(length) is not int or length < 0 for length in item.shape):
        raise ValueError("checkpoint payload shape is invalid.")
    if type(item.data) is not bytes:
        raise TypeError("checkpoint payload data must be immutable bytes.")
    if item.capacity is not None and (
        isinstance(item.capacity, bool)
        or not isinstance(item.capacity, Integral)
        or item.capacity <= 0
    ):
        raise ValueError("checkpoint payload capacity is invalid.")
    try:
        dtype = np.dtype(item.dtype)
    except TypeError as error:
        raise ValueError("checkpoint payload dtype is invalid.") from error
    size = int(np.prod(item.shape, dtype=np.int64)) * dtype.itemsize
    if len(item.data) != size:
        raise ValueError("checkpoint payload byte length is invalid.")
    return item


class ResidentCheckpointController:
    """Bind checkpoint/finalization operations to one exact resident binding."""

    def __init__(
        self, session: ResidentSession, registry: Any, guard: ResidentStepGuard
    ) -> None:
        """Bind exact active session, registry, and guard identities.

        Args:
            session: Active resident session to checkpoint.
            registry: Registry pinned to ``session``.
            guard: Closed-step guard bound to the same registry and session.
        """
        from particula.execution.gpu_resources import GPUResourceRegistry

        if type(session) is not ResidentSession:
            raise TypeError("session must be an exact ResidentSession.")
        if type(registry) is not GPUResourceRegistry:
            raise TypeError("registry must be an exact GPUResourceRegistry.")
        if type(guard) is not ResidentStepGuard:
            raise TypeError("guard must be an exact ResidentStepGuard.")
        if guard._session is not session or guard._registry is not registry:
            raise ValueError(
                "guard must match the resident session and registry."
            )
        registry.validate_pinned_session(session)
        self._session = session
        self._registry = registry
        self._guard = guard
        self._finalized: ResidentCheckpoint | None = None

    def _validate(self) -> None:
        """Validate active identity-bound lifecycle state before readback."""
        if self._session.lifecycle is not ResidentLifecycle.ACTIVE:
            raise ValueError("session.lifecycle must be ACTIVE.")
        if (
            self._guard._session is not self._session
            or self._guard._registry is not self._registry
        ):
            raise ValueError(
                "guard must match the resident session and registry."
            )
        self._registry.validate_pinned_session(self._session)
        self._guard.assert_step_closed()

    def checkpoint(self) -> ResidentCheckpoint:
        """Return a fresh immutable snapshot while leaving the session active.

        Returns:
            A new independent checkpoint record.
        """
        self._validate()
        import warp as wp

        from particula.gpu.conversion import (
            from_warp_environment_data,
            from_warp_gas_data,
            from_warp_particle_data,
        )

        wp.synchronize()
        particles = from_warp_particle_data(
            cast(Any, self._session.particles), sync=False
        )
        gas = from_warp_gas_data(
            cast(Any, self._session.gas),
            name=list(self._session.metadata.gas_names),
            sync=False,
        )
        environment = from_warp_environment_data(
            cast(Any, self._session.environment), sync=False
        )
        payloads = [
            _payload(
                family, role, getattr(getattr(self._session, family), role)
            )
            for family, role in _PRIMARY_ROLES
        ]
        payloads.extend(
            _payload(family, role, value, capacity)
            for family, role, value, capacity in (
                self._registry._enumerate_resources()
            )
        )
        return ResidentCheckpoint(
            _SCHEMA_VERSION,
            "ResidentSession",
            self._session.dimensions,
            self._session.metadata.device,
            self._session.metadata.gas_names,
            self._guard.completed_steps,
            self._guard.simulated_time,
            ResidentLifecycle.ACTIVE,
            particles,
            gas,
            environment,
            tuple(payloads),
        )

    def finalize(self) -> ResidentCheckpoint:
        """Create once, cache, and terminally finalize this resident session."""
        if self._finalized is not None:
            return self._finalized
        checkpoint = self.checkpoint()
        self._session._finalize_checkpoint()
        self._finalized = checkpoint
        return checkpoint


def restart_resident_session(  # noqa: C901
    checkpoint: ResidentCheckpoint,
    device: object,
) -> tuple[ResidentSession, Any, ResidentStepGuard]:
    """Restart a fresh compatible resident session from canonical bytes.

    The target device must exactly match the checkpoint declaration.  This
    explicit in-memory operation never migrates devices and has no rollback
    guarantee after an asynchronous device write has launched.
    """
    primary = _preflight_restart(checkpoint, device)
    target_device = cast(Device, device)

    def unpack(family: str, role: str) -> np.ndarray:
        item = primary[(family, role)]
        dtype = np.dtype(item.dtype)
        return np.frombuffer(item.data, dtype=dtype).reshape(item.shape).copy()

    from particula.gas import EnvironmentData, GasData
    from particula.particles import ParticleData

    particles = ParticleData(
        unpack("particles", "masses"),
        unpack("particles", "concentration"),
        unpack("particles", "charge"),
        unpack("particles", "density"),
        unpack("particles", "volume"),
    )
    gas = GasData(
        list(checkpoint.gas_names),
        unpack("gas", "molar_mass"),
        unpack("gas", "concentration"),
        unpack("gas", "partitioning")[0].astype(bool),
    )
    environment = EnvironmentData(
        unpack("environment", "temperature"),
        unpack("environment", "pressure"),
        unpack("environment", "saturation_ratio"),
    )
    session = setup_resident_session(particles, gas, environment, target_device)
    import warp as wp

    vapor: Any = wp.array(
        unpack("gas", "vapor_pressure"),
        dtype=wp.float64,
        device=target_device.native,
    )
    wp.copy(cast(Any, session.gas).vapor_pressure, vapor)
    from particula.execution.gpu_resources import GPUResourceRegistry

    registry = GPUResourceRegistry(session)
    resource_payloads = tuple(checkpoint.payloads[len(_PRIMARY_ROLES) :])
    if resource_payloads:
        manifests = {
            manifest.family: manifest for manifest in registry.manifests
        }
        grouped: dict[str, dict[str, Any]] = {}
        capacities: dict[str, int | None] = {}
        for item in resource_payloads:
            value = unpack(item.family, item.role)
            dtype = {"<f8": wp.float64, "<i4": wp.int32, "<u4": wp.uint32}.get(
                np.dtype(item.dtype).str
            )
            if dtype is None:
                raise ValueError("checkpoint resource dtype is invalid.")
            grouped.setdefault(item.family, {})[item.role] = wp.array(
                value, dtype=dtype, device=target_device.native
            )
            capacities[item.family] = item.capacity
        for family, bindings in grouped.items():
            manifest = manifests[family]
            if set(bindings) != {entry.role for entry in manifest.entries}:
                raise ValueError("checkpoint resource payloads are incomplete.")
            registry._acquire(manifest, bindings, capacities[family])
            if family == "condensation":
                registry.acquire_condensation()
            elif family == "coagulation":
                if capacities[family] is None:
                    raise ValueError("coagulation checkpoint lacks capacity.")
                registry.acquire_coagulation(cast(int, capacities[family]))
            elif family == "wall_loss":
                registry.acquire_wall_loss()
            else:
                registry.acquire_nucleation()
    guard = ResidentStepGuard(session, registry)
    guard._restore_checkpoint_counters(
        int(checkpoint.completed_steps), checkpoint.simulated_time
    )
    return session, registry, guard


def _preflight_restart(  # noqa: C901
    checkpoint: object, device: object
) -> dict[tuple[str, str], CheckpointPayload]:
    """Validate every descriptor before resident setup or Warp writes."""
    if type(checkpoint) is not ResidentCheckpoint:
        raise TypeError("checkpoint must be an exact ResidentCheckpoint.")
    if (
        checkpoint.schema_version != _SCHEMA_VERSION
        or checkpoint.carrier_type != "ResidentSession"
    ):
        raise ValueError("Unsupported resident checkpoint schema.")
    if checkpoint.lifecycle is not ResidentLifecycle.ACTIVE:
        raise ValueError("checkpoint must describe an active session.")
    if type(checkpoint.dimensions) is not ResidentDimensions:
        raise TypeError(
            "checkpoint dimensions must be exact ResidentDimensions."
        )
    if type(checkpoint.device) is not Device:
        raise TypeError("checkpoint device metadata is invalid.")
    if device != checkpoint.device:
        raise ValueError("device must exactly match checkpoint.device.")
    if type(checkpoint.gas_names) is not tuple or any(
        type(name) is not str for name in checkpoint.gas_names
    ):
        raise TypeError("checkpoint gas_names must be a tuple of str.")
    if len(checkpoint.gas_names) != checkpoint.dimensions.n_species:
        raise ValueError("checkpoint gas_names do not match dimensions.")
    if (
        isinstance(checkpoint.completed_steps, bool)
        or not isinstance(checkpoint.completed_steps, Integral)
        or checkpoint.completed_steps < 0
    ):
        raise ValueError(
            "completed_steps must be a non-boolean nonnegative int."
        )
    if (
        isinstance(checkpoint.simulated_time, bool)
        or not isinstance(checkpoint.simulated_time, Real)
        or not _isfinite_real(checkpoint.simulated_time)
        or checkpoint.simulated_time < 0
    ):
        raise ValueError("simulated_time must be finite and nonnegative.")
    if type(checkpoint.payloads) is not tuple:
        raise TypeError("checkpoint payloads must be a tuple.")
    payloads = tuple(_validate_payload(item) for item in checkpoint.payloads)
    keys = tuple((item.family, item.role) for item in payloads)
    if len(set(keys)) != len(keys):
        raise ValueError("checkpoint payload descriptors must be unique.")
    if keys[: len(_PRIMARY_ROLES)] != _PRIMARY_ROLES:
        raise ValueError(
            "checkpoint primary payload descriptors are incomplete."
        )
    primary = dict(zip(keys, payloads, strict=True))
    dimensions = checkpoint.dimensions
    expected = {
        ("particles", "masses"): (
            "<f8",
            (dimensions.n_boxes, dimensions.n_particles, dimensions.n_species),
        ),
        ("particles", "concentration"): (
            "<f8",
            (dimensions.n_boxes, dimensions.n_particles),
        ),
        ("particles", "charge"): (
            "<f8",
            (dimensions.n_boxes, dimensions.n_particles),
        ),
        ("particles", "density"): ("<f8", (dimensions.n_species,)),
        ("particles", "volume"): ("<f8", (dimensions.n_boxes,)),
        ("gas", "molar_mass"): ("<f8", (dimensions.n_species,)),
        ("gas", "concentration"): (
            "<f8",
            (dimensions.n_boxes, dimensions.n_species),
        ),
        ("gas", "vapor_pressure"): (
            "<f8",
            (dimensions.n_boxes, dimensions.n_species),
        ),
        ("gas", "partitioning"): (
            "<i4",
            (dimensions.n_boxes, dimensions.n_species),
        ),
        ("environment", "temperature"): ("<f8", (dimensions.n_boxes,)),
        ("environment", "pressure"): ("<f8", (dimensions.n_boxes,)),
        ("environment", "saturation_ratio"): (
            "<f8",
            (dimensions.n_boxes, dimensions.n_species),
        ),
    }
    for key, (dtype, shape) in expected.items():
        item = primary[key]
        if (
            item.dtype != dtype
            or item.shape != shape
            or item.capacity is not None
        ):
            raise ValueError("checkpoint primary payload metadata is invalid.")
    _validate_resource_payloads(payloads[len(_PRIMARY_ROLES) :], dimensions)
    return primary


def _validate_resource_payloads(  # noqa: C901
    payloads: tuple[CheckpointPayload, ...], dimensions: ResidentDimensions
) -> None:
    """Validate ordered acquired-sidecar descriptors before resident setup."""
    if not payloads:
        return
    index = 0
    seen_families: set[str] = set()
    shape_map = {
        "b": (dimensions.n_boxes,),
        "bn": (dimensions.n_boxes, dimensions.n_particles),
        "bs": (dimensions.n_boxes, dimensions.n_species),
        "bns": (
            dimensions.n_boxes,
            dimensions.n_particles,
            dimensions.n_species,
        ),
    }
    # The immutable class manifests do not require a live registry.  Their
    # deterministic order is reproduced from a lightweight temporary-free call.
    from particula.execution import gpu_resources

    for manifest in (
        gpu_resources._CONDENSATION,
        gpu_resources._COAGULATION,
        gpu_resources._WALL_LOSS,
        gpu_resources._NUCLEATION,
    ):
        if index >= len(payloads) or payloads[index].family != manifest.family:
            continue
        if manifest.family in seen_families:
            raise ValueError("checkpoint resource descriptors are duplicated.")
        seen_families.add(manifest.family)
        capacity = payloads[index].capacity
        if manifest.family == "coagulation" and capacity is None:
            raise ValueError("coagulation checkpoint lacks capacity.")
        if manifest.family != "coagulation" and capacity is not None:
            raise ValueError("checkpoint resource capacity is invalid.")
        for entry in manifest.entries:
            if index >= len(payloads):
                raise ValueError("checkpoint resource payloads are incomplete.")
            item = payloads[index]
            if entry.shape_kind == "bc2":
                shape: tuple[int, ...] = (
                    dimensions.n_boxes,
                    cast(int, capacity),
                    2,
                )
            else:
                shape = shape_map[entry.shape_kind]
            dtype = np.dtype(
                np.float64
                if entry.dtype.__name__ == "float64"
                else np.int32
                if entry.dtype.__name__ == "int32"
                else np.uint32
            ).str
            if (
                (item.family, item.role) != (manifest.family, entry.role)
                or item.dtype != dtype
                or item.shape != shape
                or item.capacity != capacity
            ):
                raise ValueError(
                    "checkpoint resource payload metadata is invalid."
                )
            index += 1
    if index != len(payloads):
        raise ValueError("checkpoint resource family is invalid.")


# The explicit aliases preserve a discoverable concrete-only recovery spelling.
restart_checkpoint = restart_resident_session
