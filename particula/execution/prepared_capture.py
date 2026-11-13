"""Private, owner-bound CUDA capture for ordered subsets of prepared physics.

The complete canonical resident scheduler keeps its stricter public graph
qualification contract. This owner exposes no transferable graph handle: only
its own capture method can install a graph, and step/run remain the sole replay
entry points. Both paths use the resident guard and native-operation leases.
"""

from dataclasses import dataclass, fields, is_dataclass
from typing import Any


def _signature(owner):  # noqa: C901
    """Retain identities and array descriptors without reading device values."""
    identities = []
    arrays = []
    seen = set()
    terminals = (owner._session, owner._registry, owner._guard)

    def visit(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        identities.append(value)
        if any(value is terminal for terminal in terminals):
            return
        if isinstance(value, owner._wp.array):
            arrays.append(
                (
                    value.ptr,
                    value.shape,
                    value.strides,
                    value.dtype,
                    value.device,
                )
            )
        elif isinstance(value, dict):
            for key, item in value.items():
                visit(key)
                visit(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
        elif is_dataclass(value) and not isinstance(value, type):
            for field in fields(value):
                visit(getattr(value, field.name))

    for value in (
        owner._session,
        owner._registry,
        owner._guard,
        owner._device,
        owner._time_step,
        owner._stream,
        owner._operations,
        owner._plans,
        owner._bindings,
        owner._auxiliary,
        owner._device_failure,
        owner._refresh,
    ):
        visit(value)
    # Primaries are frozen carriers, but identity validation also rejects
    # accidental low-level reassignment of fields beneath the private owner.
    for carrier in (
        owner._session.particles,
        owner._session.gas,
        owner._session.environment,
    ):
        visit(carrier)
    return tuple(identities), tuple(arrays)


@dataclass
class _OwnedGraph:
    """Keep native storage private and pinned until its stream has drained."""

    owner: Any
    graph: Any
    identities: tuple[Any, ...]
    arrays: tuple[Any, ...]
    valid: bool = True

    def invalidate(self):
        """Revoke replay without releasing storage used by queued work."""
        self.valid = False

    def validate(self, owner):
        """Reject stale bindings before graph launch or random draws."""
        if not self.valid or self.owner is not owner:
            raise ValueError(
                "prepared CUDA graph is invalid or belongs to another owner"
            )
        owner._registry.validate_pinned_session(owner._session)
        identities, arrays = _signature(owner)
        if (
            len(identities) != len(self.identities)
            or any(
                a is not b
                for a, b in zip(identities, self.identities, strict=True)
            )
            or arrays != self.arrays
        ):
            self.invalidate()
            raise ValueError(
                "prepared CUDA graph binding changed; create a new simulation"
            )


def _admit_capture(owner):
    """Check supported target, binding and stream before capture work."""
    owner._require_active()
    if owner._stream is None:
        raise ValueError("CUDA graph capture requires a CUDA device")
    if owner._operations is None:
        raise ValueError("capture requires a bound process plan")
    if owner._capture is not None:
        raise ValueError("simulation already owns a graph; uncapture first")
    owner.synchronize()
    if owner._stream.is_capturing:
        raise ValueError("simulation stream is already capturing")


def capture(owner):
    """Capture launches only; recording performs no step or scientific write."""
    _admit_capture(owner)
    wp = owner._wp
    identities, arrays = _signature(owner)
    lease = owner._session._acquire_native_operation("capture")
    begun = False
    try:
        with owner._scope():
            # Force loading happens before Warp begins the native recording;
            # unlike a warmup step, this consumes no random words or gas.
            wp.capture_begin(stream=owner._stream, force_module_load=True)
            begun = True
            for operation in owner._operations:
                operation()
            graph = wp.capture_end(stream=owner._stream)
            begun = False
            if graph is None:
                raise ValueError("Warp capture did not return a graph")
            owned = _OwnedGraph(owner, graph, identities, arrays)
            owned.validate(owner)
            owner._capture = owned
    except BaseException as error:
        cleanup_error = None
        if begun and owner._stream.is_capturing:
            try:
                # Warp 1.11 has no separate public abort/release operation.
                # End the recording and drop the unlaunched graph reference.
                wp.capture_end(stream=owner._stream)
            except BaseException as caught:
                cleanup_error = caught
        # Release the native lease before faulting (a terminal transition).
        owner._session._release_native_operation(lease)
        owner._fault()
        if cleanup_error is not None:
            raise error from cleanup_error
        raise
    finally:
        owner._session._release_native_operation(lease)


def replay(owner):
    """Launch only this owner's issued graph inside its single step token."""
    owned = owner._capture
    if type(owned) is not _OwnedGraph:
        raise TypeError("prepared graph was not issued by capture")
    owned.validate(owner)
    lease = owner._session._acquire_native_operation("replay")
    try:
        owner._wp.capture_launch(owned.graph, stream=owner._stream)
    finally:
        owner._session._release_native_operation(lease)
