"""Registration isolation and explicit capability admission tests."""

import pytest

from particula.execution.extension_registry import (
    PreparedOperation,
    ProcessCapabilities,
    ProcessRegistry,
)


class Process:
    """Example external process."""


class SpecializedProcess(Process):
    """A subclass can change scientific semantics."""


def test_exact_registration_and_snapshot_isolation():
    """Caller edits and subclassing cannot change prepared admission."""
    registry = ProcessRegistry()
    adapter = object()
    registry.register(Process, adapter)
    prepared = registry.snapshot()
    registry.register(SpecializedProcess, object())
    assert prepared.resolve(Process()) is adapter
    with pytest.raises(ValueError, match="no registered"):
        prepared.resolve(SpecializedProcess())
    with pytest.raises(RuntimeError, match="frozen"):
        prepared.register(SpecializedProcess, object())
    with pytest.raises(ValueError, match="already registered"):
        registry.register(Process, object())
    with pytest.raises(ValueError, match="no registered"):
        ProcessRegistry().resolve(Process())


def test_capabilities_are_independent_and_capture_requires_residency():
    """Capture is resident-only; checkpoint support is independent."""
    assert ProcessCapabilities(checkpoint=True).checkpoint
    assert ProcessCapabilities(resident=True, capture=True).capture
    with pytest.raises(ValueError, match="resident"):
        ProcessCapabilities(capture=True)
    with pytest.raises(TypeError, match="bool"):
        ProcessCapabilities(resident=1)


def test_operation_retains_resources_and_propagates_failure():
    """Delegation neither retries nor hides writer failure."""
    resource = object()
    calls = []

    def fail():
        calls.append(resource)
        raise RuntimeError("writer failed")

    operation = PreparedOperation("test.writer", fail, (resource,))
    assert operation.resources[0] is resource
    with pytest.raises(RuntimeError, match="writer failed"):
        operation()
    assert calls == [resource]
