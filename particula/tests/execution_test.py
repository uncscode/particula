"""Tests for immutable execution capability metadata."""

import os
import re
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import cast

import pytest
from particula.execution import (
    Backend,
    Capability,
    CapabilityDeclaration,
    CapabilityMatrix,
    CapabilityRequirements,
    Device,
    ExecutionContext,
    ExecutionRequest,
    Process,
    _AdapterRegistry,
)


def _device() -> Device:
    """Create the standard CPU device declaration."""
    return Device(Backend.CPU, "cpu")


def _process() -> Process:
    """Create the standard process declaration."""
    return Process("condensation")


def _requirements(*names: str) -> CapabilityRequirements:
    """Create immutable requirements from capability names."""
    return CapabilityRequirements(frozenset(Capability(name) for name in names))


def _declaration(*names: str) -> CapabilityDeclaration:
    """Create a standard capability declaration."""
    return CapabilityDeclaration(_device(), _process(), _requirements(*names))


def _require_supported(
    matrix: CapabilityMatrix,
    device: Device,
    process: Process,
    requirements: CapabilityRequirements,
) -> object:
    """Call require for a supported request and return None."""
    matrix.require(device, process, requirements)
    return None


def test_declarations_compare_hash_and_freeze_by_value() -> None:
    """Test immutable declarations use value equality and hashing."""
    declaration = _declaration("isothermal")

    assert declaration == _declaration("isothermal")
    assert hash(declaration) == hash(_declaration("isothermal"))
    with pytest.raises(FrozenInstanceError):
        declaration.device = _device()  # type: ignore[misc]


@pytest.mark.parametrize(
    ("backend", "native", "exception", "message"),
    [
        ("cpu", "cpu", TypeError, "Device.backend must be a Backend."),
        (1, "cpu", TypeError, "Device.backend must be a Backend."),
        (
            Backend.CPU,
            "",
            ValueError,
            "Device.native must be a nonempty str "
            "without surrounding whitespace.",
        ),
        (
            Backend.CPU,
            " cpu",
            ValueError,
            "Device.native must be a nonempty str "
            "without surrounding whitespace.",
        ),
        (
            Backend.CPU,
            "cpu ",
            ValueError,
            "Device.native must be a nonempty str "
            "without surrounding whitespace.",
        ),
        (Backend.CPU, 1, TypeError, "Device.native must be a str."),
    ],
)
def test_device_rejects_invalid_values(
    backend: object,
    native: object,
    exception: type[Exception],
    message: str,
) -> None:
    """Test device fields reject invalid types and native identifiers."""
    with pytest.raises(exception, match=f"^{re.escape(message)}$"):
        Device(backend, native)  # type: ignore[arg-type]


def test_device_preserves_opaque_native_identifier() -> None:
    """Test a valid native identifier is retained without parsing."""
    assert Device(Backend.WARP, "cuda:0").native == "cuda:0"


@pytest.mark.parametrize(
    "constructor, field_name",
    [(Process, "Process.name"), (Capability, "Capability.name")],
)
@pytest.mark.parametrize("value", ["", " name", "name ", "Name", "bad-name", 1])
def test_names_reject_invalid_values(
    constructor: type[Process] | type[Capability],
    field_name: str,
    value: object,
) -> None:
    """Test process and capability names follow the declaration grammar."""
    exception = TypeError if not isinstance(value, str) else ValueError
    message = (
        f"{field_name} must be a str."
        if exception is TypeError
        else f"{field_name} must match ^[a-z][a-z0-9_]*$."
    )

    with pytest.raises(exception, match=f"^{re.escape(message)}$"):
        constructor(value)  # type: ignore[arg-type]


@pytest.mark.parametrize("constructor", [Process, Capability])
def test_names_accept_lowercase_identifier_grammar(
    constructor: type[Process] | type[Capability],
) -> None:
    """Test lowercase names may contain underscores and digits."""
    assert constructor("process_2").name == "process_2"


@pytest.mark.parametrize(
    "values",
    [
        {Capability("isothermal")},
        [Capability("isothermal")],
        (Capability("isothermal"),),
        "isothermal",
    ],
)
def test_requirements_reject_iterable_coercion(values: object) -> None:
    """Test requirements accept only a frozenset without coercion."""
    with pytest.raises(
        TypeError,
        match="^CapabilityRequirements.values must be a frozenset.$",
    ):
        CapabilityRequirements(values)  # type: ignore[arg-type]


def test_requirements_validate_members_and_empty_set() -> None:
    """Test requirement members are typed and an empty set remains valid."""
    empty: frozenset[Capability] = frozenset()

    assert CapabilityRequirements(empty).values is empty
    with pytest.raises(
        TypeError,
        match=(
            "^CapabilityRequirements.values must contain only Capability "
            "instances.$"
        ),
    ):
        CapabilityRequirements(frozenset({"isothermal"}))  # type: ignore[arg-type]


def test_requirements_reject_frozenset_subclasses() -> None:
    """Test requirements require the exact frozenset collection type."""

    class CapabilitySet(frozenset[Capability]):
        """A frozenset subclass that must not be accepted implicitly."""

    with pytest.raises(
        TypeError,
        match="^CapabilityRequirements.values must be a frozenset.$",
    ):
        CapabilityRequirements(CapabilitySet())


@pytest.mark.parametrize(
    ("device", "process", "requirements", "message"),
    [
        (
            "cpu",
            _process(),
            _requirements(),
            "CapabilityDeclaration.device must be a Device.",
        ),
        (
            _device(),
            "condensation",
            _requirements(),
            "CapabilityDeclaration.process must be a Process.",
        ),
        (
            _device(),
            _process(),
            frozenset(),
            "CapabilityDeclaration.requirements must be a CapabilityRequirements.",
        ),
    ],
)
def test_declaration_rejects_invalid_fields(
    device: object,
    process: object,
    requirements: object,
    message: str,
) -> None:
    """Test declarations validate each typed field independently."""
    with pytest.raises(TypeError, match=f"^{re.escape(message)}$"):
        CapabilityDeclaration(
            cast(Device, device),
            cast(Process, process),
            cast(CapabilityRequirements, requirements),
        )


def test_matrix_validates_collection_and_members() -> None:
    """Test matrices accept only typed immutable declaration collections."""
    empty: frozenset[CapabilityDeclaration] = frozenset()

    assert CapabilityMatrix(empty).declarations is empty
    assert CapabilityMatrix(frozenset({_declaration()})).declarations
    with pytest.raises(
        TypeError,
        match="^CapabilityMatrix.declarations must be a frozenset.$",
    ):
        CapabilityMatrix({_declaration()})  # type: ignore[arg-type]
    with pytest.raises(
        TypeError,
        match=(
            "^CapabilityMatrix.declarations must contain only "
            "CapabilityDeclaration instances.$"
        ),
    ):
        CapabilityMatrix(frozenset({_device()}))  # type: ignore[arg-type]


def test_matrix_supports_exact_declarations_without_composition() -> None:
    """Test exact, base, device, and process lookup rules."""
    cpu = _device()
    process = _process()
    warp_cpu = Device(Backend.WARP, "cpu")
    coagulation = Process("coagulation")
    unlisted_device = Device(Backend.CPU, "cpu:1")
    unlisted_process = Process("nucleation")
    isothermal = _requirements("isothermal")
    latent_heat = _requirements("latent_heat")
    combined = _requirements("isothermal", "latent_heat")
    matrix = CapabilityMatrix(
        frozenset(
            {
                CapabilityDeclaration(cpu, process, isothermal),
                CapabilityDeclaration(cpu, process, latent_heat),
                CapabilityDeclaration(cpu, process, combined),
                CapabilityDeclaration(warp_cpu, process, isothermal),
                CapabilityDeclaration(cpu, coagulation, isothermal),
            }
        )
    )

    assert matrix.supports(cpu, process, isothermal)
    assert matrix.supports(cpu, process, latent_heat)
    assert matrix.supports(cpu, process, combined)
    separate_matrix = CapabilityMatrix(
        frozenset(
            {
                CapabilityDeclaration(cpu, process, isothermal),
                CapabilityDeclaration(cpu, process, latent_heat),
            }
        )
    )
    assert not separate_matrix.supports(cpu, process, combined)
    assert matrix.supports(cpu, process, _requirements())
    assert matrix.supports(warp_cpu, process, _requirements())
    assert matrix.supports(cpu, coagulation, _requirements())
    assert not matrix.supports(unlisted_device, process, isothermal)
    assert not matrix.supports(cpu, unlisted_process, isothermal)
    assert not matrix.supports(unlisted_device, process, _requirements())
    assert not matrix.supports(cpu, unlisted_process, _requirements())
    empty_matrix = CapabilityMatrix(frozenset())
    assert not empty_matrix.supports(cpu, process, _requirements())
    assert not empty_matrix.supports(cpu, process, isothermal)
    assert matrix.supports(cpu, process, isothermal)


@pytest.mark.parametrize("method_name", ["supports", "require"])
def test_matrix_validates_request_arguments_in_order(method_name: str) -> None:
    """Test request arguments fail in fixed positional validation order."""
    matrix = CapabilityMatrix(frozenset())
    method = getattr(matrix, method_name)

    with pytest.raises(TypeError, match="^device must be a Device.$"):
        method("device", "process", "requirements")
    with pytest.raises(TypeError, match="^process must be a Process.$"):
        method(_device(), "process", "requirements")
    with pytest.raises(
        TypeError,
        match="^requirements must be a CapabilityRequirements.$",
    ):
        method(_device(), _process(), "requirements")


@pytest.mark.parametrize(
    ("device", "process", "unsupported"),
    [
        (_device(), _process(), _requirements("latent_heat")),
        (_device(), _process(), _requirements("isothermal", "latent_heat")),
        (Device(Backend.WARP, "cpu"), _process(), _requirements("isothermal")),
        (_device(), Process("coagulation"), _requirements("isothermal")),
        (_device(), Process("nucleation"), _requirements()),
    ],
)
def test_matrix_require_is_pure_and_reports_unsupported_request(
    device: Device,
    process: Process,
    unsupported: CapabilityRequirements,
) -> None:
    """Test require returns None or reports unsupported declarations exactly."""
    supported = _declaration("isothermal")
    matrix = CapabilityMatrix(frozenset({supported}))
    declarations_before = matrix.declarations
    hashes_before = {hash(declaration) for declaration in matrix.declarations}

    assert matrix.supports(_device(), _process(), _requirements("isothermal"))
    assert not matrix.supports(device, process, unsupported)
    assert matrix.supports(_device(), _process(), _requirements("isothermal"))
    assert not matrix.supports(device, process, unsupported)
    assert (
        _require_supported(
            matrix,
            _device(),
            _process(),
            _requirements("isothermal"),
        )
        is None
    )
    expected = "Unsupported capability declaration: " + repr(
        CapabilityDeclaration(device, process, unsupported)
    )
    with pytest.raises(ValueError, match=f"^{re.escape(expected)}$"):
        matrix.require(device, process, unsupported)

    assert matrix.declarations is declarations_before
    assert {
        hash(declaration) for declaration in matrix.declarations
    } == hashes_before


def test_execution_import_does_not_load_optional_backend() -> None:
    """Test a fresh execution import neither imports Warp nor particula.gpu."""
    repository_root = Path(__file__).parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(repository_root), environment.get("PYTHONPATH")))
    )
    script = """
import builtins
import sys

original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == "warp" or name.startswith("warp.") or name == "particula.gpu" or name.startswith("particula.gpu."):
        raise AssertionError(f"Unexpected optional backend import: {name}")
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
import particula.execution
assert "warp" not in sys.modules
assert "particula.gpu" not in sys.modules

from particula.execution import (
    Backend,
    CapabilityDeclaration,
    CapabilityMatrix,
    CapabilityRequirements,
    Device,
    ExecutionContext,
    ExecutionRequest,
    Process,
)

class Adapter:
    def __init__(self):
        self.calls = 0

    def execute(self, *args, **kwargs):
        self.calls += 1

device = Device(Backend.CPU, "cpu")
process = Process("condensation")
requirements = CapabilityRequirements(frozenset())
matrix = CapabilityMatrix(
    frozenset({CapabilityDeclaration(device, process, requirements)})
)
context = ExecutionContext(matrix)
adapter = Adapter()
context._registry._register_adapter(process, Backend.CPU, adapter)
assert context.resolve(
    ExecutionRequest(Backend.CPU, device, process, requirements)
) is adapter
assert adapter.calls == 0
"""

    completed = subprocess.run(  # noqa: S603 -- fixed test interpreter
        [sys.executable, "-c", script],
        cwd=repository_root,
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )

    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )


class _FakeAdapter:
    """Record whether P2 selection incorrectly invokes an adapter."""

    def __init__(self) -> None:
        """Create an adapter with no prior execution calls."""
        self.calls = 0

    def execute(self, *args: object, **kwargs: object) -> object:
        """Record an invocation that P2 must never make."""
        del args, kwargs
        self.calls += 1
        return None


def _context(
    device: Device | None = None,
    process: Process | None = None,
    requirements: CapabilityRequirements | None = None,
) -> ExecutionContext:
    """Create a context with one exact declared capability entry."""
    declared_device = device or _device()
    declared_process = process or _process()
    declared_requirements = requirements or _requirements()
    matrix = CapabilityMatrix(
        frozenset(
            {
                CapabilityDeclaration(
                    declared_device,
                    declared_process,
                    declared_requirements,
                )
            }
        )
    )
    return ExecutionContext(matrix)


def _request(
    backend: Backend = Backend.CPU,
    device: Device | None = None,
    process: Process | None = None,
    requirements: CapabilityRequirements | None = None,
) -> ExecutionRequest:
    """Create a typed request with the standard CPU values by default."""
    return ExecutionRequest(
        backend,
        device or _device(),
        process or _process(),
        requirements or _requirements(),
    )


def test_execution_request_validates_fields_in_order_and_freezes() -> None:
    """Test typed P2 requests reject fields before backend pairing checks."""
    request = _request()

    with pytest.raises(FrozenInstanceError):
        request.backend = Backend.WARP  # type: ignore[misc]
    with pytest.raises(
        TypeError, match="^ExecutionRequest.backend must be a Backend.$"
    ):
        ExecutionRequest("cpu", "device", "process", "requirements")  # type: ignore[arg-type]
    with pytest.raises(
        TypeError, match="^ExecutionRequest.device must be a Device.$"
    ):
        ExecutionRequest(Backend.CPU, "device", "process", "requirements")  # type: ignore[arg-type]
    with pytest.raises(
        TypeError, match="^ExecutionRequest.process must be a Process.$"
    ):
        ExecutionRequest(Backend.CPU, _device(), "process", "requirements")  # type: ignore[arg-type]
    with pytest.raises(
        TypeError,
        match=(
            "^ExecutionRequest.requirements must be a CapabilityRequirements.$"
        ),
    ):
        ExecutionRequest(Backend.CPU, _device(), _process(), "requirements")  # type: ignore[arg-type]
    with pytest.raises(
        ValueError,
        match="^ExecutionRequest.backend must match device.backend.$",
    ):
        _request(Backend.CPU, Device(Backend.WARP, "cuda:0"))


def test_context_rejects_non_matrix_before_creating_registry() -> None:
    """Test a context requires immutable capability metadata."""
    with pytest.raises(TypeError, match="^matrix must be a CapabilityMatrix.$"):
        ExecutionContext(object())  # type: ignore[arg-type]


def test_context_selects_canonical_cpu_adapter_without_execution() -> None:
    """Test CPU selection returns the exact adapter without dispatching it."""
    context = _context()
    adapter = _FakeAdapter()
    context._registry._register_adapter(_process(), Backend.CPU, adapter)

    resolved = context.resolve(_request())

    assert resolved is adapter
    assert _request().device == Device(Backend.CPU, "cpu")
    assert adapter.calls == 0


def test_context_selects_warp_adapter_with_opaque_native_identifier() -> None:
    """Test Warp selection preserves opaque native names without a probe."""
    warp_device = Device(Backend.WARP, "cuda:0")
    context = _context(warp_device)
    adapter = _FakeAdapter()
    context._registry._register_adapter(_process(), Backend.WARP, adapter)

    resolved = context.resolve(_request(Backend.WARP, warp_device))

    assert resolved is adapter
    assert warp_device.native == "cuda:0"
    assert adapter.calls == 0


@pytest.mark.parametrize("native", ["cpu:0", "cuda:0"])
def test_cpu_native_rejection_precedes_matrix_and_registry_lookup(
    native: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test invalid CPU spelling fails before capability or adapter access."""
    context = _context()
    calls: list[str] = []
    monkeypatch.setattr(
        CapabilityMatrix,
        "require",
        lambda *args: calls.append("require"),
    )
    monkeypatch.setattr(
        _AdapterRegistry,
        "_lookup",
        lambda *args: calls.append("lookup"),
    )

    with pytest.raises(
        ValueError,
        match="^CPU execution requires Device\\(Backend.CPU, 'cpu'\\).$",
    ):
        context.resolve(_request(device=Device(Backend.CPU, native)))

    assert calls == []


def test_resolve_non_request_precedes_matrix_and_registry_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test resolve type validation occurs before all selection work."""
    context = _context()
    calls: list[str] = []
    monkeypatch.setattr(
        CapabilityMatrix,
        "require",
        lambda *args: calls.append("require"),
    )
    monkeypatch.setattr(
        _AdapterRegistry,
        "_lookup",
        lambda *args: calls.append("lookup"),
    )

    with pytest.raises(
        TypeError, match="^request must be an ExecutionRequest.$"
    ):
        context.resolve(object())  # type: ignore[arg-type]

    assert calls == []


def test_resolve_requires_capability_before_one_registry_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test successful CPU selection performs require then one exact lookup."""
    context = _context()
    adapter = _FakeAdapter()
    context._registry._register_adapter(_process(), Backend.CPU, adapter)
    calls: list[tuple[str, object]] = []
    original_require = CapabilityMatrix.require
    original_lookup = _AdapterRegistry._lookup

    def record_require(
        matrix: CapabilityMatrix,
        device: Device,
        process: Process,
        requirements: CapabilityRequirements,
    ) -> None:
        calls.append(("require", device))
        original_require(matrix, device, process, requirements)

    def record_lookup(
        registry: _AdapterRegistry,
        process: Process,
        backend: Backend,
    ) -> object:
        calls.append(("lookup", (process, backend)))
        return original_lookup(registry, process, backend)

    monkeypatch.setattr(CapabilityMatrix, "require", record_require)
    monkeypatch.setattr(_AdapterRegistry, "_lookup", record_lookup)

    assert context.resolve(_request()) is adapter
    assert calls == [
        ("require", Device(Backend.CPU, "cpu")),
        ("lookup", (_process(), Backend.CPU)),
    ]


def test_resolve_warp_requires_capability_before_one_registry_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test Warp selection preserves its opaque device through validation."""
    warp_device = Device(Backend.WARP, "cuda:0")
    context = _context(warp_device)
    adapter = _FakeAdapter()
    context._registry._register_adapter(_process(), Backend.WARP, adapter)
    calls: list[tuple[str, object]] = []
    original_require = CapabilityMatrix.require
    original_lookup = _AdapterRegistry._lookup

    def record_require(
        matrix: CapabilityMatrix,
        device: Device,
        process: Process,
        requirements: CapabilityRequirements,
    ) -> None:
        calls.append(("require", device))
        original_require(matrix, device, process, requirements)

    def record_lookup(
        registry: _AdapterRegistry,
        process: Process,
        backend: Backend,
    ) -> object:
        calls.append(("lookup", (process, backend)))
        return original_lookup(registry, process, backend)

    monkeypatch.setattr(CapabilityMatrix, "require", record_require)
    monkeypatch.setattr(_AdapterRegistry, "_lookup", record_lookup)

    assert context.resolve(_request(Backend.WARP, warp_device)) is adapter
    assert calls == [
        ("require", warp_device),
        ("lookup", (_process(), Backend.WARP)),
    ]
    assert adapter.calls == 0


def test_unsupported_request_precedes_registry_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test unsupported capability errors cannot fall back to an adapter."""
    context = _context(requirements=_requirements("isothermal"))
    alternate = _FakeAdapter()
    context._registry._register_adapter(_process(), Backend.CPU, alternate)
    calls: list[object] = []
    monkeypatch.setattr(
        _AdapterRegistry,
        "_lookup",
        lambda *args: calls.append(args),
    )

    with pytest.raises(
        ValueError, match="^Unsupported capability declaration:"
    ):
        context.resolve(_request(requirements=_requirements("latent_heat")))

    assert calls == []
    assert alternate.calls == 0


def test_supported_unregistered_request_raises_exact_lookup_error() -> None:
    """Test supported selection has no fallback adapter when unregistered."""
    with pytest.raises(
        LookupError,
        match="^No adapter registered for process and backend.$",
    ):
        _context().resolve(_request())


@pytest.mark.parametrize(
    ("process", "backend", "adapter", "message"),
    [
        ("process", "backend", object(), "process must be a Process."),
        (_process(), "backend", object(), "backend must be a Backend."),
        (
            _process(),
            Backend.CPU,
            object(),
            "adapter must have a callable execute attribute.",
        ),
        (
            _process(),
            Backend.CPU,
            type("Adapter", (), {"execute": None})(),
            "adapter must have a callable execute attribute.",
        ),
    ],
)
def test_registry_rejects_invalid_entries_without_mutation(
    process: object,
    backend: object,
    adapter: object,
    message: str,
) -> None:
    """Test registration validation order leaves local state unchanged."""
    registry = _AdapterRegistry()
    before = registry._snapshot()

    with pytest.raises(TypeError, match=f"^{re.escape(message)}$"):
        registry._register_adapter(process, backend, adapter)

    assert registry._snapshot() == before


def test_duplicate_registration_preserves_original_adapter() -> None:
    """Test duplicate private registration never replaces the original."""
    context = _context()
    original = _FakeAdapter()
    duplicate = _FakeAdapter()
    context._registry._register_adapter(_process(), Backend.CPU, original)
    before = context._registry._snapshot()

    with pytest.raises(
        ValueError,
        match="^Adapter already registered for process and backend.$",
    ):
        context._registry._register_adapter(_process(), Backend.CPU, duplicate)

    assert context._registry._snapshot() == before
    assert context.resolve(_request()) is original
    assert original.calls == duplicate.calls == 0
