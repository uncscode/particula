"""Tests for immutable execution capability metadata."""

import os
import re
import subprocess
import sys
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from typing import cast

import pytest
from particula.execution import (
    Backend,
    BackendResult,
    Capability,
    CapabilityDeclaration,
    CapabilityMatrix,
    CapabilityRequirements,
    Device,
    ExecutionAdapter,
    ExecutionContext,
    ExecutionRequest,
    ExecutionResult,
    ExecutionState,
    MutationDeclaration,
    MutationScope,
    Process,
    _AdapterRegistry,
    validate_execution_result,
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


def test_execution_warp_selection_does_not_load_optional_backend() -> None:
    """Test fresh Warp selection neither imports nor probes optional modules."""
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

device = Device(Backend.WARP, "cuda:0")
process = Process("condensation")
requirements = CapabilityRequirements(frozenset())
matrix = CapabilityMatrix(
    frozenset({CapabilityDeclaration(device, process, requirements)})
)
context = ExecutionContext(matrix)
adapter = Adapter()
context._registry._register_adapter(process, Backend.WARP, adapter)
assert context.resolve(
    ExecutionRequest(Backend.WARP, device, process, requirements)
) is adapter
assert device.native == "cuda:0"
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


def test_cpu_selection_does_not_fall_back_to_warp_adapter() -> None:
    """Test CPU selection fails without executing a registered Warp adapter."""
    context = _context()
    warp_adapter = _FakeAdapter()
    context._registry._register_adapter(_process(), Backend.WARP, warp_adapter)

    with pytest.raises(
        LookupError,
        match="^No adapter registered for process and backend.$",
    ):
        context.resolve(_request())

    assert warp_adapter.calls == 0


def test_context_registries_are_isolated_with_shared_matrix() -> None:
    """Test registrations are owned by their individual execution context."""
    matrix = CapabilityMatrix(frozenset({_declaration()}))
    registered_context = ExecutionContext(matrix)
    unregistered_context = ExecutionContext(matrix)
    adapter = _FakeAdapter()
    registered_context._registry._register_adapter(
        _process(),
        Backend.CPU,
        adapter,
    )

    assert registered_context.resolve(_request()) is adapter
    with pytest.raises(
        LookupError,
        match="^No adapter registered for process and backend.$",
    ):
        unregistered_context.resolve(_request())
    assert adapter.calls == 0


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


class _State:
    """Provide a structurally valid state with an opaque payload."""

    def __init__(self, payload: object) -> None:
        self.backend_payload = payload


class _MissingPayloadState:
    """Deliberately omit the state protocol payload."""


class _OpaqueState:
    """Provide a state whose opaque payload raises when read."""

    @property
    def backend_payload(self) -> object:
        """Reject payload inspection at the validation boundary."""
        raise AssertionError("P3 must not inspect backend_payload.")


class _TypedAdapter:
    """Provide the typed P3 adapter shape without executing it."""

    def __init__(self) -> None:
        self.calls = 0

    def execute(self, state: ExecutionState) -> ExecutionResult:
        del state
        self.calls += 1
        raise AssertionError("P2 must not execute a P3 adapter.")


class _IncompatibleAdapter:
    """Provide a callable execute member with an incompatible signature."""

    def __init__(self) -> None:
        self.calls = 0

    def execute(self) -> object:
        self.calls += 1
        return None


def _valid_result(
    state: ExecutionState,
    mutation: MutationDeclaration | None = None,
    metadata: tuple[tuple[str, str], ...] = (("phase", "p3"),),
    backend_result: BackendResult | None = None,
) -> ExecutionResult:
    """Create a valid execution result with P3 defaults."""
    return ExecutionResult(
        state,
        metadata,
        mutation or MutationDeclaration(frozenset({MutationScope.NONE})),
        backend_result,
    )


def test_p3_protocols_are_structural_and_do_not_change_p2_selection() -> None:
    """Test structural protocols permit callable-only P2 registration."""
    state = _State(object())
    typed = _TypedAdapter()
    incompatible = _IncompatibleAdapter()

    assert isinstance(state, ExecutionState)
    assert not isinstance(_MissingPayloadState(), ExecutionState)
    assert isinstance(typed, ExecutionAdapter)
    assert isinstance(incompatible, ExecutionAdapter)
    assert not isinstance(object(), ExecutionAdapter)
    assert not isinstance(
        type("NoCall", (), {"execute": None})(), ExecutionAdapter
    )

    context = _context()
    context._registry._register_adapter(_process(), Backend.CPU, incompatible)

    assert context.resolve(_request()) is incompatible
    assert incompatible.calls == typed.calls == 0


def test_p3_closed_representation_and_frozen_carriers() -> None:
    """Test closed result vocabulary, field order, defaults, and freezing."""
    state = _State(object())
    mutation = MutationDeclaration(frozenset({MutationScope.NONE}))
    backend_result = BackendResult(object())
    result = _valid_result(state, mutation)

    assert list(MutationScope) == [MutationScope.NONE, MutationScope.STATE]
    assert [scope.value for scope in MutationScope] == ["none", "state"]
    assert [field.name for field in fields(ExecutionResult)] == [
        "state",
        "metadata",
        "mutation",
        "backend_result",
    ]
    assert result.backend_result is None
    with pytest.raises(FrozenInstanceError):
        mutation.scopes = frozenset({MutationScope.STATE})  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        backend_result.value = object()  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.state = state  # type: ignore[misc]


@pytest.mark.parametrize("scope", [MutationScope.NONE, MutationScope.STATE])
def test_mutation_declaration_accepts_one_scope(scope: MutationScope) -> None:
    """Test either closed mutation permission is accepted by identity."""
    scopes = frozenset({scope})

    assert MutationDeclaration(scopes).scopes is scopes


@pytest.mark.parametrize(
    ("scopes", "exception", "message"),
    [
        (set(), TypeError, "MutationDeclaration.scopes must be a frozenset."),
        ((), TypeError, "MutationDeclaration.scopes must be a frozenset."),
        (
            frozenset({"none"}),
            TypeError,
            "MutationDeclaration.scopes must contain only MutationScope instances.",
        ),
        (
            frozenset({object()}),
            TypeError,
            "MutationDeclaration.scopes must contain only MutationScope instances.",
        ),
        (
            frozenset(),
            ValueError,
            "MutationDeclaration.scopes must contain exactly one mutation scope.",
        ),
        (
            frozenset({MutationScope.NONE, MutationScope.STATE}),
            ValueError,
            "MutationDeclaration.scopes must contain exactly one mutation scope.",
        ),
    ],
)
def test_mutation_declaration_rejects_invalid_scopes(
    scopes: object,
    exception: type[Exception],
    message: str,
) -> None:
    """Test declarations do not coerce mutable, unknown, or mixed scopes."""
    with pytest.raises(exception, match=f"^{re.escape(message)}$"):
        MutationDeclaration(scopes)  # type: ignore[arg-type]


def test_mutation_declaration_rejects_frozenset_subclass() -> None:
    """Test declarations require the exact built-in frozenset type."""

    class MutationScopes(frozenset[MutationScope]):
        """Represent an unsupported frozenset subtype."""

    with pytest.raises(
        TypeError,
        match="^MutationDeclaration.scopes must be a frozenset.$",
    ):
        MutationDeclaration(MutationScopes({MutationScope.NONE}))


def test_backend_result_retains_opaque_objects_by_identity() -> None:
    """Test opaque backend fields are retained without inspection or copying."""
    value = object()
    diagnostics = object()
    result = BackendResult(value, diagnostics)

    assert result.value is value
    assert result.diagnostics is diagnostics


@pytest.mark.parametrize("scope", [MutationScope.NONE, MutationScope.STATE])
def test_validator_returns_valid_result_and_opaque_values_by_identity(
    scope: MutationScope,
) -> None:
    """Test valid results retain state, metadata, and backend values exactly."""
    payload = object()
    value = object()
    diagnostics = object()
    state = _State(payload)
    metadata = (("first", "one"), ("second", "two"))
    backend_result = BackendResult(value, diagnostics)
    result = _valid_result(
        state,
        MutationDeclaration(frozenset({scope})),
        metadata,
        backend_result,
    )

    assert validate_execution_result(state, result) is result
    assert result.state is state
    assert state.backend_payload is payload
    assert result.metadata is metadata
    assert result.backend_result is backend_result
    assert backend_result.value is value
    assert backend_result.diagnostics is diagnostics


def test_validator_accepts_empty_metadata_and_default_backend_result() -> None:
    """Test empty immutable metadata and omitted backend result are valid."""
    state = _State(object())
    result = _valid_result(state, metadata=())

    assert validate_execution_result(state, result) is result
    assert result.backend_result is None


def test_validator_does_not_inspect_opaque_state_payload() -> None:
    """Test structural validation leaves an opaque state payload unread."""
    state = _OpaqueState()
    result = _valid_result(state)

    assert validate_execution_result(state, result) is result


@pytest.mark.parametrize("key", ["Name", "1name", "bad name", "bad-name"])
def test_validator_rejects_each_metadata_name_grammar_class(key: str) -> None:
    """Test metadata keys share the declared lowercase-name grammar."""
    state = _State(object())
    result = _valid_result(state, metadata=((key, "value"),))

    with pytest.raises(
        ValueError,
        match=(
            "^"
            + re.escape(
                "ExecutionResult.metadata key must match ^[a-z][a-z0-9_]*$."
            )
            + "$"
        ),
    ):
        validate_execution_result(state, result)


def test_validator_rejects_tuple_subclasses_and_mutable_metadata_entry() -> (
    None
):
    """Test metadata requires exact built-in outer and inner tuple types."""

    class Metadata(tuple):
        """Represent an unsupported tuple subtype."""

    class MetadataEntry(tuple):
        """Represent an unsupported metadata-entry tuple subtype."""

    state = _State(object())
    with pytest.raises(
        TypeError,
        match="^ExecutionResult.metadata must be a tuple.$",
    ):
        validate_execution_result(
            state, _valid_result(state, metadata=Metadata())
        )
    with pytest.raises(
        TypeError,
        match=r"^ExecutionResult.metadata entries must be \(str, str\) tuples.$",
    ):
        validate_execution_result(
            state,
            _valid_result(
                state,
                metadata=(MetadataEntry(("key", "value")),),  # type: ignore[arg-type]
            ),
        )
    with pytest.raises(
        TypeError,
        match=r"^ExecutionResult.metadata entries must be \(str, str\) tuples.$",
    ):
        validate_execution_result(
            state,
            _valid_result(
                state,
                metadata=cast(
                    tuple[tuple[str, str], ...],
                    (["a", "b"],),
                ),
            ),
        )


@pytest.mark.parametrize(
    ("original_state", "result_factory", "exception", "message"),
    [
        (
            _MissingPayloadState(),
            lambda state: _valid_result(state),
            TypeError,
            "original_state must be an ExecutionState.",
        ),
        (
            _State(object()),
            lambda state: object(),
            TypeError,
            "result must be an ExecutionResult.",
        ),
        (
            _State(object()),
            lambda state: _valid_result(_State(object())),
            ValueError,
            "ExecutionResult.state must be original_state.",
        ),
        (
            _State(object()),
            lambda state: _valid_result(
                state,
                metadata=cast(tuple[tuple[str, str], ...], []),
            ),
            TypeError,
            "ExecutionResult.metadata must be a tuple.",
        ),
        (
            _State(object()),
            lambda state: _valid_result(
                state,
                metadata=cast(tuple[tuple[str, str], ...], (("key",),)),
            ),
            TypeError,
            "ExecutionResult.metadata entries must be (str, str) tuples.",
        ),
        (
            _State(object()),
            lambda state: _valid_result(
                state,
                metadata=cast(tuple[tuple[str, str], ...], (("key", 1),)),
            ),
            TypeError,
            "ExecutionResult.metadata entries must be (str, str) tuples.",
        ),
        (
            _State(object()),
            lambda state: _valid_result(state, metadata=(("Name", "value"),)),
            ValueError,
            "ExecutionResult.metadata key must match ^[a-z][a-z0-9_]*$.",
        ),
        (
            _State(object()),
            lambda state: _valid_result(
                state, metadata=(("same", "one"), ("same", "two"))
            ),
            ValueError,
            "ExecutionResult.metadata keys must be unique.",
        ),
        (
            _State(object()),
            lambda state: ExecutionResult(
                state,
                (),
                cast(MutationDeclaration, object()),
            ),
            TypeError,
            "ExecutionResult.mutation must be a MutationDeclaration.",
        ),
        (
            _State(object()),
            lambda state: ExecutionResult(
                state,
                (),
                MutationDeclaration(frozenset({MutationScope.NONE})),
                cast(BackendResult, object()),
            ),
            TypeError,
            "ExecutionResult.backend_result must be a BackendResult.",
        ),
    ],
)
def test_validator_rejects_malformed_forms_without_mutation(
    original_state: object,
    result_factory: object,
    exception: type[Exception],
    message: str,
) -> None:
    """Test invalid boundaries retain inputs and do not invoke P2 adapters."""
    result = cast(object, result_factory(original_state))  # type: ignore[operator]
    payload = getattr(original_state, "backend_payload", None)
    metadata = getattr(result, "metadata", None)
    context = _context()
    adapter = _FakeAdapter()
    context._registry._register_adapter(_process(), Backend.CPU, adapter)
    registry = context._registry._snapshot()

    with pytest.raises(exception, match=f"^{re.escape(message)}$"):
        validate_execution_result(original_state, result)

    assert getattr(original_state, "backend_payload", None) is payload
    assert getattr(result, "metadata", None) is metadata
    assert context._registry._snapshot() == registry
    assert adapter.calls == 0
