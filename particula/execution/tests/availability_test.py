"""Tests for the concrete execution availability boundary."""

import subprocess
import sys
import textwrap
from collections.abc import Iterator, Mapping
from dataclasses import FrozenInstanceError
from typing import cast

import pytest

import particula.execution.availability as availability
from particula.execution import (
    Backend,
    Capability,
    CapabilityDeclaration,
    CapabilityMatrix,
    CapabilityRequirements,
    Device,
    ExecutionRequest,
    Process,
)
from particula.execution.errors import (
    ExecutionCapabilityError,
    InvalidExecutionStateError,
    UnavailableDeviceError,
    UnavailableRuntimeError,
    UnknownDeviceError,
    UnsupportedCapabilityError,
    UnsupportedProcessError,
)


class FakeProvider:
    """Record configured availability responses."""

    def __init__(
        self,
        log: list[str],
        *,
        recognizes: object = True,
        runtime: object = True,
        device: object = True,
    ) -> None:
        """Store test responses and a call log."""
        self.log = log
        self.recognizes_result = recognizes
        self.runtime_result = runtime
        self.device_result = device

    def recognizes(self, _: Device) -> object:
        """Return the configured recognition result."""
        self.log.append("recognition")
        if isinstance(self.recognizes_result, Exception):
            raise self.recognizes_result
        return self.recognizes_result

    def runtime_available(self) -> object:
        """Return the configured runtime result."""
        self.log.append("runtime")
        if isinstance(self.runtime_result, Exception):
            raise self.runtime_result
        return self.runtime_result

    def device_available(self, _: Device) -> object:
        """Return the configured device result."""
        self.log.append("device")
        if isinstance(self.device_result, Exception):
            raise self.device_result
        return self.device_result


def _request(backend: Backend = Backend.CPU) -> ExecutionRequest:
    """Build a valid request with an exact real-device declaration."""
    native = "cpu" if backend is Backend.CPU else "opaque:0"
    return ExecutionRequest(
        backend,
        Device(backend, native),
        Process("test_process"),
        CapabilityRequirements(frozenset({Capability("test_capability")})),
    )


def _matrix(request: ExecutionRequest) -> CapabilityMatrix:
    """Build the exact capability matrix for a request."""
    return CapabilityMatrix(
        frozenset(
            {
                CapabilityDeclaration(
                    request.device,
                    request.process,
                    request.requirements,
                )
            }
        )
    )


def _providers(
    log: list[str], **kwargs: object
) -> dict[Backend, availability.AvailabilityProvider]:
    """Build a complete injected provider registry."""
    return cast(
        dict[Backend, availability.AvailabilityProvider],
        {
            Backend.CPU: FakeProvider(log, **kwargs),
            Backend.WARP: FakeProvider(log, **kwargs),
        },
    )


def test_cpu_default_decision_retains_exact_frozen_request() -> None:
    """Canonical CPU resolves without optional runtime work."""
    request = _request()

    decision = availability.resolve_availability(request, _matrix(request))

    assert decision.request is request
    with pytest.raises(FrozenInstanceError):
        decision.request = request  # type: ignore[misc]


def test_injected_provider_success_runs_each_phase_in_order() -> None:
    """Injected providers and state validation run in resolver order."""
    request = _request()
    log: list[str] = []

    def valid_state(_: ExecutionRequest) -> bool:
        """Record successful request-associated state validation."""
        log.append("state")
        return True

    decision = availability.resolve_availability(
        request,
        _matrix(request),
        providers=_providers(log),
        state_validator=valid_state,
    )

    assert decision.request is request
    assert log == ["recognition", "runtime", "device", "state"]


@pytest.mark.parametrize(
    ("stage", "error_type", "expected_log"),
    [
        ("recognizes", UnknownDeviceError, ["recognition"]),
        ("runtime", UnavailableRuntimeError, ["recognition", "runtime"]),
        (
            "device",
            UnavailableDeviceError,
            ["recognition", "runtime", "device"],
        ),
    ],
)
def test_provider_phase_failures_short_circuit(
    stage: str,
    error_type: type[Exception],
    expected_log: list[str],
) -> None:
    """Provider phase failures use typed errors and stop subsequent phases."""
    request = _request()
    log: list[str] = []
    values: dict[str, object] = {
        "recognizes": True,
        "runtime": True,
        "device": True,
    }
    values[stage] = False

    with pytest.raises(error_type) as raised:
        availability.resolve_availability(
            request,
            _matrix(request),
            providers=_providers(log, **values),
        )

    assert log == expected_log
    assert cast(ExecutionCapabilityError, raised.value).backend == "cpu"
    assert raised.value.__cause__ is None


def test_provider_exception_is_chained() -> None:
    """Unexpected provider errors map to and retain their typed cause."""
    request = _request()
    source = RuntimeError("unavailable")
    log: list[str] = []

    with pytest.raises(UnavailableRuntimeError) as raised:
        availability.resolve_availability(
            request,
            _matrix(request),
            providers=_providers(log, runtime=source),
        )

    assert raised.value.__cause__ is source
    assert log == ["recognition", "runtime"]


@pytest.mark.parametrize(
    ("stage", "error_type", "expected_log"),
    [
        ("recognizes", UnknownDeviceError, ["recognition"]),
        (
            "device",
            UnavailableDeviceError,
            ["recognition", "runtime", "device"],
        ),
    ],
)
def test_provider_exceptions_map_to_their_phase_error(
    stage: str,
    error_type: type[Exception],
    expected_log: list[str],
) -> None:
    """Provider exceptions retain their cause and stop later phases."""
    request = _request()
    source = RuntimeError(stage)
    log: list[str] = []
    values: dict[str, object] = {
        "recognizes": True,
        "runtime": True,
        "device": True,
    }
    values[stage] = source

    with pytest.raises(error_type) as raised:
        availability.resolve_availability(
            request,
            _matrix(request),
            providers=_providers(log, **values),
        )

    assert raised.value.__cause__ is source
    assert log == expected_log


@pytest.mark.parametrize(
    ("stage", "error_type", "expected_log"),
    [
        ("recognizes", UnknownDeviceError, ["recognition"]),
        ("runtime", UnavailableRuntimeError, ["recognition", "runtime"]),
        (
            "device",
            UnavailableDeviceError,
            ["recognition", "runtime", "device"],
        ),
    ],
)
def test_malformed_provider_results_fail_closed(
    stage: str,
    error_type: type[Exception],
    expected_log: list[str],
) -> None:
    """Provider phases require exact boolean status results."""
    request = _request()
    log: list[str] = []
    values: dict[str, object] = {
        "recognizes": True,
        "runtime": True,
        "device": True,
    }
    values[stage] = 1

    with pytest.raises(error_type) as raised:
        availability.resolve_availability(
            request,
            _matrix(request),
            providers=_providers(log, **values),
        )

    assert raised.value.__cause__ is None
    assert log == expected_log


def test_declaration_failures_precede_runtime() -> None:
    """Structural declaration failures do not enter lazy provider phases."""
    request = _request()
    log: list[str] = []
    absent = CapabilityMatrix(frozenset())

    with pytest.raises(UnsupportedProcessError) as raised:
        availability.resolve_availability(
            request,
            absent,
            providers=_providers(log),
        )

    assert log == ["recognition"]
    assert raised.value.process == "test_process"
    assert raised.value.device == "cpu"


def test_capability_failure_precedes_runtime() -> None:
    """An existing process with different requirements is unsupported."""
    request = _request()
    log: list[str] = []
    matrix = CapabilityMatrix(
        frozenset(
            {
                CapabilityDeclaration(
                    request.device,
                    request.process,
                    CapabilityRequirements(frozenset()),
                )
            }
        )
    )

    with pytest.raises(UnsupportedCapabilityError) as raised:
        availability.resolve_availability(
            request,
            matrix,
            providers=_providers(log),
        )

    assert log == ["recognition"]
    assert raised.value.capability == repr(request.requirements)


def test_state_validator_is_last_and_fail_closed() -> None:
    """State validation runs last and maps false values to the typed error."""
    request = _request()
    log: list[str] = []

    def invalid_state(_: ExecutionRequest) -> bool:
        log.append("state")
        return False

    with pytest.raises(InvalidExecutionStateError) as raised:
        availability.resolve_availability(
            request,
            _matrix(request),
            providers=_providers(log),
            state_validator=invalid_state,
        )

    assert log == ["recognition", "runtime", "device", "state"]
    assert raised.value.state == "validation_failed"
    assert raised.value.process == "test_process"


def test_noncallable_state_validator_fails_closed() -> None:
    """State validation accepts only callable validators returning exact bools."""
    request = _request()

    with pytest.raises(InvalidExecutionStateError) as raised:
        availability.resolve_availability(
            request,
            _matrix(request),
            state_validator=1,  # type: ignore[arg-type]
        )

    assert raised.value.backend == "cpu"
    assert raised.value.device == "cpu"
    assert raised.value.capability == repr(request.requirements)


def test_state_validator_exception_is_chained() -> None:
    """State-validation exceptions map to invalid state with their cause."""
    request = _request()
    source = RuntimeError("state")

    def raising_validator(_: ExecutionRequest) -> bool:
        """Raise the configured state-validation failure."""
        raise source

    with pytest.raises(InvalidExecutionStateError) as raised:
        availability.resolve_availability(
            request,
            _matrix(request),
            state_validator=raising_validator,
        )

    assert raised.value.__cause__ is source


@pytest.mark.parametrize("result", [None, 1])
def test_malformed_state_result_fails_closed(result: object) -> None:
    """State validators must return an exact bool result."""
    request = _request()

    with pytest.raises(InvalidExecutionStateError):
        availability.resolve_availability(
            request,
            _matrix(request),
            state_validator=cast(
                availability.StateValidator,
                lambda _: result,
            ),
        )


def test_noncanonical_cpu_device_is_unknown() -> None:
    """The default CPU provider accepts only its canonical native identifier."""
    request = ExecutionRequest(
        Backend.CPU,
        Device(Backend.CPU, "other"),
        Process("test_process"),
        CapabilityRequirements(frozenset()),
    )

    with pytest.raises(UnknownDeviceError) as raised:
        availability.resolve_availability(request, _matrix(request))

    assert raised.value.device == "other"
    assert raised.value.backend == "cpu"


def test_empty_requirements_use_existing_process_declaration() -> None:
    """Empty requirements are supported for any existing device-process pair."""
    request = ExecutionRequest(
        Backend.CPU,
        Device(Backend.CPU, "cpu"),
        Process("test_process"),
        CapabilityRequirements(frozenset()),
    )
    declared = CapabilityRequirements(frozenset({Capability("declared")}))
    matrix = CapabilityMatrix(
        frozenset(
            {
                CapabilityDeclaration(
                    request.device,
                    request.process,
                    declared,
                )
            }
        )
    )

    assert availability.resolve_availability(request, matrix).request is request


@pytest.mark.parametrize(
    "providers",
    [
        {},
        {"cpu": object()},
        {Backend.CPU: object(), Backend.WARP: object()},
        {
            Backend.CPU: FakeProvider([]),
            Backend.WARP: FakeProvider([]),
            1: object(),
        },
        object(),
    ],
)
def test_malformed_registry_fails_before_provider_work(
    providers: object,
) -> None:
    """Registry malformation maps deterministically to runtime unavailable."""
    request = _request()

    with pytest.raises(UnavailableRuntimeError):
        availability.resolve_availability(
            request,
            _matrix(request),
            providers=providers,  # type: ignore[arg-type]
        )


def test_duplicate_yielding_registry_fails_before_provider_or_import_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Duplicate registry keys fail before provider access or Warp imports."""
    request = _request()
    provider_accessed = False
    imports: list[str] = []

    class DuplicateKeyRegistry(Mapping[Backend, object]):
        """Expose duplicate expected keys while rejecting provider access."""

        def __iter__(self) -> Iterator[Backend]:
            """Yield a duplicate expected backend key."""
            yield Backend.CPU
            yield Backend.WARP
            yield Backend.CPU

        def __len__(self) -> int:
            """Return the intentionally malformed yielded-key count."""
            return 3

        def __getitem__(self, _: Backend) -> object:
            """Record prohibited provider access."""
            nonlocal provider_accessed
            provider_accessed = True
            raise AssertionError("provider access must not occur")

    def record_import(name: str) -> object:
        """Record prohibited optional runtime access."""
        imports.append(name)
        raise AssertionError("optional import must not occur")

    monkeypatch.setattr(availability.importlib, "import_module", record_import)

    with pytest.raises(UnavailableRuntimeError) as raised:
        availability.resolve_availability(
            request,
            _matrix(request),
            providers=cast(
                Mapping[Backend, availability.AvailabilityProvider],
                DuplicateKeyRegistry(),
            ),
        )

    assert raised.value.backend == "cpu"
    assert not provider_accessed
    assert imports == []


def test_registry_property_failure_is_chained() -> None:
    """Inaccessible provider methods fail closed before recognition."""
    request = _request()

    class BrokenProvider:
        """Raise while retrieving a required provider method."""

        @property
        def recognizes(self) -> object:
            """Raise the configured registry-access failure."""
            raise RuntimeError("broken")

    providers = {
        Backend.CPU: BrokenProvider(),
        Backend.WARP: FakeProvider([]),
    }

    with pytest.raises(UnavailableRuntimeError) as raised:
        availability.resolve_availability(
            request,
            _matrix(request),
            providers=providers,  # type: ignore[arg-type]
        )

    assert isinstance(raised.value.__cause__, RuntimeError)


def test_provider_uses_bound_methods_validated_during_preflight() -> None:
    """Provider descriptors cannot change after registry validation."""
    request = _request()
    log: list[str] = []

    class ChangingProvider:
        """Expose every provider method once through dynamic descriptors."""

        def __init__(self, log: list[str]) -> None:
            """Initialize the provider with one shared log."""
            self.log = log

        def _read_once(self, name: str, method: object) -> object:
            """Return a bound method once and reject later descriptor access."""
            attribute = f"_{name}_read"
            if hasattr(self, attribute):
                raise RuntimeError("post-validation lookup")
            setattr(self, attribute, True)
            return method

        @property
        def recognizes(self) -> object:
            """Return the recognition method only during preflight."""
            return self._read_once("recognizes", self._recognizes)

        @property
        def runtime_available(self) -> object:
            """Return the runtime method only during preflight."""
            return self._read_once("runtime", self._runtime_available)

        @property
        def device_available(self) -> object:
            """Return the device method only during preflight."""
            return self._read_once("device", self._device_available)

        def _recognizes(self, _: Device) -> bool:
            """Record the bound recognition invocation."""
            log.append("recognition")
            return True

        def _runtime_available(self) -> bool:
            """Record the bound runtime invocation."""
            log.append("runtime")
            return True

        def _device_available(self, _: Device) -> bool:
            """Record the bound device invocation."""
            log.append("device")
            return True

    providers = cast(
        dict[Backend, availability.AvailabilityProvider],
        {
            Backend.CPU: ChangingProvider(log),
            Backend.WARP: FakeProvider(log),
        },
    )

    assert (
        availability.resolve_availability(
            request,
            _matrix(request),
            providers=providers,
        ).request
        is request
    )
    assert log == ["recognition", "runtime", "device"]


def test_forged_nested_carriers_fail_before_registry_access() -> None:
    """Forged nested request and matrix values fail in deterministic order."""
    request = _request()
    matrix = _matrix(request)
    object.__setattr__(request.device, "native", 1)

    with pytest.raises(TypeError, match="request.device.native"):
        availability.resolve_availability(request, matrix)

    request = _request()
    matrix = _matrix(request)
    declaration = next(iter(matrix.declarations))
    declaration_process = Process("matrix_process")
    object.__setattr__(declaration, "process", declaration_process)
    object.__setattr__(declaration_process, "name", 1)

    with pytest.raises(TypeError, match="matrix.declaration.process.name"):
        availability.resolve_availability(request, matrix)


def test_unselected_registry_provider_must_be_usable() -> None:
    """Malformed unselected providers fail before selected-provider work."""
    request = _request()
    log: list[str] = []
    providers = {
        Backend.CPU: FakeProvider(log),
        Backend.WARP: object(),
    }

    with pytest.raises(UnavailableRuntimeError) as raised:
        availability.resolve_availability(
            request,
            _matrix(request),
            providers=providers,  # type: ignore[arg-type]
        )

    assert raised.value.backend == "cpu"
    assert log == []


def test_warp_lazy_import_and_opaque_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default Warp resolution passes the opaque native string unchanged."""
    request = _request(Backend.WARP)
    imports: list[str] = []

    class Runtime:
        """Provide the one device lookup used by the availability provider."""

        def get_device(self, native: str) -> None:
            """Record the opaque native identifier."""
            imports.append(native)

    monkeypatch.setattr(
        availability.importlib,
        "import_module",
        lambda name: Runtime(),
    )

    availability.resolve_availability(request, _matrix(request))

    assert imports == ["opaque:0"]


def test_warp_import_failure_maps_to_unavailable_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default Warp runtime import failure stops before device lookup."""
    request = _request(Backend.WARP)

    def missing_runtime(_: str) -> object:
        """Raise the optional-runtime import failure."""
        raise ImportError("warp missing")

    monkeypatch.setattr(
        availability.importlib, "import_module", missing_runtime
    )

    with pytest.raises(UnavailableRuntimeError) as raised:
        availability.resolve_availability(request, _matrix(request))

    assert raised.value.backend == "warp"


@pytest.mark.parametrize(
    ("matrix", "error_type"),
    [
        (CapabilityMatrix(frozenset()), UnsupportedProcessError),
        (
            CapabilityMatrix(
                frozenset(
                    {
                        CapabilityDeclaration(
                            Device(Backend.WARP, "opaque:0"),
                            Process("test_process"),
                            CapabilityRequirements(frozenset()),
                        )
                    }
                )
            ),
            UnsupportedCapabilityError,
        ),
    ],
)
def test_warp_structural_failures_precede_lazy_import(
    monkeypatch: pytest.MonkeyPatch,
    matrix: CapabilityMatrix,
    error_type: type[Exception],
) -> None:
    """Default Warp structural failures occur before optional imports."""
    request = _request(Backend.WARP)
    imports: list[str] = []

    monkeypatch.setattr(
        availability.importlib,
        "import_module",
        lambda name: imports.append(name),
    )

    with pytest.raises(error_type):
        availability.resolve_availability(request, matrix)

    assert imports == []


def test_warp_device_resolution_failure_maps_to_unavailable_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default Warp device lookup maps runtime resolution errors to its phase."""
    request = _request(Backend.WARP)
    source = ValueError("unavailable device")

    class Runtime:
        """Raise while resolving an opaque native device identifier."""

        def get_device(self, _: str) -> None:
            """Raise the configured device-resolution error."""
            raise source

    monkeypatch.setattr(
        availability.importlib, "import_module", lambda _: Runtime()
    )

    with pytest.raises(UnavailableDeviceError) as raised:
        availability.resolve_availability(request, _matrix(request))

    assert raised.value.backend == "warp"
    assert raised.value.device == "opaque:0"


def test_resolver_validates_carriers_before_registry_access() -> None:
    """Invalid P1 carriers raise TypeError before optional registry work."""
    with pytest.raises(TypeError, match="request"):
        availability.resolve_availability(object(), object())  # type: ignore[arg-type]

    request = _request()
    with pytest.raises(TypeError, match="matrix"):
        availability.resolve_availability(request, object())  # type: ignore[arg-type]


def test_default_cpu_resolution_does_not_import_optional_execution_seams() -> (
    None
):
    """Availability import and CPU resolution remain optional-runtime-neutral."""
    program = textwrap.dedent(
        """
        import builtins
        import sys

        blocked = (
            "warp",
            "particula.gpu",
            "particula.execution.adapters",
            "particula.execution.gpu_session",
            "particula.execution.gpu_resources",
            "particula.execution.checkpoint",
            "particula.execution.conversion",
            "particula.execution.state",
        )
        original_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if any(name == module or name.startswith(module + ".") for module in blocked):
                raise AssertionError(f"unexpected optional import: {name}")
            return original_import(name, globals, locals, fromlist, level)

        builtins.__import__ = guarded_import
        import particula.execution.availability as availability
        from particula.execution import (
            Backend, Capability, CapabilityDeclaration, CapabilityMatrix,
            CapabilityRequirements, Device, ExecutionRequest, Process,
        )

        request = ExecutionRequest(
            Backend.CPU,
            Device(Backend.CPU, "cpu"),
            Process("test_process"),
            CapabilityRequirements(frozenset({Capability("test_capability")})),
        )
        matrix = CapabilityMatrix(frozenset({CapabilityDeclaration(
            request.device, request.process, request.requirements,
        )}))
        assert availability.resolve_availability(request, matrix).request is request
        assert not any(
            name == module or name.startswith(module + ".")
            for name in sys.modules for module in blocked
        )
        """
    )

    result = subprocess.run(  # noqa: S603
        [sys.executable, "-Werror", "-c", program],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
