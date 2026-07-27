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
    Process,
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
                CapabilityDeclaration(warp_cpu, process, isothermal),
                CapabilityDeclaration(cpu, coagulation, isothermal),
            }
        )
    )

    assert matrix.supports(cpu, process, isothermal)
    assert matrix.supports(cpu, process, latent_heat)
    assert not matrix.supports(cpu, process, combined)
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
