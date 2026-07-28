"""Tests for concrete-only GPU-resident session P1 carriers."""

import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, cast

import pytest

from particula.execution import Backend, Device
from particula.execution.gpu_session import (
    ResidentDimensions,
    ResidentLifecycle,
    ResidentMetadata,
    ResidentSession,
)


def _metadata(species: int = 1) -> ResidentMetadata:
    """Create valid CPU-owned metadata for a Warp CPU fixture."""
    return ResidentMetadata(
        Device(Backend.WARP, "cpu"),
        tuple(f"species_{index}" for index in range(species)),
    )


def _warp_resources(
    boxes: int = 1, particles_count: int = 2, species: int = 1
) -> tuple[Any, Any, Any]:
    """Construct a tiny schema-valid Warp CPU fixture lazily."""
    wp = pytest.importorskip("warp")
    from particula.gpu.warp_types import (
        WarpEnvironmentData,
        WarpGasData,
        WarpParticleData,
    )

    particles = WarpParticleData()
    particles.masses = wp.ones(
        (boxes, particles_count, species), dtype=wp.float64, device="cpu"
    )
    particles.concentration = wp.ones(
        (boxes, particles_count), dtype=wp.float64, device="cpu"
    )
    particles.charge = wp.zeros(
        (boxes, particles_count), dtype=wp.float64, device="cpu"
    )
    particles.density = wp.ones(species, dtype=wp.float64, device="cpu")
    particles.volume = wp.ones(boxes, dtype=wp.float64, device="cpu")
    gas = WarpGasData()
    gas.molar_mass = wp.ones(species, dtype=wp.float64, device="cpu")
    gas.concentration = wp.ones(
        (boxes, species), dtype=wp.float64, device="cpu"
    )
    gas.vapor_pressure = wp.ones(
        (boxes, species), dtype=wp.float64, device="cpu"
    )
    gas.partitioning = wp.ones((boxes, species), dtype=wp.int32, device="cpu")
    environment = WarpEnvironmentData()
    environment.temperature = wp.ones(boxes, dtype=wp.float64, device="cpu")
    environment.pressure = wp.ones(boxes, dtype=wp.float64, device="cpu")
    environment.saturation_ratio = wp.ones(
        (boxes, species), dtype=wp.float64, device="cpu"
    )
    return particles, gas, environment


class _MetadataArray:
    """Expose only array metadata and reject every payload operation."""

    def __init__(self, dtype: object, shape: object, device: object) -> None:
        self.dtype = dtype
        self.shape = shape
        self.device = device

    def numpy(self) -> object:
        """Fail if P1 attempts a host readback."""
        raise AssertionError("P1 must not read resident array payloads.")


def _primary_fields() -> tuple[tuple[str, str, tuple[int, ...], str], ...]:
    """Return the primary array schema for the canonical one-box fixture."""
    return (
        ("particles", "masses", (1, 2, 1), "float64"),
        ("particles", "concentration", (1, 2), "float64"),
        ("particles", "charge", (1, 2), "float64"),
        ("particles", "density", (1,), "float64"),
        ("particles", "volume", (1,), "float64"),
        ("gas", "molar_mass", (1,), "float64"),
        ("gas", "concentration", (1, 1), "float64"),
        ("gas", "vapor_pressure", (1, 1), "float64"),
        ("gas", "partitioning", (1, 1), "int32"),
        ("environment", "temperature", (1,), "float64"),
        ("environment", "pressure", (1,), "float64"),
        ("environment", "saturation_ratio", (1, 1), "float64"),
    )


def _snapshot_resources(
    particles: Any, gas: Any, environment: Any
) -> tuple[tuple[int, object, object, object, tuple[object, ...]], ...]:
    """Capture primary-array identity, metadata, and tiny fixture values."""
    snapshot = []
    for carrier, field, _, _ in _primary_fields():
        value = getattr(
            {"particles": particles, "gas": gas, "environment": environment}[
                carrier
            ],
            field,
        )
        snapshot.append(
            (
                id(value),
                value.dtype,
                value.shape,
                value.device,
                tuple(value.numpy().flat),
            )
        )
    return tuple(snapshot)


def test_carriers_validate_and_remain_immutable() -> None:
    """Test CPU-only dimensions, metadata, and lifecycle vocabulary."""
    dimensions = ResidentDimensions(1, 0, 0)
    metadata = ResidentMetadata(Device(Backend.WARP, "cpu"), ())

    assert dimensions == ResidentDimensions(1, 0, 0)
    assert metadata.gas_names is metadata.gas_names
    assert set(ResidentLifecycle) == {
        ResidentLifecycle.ACTIVE,
        ResidentLifecycle.FAULTED,
        ResidentLifecycle.FINALIZED,
        ResidentLifecycle.CLOSED,
    }
    with pytest.raises(FrozenInstanceError):
        dimensions.n_boxes = 2  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        metadata.gas_names = ()  # type: ignore[misc]


@pytest.mark.parametrize(
    ("value", "name", "positive", "error", "message"),
    [
        (True, "n_boxes", True, TypeError, "integral, not bool"),
        (1.0, "n_boxes", True, TypeError, "integral, not bool"),
        (0, "n_boxes", True, ValueError, "greater than zero"),
        (-1, "n_particles", False, ValueError, "nonnegative"),
        (-1, "n_species", False, ValueError, "nonnegative"),
    ],
)
def test_dimensions_reject_invalid_values(
    value: object,
    name: str,
    positive: bool,
    error: type[Exception],
    message: str,
) -> None:
    """Test dimension validation has stable condition-specific messages."""
    values: dict[str, object] = {
        "n_boxes": 1,
        "n_particles": 0,
        "n_species": 0,
    }
    values[name] = value
    with pytest.raises(error, match=message):
        ResidentDimensions(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("device", "names", "error", "message"),
    [
        (object(), (), TypeError, "exact Device"),
        (Device(Backend.CPU, "cpu"), (), ValueError, "Backend.WARP"),
        (Device(Backend.WARP, "cpu"), [], TypeError, "exact tuple"),
    ],
)
def test_metadata_rejects_invalid_values(
    device: Device, names: object, error: type[Exception], message: str
) -> None:
    """Test metadata does not normalize invalid device or name declarations."""
    with pytest.raises(error, match=message):
        ResidentMetadata(device, names)  # type: ignore[arg-type]


def test_metadata_preserves_valid_names_by_identity() -> None:
    """Test metadata retains valid tuple and string names by identity."""
    names = ("ok", "still_ok")

    metadata = ResidentMetadata(Device(Backend.WARP, "cpu"), names)

    assert metadata.gas_names is names


@pytest.mark.parametrize("names", [("ok", object()), ("ok", 1)])
def test_metadata_rejects_non_string_names(names: tuple[object, ...]) -> None:
    """Test metadata requires exact string gas-name entries."""
    with pytest.raises(TypeError, match="exact str instances"):
        ResidentMetadata(Device(Backend.WARP, "cpu"), names)  # type: ignore[arg-type]


def test_module_import_does_not_load_warp_or_gpu() -> None:
    """Test importing carriers and using metadata stays optional-backend free."""
    root = Path(__file__).parents[3]
    environment = os.environ | {"PYTHONPATH": str(root)}
    script = """
import sys
from particula.execution import Backend, Device
from particula.execution.gpu_session import ResidentDimensions, ResidentLifecycle, ResidentMetadata
ResidentDimensions(1, 0, 0)
ResidentMetadata(Device(Backend.WARP, 'cpu'), ())
assert ResidentLifecycle.ACTIVE.value == 'active'
assert not any(name == 'warp' or name.startswith('particula.gpu') for name in sys.modules)
"""
    completed = subprocess.run(  # noqa: S603 -- fixed interpreter and script
        [sys.executable, "-c", script],
        cwd=root,
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.warp
@pytest.mark.parametrize(
    "shape",
    [(1, 2, 1), (2, 1, 2), (1, 0, 0), (1, 0, 1), (1, 2, 0)],
)
def test_session_retains_valid_resources_by_identity(
    shape: tuple[int, int, int],
) -> None:
    """Test every declared lifecycle is accepted as immutable P1 state.

    P4 owns transition legality; P1 only accepts each declared value.
    """
    boxes, particle_count, species = shape
    particles, gas, environment = _warp_resources(
        boxes, particle_count, species
    )
    dimensions = ResidentDimensions(*shape)
    metadata = _metadata(species)
    before = _snapshot_resources(particles, gas, environment)
    for lifecycle in ResidentLifecycle:
        session = ResidentSession(
            particles, gas, environment, dimensions, metadata, lifecycle
        )
        assert session.particles is particles
        assert session.gas is gas
        assert session.environment is environment
        assert session.dimensions is dimensions
        assert session.metadata is metadata
        assert session.lifecycle is lifecycle
        assert session.metadata.gas_names is metadata.gas_names
        assert all(
            actual is expected
            for actual, expected in zip(
                session.metadata.gas_names, metadata.gas_names, strict=True
            )
        )
        assert session != ResidentSession(
            particles, gas, environment, dimensions, metadata, lifecycle
        )
    assert _snapshot_resources(particles, gas, environment) == before


@pytest.mark.warp
def test_session_preserves_gas_name_identity() -> None:
    """Test valid session construction preserves valid gas-name tuples."""
    particles, gas, environment = _warp_resources(species=2)
    names = ("species_0", "species_1")
    metadata = ResidentMetadata(Device(Backend.WARP, "cpu"), names)

    session = ResidentSession(
        particles,
        gas,
        environment,
        ResidentDimensions(1, 2, 2),
        metadata,
        ResidentLifecycle.ACTIVE,
    )

    assert session.metadata.gas_names is names
    assert session.metadata.gas_names == names


@pytest.mark.warp
def test_session_rejects_schema_and_metadata_mismatches() -> None:
    """Test isolated schema failures are deterministic and read-only."""
    particles, gas, environment = _warp_resources()
    dimensions = ResidentDimensions(1, 2, 1)

    with pytest.raises(ValueError, match="gas_names length"):
        ResidentSession(
            particles,
            gas,
            environment,
            dimensions,
            ResidentMetadata(Device(Backend.WARP, "cpu"), ("a", "b")),
            ResidentLifecycle.ACTIVE,
        )
    with pytest.raises(ValueError, match="metadata.device.native"):
        ResidentSession(
            particles,
            gas,
            environment,
            dimensions,
            ResidentMetadata(Device(Backend.WARP, "cuda:0"), ("a",)),
            ResidentLifecycle.ACTIVE,
        )
    with pytest.raises(ValueError, match="masses shape must match"):
        ResidentSession(
            particles,
            gas,
            environment,
            ResidentDimensions(1, 1, 1),
            _metadata(),
            ResidentLifecycle.ACTIVE,
        )


def test_preflight_rejects_cpu_carriers_before_warp_import() -> None:
    """Test carrier and identity failures precede optional Warp import."""
    dimensions = ResidentDimensions(1, 0, 0)
    metadata = ResidentMetadata(Device(Backend.WARP, "cpu"), ())
    with pytest.raises(TypeError, match="exact ResidentDimensions"):
        ResidentSession(
            object(),
            object(),
            object(),
            cast(ResidentDimensions, object()),
            metadata,
            ResidentLifecycle.ACTIVE,
        )
    for first, second, message in (
        ("particles", "gas", "particles and gas"),
        ("particles", "environment", "particles and environment"),
        ("gas", "environment", "gas and environment"),
    ):
        shared = object()
        resources = {
            "particles": object(),
            "gas": object(),
            "environment": object(),
        }
        resources[first] = shared
        resources[second] = shared
        with pytest.raises(ValueError, match=message):
            ResidentSession(
                resources["particles"],
                resources["gas"],
                resources["environment"],
                dimensions,
                metadata,
                ResidentLifecycle.ACTIVE,
            )


def test_missing_warp_runtime_has_stable_error_after_cpu_preflight() -> None:
    """Test only resident validation requires the optional Warp runtime."""
    root = Path(__file__).parents[3]
    environment = os.environ | {"PYTHONPATH": str(root)}
    script = """
import builtins
from particula.execution import Backend, Device
from particula.execution.gpu_session import (
    ResidentDimensions, ResidentLifecycle, ResidentMetadata, ResidentSession,
)
original_import = builtins.__import__
def blocked_import(name, *args, **kwargs):
    if name == 'warp' or name.startswith('warp.'):
        raise ModuleNotFoundError("No module named 'warp'", name='warp')
    return original_import(name, *args, **kwargs)
builtins.__import__ = blocked_import
metadata = ResidentMetadata(Device(Backend.WARP, 'cpu'), ())
try:
    ResidentSession(object(), object(), object(), object(), metadata, ResidentLifecycle.ACTIVE)
except TypeError as error:
    assert str(error) == 'dimensions must be an exact ResidentDimensions.'
else:
    raise AssertionError('carrier error did not precede Warp import')
shared = object()
try:
    ResidentSession(shared, shared, object(), ResidentDimensions(1, 0, 0), metadata, ResidentLifecycle.ACTIVE)
except ValueError as error:
    assert str(error) == 'particles and gas must not be identical.'
else:
    raise AssertionError('identity error did not precede Warp import')
try:
    ResidentSession(object(), object(), object(), ResidentDimensions(1, 0, 0), metadata, ResidentLifecycle.ACTIVE)
except RuntimeError as error:
    assert str(error) == 'ResidentSession requires the optional Warp runtime.'
else:
    raise AssertionError('resident validation did not require Warp')
"""
    completed = subprocess.run(  # noqa: S603 -- fixed interpreter and script
        [sys.executable, "-c", script],
        cwd=root,
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_missing_warp_runtime_is_covered_without_loading_generated_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test resident validation converts only a missing top-level Warp import."""
    import builtins

    original_import: Any = builtins.__import__

    def blocked_import(name: str, *args: object, **kwargs: object) -> object:
        """Raise only for the documented optional runtime import."""
        if name == "warp":
            raise ModuleNotFoundError("No module named 'warp'", name="warp")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(
        RuntimeError,
        match="ResidentSession requires the optional Warp runtime",
    ):
        ResidentSession(
            object(),
            object(),
            object(),
            ResidentDimensions(1, 0, 0),
            _metadata(0),
            ResidentLifecycle.ACTIVE,
        )


def test_session_carriers_remain_concrete_only() -> None:
    """Test P1 names do not alter public execution, adapter, or root exports."""
    import particula
    import particula.execution as execution
    import particula.execution.adapters as adapters

    names = {
        "ResidentDimensions",
        "ResidentLifecycle",
        "ResidentMetadata",
        "ResidentSession",
    }
    assert names.isdisjoint(execution.__all__)
    assert all(not hasattr(particula, name) for name in names)
    assert all(not hasattr(execution, name) for name in names)
    assert all(not hasattr(adapters, name) for name in names)


def test_session_revalidates_fabricated_cpu_carriers_before_warp_import() -> (
    None
):
    """Test bypassed carrier, metadata, name, and lifecycle errors are ordered."""
    dimensions = object.__new__(ResidentDimensions)
    object.__setattr__(dimensions, "n_boxes", 0)
    object.__setattr__(dimensions, "n_particles", 0)
    object.__setattr__(dimensions, "n_species", 0)
    metadata = _metadata(0)
    with pytest.raises(ValueError, match="n_boxes must be greater than zero"):
        ResidentSession(
            object(),
            object(),
            object(),
            dimensions,
            metadata,
            ResidentLifecycle.ACTIVE,
        )

    invalid_metadata = object.__new__(ResidentMetadata)
    object.__setattr__(invalid_metadata, "device", Device(Backend.CPU, "cpu"))
    object.__setattr__(invalid_metadata, "gas_names", ())
    with pytest.raises(ValueError, match="device.backend must be Backend.WARP"):
        ResidentSession(
            object(),
            object(),
            object(),
            ResidentDimensions(1, 0, 0),
            invalid_metadata,
            ResidentLifecycle.ACTIVE,
        )

    invalid_names = object.__new__(ResidentMetadata)
    object.__setattr__(invalid_names, "device", Device(Backend.WARP, "cpu"))
    object.__setattr__(invalid_names, "gas_names", ("ok", object()))
    with pytest.raises(TypeError, match="exact str instances"):
        ResidentSession(
            object(),
            object(),
            object(),
            ResidentDimensions(1, 0, 0),
            invalid_names,
            ResidentLifecycle.ACTIVE,
        )

    with pytest.raises(TypeError, match="lifecycle must be an exact"):
        ResidentSession(
            object(),
            object(),
            object(),
            ResidentDimensions(1, 0, 0),
            _metadata(0),
            "active",  # type: ignore[arg-type]
        )


@pytest.mark.warp
@pytest.mark.parametrize(
    ("carrier", "message"),
    [
        ("particles", "particles must be a WarpParticleData"),
        ("gas", "gas must be a WarpGasData"),
        ("environment", "environment must be a WarpEnvironmentData"),
    ],
)
def test_session_rejects_each_invalid_generated_container(
    carrier: str, message: str
) -> None:
    """Test each generated-container form has its own stable failure."""
    particles, gas, environment = _warp_resources()
    resources = {"particles": particles, "gas": gas, "environment": environment}
    resources[carrier] = object()
    with pytest.raises(TypeError, match=message):
        ResidentSession(
            resources["particles"],
            resources["gas"],
            resources["environment"],
            ResidentDimensions(1, 2, 1),
            _metadata(),
            ResidentLifecycle.ACTIVE,
        )


@pytest.mark.warp
@pytest.mark.parametrize("carrier,field,shape,dtype_name", _primary_fields())
def test_session_rejects_each_primary_array_wrong_shape_read_only(
    carrier: str, field: str, shape: tuple[int, ...], dtype_name: str
) -> None:
    """Test every primary field rejects one isolated wrong-rank or shape schema."""
    wp = pytest.importorskip("warp")
    particles, gas, environment = _warp_resources()
    before = _snapshot_resources(particles, gas, environment)
    resource = {"particles": particles, "gas": gas, "environment": environment}[
        carrier
    ]
    original = getattr(resource, field)
    wrong_shape = shape[:-1] if field == "masses" else shape + (1,)
    object.__setattr__(
        resource,
        field,
        _MetadataArray(getattr(wp, dtype_name), wrong_shape, original.device),
    )

    expected = (
        "rank 3" if field == "masses" else f"{carrier}.{field} must have shape"
    )
    with pytest.raises(ValueError, match=expected):
        ResidentSession(
            particles,
            gas,
            environment,
            ResidentDimensions(1, 2, 1),
            _metadata(),
            ResidentLifecycle.ACTIVE,
        )
    object.__setattr__(resource, field, original)
    assert _snapshot_resources(particles, gas, environment) == before


@pytest.mark.warp
@pytest.mark.parametrize("carrier,field,shape,dtype_name", _primary_fields())
def test_session_rejects_each_primary_array_wrong_dtype_read_only(
    carrier: str, field: str, shape: tuple[int, ...], dtype_name: str
) -> None:
    """Test every primary field rejects an isolated wrong dtype without writes."""
    wp = pytest.importorskip("warp")
    particles, gas, environment = _warp_resources()
    before = _snapshot_resources(particles, gas, environment)
    resource = {"particles": particles, "gas": gas, "environment": environment}[
        carrier
    ]
    original = getattr(resource, field)
    wrong_dtype = wp.int32 if dtype_name == "float64" else wp.float64
    object.__setattr__(
        resource, field, _MetadataArray(wrong_dtype, shape, original.device)
    )

    with pytest.raises(ValueError, match=f"{carrier}.{field} must use dtype"):
        ResidentSession(
            particles,
            gas,
            environment,
            ResidentDimensions(1, 2, 1),
            _metadata(),
            ResidentLifecycle.ACTIVE,
        )
    object.__setattr__(resource, field, original)
    assert _snapshot_resources(particles, gas, environment) == before


@pytest.mark.warp
@pytest.mark.parametrize(
    "carrier,field,shape,dtype_name", _primary_fields()[1:]
)
def test_session_rejects_each_primary_array_wrong_device_read_only(
    carrier: str, field: str, shape: tuple[int, ...], dtype_name: str
) -> None:
    """Test every non-anchor primary field requires the masses device exactly."""
    wp = pytest.importorskip("warp")
    particles, gas, environment = _warp_resources()
    before = _snapshot_resources(particles, gas, environment)
    resource = {"particles": particles, "gas": gas, "environment": environment}[
        carrier
    ]
    original = getattr(resource, field)
    object.__setattr__(
        resource,
        field,
        _MetadataArray(getattr(wp, dtype_name), shape, object()),
    )

    with pytest.raises(
        ValueError, match="device must match particles.masses device"
    ):
        ResidentSession(
            particles,
            gas,
            environment,
            ResidentDimensions(1, 2, 1),
            _metadata(),
            ResidentLifecycle.ACTIVE,
        )
    object.__setattr__(resource, field, original)
    assert _snapshot_resources(particles, gas, environment) == before


@pytest.mark.warp
def test_session_rejects_masses_metadata_and_native_device_read_only() -> None:
    """Test the anchor array validates array form, dtype, shape, and native id."""
    wp = pytest.importorskip("warp")
    particles, gas, environment = _warp_resources()
    original = particles.masses
    before = _snapshot_resources(particles, gas, environment)
    object.__setattr__(particles, "masses", object())
    with pytest.raises(
        ValueError, match="particles.masses must be a Warp array"
    ):
        ResidentSession(
            particles,
            gas,
            environment,
            ResidentDimensions(1, 2, 1),
            _metadata(),
            ResidentLifecycle.ACTIVE,
        )
    object.__setattr__(
        particles,
        "masses",
        _MetadataArray(wp.int32, (1, 2, 1), original.device),
    )
    with pytest.raises(ValueError, match="particles.masses must use dtype"):
        ResidentSession(
            particles,
            gas,
            environment,
            ResidentDimensions(1, 2, 1),
            _metadata(),
            ResidentLifecycle.ACTIVE,
        )
    object.__setattr__(
        particles,
        "masses",
        _MetadataArray(wp.float64, [1, 2, 1], original.device),
    )
    with pytest.raises(ValueError, match="tuple shape"):
        ResidentSession(
            particles,
            gas,
            environment,
            ResidentDimensions(1, 2, 1),
            _metadata(),
            ResidentLifecycle.ACTIVE,
        )
    object.__setattr__(particles, "masses", original)
    with pytest.raises(ValueError, match="metadata.device.native"):
        ResidentSession(
            particles,
            gas,
            environment,
            ResidentDimensions(1, 2, 1),
            ResidentMetadata(Device(Backend.WARP, "not-cpu"), ("species_0",)),
            ResidentLifecycle.ACTIVE,
        )
    assert _snapshot_resources(particles, gas, environment) == before


@pytest.mark.warp
def test_session_rejects_primary_array_without_metadata_read_only() -> None:
    """Test non-anchor arrays must expose all three required metadata fields."""
    particles, gas, environment = _warp_resources()
    before = _snapshot_resources(particles, gas, environment)
    original = gas.concentration
    object.__setattr__(gas, "concentration", object())
    with pytest.raises(
        ValueError, match="gas.concentration must be a Warp array"
    ):
        ResidentSession(
            particles,
            gas,
            environment,
            ResidentDimensions(1, 2, 1),
            _metadata(),
            ResidentLifecycle.ACTIVE,
        )
    object.__setattr__(gas, "concentration", original)
    assert _snapshot_resources(particles, gas, environment) == before


@pytest.mark.warp
def test_session_schema_validation_never_performs_runtime_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test valid and invalid schema construction performs metadata-only work."""
    wp = pytest.importorskip("warp")
    particles, gas, environment = _warp_resources()
    for carrier, field, _, _ in _primary_fields():
        resource = {
            "particles": particles,
            "gas": gas,
            "environment": environment,
        }[carrier]
        value = getattr(resource, field)
        object.__setattr__(
            resource,
            field,
            _MetadataArray(value.dtype, value.shape, value.device),
        )

    def forbidden(*_: object, **__: object) -> None:
        raise AssertionError("P1 must not perform runtime work.")

    for name in (
        "synchronize",
        "launch",
        "copy",
        "from_numpy",
        "zeros",
        "ones",
        "empty",
        "full",
        "zeros_like",
        "ones_like",
        "empty_like",
        "clone",
    ):
        if hasattr(wp, name):
            monkeypatch.setattr(wp, name, forbidden)
    session = ResidentSession(
        particles,
        gas,
        environment,
        ResidentDimensions(1, 2, 1),
        _metadata(),
        ResidentLifecycle.ACTIVE,
    )
    assert session.particles is particles

    object.__setattr__(
        particles,
        "concentration",
        _MetadataArray(wp.float64, (1, 1), particles.masses.device),
    )
    with pytest.raises(
        ValueError, match="particles.concentration must have shape"
    ):
        ResidentSession(
            particles,
            gas,
            environment,
            ResidentDimensions(1, 2, 1),
            _metadata(),
            ResidentLifecycle.ACTIVE,
        )
