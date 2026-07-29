"""Tests for concrete-only fixed-shape GPU resource acquisition."""

import os
import subprocess
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, cast

import pytest

from particula.execution import Backend, Device
from particula.execution.gpu_resources import (
    GPUResourceRegistry,
    ManifestEntry,
    _item_size,
)
from particula.execution.gpu_session import (
    ResidentDimensions,
    ResidentLifecycle,
    ResidentMetadata,
    ResidentSession,
)


def _session(
    boxes: int = 1, particle_count: int = 2, species: int = 1
) -> ResidentSession:
    """Build a small valid active Warp CPU resident session lazily."""
    wp = pytest.importorskip("warp")
    from particula.gpu.warp_types import (
        WarpEnvironmentData,
        WarpGasData,
        WarpParticleData,
    )

    particles = WarpParticleData()
    particles.masses = wp.ones(
        (boxes, particle_count, species), dtype=wp.float64, device="cpu"
    )
    particles.concentration = wp.ones(
        (boxes, particle_count), dtype=wp.float64, device="cpu"
    )
    particles.charge = wp.zeros(
        (boxes, particle_count), dtype=wp.float64, device="cpu"
    )
    particles.density = wp.ones(species, dtype=wp.float64, device="cpu")
    particles.volume = wp.ones(boxes, dtype=wp.float64, device="cpu")
    gas = WarpGasData()
    gas.molar_mass = wp.ones(species, dtype=wp.float64, device="cpu")
    gas.concentration = wp.ones(
        (boxes, species), dtype=wp.float64, device="cpu"
    )
    gas.vapor_pressure = wp.zeros(
        (boxes, species), dtype=wp.float64, device="cpu"
    )
    gas.partitioning = wp.ones((boxes, species), dtype=wp.int32, device="cpu")
    environment = WarpEnvironmentData()
    environment.temperature = wp.ones(boxes, dtype=wp.float64, device="cpu")
    environment.pressure = wp.ones(boxes, dtype=wp.float64, device="cpu")
    environment.saturation_ratio = wp.ones(
        (boxes, species), dtype=wp.float64, device="cpu"
    )
    return ResidentSession(
        particles,
        gas,
        environment,
        ResidentDimensions(boxes, particle_count, species),
        ResidentMetadata(
            Device(Backend.WARP, "cpu"),
            tuple(str(index) for index in range(species)),
        ),
        ResidentLifecycle.ACTIVE,
    )


@pytest.mark.warp
@pytest.mark.parametrize("shape", [(1, 2, 1), (2, 1, 2)])
def test_all_families_allocate_complete_stable_resources(
    shape: tuple[int, int, int],
) -> None:
    """Test every family supplies complete records and stable repeats."""
    registry = GPUResourceRegistry(_session(*shape))

    condensation = registry.acquire_condensation()
    coagulation = registry.acquire_coagulation(3)
    wall_loss = registry.acquire_wall_loss()
    nucleation = registry.acquire_nucleation()

    condensation_buffers = cast(Any, condensation.scratch_buffers)
    nucleation_exhaustion = cast(Any, nucleation.exhaustion)
    nucleation_finalized_demand = cast(Any, nucleation.finalized_demand)

    assert registry.acquire_condensation() is condensation
    assert registry.acquire_coagulation(3) is coagulation
    assert registry.acquire_wall_loss() is wall_loss
    assert registry.acquire_nucleation() is nucleation
    assert condensation_buffers.work_mass_transfer.shape == shape
    assert coagulation.collision_pairs.shape == (shape[0], 3, 2)
    assert coagulation.n_collisions.shape == (shape[0],)
    assert wall_loss.rng_states is not coagulation.rng_states
    assert (
        nucleation_exhaustion.resampling_buffers.replacement_masses.shape
        == shape
    )
    assert nucleation_finalized_demand.precursor_mass_change.shape == (
        shape[0],
        shape[2],
    )


@pytest.mark.warp
@pytest.mark.parametrize("shape", [(1, 0, 0), (1, 0, 1), (1, 2, 0)])
def test_registry_allocates_canonical_zero_dimension_schemas(
    shape: tuple[int, int, int],
) -> None:
    """Test zero particle/species dimensions retain complete schemas."""
    wp = pytest.importorskip("warp")
    registry = GPUResourceRegistry(_session(*shape))

    condensation = registry.acquire_condensation()
    coagulation = registry.acquire_coagulation(1)
    nucleation = registry.acquire_nucleation()
    condensation_buffers = cast(Any, condensation.scratch_buffers)
    nucleation_diagnostics = cast(Any, nucleation.diagnostics)
    nucleation_finalized_demand = cast(Any, nucleation.finalized_demand)

    assert condensation_buffers.work_mass_transfer.shape == shape
    assert condensation_buffers.work_mass_transfer.strides == (
        shape[1] * shape[2] * 8,
        shape[2] * 8,
        8,
    )
    assert coagulation.collision_pairs.shape == (shape[0], 1, 2)
    assert coagulation.collision_pairs.dtype == wp.int32
    assert nucleation_diagnostics.selected_slot_indices.shape == (
        shape[0],
        shape[1],
    )
    assert nucleation_finalized_demand.precursor_mass_change.shape == (
        shape[0],
        shape[2],
    )


@pytest.mark.warp
def test_registry_rejects_replacement_capacity_and_primary_alias() -> None:
    """Test established role identity and protected primary ownership checks."""
    session = _session()
    registry = GPUResourceRegistry(session)
    view = registry.acquire_wall_loss()
    session_temperature = cast(Any, session.environment).temperature
    with pytest.raises(ValueError, match="replaced"):
        registry.acquire_wall_loss(rng_states=session_temperature)
    registry.acquire_coagulation(1)
    with pytest.raises(ValueError, match="cannot change"):
        registry.acquire_coagulation(2)
    assert registry.acquire_wall_loss() is view


@pytest.mark.warp
def test_registry_rejects_session_drift_before_acquisition() -> None:
    """Test a fabricated changed frozen session cannot silently resize."""
    session = _session()
    registry = GPUResourceRegistry(session)
    object.__setattr__(session, "lifecycle", ResidentLifecycle.FAULTED)
    with pytest.raises(ValueError, match="ACTIVE"):
        registry.acquire_condensation()


@pytest.mark.warp
def test_registry_rejects_non_warp_and_invalid_capacity() -> None:
    """Test public metadata validation has no permissive binding path."""
    registry = GPUResourceRegistry(_session())
    with pytest.raises(TypeError, match="Warp array"):
        registry.acquire_wall_loss(rng_states=object())
    with pytest.raises(TypeError, match="non-boolean integral"):
        registry.acquire_coagulation(True)
    with pytest.raises(ValueError, match="positive"):
        registry.acquire_coagulation(0)
    with pytest.raises(TypeError):
        registry.acquire_wall_loss(unexpected=object())  # type: ignore[call-arg]


@pytest.mark.warp
def test_registry_accepts_complete_supplied_condensation_record() -> None:
    """Test complete native records retain their caller-owned array identity."""
    source_resources = GPUResourceRegistry(_session()).acquire_condensation()
    registry = GPUResourceRegistry(_session())

    acquired = registry.acquire_condensation(
        buffers=source_resources.scratch_buffers,
    )

    assert acquired.scratch_buffers is not source_resources.scratch_buffers
    assert (
        acquired.scratch_buffers.work_mass_transfer
        is source_resources.scratch_buffers.work_mass_transfer
    )
    assert (
        registry.acquire_condensation(
            buffers=source_resources.scratch_buffers,
        )
        is acquired
    )


@pytest.mark.warp
def test_registry_rejects_incomplete_and_inexact_condensation_records() -> None:
    """Test supplied records must be complete exact native scratch records."""
    source_resources = GPUResourceRegistry(_session()).acquire_condensation()
    buffers = source_resources.scratch_buffers
    object.__setattr__(buffers, "work_mass_transfer", None)
    registry = GPUResourceRegistry(_session())

    with pytest.raises(ValueError, match="complete"):
        registry.acquire_condensation(buffers=buffers)
    with pytest.raises(TypeError, match="exact CondensationScratchBuffers"):
        registry.acquire_condensation(buffers=object())  # type: ignore[arg-type]


@pytest.mark.warp
def test_registry_rejects_invalid_sidecar_schema_and_session_signature() -> (
    None
):
    """Test sidecar schemas and immutable session identity are enforced."""
    session = _session()
    registry = GPUResourceRegistry(session)
    wp = pytest.importorskip("warp")
    wrong_dtype = wp.zeros((1,), dtype=wp.float64, device="cpu")

    with pytest.raises(ValueError, match="incompatible schema"):
        registry.acquire_wall_loss(rng_states=wrong_dtype)
    object.__setattr__(session, "dimensions", ResidentDimensions(1, 3, 1))
    with pytest.raises(ValueError, match="signature changed"):
        registry.acquire_wall_loss()


@pytest.mark.warp
def test_registry_rejects_replaced_primary_identity_before_publication() -> (
    None
):
    """Test changing a protected primary rejects before another family binds."""
    wp = pytest.importorskip("warp")
    session = _session()
    registry = GPUResourceRegistry(session)
    object.__setattr__(
        cast(Any, session.environment),
        "temperature",
        wp.ones((1,), dtype=wp.float64, device="cpu"),
    )

    with pytest.raises(ValueError, match="signature changed"):
        registry.acquire_wall_loss()

    assert registry._bindings == {}


@pytest.mark.warp
@pytest.mark.parametrize(
    ("carrier", "field", "role"),
    [
        ("particles", "masses", "work_mass_transfer"),
        ("gas", "concentration", "positive_mass_transfer_demand"),
        ("environment", "temperature", "dynamic_viscosity"),
    ],
)
def test_registry_rejects_particle_gas_and_environment_primary_aliases(
    carrier: str,
    field: str,
    role: str,
) -> None:
    """Test each protected primary category cannot become a sidecar."""
    session = _session()
    registry = GPUResourceRegistry(session)
    buffers = GPUResourceRegistry(_session()).acquire_condensation()
    object.__setattr__(
        buffers.scratch_buffers,
        role,
        getattr(getattr(session, carrier), field),
    )

    with pytest.raises(ValueError, match="alias"):
        registry.acquire_condensation(buffers=buffers.scratch_buffers)

    assert registry._bindings == {}


@pytest.mark.warp
def test_registry_rejects_cross_family_duplicate_sidecar_identity() -> None:
    """Test persistent RNG storage cannot be reused across resource families."""
    registry = GPUResourceRegistry(_session())
    wall_loss = registry.acquire_wall_loss()

    with pytest.raises(ValueError, match="share identity"):
        registry.acquire_coagulation(1, rng_states=wall_loss.rng_states)

    assert set(registry._bindings) == {"wall_loss"}


@pytest.mark.warp
def test_registry_accepts_complete_supplied_nucleation_records() -> None:
    """Test every supplied nucleation record is retained by array identity."""
    source = GPUResourceRegistry(_session()).acquire_nucleation()
    registry = GPUResourceRegistry(_session())

    acquired = registry.acquire_nucleation(
        scratch=source.scratch,
        finalized_demand=source.finalized_demand,
        diagnostics=source.diagnostics,
        exhaustion=source.exhaustion,
    )

    for record_name in (
        "scratch",
        "finalized_demand",
        "diagnostics",
        "exhaustion",
    ):
        supplied = getattr(source, record_name)
        retained = getattr(acquired, record_name)
        for field in fields(supplied):
            if field.name == "resampling_buffers":
                for nested in fields(getattr(supplied, field.name)):
                    assert getattr(
                        getattr(retained, field.name), nested.name
                    ) is getattr(getattr(supplied, field.name), nested.name)
            else:
                assert getattr(retained, field.name) is getattr(
                    supplied, field.name
                )
    assert registry.acquire_nucleation() is acquired


@pytest.mark.warp
def test_registry_rejects_incomplete_and_inexact_nucleation_records() -> None:
    """Test native nucleation records must be exact and complete."""
    source = GPUResourceRegistry(_session()).acquire_nucleation()
    object.__setattr__(source.diagnostics, "gate_codes", None)
    registry = GPUResourceRegistry(_session())

    with pytest.raises(ValueError, match="complete"):
        registry.acquire_nucleation(diagnostics=source.diagnostics)
    with pytest.raises(TypeError, match="exact native types"):
        registry.acquire_nucleation(diagnostics=object())  # type: ignore[arg-type]


def test_execution_package_remains_dependency_neutral() -> None:
    """Test the package import neither exports nor eagerly loads resources."""
    root = Path(__file__).parents[3]
    environment = os.environ | {"PYTHONPATH": str(root)}
    script = """
import sys
import particula.execution as execution
assert 'GPUResourceRegistry' not in execution.__all__
assert 'particula.execution.gpu_resources' not in sys.modules
assert 'warp' not in sys.modules
assert not any(name == 'particula.gpu' or name.startswith('particula.gpu.') for name in sys.modules)
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


def test_manifest_helpers_reject_unsupported_dtype_and_size() -> None:
    """Test private manifest helpers reject unsupported allocation metadata."""
    with pytest.raises(ValueError, match="Unsupported manifest dtype"):
        _item_size(object())

    registry = object.__new__(GPUResourceRegistry)
    with pytest.raises(ValueError, match="exceeds supported range"):
        registry._checked_product(2**63, 2)
    with pytest.raises(ValueError, match="exceeds supported range"):
        registry._checked_product(1, -1)
    entry = ManifestEntry("pairs", "coagulation", object(), "bc2")
    registry._session = _session()
    with pytest.raises(ValueError, match="collision capacity"):
        registry._shape(entry)


def test_registry_requires_exact_active_session() -> None:
    """Test construction rejects values outside the exact active boundary."""
    with pytest.raises(TypeError, match="exact ResidentSession"):
        GPUResourceRegistry(object())  # type: ignore[arg-type]


@pytest.mark.warp
def test_registry_requires_active_session_and_exposes_all_manifests() -> None:
    """Test construction enforces ACTIVE state and publishes fixed schemas."""
    session = _session()
    object.__setattr__(session, "lifecycle", ResidentLifecycle.CLOSED)
    with pytest.raises(ValueError, match="ACTIVE"):
        GPUResourceRegistry(session)

    manifests = GPUResourceRegistry(_session()).manifests
    assert tuple(manifest.family for manifest in manifests) == (
        "condensation",
        "coagulation",
        "wall_loss",
        "nucleation",
    )
    assert all(
        entry.role and entry.dtype
        for manifest in manifests
        for entry in manifest.entries
    )


@pytest.mark.warp
def test_registry_allocation_failure_does_not_publish_partial_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test a failed allocation leaves all registry bindings unpublished."""
    registry = GPUResourceRegistry(_session())

    def fail_allocation(*_args: object, **_kwargs: object) -> object:
        """Raise a deterministic allocation failure for the local candidate."""
        raise RuntimeError("allocation failed")

    monkeypatch.setattr(registry, "_allocate", fail_allocation)
    with pytest.raises(RuntimeError, match="allocation failed"):
        registry.acquire_wall_loss()

    assert registry._bindings == {}
    assert registry._views == {}
