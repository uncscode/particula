"""Test P1 fixtures/helpers and private P2 resident GPU process sequences.

This private module supplies deterministic fixtures and independent accounting
helpers for process-sequence tests, plus private resident composition evidence
for shipped direct GPU boundaries. It defines no production or user-facing API.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.testing as npt
import pytest

from particula.gpu.tests.cuda_availability import warp_devices
from particula.particles.exhaustion import (
    POLICY_ACTIVATE,
    POLICY_RESAMPLE_DEFERRED,
    POLICY_SCALE_DEFERRED,
    ExhaustionControls,
    ExhaustionInputs,
    resolve_exhaustion,
)


@dataclass(frozen=True)
class ProcessFixture:
    """Store deterministic, fixed-shape process input state.

    Attributes:
        name: Stable identifier used for parametrized test cases.
        masses: Particle mass storage with shape ``(boxes, particles, species)``.
        particle_concentration: Particle concentrations with shape
            ``(boxes, particles)``.
        charge: Particle charge state with shape ``(boxes, particles)``.
        density: Species densities in kg/m³.
        volume: Per-box volumes in m³.
        gas_concentration: Gas concentrations with shape ``(boxes, species)``.
        molar_mass: Species molar masses in kg/mol.
        partitioning: Per-box, per-species partitioning flags.
        temperature: Per-box temperatures in K.
        pressure: Per-box pressures in Pa.
    """

    name: str
    masses: np.ndarray
    particle_concentration: np.ndarray
    charge: np.ndarray
    density: np.ndarray
    volume: np.ndarray
    gas_concentration: np.ndarray
    molar_mass: np.ndarray
    partitioning: np.ndarray
    temperature: np.ndarray
    pressure: np.ndarray


@dataclass(frozen=True)
class SlotExpectation:
    """Store independently calculated fixed-slot diagnostics.

    Attributes:
        free_indices: Ascending free-slot indices with ``-1`` tail padding.
        active_counts: Number of active slots in each box.
        free_counts: Number of free slots in each box.
    """

    free_indices: np.ndarray
    active_counts: np.ndarray
    free_counts: np.ndarray


@dataclass(frozen=True)
class _ArraySnapshot:
    """Store an array's identity, metadata, and copied values for assertions.

    Attributes:
        identity: Python object identity of the captured array.
        shape: Captured array shape.
        dtype: Captured array data type.
        device: Captured Warp device, or ``None`` for NumPy arrays.
        bytes_value: Byte representation used by exact-value assertions.
    """

    identity: int
    shape: tuple[int, ...]
    dtype: Any
    device: Any
    bytes_value: bytes


def _build_process_fixtures() -> tuple[ProcessFixture, ...]:
    """Build fresh deterministic one- and multi-box sparse fixtures."""
    density = np.array([1000.0, 1500.0], dtype=np.float64)
    molar_mass = np.array([0.018, 0.098], dtype=np.float64)
    one = ProcessFixture(
        name="one_box_sparse",
        masses=np.array(
            [[[1.0e-18, 2.0e-18], [0.0, 0.0], [3.0e-18, 0.0], [0.0, 0.0]]],
            dtype=np.float64,
        ),
        particle_concentration=np.array(
            [[2.0, 0.0, 5.0, 0.0]], dtype=np.float64
        ),
        charge=np.array([[1.0, 0.0, -2.0, 0.0]], dtype=np.float64),
        volume=np.array([1.0e-6], dtype=np.float64),
        gas_concentration=np.array([[1.0e-9, 2.0e-9]], dtype=np.float64),
        partitioning=np.array([[True, False]], dtype=np.bool_),
        temperature=np.array([298.15], dtype=np.float64),
        pressure=np.array([101325.0], dtype=np.float64),
        density=density.copy(),
        molar_mass=molar_mass.copy(),
    )
    multi = ProcessFixture(
        name="multi_box_sparse",
        masses=np.array(
            [
                [[1.0e-18, 0.0], [0.0, 0.0], [2.0e-18, 4.0e-18], [0.0, 0.0]],
                [[0.0, 0.0], [5.0e-18, 1.0e-18], [0.0, 0.0], [7.0e-18, 0.0]],
            ],
            dtype=np.float64,
        ),
        particle_concentration=np.array(
            [[1.0, 0.0, 3.0, 0.0], [0.0, 4.0, 0.0, 2.0]],
            dtype=np.float64,
        ),
        charge=np.array(
            [[0.0, 0.0, -1.0, 0.0], [0.0, 2.0, 0.0, -3.0]], dtype=np.float64
        ),
        volume=np.array([2.0e-6, 5.0e-6], dtype=np.float64),
        gas_concentration=np.array(
            [[3.0e-9, 1.0e-9], [4.0e-9, 6.0e-9]],
            dtype=np.float64,
        ),
        partitioning=np.array([[True, False], [False, True]], dtype=np.bool_),
        temperature=np.array([280.0, 310.0], dtype=np.float64),
        pressure=np.array([90000.0, 110000.0], dtype=np.float64),
        density=density.copy(),
        molar_mass=molar_mass.copy(),
    )
    return one, multi


def _resident_sequence_fixture(fixture: ProcessFixture) -> ProcessFixture:
    """Copy a fixture with every gas lane enabled for resident nucleation.

    The original sparse masks remain the P1 fixtures for disabled-condensation
    coverage. Nucleation requires its selected precursor to participate in every
    box, so resident sequencing deliberately uses this test-local derivative.
    """
    return replace(
        fixture,
        masses=fixture.masses.copy(),
        particle_concentration=fixture.particle_concentration.copy(),
        charge=fixture.charge.copy(),
        density=fixture.density.copy(),
        volume=fixture.volume.copy(),
        gas_concentration=fixture.gas_concentration.copy(),
        molar_mass=fixture.molar_mass.copy(),
        partitioning=np.ones(fixture.gas_concentration.shape, dtype=np.bool_),
        temperature=fixture.temperature.copy(),
        pressure=fixture.pressure.copy(),
    )


def _resident_state(fixture: ProcessFixture, device: str) -> SimpleNamespace:
    """Build one complete caller-owned resident state for a direct sequence."""
    wp = pytest.importorskip("warp")
    from particula.gpu.kernels.condensation import CondensationScratchBuffers
    from particula.gpu.kernels.exhaustion import ResamplingBuffers
    from particula.gpu.kernels.nucleation import (
        NucleationDiagnosticBuffers,
        NucleationExhaustionBuffers,
        NucleationFinalizedDemandBuffers,
        NucleationScratchBuffers,
    )
    from particula.gpu.kernels.thermodynamics import ThermodynamicsConfig
    from particula.gpu.warp_types import (
        WarpEnvironmentData,
        WarpGasData,
        WarpParticleData,
    )

    boxes, particles_count, species = fixture.masses.shape
    particles = WarpParticleData()
    particles.masses = wp.array(fixture.masses, dtype=wp.float64, device=device)
    particles.concentration = wp.array(
        fixture.particle_concentration, dtype=wp.float64, device=device
    )
    particles.charge = wp.array(fixture.charge, dtype=wp.float64, device=device)
    particles.density = wp.array(
        fixture.density, dtype=wp.float64, device=device
    )
    particles.volume = wp.array(fixture.volume, dtype=wp.float64, device=device)
    gas = WarpGasData()
    gas.molar_mass = wp.array(
        fixture.molar_mass, dtype=wp.float64, device=device
    )
    gas.concentration = wp.array(
        fixture.gas_concentration, dtype=wp.float64, device=device
    )
    gas.vapor_pressure = wp.zeros(
        (boxes, species), dtype=wp.float64, device=device
    )
    gas.partitioning = wp.array(
        fixture.partitioning.astype(np.int32), dtype=wp.int32, device=device
    )
    environment = WarpEnvironmentData()
    environment.temperature = wp.array(
        fixture.temperature, dtype=wp.float64, device=device
    )
    environment.pressure = wp.array(
        fixture.pressure, dtype=wp.float64, device=device
    )
    environment.saturation_ratio = wp.ones(
        (boxes, species), dtype=wp.float64, device=device
    )
    transfer_shape = (boxes, particles_count, species)
    mass_transfer = wp.zeros(transfer_shape, dtype=wp.float64, device=device)
    scratch = CondensationScratchBuffers(
        work_mass_transfer=None,
        total_mass_transfer=None,
        dynamic_viscosity=wp.zeros(boxes, dtype=wp.float64, device=device),
        mean_free_path=wp.zeros(boxes, dtype=wp.float64, device=device),
        positive_mass_transfer_demand=wp.zeros(
            (boxes, species), dtype=wp.float64, device=device
        ),
        negative_mass_transfer_release=wp.zeros(
            (boxes, species), dtype=wp.float64, device=device
        ),
        positive_mass_transfer_scale=wp.zeros(
            (boxes, species), dtype=wp.float64, device=device
        ),
    )
    thermodynamics = ThermodynamicsConfig(
        modes=wp.zeros(species, dtype=wp.int32, device=device),
        parameters=wp.array(
            np.column_stack(
                (np.full(species, 1.0e-12), np.zeros((species, 3)))
            ),
            dtype=wp.float64,
            device=device,
        ),
        molar_mass_reference=wp.array(
            fixture.molar_mass, dtype=wp.float64, device=device
        ),
    )
    resampling = ResamplingBuffers(
        retained_counts=wp.zeros(boxes, dtype=wp.int32, device=device),
        released_counts=wp.zeros(boxes, dtype=wp.int32, device=device),
        retained_indices=wp.zeros(
            (boxes, particles_count), dtype=wp.int32, device=device
        ),
        released_indices=wp.zeros(
            (boxes, particles_count), dtype=wp.int32, device=device
        ),
        sorted_indices=wp.zeros(
            (boxes, particles_count), dtype=wp.int32, device=device
        ),
        replacement_masses=wp.zeros(
            transfer_shape, dtype=wp.float64, device=device
        ),
        replacement_concentration=wp.zeros(
            (boxes, particles_count), dtype=wp.float64, device=device
        ),
        replacement_charge=wp.zeros(
            (boxes, particles_count), dtype=wp.float64, device=device
        ),
        source_radii=wp.zeros(
            (boxes, particles_count), dtype=wp.float64, device=device
        ),
        radius_cubed_relative_error=wp.zeros(
            boxes, dtype=wp.float64, device=device
        ),
        mean_radius_relative_error=wp.zeros(
            boxes, dtype=wp.float64, device=device
        ),
        surface_relative_error=wp.zeros(boxes, dtype=wp.float64, device=device),
        diversity_absolute_error=wp.zeros(
            boxes, dtype=wp.float64, device=device
        ),
        planning_status=wp.zeros(boxes, dtype=wp.int32, device=device),
    )
    exhaustion = NucleationExhaustionBuffers(
        resampling_buffers=resampling,
        demand_workspace=wp.zeros(boxes, dtype=wp.float64, device=device),
        final_demand=wp.zeros(boxes, dtype=wp.float64, device=device),
        requested_scale=wp.ones(boxes, dtype=wp.float64, device=device),
        minimum_scale=wp.ones(boxes, dtype=wp.float64, device=device),
        minimum_volume=wp.full(boxes, 1.0e-12, dtype=wp.float64, device=device),
        resolved_scale=wp.zeros(boxes, dtype=wp.float64, device=device),
        resampling_releasable_counts=wp.zeros(
            boxes, dtype=wp.int32, device=device
        ),
        required_release_counts=wp.zeros(boxes, dtype=wp.int32, device=device),
        scaling_required=wp.zeros(boxes, dtype=wp.int32, device=device),
        final_counts=wp.zeros(boxes, dtype=wp.int32, device=device),
        final_selected_slot_indices=wp.zeros(
            (boxes, particles_count), dtype=wp.int32, device=device
        ),
    )
    return SimpleNamespace(
        particles=particles,
        gas=gas,
        environment=environment,
        scratch=scratch,
        mass_transfer=mass_transfer,
        thermodynamics=thermodynamics,
        collision_pairs=wp.full(
            (boxes, particles_count, 2), -1, dtype=wp.int32, device=device
        ),
        collision_counts=wp.zeros(boxes, dtype=wp.int32, device=device),
        coagulation_rng=wp.zeros(boxes, dtype=wp.uint32, device=device),
        wall_rng=wp.zeros(boxes, dtype=wp.uint32, device=device),
        nucleation_scratch=NucleationScratchBuffers(
            *[
                wp.zeros(boxes, dtype=wp.float64, device=device)
                for _ in range(3)
            ]
        ),
        finalized=NucleationFinalizedDemandBuffers(
            wp.zeros(boxes, dtype=wp.int32, device=device),
            wp.zeros(boxes, dtype=wp.float64, device=device),
            wp.zeros((boxes, species), dtype=wp.float64, device=device),
        ),
        diagnostics=NucleationDiagnosticBuffers(
            wp.zeros(boxes, dtype=wp.int32, device=device),
            wp.full(
                (boxes, particles_count), -1, dtype=wp.int32, device=device
            ),
            wp.full(
                (boxes, particles_count), -1, dtype=wp.int32, device=device
            ),
            wp.zeros(boxes, dtype=wp.int32, device=device),
            wp.zeros(boxes, dtype=wp.int32, device=device),
        ),
        exhaustion=exhaustion,
    )


def _assert_fixture_schema(fixture: ProcessFixture) -> None:
    """Validate a fixture's fixed-shape schema, dtypes, and physical values.

    Args:
        fixture: Test-local fixture to validate.

    Raises:
        ValueError: If a field has an invalid shape, dtype, or physical value.
    """
    if fixture.masses.ndim != 3:
        raise ValueError("masses must have shape (B, N, S)")
    boxes, particles, species = fixture.masses.shape
    _assert_fixture_shapes(fixture, boxes, particles, species)
    _assert_fixture_dtypes(fixture)
    _assert_fixture_values(fixture)


def _assert_fixture_shapes(
    fixture: ProcessFixture,
    boxes: int,
    particles: int,
    species: int,
) -> None:
    """Validate fields against the supplied fixed particle-storage dimensions.

    Args:
        fixture: Test-local fixture whose fields are checked.
        boxes: Expected number of boxes.
        particles: Expected particle capacity per box.
        species: Expected number of species.

    Raises:
        ValueError: If a fixture field does not have its expected shape.
    """
    expected = {
        "particle_concentration": (boxes, particles),
        "charge": (boxes, particles),
        "density": (species,),
        "volume": (boxes,),
        "gas_concentration": (boxes, species),
        "molar_mass": (species,),
        "partitioning": (boxes, species),
        "temperature": (boxes,),
        "pressure": (boxes,),
    }
    for name, shape in expected.items():
        if getattr(fixture, name).shape != shape:
            raise ValueError(f"{name} must have shape {shape}")


def _assert_fixture_dtypes(fixture: ProcessFixture) -> None:
    """Validate required float64 and boolean fixture dtypes.

    Args:
        fixture: Test-local fixture whose array dtypes are checked.

    Raises:
        ValueError: If a floating field is not float64 or flags are not bool.
    """
    for item in fields(fixture):
        if item.name not in {"name", "partitioning"}:
            value = getattr(fixture, item.name)
            if isinstance(value, np.ndarray) and value.dtype != np.float64:
                raise ValueError(f"{item.name} must use np.float64")
    if fixture.partitioning.dtype != np.bool_:
        raise ValueError("partitioning must use np.bool_")


def _assert_fixture_values(fixture: ProcessFixture) -> None:
    """Validate finite, nonnegative or positive fixture physical values.

    Args:
        fixture: Test-local fixture whose numerical fields are checked.

    Raises:
        ValueError: If a field contains values outside its required domain.
    """
    nonnegative = ("masses", "particle_concentration", "gas_concentration")
    for name in nonnegative:
        value = getattr(fixture, name)
        if not np.all(np.isfinite(value)) or np.any(value < 0.0):
            raise ValueError(f"{name} must be finite and nonnegative")
    if not np.all(np.isfinite(fixture.charge)):
        raise ValueError("charge must be finite")
    for name in ("density", "molar_mass", "volume", "temperature", "pressure"):
        value = getattr(fixture, name)
        if not np.all(np.isfinite(value)) or np.any(value <= 0.0):
            raise ValueError(f"{name} must be finite and positive")


def _snapshot_array(value: Any) -> _ArraySnapshot:
    """Capture NumPy- or Warp-like array metadata and exact bytes.

    Args:
        value: Array-like value exposing either NumPy conversion or array data.

    Returns:
        Snapshot retaining identity, metadata, and raw bytes.
    """
    values = value.numpy() if hasattr(value, "numpy") else np.asarray(value)
    copied = np.array(values, copy=True)
    return _ArraySnapshot(
        identity=id(value),
        shape=tuple(value.shape),
        dtype=value.dtype,
        device=getattr(value, "device", None),
        bytes_value=copied.tobytes(),
    )


def _snapshot_owners(**owners: Any) -> dict[str, dict[str, _ArraySnapshot]]:
    """Snapshot named arrays or array fields owned by named objects.

    Args:
        **owners: Named arrays, fixture records, or namespaces to snapshot.

    Returns:
        Snapshots keyed by owner name and field name.
    """
    result: dict[str, dict[str, _ArraySnapshot]] = {}
    for owner_name, owner in owners.items():
        if isinstance(owner, np.ndarray) or hasattr(owner, "numpy"):
            result[owner_name] = {"value": _snapshot_array(owner)}
            continue
        result[owner_name] = {
            name: _snapshot_array(value)
            for name, value in vars(owner).items()
            if isinstance(value, np.ndarray) or hasattr(value, "numpy")
        }
    return result


def _assert_snapshot_unchanged(
    snapshot: dict[str, dict[str, _ArraySnapshot]], **owners: Any
) -> None:
    """Assert captured fields retain identity, metadata, and exact bytes.

    Args:
        snapshot: Previously captured owner and field snapshots.
        **owners: Current named owners corresponding to ``snapshot``.

    Raises:
        AssertionError: If a captured field was replaced or changed.
    """
    for owner_name, captured_fields in snapshot.items():
        owner = owners[owner_name]
        for field_name, before in captured_fields.items():
            value = (
                owner if field_name == "value" else getattr(owner, field_name)
            )
            after = _snapshot_array(value)
            assert after.identity == before.identity, (
                f"{owner_name}.{field_name} was replaced"
            )
            assert after.shape == before.shape, (
                f"{owner_name}.{field_name} shape changed"
            )
            assert after.dtype == before.dtype, (
                f"{owner_name}.{field_name} dtype changed"
            )
            assert after.device == before.device, (
                f"{owner_name}.{field_name} device changed"
            )
            assert after.bytes_value == before.bytes_value, (
                f"{owner_name}.{field_name} changed"
            )


def _assert_only_fields_changed(
    before: dict[str, dict[str, _ArraySnapshot]],
    allowed_fields: set[str],
    **owners: Any,
) -> None:
    """Assert that only explicitly permitted owner fields changed.

    Args:
        before: Previously captured owner and field snapshots.
        allowed_fields: Permitted changes named as ``"owner.field"`` keys.
        **owners: Current named owners corresponding to ``before``.

    Raises:
        AssertionError: If an unpermitted field was replaced or changed.
    """
    for owner_name, captured in before.items():
        owner = owners[owner_name]
        for field_name, prior in captured.items():
            key = f"{owner_name}.{field_name}"
            value = (
                owner if field_name == "value" else getattr(owner, field_name)
            )
            after = _snapshot_array(value)
            changed = (
                after.identity != prior.identity
                or after.shape != prior.shape
                or after.dtype != prior.dtype
                or after.device != prior.device
                or after.bytes_value != prior.bytes_value
            )
            assert not changed or key in allowed_fields, f"{key} changed"


def _particle_inventory(
    masses: np.ndarray, concentration: np.ndarray
) -> np.ndarray:
    """Return concentration-weighted particle mass by box and species.

    Args:
        masses: Particle masses with shape ``(boxes, particles, species)``.
        concentration: Particle concentrations with shape ``(boxes, particles)``.

    Returns:
        Particle inventory with shape ``(boxes, species)``.
    """
    return np.sum(masses * concentration[..., None], axis=1)


def _particle_plus_gas_inventory(
    masses: np.ndarray, concentration: np.ndarray, gas: np.ndarray
) -> np.ndarray:
    """Return per-box, per-species particle-plus-gas inventory.

    Args:
        masses: Particle masses with shape ``(boxes, particles, species)``.
        concentration: Particle concentrations with shape ``(boxes, particles)``.
        gas: Gas concentrations with shape ``(boxes, species)``.

    Returns:
        Combined particle and gas inventory with shape ``(boxes, species)``.
    """
    return _particle_inventory(masses, concentration) + gas


def _active_slot_mass_and_charge(
    masses: np.ndarray, concentration: np.ndarray, charge: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return concentration-weighted active-slot mass and signed charge."""
    active = concentration > 0.0
    weighted_concentration = concentration * active
    return np.sum(masses * weighted_concentration[..., None], axis=1), np.sum(
        charge * weighted_concentration, axis=1
    )


def _dilution_expectation(
    concentration: np.ndarray, coefficient: float | np.ndarray, time_step: float
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate independent finite-step dilution concentrations and loss."""
    if concentration.dtype != np.float64 or concentration.ndim != 2:
        raise ValueError("concentration must be a rank-2 np.float64 array")
    if not np.all(np.isfinite(concentration)) or np.any(concentration < 0.0):
        raise ValueError("concentration must be finite and nonnegative")
    coefficient_values = np.asarray(coefficient)
    boxes = concentration.shape[0]
    if (
        coefficient_values.dtype != np.float64
        or coefficient_values.shape not in {(), (boxes,)}
    ):
        raise ValueError("coefficient must be float64 scalar or shape (B,)")
    if not np.all(np.isfinite(coefficient_values)) or np.any(
        coefficient_values < 0.0
    ):
        raise ValueError("coefficient must be finite and nonnegative")
    if (
        not isinstance(time_step, (float, np.floating))
        or not np.isfinite(time_step)
        or time_step < 0.0
    ):
        raise ValueError("time_step must be finite and nonnegative")
    rates = (
        np.full(boxes, coefficient_values, dtype=np.float64)
        if coefficient_values.ndim == 0
        else coefficient_values
    )
    final = concentration * np.exp(-rates[:, None] * time_step)
    return final, concentration - final


def _wall_loss_budget(
    masses: np.ndarray, concentration: np.ndarray, removed_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return retained and removed concentration-weighted mass budgets."""
    weighted = masses * concentration[..., None]
    removed = np.sum(weighted * removed_mask[..., None], axis=1)
    return np.sum(weighted, axis=1) - removed, removed


def _binomial_three_sigma_bound(probability: float, trials: int) -> float:
    """Return the three-sigma count bound for independent Bernoulli trials."""
    if (
        not isinstance(probability, (float, np.floating))
        or not np.isfinite(probability)
        or not 0.0 <= probability <= 1.0
    ):
        raise ValueError("probability must be finite and in [0, 1]")
    if not isinstance(trials, (int, np.integer)) or trials < 1:
        raise ValueError("trials must be a positive integer")
    return 3.0 * np.sqrt(trials * probability * (1.0 - probability))


def _slot_expectation(
    masses: np.ndarray, concentration: np.ndarray, charge: np.ndarray
) -> SlotExpectation:
    """Classify active/free slots with the documented fixed-slot definitions."""
    if (
        masses.ndim != 3
        or concentration.shape != masses.shape[:2]
        or charge.shape != masses.shape[:2]
    ):
        raise ValueError("Invalid particle slot state.")
    mass_valid = np.all(np.isfinite(masses) & (masses >= 0.0), axis=-1)
    total = np.sum(masses, axis=-1, dtype=np.float64)
    active = (
        np.isfinite(concentration)
        & (concentration > 0.0)
        & mass_valid
        & np.isfinite(total)
        & (total > 0.0)
        & np.isfinite(charge)
    )
    free = (
        (concentration == 0.0)
        & np.all(masses == 0.0, axis=-1)
        & (charge == 0.0)
        & np.isfinite(charge)
    )
    if np.any(~(active | free)):
        raise ValueError("Invalid particle slot state.")
    free_indices = np.full(concentration.shape, -1, dtype=np.int32)
    for box in range(concentration.shape[0]):
        free_indices[box, : np.sum(free[box])] = np.flatnonzero(free[box])
    return SlotExpectation(
        free_indices,
        np.sum(active, axis=1, dtype=np.int32),
        np.sum(free, axis=1, dtype=np.int32),
    )


def _assert_no_alias(*arrays: np.ndarray) -> None:
    """Reject deliberately aliased local sidecars before mutation."""
    if any(
        np.shares_memory(first, second)
        for index, first in enumerate(arrays)
        for second in arrays[index + 1 :]
    ):
        raise ValueError("caller-owned sidecars must not share storage")


def _require_warp_device(wp: Any, device: str) -> str:
    """Return a requested Warp test device or skip when unavailable."""
    if device not in warp_devices(wp):
        pytest.skip(f"{device} not available")
    return device


def _install_conversion_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    """Patch GPU conversion helpers to block intermediate host restores."""
    import particula.gpu as gpu_module
    import particula.gpu.conversion as conversion_module

    guard = SimpleNamespace(
        allow_final=False,
        particle_calls=0,
        gas_calls=0,
        environment_calls=0,
    )
    wrapper_names = (
        "from_warp_particle_data",
        "from_warp_gas_data",
        "from_warp_environment_data",
    )
    counters = {
        "from_warp_particle_data": "particle_calls",
        "from_warp_gas_data": "gas_calls",
        "from_warp_environment_data": "environment_calls",
    }

    for name in wrapper_names:
        original = getattr(conversion_module, name)

        def wrapper(*args: Any, _name=name, _original=original, **kwargs: Any):
            setattr(guard, counters[_name], getattr(guard, counters[_name]) + 1)
            if not guard.allow_final:
                raise AssertionError("intermediate from_warp_* restore blocked")
            return _original(*args, **kwargs)

        monkeypatch.setattr(conversion_module, name, wrapper)
        monkeypatch.setattr(gpu_module, name, wrapper)

    return guard


def _assert_restored_cpu_conversion_matches(
    fixture: ProcessFixture,
    state: SimpleNamespace,
    restored_particles: Any,
    restored_gas: Any,
    restored_environment: Any,
) -> None:
    """Compare restored CPU containers with explicit raw Warp snapshots."""
    assert restored_particles.masses.shape == fixture.masses.shape
    assert (
        restored_particles.concentration.shape
        == fixture.particle_concentration.shape
    )
    assert restored_particles.charge.shape == fixture.charge.shape
    assert restored_particles.density.shape == fixture.density.shape
    assert restored_particles.volume.shape == fixture.volume.shape
    assert restored_gas.molar_mass.shape == fixture.molar_mass.shape
    assert restored_gas.concentration.shape == fixture.gas_concentration.shape
    assert (
        restored_gas.partitioning.shape == fixture.gas_concentration.shape[1:]
    )
    assert restored_environment.temperature.shape == fixture.temperature.shape
    assert restored_environment.pressure.shape == fixture.pressure.shape
    assert (
        restored_environment.saturation_ratio.shape
        == fixture.gas_concentration.shape
    )
    npt.assert_array_equal(
        restored_particles.masses, state.particles.masses.numpy()
    )
    npt.assert_array_equal(
        restored_particles.concentration,
        state.particles.concentration.numpy(),
    )
    npt.assert_array_equal(
        restored_particles.charge, state.particles.charge.numpy()
    )
    npt.assert_array_equal(
        restored_particles.density, state.particles.density.numpy()
    )
    npt.assert_array_equal(
        restored_particles.volume, state.particles.volume.numpy()
    )
    npt.assert_array_equal(
        restored_gas.molar_mass, state.gas.molar_mass.numpy()
    )
    npt.assert_array_equal(
        restored_gas.concentration,
        state.gas.concentration.numpy(),
    )
    npt.assert_array_equal(
        restored_gas.partitioning,
        state.gas.partitioning.numpy()[0].astype(bool),
    )
    assert restored_gas.partitioning.dtype == np.bool_
    npt.assert_array_equal(
        restored_environment.temperature,
        state.environment.temperature.numpy(),
    )
    npt.assert_array_equal(
        restored_environment.pressure,
        state.environment.pressure.numpy(),
    )
    npt.assert_array_equal(
        restored_environment.saturation_ratio,
        state.environment.saturation_ratio.numpy(),
    )


@pytest.mark.warp
def test_conversion_guard_blocks_intermediate_restore_and_allows_final_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The conversion guard fails fast before the final restore phase."""
    pytest.importorskip("warp")
    import particula.gpu as gpu_module

    fixture = _resident_sequence_fixture(_build_process_fixtures()[0])
    state = _resident_state(fixture, "cpu")
    guard = _install_conversion_guard(monkeypatch)

    with pytest.raises(
        AssertionError, match="intermediate from_warp_\\* restore blocked"
    ):
        gpu_module.from_warp_particle_data(state.particles, sync=False)

    assert guard.particle_calls == 1
    assert guard.gas_calls == 0
    assert guard.environment_calls == 0

    guard.allow_final = True
    restored_particles = gpu_module.from_warp_particle_data(
        state.particles, sync=False
    )
    restored_gas = gpu_module.from_warp_gas_data(state.gas, sync=False)
    restored_environment = gpu_module.from_warp_environment_data(
        state.environment, sync=False
    )

    assert guard.particle_calls == 2
    assert guard.gas_calls == 1
    assert guard.environment_calls == 1
    _assert_restored_cpu_conversion_matches(
        fixture,
        state,
        restored_particles,
        restored_gas,
        restored_environment,
    )


def _resident_snapshot_owners(
    state: SimpleNamespace,
) -> dict[str, dict[str, _ArraySnapshot]]:
    """Snapshot all resident caller-owned arrays and nested sidecars."""
    return _snapshot_owners(
        particles=state.particles,
        gas=state.gas,
        environment=state.environment,
        scratch=state.scratch,
        mass_transfer=state.mass_transfer,
        thermodynamics=state.thermodynamics,
        collision_pairs=state.collision_pairs,
        collision_counts=state.collision_counts,
        coagulation_rng=state.coagulation_rng,
        wall_rng=state.wall_rng,
        nucleation_scratch=state.nucleation_scratch,
        finalized=state.finalized,
        diagnostics=state.diagnostics,
        exhaustion=state.exhaustion,
        resampling_buffers=state.exhaustion.resampling_buffers,
    )


@pytest.mark.parametrize(
    "fixture", _build_process_fixtures(), ids=lambda item: item.name
)
def test_process_fixtures_have_repeatable_valid_schema(
    fixture: ProcessFixture,
) -> None:
    """Canonical fixtures are fresh, deterministic fp64 fixed-shape inputs."""
    _assert_fixture_schema(fixture)
    matching = next(
        item for item in _build_process_fixtures() if item.name == fixture.name
    )
    for item in fields(fixture):
        first, second = (
            getattr(fixture, item.name),
            getattr(matching, item.name),
        )
        if isinstance(first, np.ndarray):
            npt.assert_array_equal(first, second)
            assert first is not second


def test_process_fixture_species_arrays_are_independently_owned() -> None:
    """Canonical fixtures never share mutable density or molar-mass arrays."""
    one, multi = _build_process_fixtures()
    rebuilt_one, _ = _build_process_fixtures()
    original_density = multi.density.copy()
    original_molar_mass = multi.molar_mass.copy()

    one.density[0] = 999.0
    one.molar_mass[1] = 999.0

    npt.assert_array_equal(multi.density, original_density)
    npt.assert_array_equal(multi.molar_mass, original_molar_mass)
    npt.assert_array_equal(rebuilt_one.density, original_density)
    npt.assert_array_equal(rebuilt_one.molar_mass, original_molar_mass)


@pytest.mark.parametrize("shape", [(0, 3, 2), (2, 0, 2)])
def test_fixture_schema_accepts_zero_box_and_zero_capacity(
    shape: tuple[int, int, int],
) -> None:
    """Auxiliary fixed-shape edge inputs retain empty diagnostic dimensions."""
    boxes, particles, species = shape
    fixture = ProcessFixture(
        "edge",
        np.zeros(shape, dtype=np.float64),
        np.zeros((boxes, particles), dtype=np.float64),
        np.zeros((boxes, particles), dtype=np.float64),
        np.ones(species, dtype=np.float64),
        np.ones(boxes, dtype=np.float64),
        np.zeros((boxes, species), dtype=np.float64),
        np.ones(species, dtype=np.float64),
        np.zeros((boxes, species), dtype=np.bool_),
        np.ones(boxes, dtype=np.float64),
        np.ones(boxes, dtype=np.float64),
    )
    _assert_fixture_schema(fixture)
    assert _particle_inventory(
        fixture.masses, fixture.particle_concentration
    ).shape == (boxes, species)
    assert _slot_expectation(
        fixture.masses, fixture.particle_concentration, fixture.charge
    ).free_indices.shape == (boxes, particles)


def test_snapshot_helpers_detect_replacement_mutation_and_nan_safely() -> None:
    """Snapshots preserve stale sidecars and unchanged NaN representations."""
    owner = SimpleNamespace(
        diagnostic=np.array([7], dtype=np.int32),
        work=np.array([np.nan]),
        rng=np.array([3], dtype=np.uint32),
    )
    before = _snapshot_owners(particle=owner)
    _assert_snapshot_unchanged(before, particle=owner)
    owner.diagnostic = np.array([7], dtype=np.int32)
    with pytest.raises(
        AssertionError, match="particle.diagnostic was replaced"
    ):
        _assert_snapshot_unchanged(before, particle=owner)
    owner.diagnostic = np.array([7], dtype=np.int32)
    owner.work[0] = 2.0
    with pytest.raises(AssertionError, match="particle.work changed"):
        _assert_only_fields_changed(
            before, {"particle.diagnostic"}, particle=owner
        )


def test_snapshot_helpers_detect_representation_level_mutations() -> None:
    """Byte-exact snapshots reject signed-zero and NaN-payload changes."""
    signed_zero = SimpleNamespace(data=np.array([0.0], dtype=np.float64))
    signed_zero_before = _snapshot_owners(array=signed_zero)
    signed_zero.data[0] = -0.0
    with pytest.raises(AssertionError, match="array.data changed"):
        _assert_snapshot_unchanged(signed_zero_before, array=signed_zero)

    nan_payload = SimpleNamespace(
        data=np.array([0x7FF8000000000001], dtype=np.uint64).view(np.float64)
    )
    nan_before = _snapshot_owners(array=nan_payload)
    _assert_snapshot_unchanged(nan_before, array=nan_payload)
    nan_payload.data[:] = np.array([0x7FF8000000000002], dtype=np.uint64).view(
        np.float64
    )
    with pytest.raises(AssertionError, match="array.data changed"):
        _assert_snapshot_unchanged(nan_before, array=nan_payload)


def test_rejected_local_alias_validation_preserves_all_named_owners() -> None:
    """Local pre-mutation rejection keeps particle, gas, and sidecars exact."""
    particle = SimpleNamespace(
        masses=np.ones((1, 1, 1)), concentration=np.ones((1, 1))
    )
    gas = SimpleNamespace(concentration=np.ones((1, 1)))
    sidecar = np.ones(2)
    diagnostic = np.ones(2, dtype=np.int32)
    work = np.ones(2)
    rng = np.ones(2, dtype=np.uint32)
    before = _snapshot_owners(
        particle=particle,
        gas=gas,
        sidecar=sidecar,
        diagnostic=diagnostic,
        work=work,
        rng=rng,
    )
    with pytest.raises(ValueError, match="must not share storage"):
        _assert_no_alias(sidecar, sidecar)
    _assert_snapshot_unchanged(
        before,
        particle=particle,
        gas=gas,
        sidecar=sidecar,
        diagnostic=diagnostic,
        work=work,
        rng=rng,
    )


@pytest.mark.parametrize(
    "fixture", _build_process_fixtures(), ids=lambda item: item.name
)
def test_inventory_transfers_conserve_per_box_and_species(
    fixture: ProcessFixture,
) -> None:
    """Authored condensation and nucleation transfers conserve total inventory."""
    before = _particle_plus_gas_inventory(
        fixture.masses,
        fixture.particle_concentration,
        fixture.gas_concentration,
    )
    transfer = (
        np.full(before.shape, 1.0e-20, dtype=np.float64) * fixture.partitioning
    )
    after_mass = fixture.masses.copy()
    for box in range(fixture.masses.shape[0]):
        particle = np.flatnonzero(fixture.particle_concentration[box] > 0.0)[0]
        after_mass[box, particle, :] += (
            transfer[box] / fixture.particle_concentration[box, particle]
        )
    after_gas = fixture.gas_concentration - transfer
    npt.assert_allclose(
        _particle_plus_gas_inventory(
            after_mass, fixture.particle_concentration, after_gas
        ),
        before,
        rtol=1e-12,
        atol=1e-30,
    )
    disabled_mass = fixture.masses.copy()
    disabled_gas = fixture.gas_concentration.copy()
    disabled_transfer = np.full(before.shape, 1.0e-20, dtype=np.float64)
    disabled_transfer[~fixture.partitioning] = 0.0
    for box in range(fixture.masses.shape[0]):
        particle = np.flatnonzero(fixture.particle_concentration[box] > 0.0)[0]
        disabled_mass[box, particle, :] += (
            disabled_transfer[box]
            / fixture.particle_concentration[box, particle]
        )
    disabled_gas -= disabled_transfer
    disabled_lanes = ~fixture.partitioning
    npt.assert_array_equal(disabled_transfer[disabled_lanes], 0.0)
    for box in range(fixture.masses.shape[0]):
        disabled_species = ~fixture.partitioning[box]
        npt.assert_array_equal(
            disabled_mass[box, :, disabled_species],
            fixture.masses[box, :, disabled_species],
        )
        npt.assert_array_equal(
            disabled_gas[box, disabled_species],
            fixture.gas_concentration[box, disabled_species],
        )


@pytest.mark.parametrize(
    "fixture", _build_process_fixtures(), ids=lambda item: item.name
)
def test_dilution_wall_loss_and_slot_helpers_match_direct_expectations(
    fixture: ProcessFixture,
) -> None:
    """Independent direct-equation helpers retain budgets and sparse slots."""
    final, loss = _dilution_expectation(
        fixture.particle_concentration, np.float64(0.2), 3.0
    )
    npt.assert_allclose(final + loss, fixture.particle_concentration)
    retained, removed = _wall_loss_budget(
        fixture.masses,
        fixture.particle_concentration,
        fixture.particle_concentration > 0.0,
    )
    npt.assert_allclose(
        retained + removed,
        _particle_inventory(fixture.masses, fixture.particle_concentration),
    )
    expected = _slot_expectation(
        fixture.masses, fixture.particle_concentration, fixture.charge
    )
    assert np.all(
        expected.active_counts + expected.free_counts == fixture.masses.shape[1]
    )


def test_accounting_helpers_cover_charge_coagulation_and_boundaries() -> None:
    """Active accounting conserves authored coagulation mass and signed charge."""
    masses = np.array([[[1.0], [2.0], [0.0]]], dtype=np.float64)
    concentration = np.array([[1.0, 1.0, 0.0]], dtype=np.float64)
    charge = np.array([[2.0, -1.0, 0.0]], dtype=np.float64)
    before = _active_slot_mass_and_charge(masses, concentration, charge)
    after_masses = np.array([[[3.0], [0.0], [0.0]]], dtype=np.float64)
    after_concentration = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    after_charge = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    after = _active_slot_mass_and_charge(
        after_masses, after_concentration, after_charge
    )
    npt.assert_array_equal(after[0], before[0])
    npt.assert_array_equal(after[1], before[1])
    assert _binomial_three_sigma_bound(0.0, 1) == 0.0
    assert _binomial_three_sigma_bound(1.0, 1) == 0.0
    with pytest.raises(ValueError, match="probability"):
        _binomial_three_sigma_bound(1.1, 3)
    with pytest.raises(ValueError, match="trials"):
        _binomial_three_sigma_bound(0.5, 0)


def test_dilution_and_slot_local_validation_rejects_without_mutation() -> None:
    """Local invalid values preserve caller arrays and slot error wording."""
    concentration = np.ones((1, 2), dtype=np.float64)
    before = _snapshot_array(concentration)
    for coefficient, time_step in (
        (np.array([1], dtype=np.int32), 1.0),
        (np.array([-1.0]), 1.0),
        (np.array([np.nan]), 1.0),
        (np.array([1.0]), -1.0),
    ):
        with pytest.raises(ValueError):
            _dilution_expectation(concentration, coefficient, time_step)
        assert _snapshot_array(concentration).bytes_value == before.bytes_value
    with pytest.raises(ValueError, match="Invalid particle slot state."):
        _slot_expectation(
            np.zeros((1, 1, 1)), np.array([[1.0]]), np.zeros((1, 1))
        )


def test_dilution_expectation_handles_scalar_per_box_and_no_op_cases() -> None:
    """Dilution applies scalar and per-box rates without altering inputs."""
    concentration = np.array([[2.0, 4.0], [3.0, 6.0]], dtype=np.float64)
    before = _snapshot_array(concentration)
    scalar_final, scalar_loss = _dilution_expectation(
        concentration, np.float64(0.5), 0.0
    )
    per_box_final, per_box_loss = _dilution_expectation(
        concentration, np.array([0.0, 0.5], dtype=np.float64), 2.0
    )
    npt.assert_array_equal(scalar_final, concentration)
    npt.assert_array_equal(scalar_loss, np.zeros_like(concentration))
    npt.assert_allclose(per_box_final[0], concentration[0])
    npt.assert_allclose(per_box_final[1], concentration[1] * np.exp(-1.0))
    npt.assert_allclose(per_box_final + per_box_loss, concentration)
    assert _snapshot_array(concentration).bytes_value == before.bytes_value


def test_wall_loss_and_slot_expectations_cover_boundary_layouts() -> None:
    """Removal budgets and fixed-slot diagnostics cover empty boundary rows."""
    masses = np.array([[[1.0], [2.0]], [[0.0], [0.0]]], dtype=np.float64)
    concentration = np.array([[1.0, 3.0], [0.0, 0.0]], dtype=np.float64)
    charge = np.array([[1.0, -2.0], [0.0, 0.0]], dtype=np.float64)
    none_removed = np.zeros((2, 2), dtype=bool)
    all_removed = np.ones((2, 2), dtype=bool)
    retained, removed = _wall_loss_budget(masses, concentration, none_removed)
    npt.assert_array_equal(removed, np.zeros((2, 1), dtype=np.float64))
    npt.assert_array_equal(retained, _particle_inventory(masses, concentration))
    retained, removed = _wall_loss_budget(masses, concentration, all_removed)
    npt.assert_array_equal(retained, np.zeros((2, 1), dtype=np.float64))
    npt.assert_array_equal(removed, _particle_inventory(masses, concentration))
    expectation = _slot_expectation(masses, concentration, charge)
    npt.assert_array_equal(expectation.active_counts, np.array([2, 0]))
    npt.assert_array_equal(expectation.free_counts, np.array([0, 2]))
    npt.assert_array_equal(
        expectation.free_indices, np.array([[-1, -1], [0, 1]])
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("masses", np.zeros((1, 4, 2), dtype=np.float32), "masses must use"),
        ("volume", np.array([-1.0], dtype=np.float64), "volume must be"),
        (
            "gas_concentration",
            np.array([[-1.0, 0.0]], dtype=np.float64),
            "gas_concentration must be",
        ),
    ],
)
def test_malformed_local_fixtures_reject_without_mutation(
    field: str, value: np.ndarray, message: str
) -> None:
    """Malformed test-local fixture fields fail before caller state changes."""
    fixture = _build_process_fixtures()[0]
    if field == "masses":
        malformed = replace(fixture, masses=value)
    elif field == "volume":
        malformed = replace(fixture, volume=value)
    else:
        malformed = replace(fixture, gas_concentration=value)
    particle = SimpleNamespace(
        masses=fixture.masses,
        concentration=fixture.particle_concentration,
    )
    gas = fixture.gas_concentration
    sidecar = np.ones(1)
    diagnostic = np.ones(1, dtype=np.int32)
    work = np.ones(1)
    rng = np.ones(1, dtype=np.uint32)
    before = _snapshot_owners(
        particle=particle,
        gas=gas,
        sidecar=sidecar,
        diagnostic=diagnostic,
        work=work,
        rng=rng,
    )
    with pytest.raises(ValueError, match=message):
        _assert_fixture_schema(malformed)
    _assert_snapshot_unchanged(
        before,
        particle=particle,
        gas=gas,
        sidecar=sidecar,
        diagnostic=diagnostic,
        work=work,
        rng=rng,
    )


def test_exhaustion_expectations_select_policies_and_preserve_failed_inputs() -> (
    None
):
    """CPU planner activates, resamples first, scales, and fails closed."""

    def inputs(requested: int, free: int, releasable: int) -> ExhaustionInputs:
        indices = np.full((1, 4), -1, dtype=np.int32)
        indices[0, :free] = np.arange(free, dtype=np.int32)
        return ExhaustionInputs(
            np.array([requested], dtype=np.int32),
            np.array([free], dtype=np.int32),
            np.array([releasable], dtype=np.int32),
            indices,
        )

    activation = resolve_exhaustion(
        inputs(1, 2, 0), ExhaustionControls()
    ).box_plans[0]
    assert (
        activation.policy_code == POLICY_ACTIVATE
        and activation.activation_indices == (0,)
    )
    resample = resolve_exhaustion(
        inputs(3, 1, 2), ExhaustionControls()
    ).box_plans[0]
    assert (
        resample.policy_code == POLICY_RESAMPLE_DEFERRED
        and resample.admitted_count == 3
    )
    scaling = resolve_exhaustion(
        inputs(3, 1, 1),
        ExhaustionControls(
            resampling=False, representative_volume_scaling=True
        ),
    ).box_plans[0]
    assert (
        scaling.policy_code == POLICY_SCALE_DEFERRED
        and scaling.admitted_count == 3
    )
    rejected = inputs(3, 1, 1)
    snapshot = _snapshot_owners(
        requested=rejected.requested_count,
        free=rejected.free_count,
        releasable=rejected.resampling_releasable_count,
        indices=rejected.free_indices,
    )
    with pytest.raises(ValueError, match="cannot represent"):
        resolve_exhaustion(rejected, ExhaustionControls(resampling=False))
    _assert_snapshot_unchanged(
        snapshot,
        requested=rejected.requested_count,
        free=rejected.free_count,
        releasable=rejected.resampling_releasable_count,
        indices=rejected.free_indices,
    )


@pytest.mark.warp
@pytest.mark.gpu_parity
def test_warp_mirrors_fresh_fixture_and_preserves_stale_sidecars() -> None:
    """Warp CPU and optional CUDA retain local mirror identity without steps."""
    wp = pytest.importorskip("warp")
    from particula.gpu.warp_types import WarpGasData, WarpParticleData

    fixture = _build_process_fixtures()[0]
    for device in warp_devices(wp):
        particles = WarpParticleData()
        gas = WarpGasData()
        particles.masses = wp.array(
            fixture.masses, dtype=wp.float64, device=device
        )
        particles.concentration = wp.array(
            fixture.particle_concentration, dtype=wp.float64, device=device
        )
        particles.charge = wp.array(
            fixture.charge, dtype=wp.float64, device=device
        )
        particles.density = wp.array(
            fixture.density, dtype=wp.float64, device=device
        )
        particles.volume = wp.array(
            fixture.volume, dtype=wp.float64, device=device
        )
        gas.molar_mass = wp.array(
            fixture.molar_mass, dtype=wp.float64, device=device
        )
        gas.concentration = wp.array(
            fixture.gas_concentration, dtype=wp.float64, device=device
        )
        gas.vapor_pressure = wp.zeros(
            fixture.gas_concentration.shape, dtype=wp.float64, device=device
        )
        gas.partitioning = wp.array(
            fixture.partitioning.astype(np.int32), dtype=wp.int32, device=device
        )
        diagnostic = wp.full((1,), 7, dtype=wp.int32, device=device)
        rng = wp.full((1,), 3, dtype=wp.uint32, device=device)
        before = _snapshot_owners(
            particles=particles, gas=gas, diagnostic=diagnostic, rng=rng
        )
        _assert_snapshot_unchanged(
            before, particles=particles, gas=gas, diagnostic=diagnostic, rng=rng
        )
        assert (
            particles.masses.dtype == wp.float64
            and particles.masses.shape == fixture.masses.shape
        )


def _run_resident_sequence_on_device(
    fixture: ProcessFixture,
    device: str,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[SimpleNamespace, SimpleNamespace]:
    """Run the full direct-GPU resident sequence on one Warp device."""
    wp = pytest.importorskip("warp")
    import particula.gpu as gpu_module
    from particula.gpu.kernels import (
        coagulation_step_gpu,
        condensation_step_gpu,
        dilution_step_gpu,
        nucleation_step_gpu,
        wall_loss_step_gpu,
    )
    from particula.gpu.kernels.nucleation import (
        NucleationConfig,
        NucleationExhaustionControls,
    )
    from particula.gpu.kernels.wall_loss import NeutralWallLossConfig

    state = _resident_state(_resident_sequence_fixture(fixture), device)
    guard = _install_conversion_guard(monkeypatch)
    wp.synchronize()
    initial_inventory = _particle_plus_gas_inventory(
        state.particles.masses.numpy(),
        state.particles.concentration.numpy(),
        state.gas.concentration.numpy(),
    )

    condensed_particles, transfer = condensation_step_gpu(
        state.particles,
        state.gas,
        None,
        None,
        0.01,
        mass_transfer=state.mass_transfer,
        environment=state.environment,
        thermodynamics=state.thermodynamics,
        scratch_buffers=state.scratch,
    )
    assert condensed_particles is state.particles
    assert transfer is state.mass_transfer
    wp.synchronize()
    condensed_inventory = _particle_plus_gas_inventory(
        state.particles.masses.numpy(),
        state.particles.concentration.numpy(),
        state.gas.concentration.numpy(),
    )
    npt.assert_allclose(
        condensed_inventory,
        initial_inventory,
        rtol=1e-12,
        atol=1e-30,
    )

    mass_charge_before = _active_slot_mass_and_charge(
        state.particles.masses.numpy(),
        state.particles.concentration.numpy(),
        state.particles.charge.numpy(),
    )
    assert np.all(np.isfinite(mass_charge_before[0]))
    assert np.all(np.isfinite(mass_charge_before[1]))
    returned = coagulation_step_gpu(
        state.particles,
        None,
        None,
        1.0,
        max_collisions=state.collision_pairs.shape[1],
        collision_pairs=state.collision_pairs,
        n_collisions=state.collision_counts,
        rng_states=state.coagulation_rng,
        initialize_rng=True,
        environment=state.environment,
    )
    assert returned[0] is state.particles
    assert returned[1] is state.collision_pairs
    assert returned[2] is state.collision_counts
    wp.synchronize()
    mass_charge_after = _active_slot_mass_and_charge(
        state.particles.masses.numpy(),
        state.particles.concentration.numpy(),
        state.particles.charge.numpy(),
    )
    npt.assert_allclose(mass_charge_after[0], mass_charge_before[0])
    npt.assert_allclose(mass_charge_after[1], mass_charge_before[1])
    collision_counts = state.collision_counts.numpy()
    collision_pairs = state.collision_pairs.numpy()
    assert np.all((collision_counts >= 0) & (collision_counts <= 4))
    for box, count in enumerate(collision_counts):
        pairs = collision_pairs[box, :count]
        assert np.all((pairs >= 0) & (pairs < fixture.masses.shape[1]))
        assert np.all(state.particles.concentration.numpy()[box, pairs] >= 0.0)

    rng_after_first_call = state.coagulation_rng.numpy().copy()
    repeat = coagulation_step_gpu(
        state.particles,
        None,
        None,
        1.0,
        max_collisions=state.collision_pairs.shape[1],
        collision_pairs=state.collision_pairs,
        n_collisions=state.collision_counts,
        rng_states=state.coagulation_rng,
        initialize_rng=False,
        environment=state.environment,
    )
    assert repeat[1] is state.collision_pairs
    assert repeat[2] is state.collision_counts
    wp.synchronize()
    assert np.any(state.coagulation_rng.numpy() != rng_after_first_call)

    concentration_before = state.particles.concentration.numpy().copy()
    gas_before = state.gas.concentration.numpy().copy()
    expected_particle, _ = _dilution_expectation(
        concentration_before, np.float64(0.2), 0.1
    )
    expected_gas, _ = _dilution_expectation(gas_before, np.float64(0.2), 0.1)
    diluted_particles, diluted_gas = dilution_step_gpu(
        state.particles, state.gas, 0.2, 0.1
    )
    assert diluted_particles is state.particles
    assert diluted_gas is state.gas
    wp.synchronize()
    npt.assert_allclose(
        state.particles.concentration.numpy(), expected_particle
    )
    npt.assert_allclose(state.gas.concentration.numpy(), expected_gas)

    wall_masses_before = state.particles.masses.numpy().copy()
    wall_concentration_before = state.particles.concentration.numpy().copy()
    wall_rng_before = state.wall_rng.numpy().copy()
    wall_config = NeutralWallLossConfig(
        "spherical",
        0.01,
        chamber_radius=np.nextafter(0.0, 1.0),
        mode="charged",
        wall_potential=-12.0,
        wall_electric_field=3.0,
    )
    assert (
        wall_loss_step_gpu(
            state.particles,
            None,
            None,
            1.0,
            config=wall_config,
            rng_states=state.wall_rng,
            initialize_rng=True,
            environment=state.environment,
        )
        is state.particles
    )
    wp.synchronize()
    assert state.wall_rng.shape == wall_rng_before.shape
    removed_mask = (wall_concentration_before > 0.0) & (
        state.particles.concentration.numpy() == 0.0
    )
    retained, removed = _wall_loss_budget(
        wall_masses_before,
        wall_concentration_before,
        removed_mask,
    )
    npt.assert_array_equal(
        state.particles.masses.numpy()[removed_mask],
        np.zeros((np.sum(removed_mask), fixture.masses.shape[2])),
    )
    npt.assert_array_equal(
        state.particles.charge.numpy()[removed_mask],
        np.zeros(np.sum(removed_mask)),
    )
    npt.assert_allclose(
        retained + removed,
        _particle_inventory(wall_masses_before, wall_concentration_before),
    )
    assert np.all(np.isfinite(state.gas.concentration.numpy()))
    assert np.all(state.gas.concentration.numpy() >= 0.0)
    _slot_expectation(
        state.particles.masses.numpy(),
        state.particles.concentration.numpy(),
        state.particles.charge.numpy(),
    )

    inventory_before_nucleation = _particle_plus_gas_inventory(
        state.particles.masses.numpy(),
        state.particles.concentration.numpy(),
        state.gas.concentration.numpy(),
    )
    config = NucleationConfig(
        rate_law="activation",
        coefficient=1.0e-12,
        survival_factor=1.0,
        precursor_index=0,
        molecule_counts=(1, 0),
        formation_diameter=1.0e-9,
        precursor_number_concentration_lower=0.0,
        precursor_number_concentration_upper=1.0e30,
        temperature_lower=200.0,
        temperature_upper=400.0,
    )
    result_particles, result_gas = nucleation_step_gpu(
        state.particles,
        state.gas,
        config,
        0.0,
        scratch=state.nucleation_scratch,
        finalized_demand=state.finalized,
        diagnostics=state.diagnostics,
        exhaustion_controls=NucleationExhaustionControls(True, True),
        exhaustion_buffers=state.exhaustion,
        environment=state.environment,
    )
    assert result_particles is state.particles
    assert result_gas is state.gas
    wp.synchronize()
    npt.assert_allclose(
        _particle_plus_gas_inventory(
            state.particles.masses.numpy(),
            state.particles.concentration.numpy(),
            state.gas.concentration.numpy(),
        ),
        inventory_before_nucleation,
        rtol=1e-12,
        atol=1e-30,
    )
    assert np.all(
        state.diagnostics.free_slot_counts.numpy() <= fixture.masses.shape[1]
    )

    wp.synchronize()
    raw_particle = state.particles.masses.numpy().copy()
    raw_concentration = state.particles.concentration.numpy().copy()
    raw_charge = state.particles.charge.numpy().copy()
    raw_density = state.particles.density.numpy().copy()
    raw_volume = state.particles.volume.numpy().copy()
    raw_gas_molar_mass = state.gas.molar_mass.numpy().copy()
    raw_gas_concentration = state.gas.concentration.numpy().copy()
    raw_partitioning = state.gas.partitioning.numpy().copy()
    raw_temperature = state.environment.temperature.numpy().copy()
    raw_pressure = state.environment.pressure.numpy().copy()
    raw_saturation_ratio = state.environment.saturation_ratio.numpy().copy()
    assert guard.particle_calls == 0
    assert guard.gas_calls == 0
    assert guard.environment_calls == 0
    guard.allow_final = True
    restored_particles = gpu_module.from_warp_particle_data(
        state.particles, sync=False
    )
    restored_gas = gpu_module.from_warp_gas_data(state.gas, sync=False)
    restored_environment = gpu_module.from_warp_environment_data(
        state.environment, sync=False
    )
    _assert_restored_cpu_conversion_matches(
        fixture,
        state,
        restored_particles,
        restored_gas,
        restored_environment,
    )
    npt.assert_array_equal(restored_particles.masses, raw_particle)
    npt.assert_array_equal(restored_particles.concentration, raw_concentration)
    npt.assert_array_equal(restored_particles.charge, raw_charge)
    npt.assert_array_equal(restored_particles.density, raw_density)
    npt.assert_array_equal(restored_particles.volume, raw_volume)
    npt.assert_array_equal(restored_gas.molar_mass, raw_gas_molar_mass)
    npt.assert_array_equal(restored_gas.concentration, raw_gas_concentration)
    npt.assert_array_equal(
        restored_gas.partitioning,
        raw_partitioning[0].astype(bool),
    )
    npt.assert_array_equal(
        restored_environment.temperature,
        raw_temperature,
    )
    npt.assert_array_equal(restored_environment.pressure, raw_pressure)
    npt.assert_array_equal(
        restored_environment.saturation_ratio,
        raw_saturation_ratio,
    )
    assert guard.particle_calls == 1
    assert guard.gas_calls == 1
    assert guard.environment_calls == 1
    return state, guard


@pytest.mark.warp
@pytest.mark.gpu_parity
@pytest.mark.parametrize(
    "fixture", _build_process_fixtures(), ids=lambda item: item.name
)
def test_resident_sequence_composes_five_direct_boundaries(
    fixture: ProcessFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compose every shipped direct boundary without restoring host state."""
    _run_resident_sequence_on_device(fixture, "cpu", monkeypatch)


@pytest.mark.warp
@pytest.mark.gpu_parity
@pytest.mark.cuda
@pytest.mark.parametrize(
    "fixture",
    [_build_process_fixtures()[1]],
    ids=lambda item: item.name,
)
def test_resident_sequence_composes_five_direct_boundaries_cuda(
    fixture: ProcessFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compose the resident direct sequence on the optional CUDA row."""
    wp = pytest.importorskip("warp")
    _require_warp_device(wp, "cuda")
    _run_resident_sequence_on_device(fixture, "cuda", monkeypatch)


@pytest.mark.warp
@pytest.mark.gpu_parity
@pytest.mark.parametrize(
    "fixture", _build_process_fixtures(), ids=lambda item: item.name
)
def test_condensation_preserves_disabled_original_fixture_lanes(
    fixture: ProcessFixture,
) -> None:
    """Original P1 sparse partitioning leaves disabled condensation lanes exact."""
    wp = pytest.importorskip("warp")
    from particula.gpu.kernels import condensation_step_gpu

    state = _resident_state(fixture, "cpu")
    disabled = ~fixture.partitioning
    wp.synchronize()
    mass_before = state.particles.masses.numpy().copy()
    gas_before = state.gas.concentration.numpy().copy()
    condensation_step_gpu(
        state.particles,
        state.gas,
        None,
        None,
        0.01,
        mass_transfer=state.mass_transfer,
        environment=state.environment,
        thermodynamics=state.thermodynamics,
        scratch_buffers=state.scratch,
    )
    wp.synchronize()
    for box in range(fixture.masses.shape[0]):
        npt.assert_array_equal(
            state.particles.masses.numpy()[box, :, disabled[box]],
            mass_before[box, :, disabled[box]],
        )
        npt.assert_array_equal(
            state.gas.concentration.numpy()[box, disabled[box]],
            gas_before[box, disabled[box]],
        )


@pytest.mark.warp
@pytest.mark.gpu_parity
@pytest.mark.parametrize(
    "fixture", _build_process_fixtures(), ids=lambda item: item.name
)
def test_zero_time_direct_calls_are_byte_exact_no_op(
    fixture: ProcessFixture,
) -> None:
    """Zero-time calls preserve process fields but refresh vapor pressure."""
    pytest.importorskip("warp")
    from particula.gpu.kernels import (
        coagulation_step_gpu,
        condensation_step_gpu,
        dilution_step_gpu,
        nucleation_step_gpu,
        wall_loss_step_gpu,
    )
    from particula.gpu.kernels.nucleation import (
        NucleationConfig,
        NucleationExhaustionControls,
    )
    from particula.gpu.kernels.wall_loss import NeutralWallLossConfig

    state = _resident_state(_resident_sequence_fixture(fixture), "cpu")
    particle_before = {
        "masses": state.particles.masses.numpy().copy(),
        "concentration": state.particles.concentration.numpy().copy(),
        "charge": state.particles.charge.numpy().copy(),
        "density": state.particles.density.numpy().copy(),
        "volume": state.particles.volume.numpy().copy(),
    }
    gas_before = {
        "concentration": state.gas.concentration.numpy().copy(),
        "molar_mass": state.gas.molar_mass.numpy().copy(),
        "partitioning": state.gas.partitioning.numpy().copy(),
    }
    environment_before = {
        "temperature": state.environment.temperature.numpy().copy(),
        "pressure": state.environment.pressure.numpy().copy(),
        "saturation_ratio": state.environment.saturation_ratio.numpy().copy(),
    }
    coagulation_rng_before = state.coagulation_rng.numpy().copy()
    wall_rng_before = state.wall_rng.numpy().copy()

    condensation_step_gpu(
        state.particles,
        state.gas,
        None,
        None,
        0.0,
        mass_transfer=state.mass_transfer,
        environment=state.environment,
        thermodynamics=state.thermodynamics,
        scratch_buffers=state.scratch,
    )
    npt.assert_allclose(
        state.gas.vapor_pressure.numpy(),
        np.full_like(state.gas.vapor_pressure.numpy(), 1.0e-12),
    )
    coagulation_step_gpu(
        state.particles,
        None,
        None,
        0.0,
        max_collisions=state.collision_pairs.shape[1],
        collision_pairs=state.collision_pairs,
        n_collisions=state.collision_counts,
        rng_states=state.coagulation_rng,
        initialize_rng=False,
        environment=state.environment,
    )
    dilution_step_gpu(state.particles, state.gas, 0.2, 0.0)
    wall_loss_step_gpu(
        state.particles,
        None,
        None,
        0.0,
        config=NeutralWallLossConfig(
            "spherical",
            0.01,
            chamber_radius=np.nextafter(0.0, 1.0),
            mode="charged",
            wall_potential=-12.0,
            wall_electric_field=3.0,
        ),
        rng_states=state.wall_rng,
        initialize_rng=False,
        environment=state.environment,
    )
    nucleation_step_gpu(
        state.particles,
        state.gas,
        NucleationConfig(
            rate_law="activation",
            coefficient=1.0e-12,
            survival_factor=1.0,
            precursor_index=0,
            molecule_counts=(1, 0),
            formation_diameter=1.0e-9,
            precursor_number_concentration_lower=0.0,
            precursor_number_concentration_upper=1.0e30,
            temperature_lower=200.0,
            temperature_upper=400.0,
        ),
        0.0,
        scratch=state.nucleation_scratch,
        finalized_demand=state.finalized,
        diagnostics=state.diagnostics,
        exhaustion_controls=NucleationExhaustionControls(True, True),
        exhaustion_buffers=state.exhaustion,
        environment=state.environment,
    )

    npt.assert_array_equal(
        state.particles.masses.numpy(), particle_before["masses"]
    )
    npt.assert_array_equal(
        state.particles.concentration.numpy(), particle_before["concentration"]
    )
    npt.assert_array_equal(
        state.particles.charge.numpy(), particle_before["charge"]
    )
    npt.assert_array_equal(
        state.particles.density.numpy(), particle_before["density"]
    )
    npt.assert_array_equal(
        state.particles.volume.numpy(), particle_before["volume"]
    )
    npt.assert_array_equal(
        state.gas.concentration.numpy(), gas_before["concentration"]
    )
    npt.assert_array_equal(
        state.gas.molar_mass.numpy(), gas_before["molar_mass"]
    )
    npt.assert_array_equal(
        state.gas.partitioning.numpy(), gas_before["partitioning"]
    )
    npt.assert_array_equal(
        state.environment.temperature.numpy(), environment_before["temperature"]
    )
    npt.assert_array_equal(
        state.environment.pressure.numpy(), environment_before["pressure"]
    )
    npt.assert_array_equal(
        state.environment.saturation_ratio.numpy(),
        environment_before["saturation_ratio"],
    )
    npt.assert_array_equal(
        state.coagulation_rng.numpy(), coagulation_rng_before
    )
    npt.assert_array_equal(state.wall_rng.numpy(), wall_rng_before)


@pytest.mark.warp
@pytest.mark.gpu_parity
def test_positive_time_nucleation_admits_slots_removes_gas_and_conserves() -> (
    None
):
    """A viable resident nucleation call activates a slot and conserves mass."""
    wp = pytest.importorskip("warp")
    from particula.gpu.kernels import nucleation_step_gpu
    from particula.gpu.kernels.nucleation import (
        NucleationConfig,
        NucleationExhaustionControls,
    )
    from particula.util.constants import AVOGADRO_NUMBER

    fixture = _resident_sequence_fixture(_build_process_fixtures()[0])
    state = _resident_state(fixture, "cpu")
    precursor_concentration = 1.0e-12
    state.gas.concentration = wp.array(
        [[precursor_concentration, 0.0]],
        dtype=wp.float64,
        device="cpu",
    )
    state.particles.masses = wp.zeros(
        state.particles.masses.shape, dtype=wp.float64, device="cpu"
    )
    state.particles.concentration = wp.zeros(
        state.particles.concentration.shape, dtype=wp.float64, device="cpu"
    )
    state.particles.charge = wp.zeros(
        state.particles.charge.shape, dtype=wp.float64, device="cpu"
    )
    inventory_before = _particle_plus_gas_inventory(
        state.particles.masses.numpy(),
        state.particles.concentration.numpy(),
        state.gas.concentration.numpy(),
    )
    gas_before = state.gas.concentration.numpy().copy()
    config = NucleationConfig(
        rate_law="activation",
        coefficient=(
            fixture.molar_mass[0]
            / (AVOGADRO_NUMBER * fixture.volume[0] * precursor_concentration)
        ),
        survival_factor=1.0,
        precursor_index=0,
        molecule_counts=(1, 0),
        formation_diameter=1.0e-9,
        precursor_number_concentration_lower=0.0,
        precursor_number_concentration_upper=1.0e30,
        temperature_lower=200.0,
        temperature_upper=400.0,
    )

    particles, gas = nucleation_step_gpu(
        state.particles,
        state.gas,
        config,
        1.0,
        scratch=state.nucleation_scratch,
        finalized_demand=state.finalized,
        diagnostics=state.diagnostics,
        exhaustion_controls=NucleationExhaustionControls(False, False),
        exhaustion_buffers=state.exhaustion,
        environment=state.environment,
    )

    assert particles is state.particles
    assert gas is state.gas
    wp.synchronize()
    npt.assert_array_equal(state.finalized.accepted_counts.numpy(), [1])
    npt.assert_array_equal(
        state.particles.concentration.numpy() > 0.0,
        [[True, False, False, False]],
    )
    assert state.gas.concentration.numpy()[0, 0] < gas_before[0, 0]
    npt.assert_allclose(
        _particle_plus_gas_inventory(
            state.particles.masses.numpy(),
            state.particles.concentration.numpy(),
            state.gas.concentration.numpy(),
        ),
        inventory_before,
        rtol=1e-12,
        atol=1e-30,
    )


@pytest.mark.warp
@pytest.mark.gpu_parity
@pytest.mark.parametrize(
    "fixture", _build_process_fixtures(), ids=lambda item: item.name
)
def test_zero_dilution_coefficient_is_byte_exact_no_op(
    fixture: ProcessFixture,
) -> None:
    """A zero dilution coefficient leaves resident state byte-for-byte exact."""
    pytest.importorskip("warp")
    from particula.gpu.kernels import dilution_step_gpu

    state = _resident_state(_resident_sequence_fixture(fixture), "cpu")
    before = _resident_snapshot_owners(state)
    dilution_step_gpu(state.particles, state.gas, 0.0, 1.0)
    _assert_snapshot_unchanged(
        before,
        particles=state.particles,
        gas=state.gas,
        environment=state.environment,
        scratch=state.scratch,
        mass_transfer=state.mass_transfer,
        thermodynamics=state.thermodynamics,
        collision_pairs=state.collision_pairs,
        collision_counts=state.collision_counts,
        coagulation_rng=state.coagulation_rng,
        wall_rng=state.wall_rng,
        nucleation_scratch=state.nucleation_scratch,
        finalized=state.finalized,
        diagnostics=state.diagnostics,
        exhaustion=state.exhaustion,
        resampling_buffers=state.exhaustion.resampling_buffers,
    )


@pytest.mark.warp
@pytest.mark.gpu_parity
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "scratch_buffers",
            object(),
            "scratch_buffers must be a CondensationScratchBuffers",
        ),
        (
            "thermodynamics",
            object(),
            "thermodynamics must be a ThermodynamicsConfig",
        ),
    ],
)
def test_condensation_preflight_rejects_invalid_metadata_without_mutation(
    field: str,
    value: Any,
    message: str,
) -> None:
    """Condensation rejects invalid metadata before any caller-owned mutation."""
    pytest.importorskip("warp")
    from particula.gpu.kernels import condensation_step_gpu

    fixture = _build_process_fixtures()[0]
    state = _resident_state(_resident_sequence_fixture(fixture), "cpu")
    before = _resident_snapshot_owners(state)
    kwargs: dict[str, Any] = {
        "mass_transfer": state.mass_transfer,
        "environment": state.environment,
        "thermodynamics": state.thermodynamics,
        "scratch_buffers": state.scratch,
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match=message):
        condensation_step_gpu(
            state.particles,
            state.gas,
            None,
            None,
            0.0,
            **kwargs,
        )
    _assert_snapshot_unchanged(
        before,
        particles=state.particles,
        gas=state.gas,
        environment=state.environment,
        scratch=state.scratch,
        mass_transfer=state.mass_transfer,
        thermodynamics=state.thermodynamics,
        collision_pairs=state.collision_pairs,
        collision_counts=state.collision_counts,
        coagulation_rng=state.coagulation_rng,
        wall_rng=state.wall_rng,
        nucleation_scratch=state.nucleation_scratch,
        finalized=state.finalized,
        diagnostics=state.diagnostics,
        exhaustion=state.exhaustion,
        resampling_buffers=state.exhaustion.resampling_buffers,
    )


@pytest.mark.warp
@pytest.mark.gpu_parity
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "collision_pairs",
            "warp_float64",
            "collision_pairs buffer must use dtype int32",
        ),
        (
            "rng_states",
            "warp_float64",
            "rng_states buffer must use dtype uint32",
        ),
    ],
)
def test_coagulation_preflight_rejects_invalid_output_schema_without_mutation(
    field: str,
    value: Any,
    message: str,
) -> None:
    """Coagulation rejects malformed output buffers before setup."""
    wp = pytest.importorskip("warp")
    from particula.gpu.kernels import coagulation_step_gpu

    fixture = _build_process_fixtures()[0]
    state = _resident_state(_resident_sequence_fixture(fixture), "cpu")
    before = _resident_snapshot_owners(state)
    kwargs: dict[str, Any] = {
        "max_collisions": state.collision_pairs.shape[1],
        "collision_pairs": state.collision_pairs,
        "n_collisions": state.collision_counts,
        "rng_states": state.coagulation_rng,
        "environment": state.environment,
    }
    if value == "warp_float64" and field == "collision_pairs":
        value = wp.zeros((1, 4, 2), dtype=wp.float64, device="cpu")
    elif value == "warp_float64":
        value = wp.zeros(1, dtype=wp.float64, device="cpu")
    kwargs[field] = value
    with pytest.raises(ValueError, match=message):
        coagulation_step_gpu(
            state.particles,
            None,
            None,
            0.0,
            **kwargs,
        )
    _assert_snapshot_unchanged(
        before,
        particles=state.particles,
        gas=state.gas,
        environment=state.environment,
        scratch=state.scratch,
        mass_transfer=state.mass_transfer,
        thermodynamics=state.thermodynamics,
        collision_pairs=state.collision_pairs,
        collision_counts=state.collision_counts,
        coagulation_rng=state.coagulation_rng,
        wall_rng=state.wall_rng,
        nucleation_scratch=state.nucleation_scratch,
        finalized=state.finalized,
        diagnostics=state.diagnostics,
        exhaustion=state.exhaustion,
        resampling_buffers=state.exhaustion.resampling_buffers,
    )


@pytest.mark.warp
@pytest.mark.gpu_parity
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("coefficient", "warp_int32", "coefficient must use"),
        ("coefficient", -0.1, "coefficient must be finite and nonnegative"),
        ("time_step", -1.0, "time_step"),
    ],
)
def test_dilution_preflight_rejects_invalid_inputs_without_mutation(
    field: str,
    value: Any,
    message: str,
) -> None:
    """Dilution rejects malformed coefficient or time-step inputs first."""
    wp = pytest.importorskip("warp")
    from particula.gpu.kernels import dilution_step_gpu

    fixture = _build_process_fixtures()[0]
    state = _resident_state(_resident_sequence_fixture(fixture), "cpu")
    before = _resident_snapshot_owners(state)
    args: dict[str, Any] = {
        "coefficient": np.float64(0.0),
        "time_step": 1.0,
    }
    if field == "coefficient" and value == "warp_int32":
        value = wp.array([1], dtype=wp.int32, device="cpu")
    args[field] = value
    with pytest.raises((TypeError, ValueError), match=message):
        dilution_step_gpu(state.particles, state.gas, **args)
    _assert_snapshot_unchanged(
        before,
        particles=state.particles,
        gas=state.gas,
        environment=state.environment,
        scratch=state.scratch,
        mass_transfer=state.mass_transfer,
        thermodynamics=state.thermodynamics,
        collision_pairs=state.collision_pairs,
        collision_counts=state.collision_counts,
        coagulation_rng=state.coagulation_rng,
        wall_rng=state.wall_rng,
        nucleation_scratch=state.nucleation_scratch,
        finalized=state.finalized,
        diagnostics=state.diagnostics,
        exhaustion=state.exhaustion,
        resampling_buffers=state.exhaustion.resampling_buffers,
    )


@pytest.mark.warp
@pytest.mark.gpu_parity
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "config",
            None,
            "config must be a NeutralWallLossConfig",
        ),
        (
            "rng_states",
            "warp_float64",
            "rng_states must use dtype uint32",
        ),
    ],
)
def test_wall_loss_preflight_rejects_invalid_configuration_without_mutation(
    field: str,
    value: Any,
    message: str,
) -> None:
    """Wall loss rejects invalid configuration and RNG schema before launch."""
    wp = pytest.importorskip("warp")
    from particula.gpu.kernels import wall_loss_step_gpu
    from particula.gpu.kernels.wall_loss import NeutralWallLossConfig

    fixture = _build_process_fixtures()[0]
    state = _resident_state(_resident_sequence_fixture(fixture), "cpu")
    before = _resident_snapshot_owners(state)
    config = NeutralWallLossConfig(
        "spherical",
        0.01,
        chamber_radius=1.0,
        mode="charged",
        wall_potential=-12.0,
        wall_electric_field=3.0,
    )
    kwargs: dict[str, Any] = {
        "config": config,
        "rng_states": state.wall_rng,
        "initialize_rng": True,
        "environment": state.environment,
    }
    if field == "rng_states" and value == "warp_float64":
        kwargs[field] = wp.zeros(1, dtype=wp.float64, device="cpu")
    else:
        kwargs[field] = value
    with pytest.raises((TypeError, ValueError), match=message):
        wall_loss_step_gpu(
            state.particles,
            None,
            None,
            0.0,
            **kwargs,
        )
    _assert_snapshot_unchanged(
        before,
        particles=state.particles,
        gas=state.gas,
        environment=state.environment,
        scratch=state.scratch,
        mass_transfer=state.mass_transfer,
        thermodynamics=state.thermodynamics,
        collision_pairs=state.collision_pairs,
        collision_counts=state.collision_counts,
        coagulation_rng=state.coagulation_rng,
        wall_rng=state.wall_rng,
        nucleation_scratch=state.nucleation_scratch,
        finalized=state.finalized,
        diagnostics=state.diagnostics,
        exhaustion=state.exhaustion,
        resampling_buffers=state.exhaustion.resampling_buffers,
    )


@pytest.mark.warp
@pytest.mark.gpu_parity
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("scratch", object(), "scratch must be NucleationScratchBuffers"),
        (
            "exhaustion_controls",
            object(),
            "controls must be NucleationExhaustionControls",
        ),
    ],
)
def test_nucleation_preflight_rejects_invalid_inputs_without_mutation(
    field: str,
    value: Any,
    message: str,
) -> None:
    """Nucleation rejects bad P1 or public-P4 inputs before any mutation."""
    pytest.importorskip("warp")
    from particula.gpu.kernels import nucleation_step_gpu
    from particula.gpu.kernels.nucleation import (
        NucleationConfig,
        NucleationExhaustionControls,
    )

    fixture = _build_process_fixtures()[0]
    state = _resident_state(_resident_sequence_fixture(fixture), "cpu")
    before = _resident_snapshot_owners(state)
    kwargs: dict[str, Any] = {
        "scratch": state.nucleation_scratch,
        "finalized_demand": state.finalized,
        "diagnostics": state.diagnostics,
        "exhaustion_controls": NucleationExhaustionControls(True, True),
        "exhaustion_buffers": state.exhaustion,
        "environment": state.environment,
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match=message):
        nucleation_step_gpu(
            state.particles,
            state.gas,
            NucleationConfig(
                rate_law="activation",
                coefficient=1.0e-12,
                survival_factor=1.0,
                precursor_index=0,
                molecule_counts=(1, 0),
                formation_diameter=1.0e-9,
                precursor_number_concentration_lower=0.0,
                precursor_number_concentration_upper=1.0e30,
                temperature_lower=200.0,
                temperature_upper=400.0,
            ),
            0.0,
            **kwargs,
        )
    _assert_snapshot_unchanged(
        before,
        particles=state.particles,
        gas=state.gas,
        environment=state.environment,
        scratch=state.scratch,
        mass_transfer=state.mass_transfer,
        thermodynamics=state.thermodynamics,
        collision_pairs=state.collision_pairs,
        collision_counts=state.collision_counts,
        coagulation_rng=state.coagulation_rng,
        wall_rng=state.wall_rng,
        nucleation_scratch=state.nucleation_scratch,
        finalized=state.finalized,
        diagnostics=state.diagnostics,
        exhaustion=state.exhaustion,
        resampling_buffers=state.exhaustion.resampling_buffers,
    )


@pytest.mark.warp
@pytest.mark.stochastic
def test_neutral_wall_loss_aggregate_removal_stays_within_binomial_band() -> (
    None
):
    """Aggregate neutral wall-loss removals stay within a 3-sigma band."""
    wp = pytest.importorskip("warp")
    from particula.dynamics.properties.wall_loss_coefficient import (
        get_spherical_wall_loss_coefficient_via_system_state,
    )
    from particula.gpu.kernels import wall_loss_step_gpu
    from particula.gpu.kernels.wall_loss import NeutralWallLossConfig

    trials = 100
    time_step = 100.0
    particle_radius = 1.0e-7
    particle_density = 1000.0
    temperature = 298.15
    pressure = 101325.0
    chamber_radius = 0.5
    wall_eddy_diffusivity = 0.01
    coefficient = get_spherical_wall_loss_coefficient_via_system_state(
        wall_eddy_diffusivity=wall_eddy_diffusivity,
        particle_radius=particle_radius,
        particle_density=particle_density,
        temperature=temperature,
        pressure=pressure,
        chamber_radius=chamber_radius,
    )
    expected_probability = 1.0 - np.exp(-float(coefficient) * time_step)
    expected_mean = trials * expected_probability
    bound = _binomial_three_sigma_bound(expected_probability, trials)
    config = NeutralWallLossConfig(
        "spherical",
        wall_eddy_diffusivity,
        chamber_radius=chamber_radius,
        mode="neutral",
    )
    particles = SimpleNamespace(
        masses=wp.full(
            (trials, 1, 1),
            4.0 * np.pi * particle_radius**3 * particle_density / 3.0,
            dtype=wp.float64,
            device="cpu",
        ),
        concentration=wp.ones((trials, 1), dtype=wp.float64, device="cpu"),
        charge=wp.ones((trials, 1), dtype=wp.float64, device="cpu"),
        density=wp.array([particle_density], dtype=wp.float64, device="cpu"),
        volume=wp.ones(trials, dtype=wp.float64, device="cpu"),
    )
    wall_loss_step_gpu(
        particles,
        temperature,
        pressure,
        time_step,
        config=config,
        rng_states=wp.array(
            np.arange(1, trials + 1, dtype=np.uint32),
            dtype=wp.uint32,
            device="cpu",
        ),
    )
    removed = int(np.sum(particles.concentration.numpy() == 0.0))

    assert abs(removed - expected_mean) <= bound
