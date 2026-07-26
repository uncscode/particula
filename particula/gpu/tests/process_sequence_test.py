"""P1 fixed-shape process-fixture and invariant evidence.

This private test module supplies deterministic fixtures and independent
accounting helpers.  It is not a resident process-sequence executor.
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
    """Hold deterministic, fixed-shape process input state."""

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
    """Hold independent fixed-slot diagnostics."""

    free_indices: np.ndarray
    active_counts: np.ndarray
    free_counts: np.ndarray


@dataclass(frozen=True)
class _ArraySnapshot:
    """Capture array identity, metadata, and values."""

    identity: int
    shape: tuple[int, ...]
    dtype: Any
    device: Any
    values: np.ndarray
    bytes_value: bytes


def _build_process_fixtures() -> tuple[ProcessFixture, ...]:
    """Build fresh deterministic one- and multi-box sparse fixtures."""
    common = dict(
        density=np.array([1000.0, 1500.0], dtype=np.float64),
        molar_mass=np.array([0.018, 0.098], dtype=np.float64),
    )
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
        **common,
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
        **common,
    )
    return one, multi


def _assert_fixture_schema(fixture: ProcessFixture) -> None:
    """Validate fixture schema, dtypes, and physical values."""
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
    """Validate array shapes against fixed particle storage dimensions."""
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
    """Validate exact float64 and bool fixture dtypes."""
    for item in fields(fixture):
        if item.name not in {"name", "partitioning"}:
            value = getattr(fixture, item.name)
            if isinstance(value, np.ndarray) and value.dtype != np.float64:
                raise ValueError(f"{item.name} must use np.float64")
    if fixture.partitioning.dtype != np.bool_:
        raise ValueError("partitioning must use np.bool_")


def _assert_fixture_values(fixture: ProcessFixture) -> None:
    """Validate finite physical values in fixture arrays."""
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
    """Capture NumPy- or Warp-like array metadata and copied values."""
    values = value.numpy() if hasattr(value, "numpy") else np.asarray(value)
    copied = np.array(values, copy=True)
    return _ArraySnapshot(
        identity=id(value),
        shape=tuple(value.shape),
        dtype=value.dtype,
        device=getattr(value, "device", None),
        values=copied,
        bytes_value=copied.tobytes(),
    )


def _snapshot_owners(**owners: Any) -> dict[str, dict[str, _ArraySnapshot]]:
    """Snapshot named arrays or each array field on named owners."""
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
    """Assert every captured owner field retains identity and exact values."""
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
            assert np.array_equal(
                after.values, before.values, equal_nan=True
            ), f"{owner_name}.{field_name} changed"


def _assert_only_fields_changed(
    before: dict[str, dict[str, _ArraySnapshot]],
    allowed_fields: set[str],
    **owners: Any,
) -> None:
    """Assert changed fields are explicitly permitted as ``owner.field`` keys."""
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
                or not np.array_equal(
                    after.values, prior.values, equal_nan=True
                )
            )
            assert not changed or key in allowed_fields, f"{key} changed"


def _particle_inventory(
    masses: np.ndarray, concentration: np.ndarray
) -> np.ndarray:
    """Return concentration-weighted particle mass by box and species."""
    return np.sum(masses * concentration[..., None], axis=1)


def _particle_plus_gas_inventory(
    masses: np.ndarray, concentration: np.ndarray, gas: np.ndarray
) -> np.ndarray:
    """Return per-box, per-species particle-plus-gas inventory."""
    return _particle_inventory(masses, concentration) + gas


def _active_slot_mass_and_charge(
    masses: np.ndarray, concentration: np.ndarray, charge: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return unweighted active-slot mass and signed charge."""
    active = concentration > 0.0
    return np.sum(masses * active[..., None], axis=1), np.sum(
        charge * active, axis=1
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
    """Snapshots preserve stale sidecars and intentionally compare NaNs safely."""
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
    disabled_transfer = np.where(
        fixture.partitioning,
        np.float64(0.0),
        np.float64(1.0e-20),
    )
    disabled_mass = fixture.masses.copy()
    disabled_gas = fixture.gas_concentration.copy()
    npt.assert_array_equal(disabled_mass, fixture.masses)
    npt.assert_array_equal(
        disabled_gas + disabled_transfer * ~fixture.partitioning,
        fixture.gas_concentration + disabled_transfer * ~fixture.partitioning,
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
    malformed = replace(fixture, **{field: value})
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
