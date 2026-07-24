"""Tests for fixed-shape particle slot discovery."""

from typing import Any

import numpy as np
import numpy.testing as npt
import pytest
from particula import particles
from particula.particles.particle_data import ParticleData
from particula.particles.slot_management import (
    activate_slots,
    get_slot_diagnostics,
)


def _make_data(
    masses: np.ndarray,
    concentration: np.ndarray,
    charge: np.ndarray,
) -> ParticleData:
    """Create particle data with density and volume matching its shapes."""
    return ParticleData(
        masses=np.asarray(masses, dtype=np.float64),
        concentration=np.asarray(concentration, dtype=np.float64),
        charge=np.asarray(charge, dtype=np.float64),
        density=np.ones(masses.shape[-1], dtype=np.float64),
        volume=np.ones(masses.shape[0], dtype=np.float64),
    )


def _assert_arrays_unchanged(
    fields: tuple[np.ndarray, ...],
    snapshots: tuple[np.ndarray, ...],
) -> None:
    """Assert caller-owned arrays retain values, schema, and identity."""
    for field, snapshot in zip(fields, snapshots, strict=True):
        assert field is not None
        assert field.shape == snapshot.shape
        assert field.dtype == snapshot.dtype
        npt.assert_array_equal(field, snapshot)


@pytest.mark.parametrize(
    ("masses", "concentration", "charge", "expected_active"),
    [
        (np.array([[[1.0, 2.0]]]), np.array([[1.0]]), np.array([[0.0]]), 1),
        (np.array([[[1.0, 0.0]]]), np.array([[2.0]]), np.array([[-3.0]]), 1),
        (np.zeros((1, 1, 2)), np.array([[0.0]]), np.array([[0.0]]), 0),
    ],
)
def test_get_slot_diagnostics_classifies_active_and_free_slots(
    masses: np.ndarray,
    concentration: np.ndarray,
    charge: np.ndarray,
    expected_active: int,
) -> None:
    """Classify valid active records and exactly-zero free records."""
    free_indices, active_counts, free_counts = get_slot_diagnostics(
        _make_data(masses, concentration, charge)
    )

    assert free_indices.dtype == np.int32
    assert active_counts.dtype == np.int32
    assert free_counts.dtype == np.int32
    npt.assert_array_equal(active_counts, [expected_active])
    npt.assert_array_equal(free_counts, [1 - expected_active])
    if expected_active:
        npt.assert_array_equal(free_indices, [[-1]])
    else:
        npt.assert_array_equal(free_indices, [[0]])


@pytest.mark.parametrize(
    ("masses", "concentration", "charge"),
    [
        (np.array([[[-1.0]]]), np.array([[1.0]]), np.array([[0.0]])),
        (np.array([[[np.nan]]]), np.array([[1.0]]), np.array([[0.0]])),
        (np.array([[[np.inf]]]), np.array([[1.0]]), np.array([[0.0]])),
        (np.array([[[1.0]]]), np.array([[-1.0]]), np.array([[0.0]])),
        (np.array([[[1.0]]]), np.array([[np.nan]]), np.array([[0.0]])),
        (np.array([[[1.0]]]), np.array([[np.inf]]), np.array([[0.0]])),
        (np.array([[[1.0]]]), np.array([[0.0]]), np.array([[0.0]])),
        (np.array([[[0.0]]]), np.array([[1.0]]), np.array([[0.0]])),
        (np.array([[[1.0, -1.0]]]), np.array([[1.0]]), np.array([[0.0]])),
        (np.zeros((1, 1, 1)), np.array([[0.0]]), np.array([[1.0]])),
        (np.array([[[1.0]]]), np.array([[1.0]]), np.array([[np.nan]])),
        (np.array([[[1.0]]]), np.array([[1.0]]), np.array([[np.inf]])),
    ],
)
def test_get_slot_diagnostics_rejects_invalid_slot_states(
    masses: np.ndarray,
    concentration: np.ndarray,
    charge: np.ndarray,
) -> None:
    """Reject contradictory, non-finite, and negative field values."""
    with pytest.raises(ValueError, match="^Invalid particle slot state[.]$"):
        get_slot_diagnostics(_make_data(masses, concentration, charge))


def test_get_slot_diagnostics_handles_zero_species_slots() -> None:
    """Treat zero-species zero records as free and positive records invalid."""
    free_data = _make_data(
        np.zeros((1, 1, 0)), np.array([[0.0]]), np.array([[0.0]])
    )
    free_indices, active_counts, free_counts = get_slot_diagnostics(free_data)

    npt.assert_array_equal(free_indices, [[0]])
    npt.assert_array_equal(active_counts, [0])
    npt.assert_array_equal(free_counts, [1])

    invalid_data = _make_data(
        np.zeros((1, 1, 0)), np.array([[1.0]]), np.array([[0.0]])
    )
    with pytest.raises(ValueError, match="^Invalid particle slot state[.]$"):
        get_slot_diagnostics(invalid_data)


def test_get_slot_diagnostics_accumulates_integer_masses_as_float64() -> None:
    """Avoid source-integer overflow when classifying active slots."""
    maximum = np.iinfo(np.int64).max
    data = ParticleData(
        masses=np.array([[[maximum, maximum]]], dtype=np.int64),
        concentration=np.array([[1.0]]),
        charge=np.array([[0.0]]),
        density=np.ones(2),
        volume=np.ones(1),
    )

    free_indices, active_counts, free_counts = get_slot_diagnostics(data)

    npt.assert_array_equal(free_indices, [[-1]])
    npt.assert_array_equal(active_counts, [1])
    npt.assert_array_equal(free_counts, [0])


def test_get_slot_diagnostics_rejects_finite_mass_lanes_with_infinite_total() -> (
    None
):
    """Reject finite mass lanes whose float64 aggregate overflows."""
    largest_finite = np.finfo(np.float64).max
    data = _make_data(
        np.array([[[largest_finite, largest_finite]]]),
        np.array([[1.0]]),
        np.array([[0.0]]),
    )
    sources = (data.masses, data.concentration, data.charge)
    snapshots = tuple(source.copy() for source in sources)

    with pytest.raises(ValueError, match="^Invalid particle slot state[.]$"):
        get_slot_diagnostics(data)

    for source, snapshot in zip(sources, snapshots, strict=True):
        npt.assert_array_equal(source, snapshot)


@pytest.mark.parametrize("field", ["concentration", "charge"])
def test_get_slot_diagnostics_rejects_mutated_field_shape_mismatches(
    field: str,
) -> None:
    """Fail closed when post-construction fields no longer match masses."""
    data = _make_data(
        np.array([[[1.0], [0.0]]]),
        np.array([[1.0, 0.0]]),
        np.array([[0.0, 0.0]]),
    )
    setattr(data, field, np.zeros((1, 1)))
    sources = (data.masses, data.concentration, data.charge)
    snapshots = tuple(source.copy() for source in sources)

    with pytest.raises(ValueError, match="^Invalid particle slot state[.]$"):
        get_slot_diagnostics(data)

    for source, snapshot in zip(sources, snapshots, strict=True):
        npt.assert_array_equal(source, snapshot)


def test_get_slot_diagnostics_returns_ordered_fixed_shape_sidecars() -> None:
    """Return ascending free prefixes, tails, and exact per-box diagnostics."""
    data = _make_data(
        np.array(
            [
                [[0.0, 0.0], [1.0, 2.0], [0.0, 0.0], [3.0, 0.0]],
                [[1.0, 0.0], [0.0, 0.0], [2.0, 1.0], [0.0, 0.0]],
            ]
        ),
        np.array([[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 2.0, 0.0]]),
        np.array([[0.0, -1.0, 0.0, 2.0], [1.0, 0.0, -3.0, 0.0]]),
    )

    free_indices, active_counts, free_counts = get_slot_diagnostics(data)

    assert free_indices.shape == (2, 4)
    assert active_counts.shape == (2,)
    assert free_counts.shape == (2,)
    npt.assert_array_equal(free_indices, [[0, 2, -1, -1], [1, 3, -1, -1]])
    npt.assert_array_equal(active_counts, [2, 2])
    npt.assert_array_equal(free_counts, [2, 2])


def test_get_slot_diagnostics_handles_full_and_zero_slot_boxes() -> None:
    """Support full boxes and valid fixed storage with no particle slots."""
    full_data = _make_data(
        np.array([[[1.0], [2.0]]]),
        np.array([[1.0, 1.0]]),
        np.array([[0.0, -1.0]]),
    )
    free_indices, active_counts, free_counts = get_slot_diagnostics(full_data)
    npt.assert_array_equal(free_indices, [[-1, -1]])
    npt.assert_array_equal(active_counts, [2])
    npt.assert_array_equal(free_counts, [0])

    zero_slot_data = _make_data(
        np.zeros((2, 0, 1)), np.zeros((2, 0)), np.zeros((2, 0))
    )
    free_indices, active_counts, free_counts = get_slot_diagnostics(
        zero_slot_data
    )
    assert free_indices.shape == (2, 0)
    npt.assert_array_equal(active_counts, [0, 0])
    npt.assert_array_equal(free_counts, [0, 0])


def test_get_slot_diagnostics_is_read_only_and_returns_fresh_arrays() -> None:
    """Preserve sources on success and failure and never reuse diagnostics."""
    data = _make_data(
        np.array([[[1.0], [0.0]]]),
        np.array([[1.0, 0.0]]),
        np.array([[-2.0, 0.0]]),
    )
    sources = (
        data.masses,
        data.concentration,
        data.charge,
        data.density,
        data.volume,
    )
    snapshots = tuple(source.copy() for source in sources)

    first = get_slot_diagnostics(data)
    second = get_slot_diagnostics(data)
    assert len({id(output) for output in (*first, *second)}) == 6
    first[0][0, 0] = 99
    first[1][0] = 99
    first[2][0] = 99
    third = get_slot_diagnostics(data)
    npt.assert_array_equal(third[0], [[1, -1]])
    npt.assert_array_equal(third[1], [1])
    npt.assert_array_equal(third[2], [1])

    for source, snapshot in zip(sources, snapshots, strict=True):
        npt.assert_array_equal(source, snapshot)
    assert sources[0] is data.masses
    assert sources[1] is data.concentration
    assert sources[2] is data.charge
    assert sources[3] is data.density
    assert sources[4] is data.volume

    data.concentration[0, 0] = 0.0
    error_snapshots = tuple(source.copy() for source in sources)
    with pytest.raises(ValueError, match="^Invalid particle slot state[.]$"):
        get_slot_diagnostics(data)
    for source, snapshot in zip(sources, error_snapshots, strict=True):
        npt.assert_array_equal(source, snapshot)


def test_get_slot_diagnostics_is_exported_through_particles_package() -> None:
    """Expose the concrete discovery callable through the particles package."""
    assert particles.get_slot_diagnostics is get_slot_diagnostics


def test_activate_slots_zero_prefix_is_exact_no_op() -> None:
    """Leave all storage untouched for valid zero-length request prefixes."""
    data = _make_data(
        np.array([[[1.0], [0.0]]]),
        np.array([[1.0, 0.0]]),
        np.array([[2.0, 0.0]]),
    )
    request_masses = np.array([[[3.0]]])
    request_concentration = np.array([[4.0]])
    request_charge = np.array([[5.0]])
    requested_counts = np.array([0], dtype=np.int64)
    fields = (
        data.masses,
        data.concentration,
        data.charge,
        data.density,
        data.volume,
        request_masses,
        request_concentration,
        request_charge,
        requested_counts,
    )
    snapshots = tuple(field.copy() for field in fields)

    result = activate_slots(
        data,
        request_masses,
        request_concentration,
        request_charge,
        requested_counts,
    )

    assert result.dtype == np.int32
    assert not np.shares_memory(result, requested_counts)
    npt.assert_array_equal(result, [0])
    for field, snapshot in zip(fields, snapshots, strict=True):
        npt.assert_array_equal(field, snapshot)


def test_activate_slots_maps_sparse_multi_box_requests_to_free_slots() -> None:
    """Map each request prefix rank to its box's ascending free slot."""
    data = _make_data(
        np.array(
            [
                [[0.0, 0.0], [1.0, 2.0], [0.0, 0.0]],
                [[3.0, 4.0], [0.0, 0.0], [0.0, 0.0]],
            ]
        ),
        np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
        np.array([[0.0, 2.0, 0.0], [-1.0, 0.0, 0.0]]),
    )
    request_masses = np.array(
        [
            [[10.0, 11.0], [12.0, 13.0]],
            [[20.0, 21.0], [22.0, 23.0]],
        ]
    )
    request_concentration = np.array([[2.0, 3.0], [4.0, 5.0]])
    request_charge = np.array([[6.0, 7.0], [8.0, 9.0]])
    requested_counts = np.array([1, 2], dtype=np.int64)

    result = activate_slots(
        data,
        request_masses,
        request_concentration,
        request_charge,
        requested_counts,
    )

    npt.assert_array_equal(result, [1, 2])
    npt.assert_array_equal(data.masses[0, 0], [10.0, 11.0])
    npt.assert_array_equal(data.masses[0, 2], [0.0, 0.0])
    npt.assert_array_equal(data.masses[1, 1:], [[20.0, 21.0], [22.0, 23.0]])
    npt.assert_array_equal(
        data.concentration, [[2.0, 1.0, 0.0], [1.0, 4.0, 5.0]]
    )
    npt.assert_array_equal(data.charge, [[6.0, 2.0, 0.0], [-1.0, 8.0, 9.0]])
    npt.assert_array_equal(request_masses[0, 1], [12.0, 13.0])


def test_activate_slots_fills_exact_free_capacity_and_produces_active_slots() -> (
    None
):
    """Fill every free slot and leave no free records behind."""
    data = _make_data(
        np.array([[[0.0], [2.0], [0.0]]]),
        np.array([[0.0, 1.0, 0.0]]),
        np.array([[0.0, -1.0, 0.0]]),
    )
    requested_counts = np.array([2], dtype=np.int64)

    result = activate_slots(
        data,
        np.array([[[3.0], [4.0]]]),
        np.array([[5.0, 6.0]]),
        np.array([[7.0, 8.0]]),
        requested_counts,
    )
    free_indices, active_counts, free_counts = get_slot_diagnostics(data)

    assert result.dtype == np.int32
    assert not np.shares_memory(result, requested_counts)
    npt.assert_array_equal(result, [2])
    npt.assert_array_equal(free_indices, [[-1, -1, -1]])
    npt.assert_array_equal(active_counts, [3])
    npt.assert_array_equal(free_counts, [0])


def test_activate_slots_accepts_empty_request_capacity_without_writes() -> None:
    """Accept zero request capacity without inspecting or writing storage."""
    data = _make_data(np.zeros((1, 2, 1)), np.zeros((1, 2)), np.zeros((1, 2)))
    request_masses = np.zeros((1, 0, 1), dtype=np.float64)
    request_concentration = np.zeros((1, 0), dtype=np.float64)
    request_charge = np.zeros((1, 0), dtype=np.float64)
    requested_counts = np.zeros(1, dtype=np.int64)
    fields = (
        data.masses,
        data.concentration,
        data.charge,
        data.density,
        data.volume,
        request_masses,
        request_concentration,
        request_charge,
        requested_counts,
    )
    identities = tuple(id(field) for field in fields)
    snapshots = tuple(field.copy() for field in fields)
    result = activate_slots(
        data,
        request_masses,
        request_concentration,
        request_charge,
        requested_counts,
    )

    npt.assert_array_equal(result, [0])
    assert result.dtype == np.int32
    assert tuple(id(field) for field in fields) == identities
    _assert_arrays_unchanged(fields, snapshots)


def test_activate_slots_accepts_zero_particle_capacity_without_writes() -> None:
    """Accept zero particle capacity when every requested prefix is empty."""
    data = _make_data(np.zeros((1, 0, 1)), np.zeros((1, 0)), np.zeros((1, 0)))
    request_masses = np.zeros((1, 0, 1), dtype=np.float64)
    request_concentration = np.zeros((1, 0), dtype=np.float64)
    request_charge = np.zeros((1, 0), dtype=np.float64)
    requested_counts = np.zeros(1, dtype=np.int64)
    fields = (
        data.masses,
        data.concentration,
        data.charge,
        data.density,
        data.volume,
        request_masses,
        request_concentration,
        request_charge,
        requested_counts,
    )
    identities = tuple(id(field) for field in fields)
    snapshots = tuple(field.copy() for field in fields)
    result = activate_slots(
        data,
        request_masses,
        request_concentration,
        request_charge,
        requested_counts,
    )

    npt.assert_array_equal(result, [0])
    assert result.dtype == np.int32
    assert tuple(id(field) for field in fields) == identities
    _assert_arrays_unchanged(fields, snapshots)


@pytest.mark.parametrize("invalid_data", [None, object()])
def test_activate_slots_rejects_invalid_data_before_field_access(
    invalid_data: object,
) -> None:
    """Reject non-container inputs with ValueError rather than field errors."""
    with pytest.raises(ValueError):
        activate_slots(
            invalid_data,  # type: ignore[arg-type]
            np.ones((1, 1, 1)),
            np.ones((1, 1)),
            np.zeros((1, 1)),
            np.ones(1, dtype=np.int64),
        )


def test_activate_slots_preflight_is_atomic_for_later_box_failure() -> None:
    """Reject a later invalid record before activating an earlier valid box."""
    data = _make_data(np.zeros((2, 1, 1)), np.zeros((2, 1)), np.zeros((2, 1)))
    request_masses = np.array([[[1.0]], [[-1.0]]])
    request_concentration = np.ones((2, 1))
    request_charge = np.zeros((2, 1))
    fields = (
        data.masses,
        data.concentration,
        data.charge,
        data.density,
        data.volume,
        request_masses,
        request_concentration,
        request_charge,
    )
    identities = tuple(id(field) for field in fields)
    snapshots = tuple(field.copy() for field in fields)

    with pytest.raises(ValueError):
        activate_slots(
            data,
            request_masses,
            request_concentration,
            request_charge,
            np.ones(2, dtype=np.int64),
        )

    assert tuple(id(field) for field in fields) == identities
    _assert_arrays_unchanged(fields, snapshots)


@pytest.mark.parametrize(
    ("first_field", "second_field"),
    [
        ("masses", "concentration"),
        ("masses", "charge"),
        ("concentration", "charge"),
    ],
)
def test_activate_slots_rejects_overlapping_mutable_destinations_atomically(
    first_field: str,
    second_field: str,
) -> None:
    """Reject every mutable destination alias before observable writes."""
    data = _make_data(np.zeros((1, 1, 1)), np.zeros((1, 1)), np.zeros((1, 1)))
    setattr(data, second_field, getattr(data, first_field).reshape(1, 1))
    request_masses = np.ones((1, 1, 1))
    request_concentration = np.ones((1, 1))
    request_charge = np.zeros((1, 1))
    requested_counts = np.ones(1, dtype=np.int64)
    fields = (
        data.masses,
        data.concentration,
        data.charge,
        data.density,
        data.volume,
        request_masses,
        request_concentration,
        request_charge,
        requested_counts,
    )
    identities = tuple(id(field) for field in fields)
    snapshots = tuple(field.copy() for field in fields)

    with pytest.raises(
        ValueError,
        match="^Particle destination fields must not share storage[.]$",
    ):
        activate_slots(
            data,
            request_masses,
            request_concentration,
            request_charge,
            requested_counts,
        )

    assert tuple(id(field) for field in fields) == identities
    _assert_arrays_unchanged(fields, snapshots)


@pytest.mark.parametrize(
    ("destination_field", "protected_field"),
    [
        ("masses", "density"),
        ("masses", "volume"),
        ("concentration", "density"),
        ("concentration", "volume"),
        ("charge", "density"),
        ("charge", "volume"),
    ],
)
def test_activate_slots_rejects_overlapping_protected_storage_atomically(
    destination_field: str,
    protected_field: str,
) -> None:
    """Reject destination views into density or volume before writes."""
    data = _make_data(np.zeros((1, 1, 1)), np.zeros((1, 1)), np.zeros((1, 1)))
    setattr(data, protected_field, getattr(data, destination_field).reshape(1))
    request_masses = np.ones((1, 1, 1))
    request_concentration = np.ones((1, 1))
    request_charge = np.zeros((1, 1))
    requested_counts = np.ones(1, dtype=np.int64)
    fields = (
        data.masses,
        data.concentration,
        data.charge,
        data.density,
        data.volume,
        request_masses,
        request_concentration,
        request_charge,
        requested_counts,
    )
    identities = tuple(id(field) for field in fields)
    snapshots = tuple(field.copy() for field in fields)

    with pytest.raises(
        ValueError,
        match="^Particle destination fields must not share storage[.]$",
    ):
        activate_slots(
            data,
            request_masses,
            request_concentration,
            request_charge,
            requested_counts,
        )

    assert tuple(id(field) for field in fields) == identities
    _assert_arrays_unchanged(fields, snapshots)


def test_activate_slots_ignores_invalid_request_tail() -> None:
    """Inspect only declared prefixes and leave ignored request tails untouched."""
    data = _make_data(np.zeros((1, 1, 1)), np.zeros((1, 1)), np.zeros((1, 1)))
    request_masses = np.array([[[2.0], [np.nan]]])
    request_concentration = np.array([[3.0, np.inf]])
    request_charge = np.array([[4.0, np.nan]])

    result = activate_slots(
        data,
        request_masses,
        request_concentration,
        request_charge,
        np.array([1], dtype=np.int64),
    )

    npt.assert_array_equal(result, [1])
    npt.assert_array_equal(data.masses, [[[2.0]]])
    npt.assert_array_equal(data.concentration, [[3.0]])
    npt.assert_array_equal(data.charge, [[4.0]])


def test_activate_slots_ignores_invalid_tail_for_zero_prefix() -> None:
    """Do not inspect invalid request storage when its declared prefix is empty."""
    data = _make_data(np.zeros((1, 1, 1)), np.zeros((1, 1)), np.zeros((1, 1)))
    request_masses = np.array([[[np.nan]]])
    request_concentration = np.array([[np.inf]])
    request_charge = np.array([[np.nan]])
    fields = (
        data.masses,
        data.concentration,
        data.charge,
        request_masses,
        request_concentration,
        request_charge,
    )
    snapshots = tuple(field.copy() for field in fields)

    result = activate_slots(
        data,
        request_masses,
        request_concentration,
        request_charge,
        np.array([0], dtype=np.int64),
    )

    npt.assert_array_equal(result, [0])
    for field, snapshot in zip(fields, snapshots, strict=True):
        npt.assert_array_equal(field, snapshot)


def test_activate_slots_rejects_aliases_and_preserves_destinations() -> None:
    """Reject request views that overlap any mutable destination field."""
    data = _make_data(np.zeros((1, 2, 1)), np.zeros((1, 2)), np.zeros((1, 2)))
    snapshots = tuple(
        field.copy() for field in (data.masses, data.concentration, data.charge)
    )

    with pytest.raises(ValueError):
        activate_slots(
            data,
            data.masses[:, :1],
            np.ones((1, 1)),
            np.zeros((1, 1)),
            np.array([1], dtype=np.int64),
        )

    for destination, snapshot in zip(
        (data.masses, data.concentration, data.charge), snapshots, strict=True
    ):
        npt.assert_array_equal(destination, snapshot)


@pytest.mark.parametrize(
    ("aliased_field", "destination_field"),
    [
        ("request_masses", "masses"),
        ("request_concentration", "charge"),
        ("request_charge", "concentration"),
    ],
)
def test_activate_slots_rejects_each_request_destination_alias(
    aliased_field: str,
    destination_field: str,
) -> None:
    """Reject aliases from every request field without changing storage."""
    data = _make_data(np.zeros((1, 1, 1)), np.zeros((1, 1)), np.zeros((1, 1)))
    requests = {
        "request_masses": np.ones((1, 1, 1)),
        "request_concentration": np.ones((1, 1)),
        "request_charge": np.zeros((1, 1)),
    }
    requests[aliased_field] = getattr(data, destination_field)
    destinations = (data.masses, data.concentration, data.charge)
    snapshots = tuple(destination.copy() for destination in destinations)

    with pytest.raises(
        ValueError,
        match="^Request fields must not share destination storage[.]$",
    ):
        activate_slots(
            data,
            requests["request_masses"],
            requests["request_concentration"],
            requests["request_charge"],
            np.array([1], dtype=np.int64),
        )

    for destination, snapshot in zip(destinations, snapshots, strict=True):
        npt.assert_array_equal(destination, snapshot)


@pytest.mark.parametrize(
    ("request_field", "destination_field"),
    [
        ("request_masses", "masses"),
        ("request_concentration", "concentration"),
        ("request_charge", "charge"),
    ],
)
def test_activate_slots_rejects_partially_overlapping_request_views(
    request_field: str,
    destination_field: str,
) -> None:
    """Reject views overlapping only one slot of each destination field."""
    data = _make_data(np.zeros((1, 2, 1)), np.zeros((1, 2)), np.zeros((1, 2)))
    requests = {
        "request_masses": np.ones((1, 1, 1)),
        "request_concentration": np.ones((1, 1)),
        "request_charge": np.zeros((1, 1)),
    }
    requests[request_field] = getattr(data, destination_field)[:, 1:]
    destinations = (data.masses, data.concentration, data.charge)
    snapshots = tuple(destination.copy() for destination in destinations)

    with pytest.raises(
        ValueError,
        match="^Request fields must not share destination storage[.]$",
    ):
        activate_slots(
            data,
            requests["request_masses"],
            requests["request_concentration"],
            requests["request_charge"],
            np.array([1], dtype=np.int64),
        )

    for destination, snapshot in zip(destinations, snapshots, strict=True):
        npt.assert_array_equal(destination, snapshot)


@pytest.mark.parametrize(
    "requested_counts",
    [
        np.array([-1], dtype=np.int64),
        np.array([2], dtype=np.int64),
        np.array([True]),
        np.array([1.0]),
        np.array([[1]], dtype=np.int64),
    ],
)
def test_activate_slots_rejects_invalid_requested_count_schema_atomically(
    requested_counts: np.ndarray,
) -> None:
    """Reject invalid prefix count schemas before mutating destinations."""
    data = _make_data(np.zeros((1, 1, 1)), np.zeros((1, 1)), np.zeros((1, 1)))
    destinations = (data.masses, data.concentration, data.charge)
    snapshots = tuple(destination.copy() for destination in destinations)

    with pytest.raises(ValueError):
        activate_slots(
            data,
            np.ones((1, 1, 1)),
            np.ones((1, 1)),
            np.zeros((1, 1)),
            requested_counts,
        )

    for destination, snapshot in zip(destinations, snapshots, strict=True):
        npt.assert_array_equal(destination, snapshot)


def test_activate_slots_rejects_invalid_request_schema_atomically() -> None:
    """Reject non-float64 and mismatched request schemas before any writes."""
    data = _make_data(np.zeros((1, 1, 1)), np.zeros((1, 1)), np.zeros((1, 1)))
    destinations = (data.masses, data.concentration, data.charge)
    snapshots = tuple(destination.copy() for destination in destinations)

    with pytest.raises(
        ValueError, match="^Request fields must be float64 NumPy arrays[.]$"
    ):
        activate_slots(
            data,
            np.ones((1, 1, 1), dtype=np.float32),
            np.ones((1, 1)),
            np.zeros((1, 1)),
            np.array([1], dtype=np.int64),
        )
    with pytest.raises(
        ValueError, match="^Request fields have invalid shapes[.]$"
    ):
        activate_slots(
            data,
            np.ones((1, 2, 1)),
            np.ones((1, 1)),
            np.zeros((1, 2)),
            np.array([1], dtype=np.int64),
        )

    for destination, snapshot in zip(destinations, snapshots, strict=True):
        npt.assert_array_equal(destination, snapshot)


@pytest.mark.parametrize(
    ("request_masses", "request_concentration", "request_charge"),
    [
        (None, np.ones((1, 1)), np.zeros((1, 1))),
        (np.ones((1, 1, 1)), [1.0], np.zeros((1, 1))),
        (
            np.ones((1, 1, 1)),
            np.ones((1, 1)),
            np.zeros((1, 1), dtype=np.float32),
        ),
        (np.ones((1, 1)), np.ones((1, 1)), np.zeros((1, 1))),
    ],
)
def test_activate_slots_rejects_non_array_or_wrong_rank_requests_atomically(
    request_masses: Any,
    request_concentration: Any,
    request_charge: Any,
) -> None:
    """Reject malformed request fields before a destination write."""
    data = _make_data(np.zeros((1, 1, 1)), np.zeros((1, 1)), np.zeros((1, 1)))
    destinations = (data.masses, data.concentration, data.charge)
    snapshots = tuple(destination.copy() for destination in destinations)

    with pytest.raises(ValueError):
        activate_slots(  # type: ignore[arg-type]
            data,
            request_masses,
            request_concentration,
            request_charge,
            np.array([1], dtype=np.int64),
        )

    for destination, snapshot in zip(destinations, snapshots, strict=True):
        npt.assert_array_equal(destination, snapshot)


@pytest.mark.parametrize(
    ("request_masses", "request_concentration", "request_charge"),
    [
        (np.array([[[-1.0]]]), np.ones((1, 1)), np.zeros((1, 1))),
        (np.array([[[np.nan]]]), np.ones((1, 1)), np.zeros((1, 1))),
        (np.array([[[np.inf]]]), np.ones((1, 1)), np.zeros((1, 1))),
        (np.zeros((1, 1, 1)), np.ones((1, 1)), np.zeros((1, 1))),
        (
            np.full((1, 1, 2), np.finfo(np.float64).max),
            np.ones((1, 1)),
            np.zeros((1, 1)),
        ),
        (np.ones((1, 1, 1)), np.array([[0.0]]), np.zeros((1, 1))),
        (np.ones((1, 1, 1)), np.array([[np.nan]]), np.zeros((1, 1))),
        (np.ones((1, 1, 1)), np.ones((1, 1)), np.array([[np.inf]])),
    ],
)
def test_activate_slots_rejects_invalid_selected_prefix_atomically(
    request_masses: np.ndarray,
    request_concentration: np.ndarray,
    request_charge: np.ndarray,
) -> None:
    """Reject every invalid selected record before mutating any slot."""
    n_species = request_masses.shape[-1]
    data = _make_data(
        np.zeros((1, 1, n_species)), np.zeros((1, 1)), np.zeros((1, 1))
    )
    destinations = (data.masses, data.concentration, data.charge)
    snapshots = tuple(destination.copy() for destination in destinations)

    with pytest.raises(
        ValueError, match="^Requested activation record is invalid[.]$"
    ):
        activate_slots(
            data,
            request_masses,
            request_concentration,
            request_charge,
            np.array([1], dtype=np.int64),
        )

    for destination, snapshot in zip(destinations, snapshots, strict=True):
        npt.assert_array_equal(destination, snapshot)


@pytest.mark.parametrize("field", ["masses", "concentration", "charge"])
def test_activate_slots_rejects_destination_dtype_and_shape_mutations(
    field: str,
) -> None:
    """Reject post-construction destination schema changes before mutation."""
    data = _make_data(np.zeros((1, 1, 1)), np.zeros((1, 1)), np.zeros((1, 1)))
    if field == "masses":
        data.masses = np.zeros((1, 1), dtype=np.float64)
    elif field == "concentration":
        data.concentration = np.zeros((1, 2), dtype=np.float64)
    else:
        data.charge = np.zeros((1, 2), dtype=np.float64)

    with pytest.raises(ValueError):
        activate_slots(
            data,
            np.ones((1, 1, 1)),
            np.ones((1, 1)),
            np.zeros((1, 1)),
            np.array([1], dtype=np.int64),
        )


def test_activate_slots_propagates_invalid_existing_slot_state() -> None:
    """Preserve the frozen diagnostic error for malformed existing slots."""
    data = _make_data(np.array([[[-1.0]]]), np.ones((1, 1)), np.zeros((1, 1)))

    with pytest.raises(ValueError, match="^Invalid particle slot state[.]$"):
        activate_slots(
            data,
            np.ones((1, 1, 1)),
            np.ones((1, 1)),
            np.zeros((1, 1)),
            np.array([1], dtype=np.int64),
        )


@pytest.mark.parametrize("field", ["masses", "concentration", "charge"])
def test_activate_slots_rejects_read_only_destination_schema(
    field: str,
) -> None:
    """Reject every read-only destination field before an activation attempt."""
    data = _make_data(np.zeros((1, 1, 1)), np.zeros((1, 1)), np.zeros((1, 1)))
    getattr(data, field).flags.writeable = False
    destinations = (data.masses, data.concentration, data.charge)
    snapshots = tuple(destination.copy() for destination in destinations)

    with pytest.raises(
        ValueError,
        match="^Particle destination fields must be writable float64 arrays[.]$",
    ):
        activate_slots(
            data,
            np.ones((1, 1, 1)),
            np.ones((1, 1)),
            np.zeros((1, 1)),
            np.array([1], dtype=np.int64),
        )

    for destination, snapshot in zip(destinations, snapshots, strict=True):
        npt.assert_array_equal(destination, snapshot)


@pytest.mark.parametrize("field", ["masses", "concentration", "charge"])
def test_activate_slots_rejects_each_non_float64_destination_atomically(
    field: str,
) -> None:
    """Reject every mutable destination field with a non-float64 dtype."""
    data = _make_data(np.zeros((1, 1, 1)), np.zeros((1, 1)), np.zeros((1, 1)))
    setattr(data, field, getattr(data, field).astype(np.float32))
    destinations = (data.masses, data.concentration, data.charge)
    snapshots = tuple(destination.copy() for destination in destinations)

    with pytest.raises(
        ValueError,
        match="^Particle destination fields must be writable float64 arrays[.]$",
    ):
        activate_slots(
            data,
            np.ones((1, 1, 1)),
            np.ones((1, 1)),
            np.zeros((1, 1)),
            np.array([1], dtype=np.int64),
        )

    for destination, snapshot in zip(destinations, snapshots, strict=True):
        npt.assert_array_equal(destination, snapshot)


def test_activate_slots_rejects_later_box_capacity_without_partial_write() -> (
    None
):
    """Preflight free capacity in every box before activating any request."""
    data = _make_data(
        np.array([[[0.0]], [[1.0]]]),
        np.array([[0.0], [1.0]]),
        np.zeros((2, 1)),
    )
    destinations = (data.masses, data.concentration, data.charge)
    snapshots = tuple(destination.copy() for destination in destinations)

    with pytest.raises(
        ValueError, match="^Requested activation exceeds free slot capacity[.]$"
    ):
        activate_slots(
            data,
            np.ones((2, 1, 1)),
            np.ones((2, 1)),
            np.zeros((2, 1)),
            np.array([1, 1], dtype=np.int64),
        )

    for destination, snapshot in zip(destinations, snapshots, strict=True):
        npt.assert_array_equal(destination, snapshot)


def test_activate_slots_rejects_positive_request_with_zero_particle_capacity() -> (
    None
):
    """Reject a nonempty request when fixed particle storage has no slots."""
    data = _make_data(np.zeros((1, 0, 1)), np.zeros((1, 0)), np.zeros((1, 0)))
    destinations = (data.masses, data.concentration, data.charge)
    snapshots = tuple(destination.copy() for destination in destinations)

    with pytest.raises(
        ValueError, match="^Requested activation exceeds free slot capacity[.]$"
    ):
        activate_slots(
            data,
            np.ones((1, 1, 1)),
            np.ones((1, 1)),
            np.zeros((1, 1)),
            np.array([1], dtype=np.int64),
        )

    for destination, snapshot in zip(destinations, snapshots, strict=True):
        npt.assert_array_equal(destination, snapshot)
