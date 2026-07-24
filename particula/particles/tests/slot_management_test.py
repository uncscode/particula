"""Tests for fixed-shape particle slot discovery."""

import numpy as np
import numpy.testing as npt
import pytest
from particula import particles
from particula.particles.particle_data import ParticleData
from particula.particles.slot_management import get_slot_diagnostics


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
