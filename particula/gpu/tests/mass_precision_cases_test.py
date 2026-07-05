"""Deterministic mass-precision baseline cases for particle data studies."""

from dataclasses import dataclass

import numpy as np
import numpy.testing as npt
import pytest

from particula.gpu.conversion import (
    from_warp_particle_data,
    to_warp_particle_data,
)
from particula.particles.particle_data import ParticleData


@dataclass(frozen=True)
class _MassPrecisionCase:
    """Container for a deterministic mass-precision baseline case."""

    case_name: str
    size_band: str
    radius_unit: str
    density_unit: str
    volume_fraction_unit: str
    target_radius_m: float
    density_kg_m3: np.ndarray
    volume_fractions: np.ndarray
    masses: np.ndarray
    concentration: np.ndarray
    charge: np.ndarray
    volume: np.ndarray


def _build_case(
    *,
    case_name: str,
    size_band: str,
    target_radius_m: float,
    density_kg_m3: list[float],
    volume_fractions: list[float],
    concentration_offset: float,
    charge_offset: float,
    volume_scale: float,
) -> _MassPrecisionCase:
    """Build one deterministic mass-precision case."""
    density = np.asarray(density_kg_m3, dtype=np.float64)
    fractions = np.asarray(volume_fractions, dtype=np.float64)
    total_volume = (4.0 / 3.0) * np.pi * target_radius_m**3
    species_volumes = total_volume * fractions
    per_species_mass = species_volumes * density

    box_scaling = np.array([[1.0], [1.25]], dtype=np.float64)
    particle_scaling = np.array([[1.0, 1.5, 2.0]], dtype=np.float64)
    scaling = box_scaling * particle_scaling
    masses = (
        scaling[..., np.newaxis] * per_species_mass[np.newaxis, np.newaxis, :]
    )

    concentration = (
        np.array(
            [
                [100.0, 125.0, 150.0],
                [175.0, 200.0, 225.0],
            ],
            dtype=np.float64,
        )
        + concentration_offset
    )
    charge = (
        np.array(
            [
                [0.0, 1.0, -1.0],
                [2.0, -2.0, 0.0],
            ],
            dtype=np.float64,
        )
        + charge_offset
    )
    volume = np.array([1.0, 1.4], dtype=np.float64) * volume_scale

    return _MassPrecisionCase(
        case_name=case_name,
        size_band=size_band,
        radius_unit="m",
        density_unit="kg/m^3",
        volume_fraction_unit="fraction",
        target_radius_m=target_radius_m,
        density_kg_m3=density,
        volume_fractions=fractions,
        masses=masses.astype(np.float64),
        concentration=concentration,
        charge=charge,
        volume=volume,
    )


def _build_mass_precision_cases() -> list[_MassPrecisionCase]:
    """Build the deterministic study cases in ascending size order."""
    return [
        _build_case(
            case_name="npf_cluster",
            size_band="new-particle-formation cluster",
            target_radius_m=1.5e-9,
            density_kg_m3=[1000.0],
            volume_fractions=[1.0],
            concentration_offset=0.0,
            charge_offset=0.0,
            volume_scale=1.0e-6,
        ),
        _build_case(
            case_name="five_to_ten_nm",
            size_band="5-10 nm particle",
            target_radius_m=7.0e-9,
            density_kg_m3=[1100.0],
            volume_fractions=[1.0],
            concentration_offset=25.0,
            charge_offset=0.0,
            volume_scale=1.5e-6,
        ),
        _build_case(
            case_name="accumulation_mode",
            size_band="accumulation mode",
            target_radius_m=1.5e-7,
            density_kg_m3=[1200.0, 1800.0],
            volume_fractions=[0.65, 0.35],
            concentration_offset=50.0,
            charge_offset=1.0,
            volume_scale=2.0e-6,
        ),
        _build_case(
            case_name="cloud_droplet",
            size_band="cloud droplet",
            target_radius_m=1.0e-5,
            density_kg_m3=[1000.0, 1770.0],
            volume_fractions=[0.92, 0.08],
            concentration_offset=75.0,
            charge_offset=-1.0,
            volume_scale=2.5e-6,
        ),
    ]


def _validate_case_shapes(case: _MassPrecisionCase) -> None:
    """Validate canonical array shapes for one deterministic case."""
    if case.masses.ndim != 3:
        raise ValueError(
            "baseline masses must be 3D (n_boxes, n_particles, n_species)"
        )

    n_boxes, n_particles, n_species = case.masses.shape

    if case.concentration.shape != (n_boxes, n_particles):
        raise ValueError(
            "baseline concentration shape must match "
            f"({n_boxes}, {n_particles})"
        )
    if case.charge.shape != (n_boxes, n_particles):
        raise ValueError(
            f"baseline charge shape must match ({n_boxes}, {n_particles})"
        )
    if case.density_kg_m3.shape != (n_species,):
        raise ValueError(f"baseline density shape must match ({n_species},)")
    if case.volume_fractions.shape != (n_species,):
        raise ValueError(
            f"baseline volume-fraction shape must match ({n_species},)"
        )
    if case.volume.shape != (n_boxes,):
        raise ValueError(f"baseline volume shape must match ({n_boxes},)")


def _validate_case_values(case: _MassPrecisionCase) -> None:
    """Validate deterministic value, dtype, and physical invariants."""
    _validate_mass_arrays(case)
    _validate_support_arrays(case)


def _validate_mass_arrays(case: _MassPrecisionCase) -> None:
    """Validate mass-array-specific baseline invariants."""
    if case.masses.dtype != np.float64:
        raise ValueError("baseline masses must use np.float64")
    if not np.all(np.isfinite(case.masses)):
        raise ValueError("baseline masses must be finite")
    if np.any(case.masses < 0.0):
        raise ValueError("baseline masses must be nonnegative")


def _validate_support_arrays(case: _MassPrecisionCase) -> None:
    """Validate non-mass baseline arrays and metadata values."""
    _validate_concentration_and_charge(case)
    _validate_density_fractions_and_volume(case)


def _validate_concentration_and_charge(case: _MassPrecisionCase) -> None:
    """Validate concentration and charge arrays."""
    if not np.all(np.isfinite(case.concentration)):
        raise ValueError("baseline concentration must be finite")
    if np.any(case.concentration < 0.0):
        raise ValueError("baseline concentration must be nonnegative")
    if not np.all(np.isfinite(case.charge)):
        raise ValueError("baseline charge must be finite")


def _validate_density_fractions_and_volume(case: _MassPrecisionCase) -> None:
    """Validate density, volume-fraction, and volume metadata arrays."""
    if not np.all(np.isfinite(case.density_kg_m3)):
        raise ValueError("baseline density must be finite")
    if np.any(case.density_kg_m3 <= 0.0):
        raise ValueError("baseline density must be positive")
    if not np.all(np.isfinite(case.volume_fractions)):
        raise ValueError("baseline volume fractions must be finite")
    if np.any(case.volume_fractions < 0.0):
        raise ValueError("baseline volume fractions must be nonnegative")
    if not np.isclose(np.sum(case.volume_fractions), 1.0):
        raise ValueError("baseline volume fractions must sum to 1.0")
    if not np.all(np.isfinite(case.volume)):
        raise ValueError("baseline volume must be finite")
    if np.any(case.volume <= 0.0):
        raise ValueError("baseline volume must be positive")


def _validate_case(case: _MassPrecisionCase) -> None:
    """Validate deterministic baseline case shape and value invariants."""
    _validate_case_shapes(case)
    _validate_case_values(case)


def _copy_case(
    case: _MassPrecisionCase,
    **updates: np.ndarray,
) -> _MassPrecisionCase:
    """Create a copied case with selected array fields replaced."""
    return _MassPrecisionCase(
        case_name=case.case_name,
        size_band=case.size_band,
        radius_unit=case.radius_unit,
        density_unit=case.density_unit,
        volume_fraction_unit=case.volume_fraction_unit,
        target_radius_m=case.target_radius_m,
        density_kg_m3=updates.get("density_kg_m3", case.density_kg_m3.copy()),
        volume_fractions=updates.get(
            "volume_fractions", case.volume_fractions.copy()
        ),
        masses=updates.get("masses", case.masses.copy()),
        concentration=updates.get("concentration", case.concentration.copy()),
        charge=updates.get("charge", case.charge.copy()),
        volume=updates.get("volume", case.volume.copy()),
    )


def _expected_case_radii(case: _MassPrecisionCase) -> np.ndarray:
    """Reconstruct expected radii from the authored case construction rule."""
    scaled_total_volume = np.sum(
        case.masses / case.density_kg_m3[np.newaxis, np.newaxis, :],
        axis=-1,
    )
    return np.cbrt(scaled_total_volume * 3.0 / (4.0 * np.pi))


@pytest.fixture(name="mass_precision_cases")
def fixture_mass_precision_cases() -> list[_MassPrecisionCase]:
    """Provide deterministic mass-precision study cases."""
    return _build_mass_precision_cases()


def test_mass_precision_cases_cover_expected_study_bands(
    mass_precision_cases: list[_MassPrecisionCase],
) -> None:
    """Cases span the expected NPF-to-droplet study range."""
    assert [case.case_name for case in mass_precision_cases] == [
        "npf_cluster",
        "five_to_ten_nm",
        "accumulation_mode",
        "cloud_droplet",
    ]
    assert [case.size_band for case in mass_precision_cases] == [
        "new-particle-formation cluster",
        "5-10 nm particle",
        "accumulation mode",
        "cloud droplet",
    ]
    assert [case.radius_unit for case in mass_precision_cases] == ["m"] * 4
    assert [case.density_unit for case in mass_precision_cases] == [
        "kg/m^3"
    ] * 4
    assert [case.volume_fraction_unit for case in mass_precision_cases] == [
        "fraction"
    ] * 4


def test_mass_precision_case_builders_are_deterministic() -> None:
    """Rebuilding the full case set produces identical arrays and metadata."""
    first_build = _build_mass_precision_cases()
    second_build = _build_mass_precision_cases()

    assert len(first_build) == len(second_build)
    for first_case, second_case in zip(first_build, second_build, strict=True):
        assert first_case.case_name == second_case.case_name
        assert first_case.size_band == second_case.size_band
        assert first_case.radius_unit == second_case.radius_unit
        assert first_case.density_unit == second_case.density_unit
        assert (
            first_case.volume_fraction_unit == second_case.volume_fraction_unit
        )
        assert first_case.target_radius_m == second_case.target_radius_m
        npt.assert_array_equal(
            first_case.density_kg_m3,
            second_case.density_kg_m3,
        )
        npt.assert_array_equal(
            first_case.volume_fractions,
            second_case.volume_fractions,
        )
        npt.assert_array_equal(first_case.masses, second_case.masses)
        npt.assert_array_equal(
            first_case.concentration,
            second_case.concentration,
        )
        npt.assert_array_equal(first_case.charge, second_case.charge)
        npt.assert_array_equal(first_case.volume, second_case.volume)


@pytest.mark.parametrize(
    "case",
    _build_mass_precision_cases(),
    ids=lambda case: case.case_name,
)
def test_mass_precision_case_shapes_and_values(
    case: _MassPrecisionCase,
) -> None:
    """Each deterministic case preserves the canonical particle-data schema."""
    _validate_case(case)

    n_boxes, n_particles, n_species = case.masses.shape
    assert case.masses.shape == (n_boxes, n_particles, n_species)
    assert case.concentration.shape == (n_boxes, n_particles)
    assert case.charge.shape == (n_boxes, n_particles)
    assert case.density_kg_m3.shape == (n_species,)
    assert case.volume_fractions.shape == (n_species,)
    assert case.volume.shape == (n_boxes,)
    assert case.masses.dtype == np.float64
    assert np.all(np.isfinite(case.masses))
    assert np.all(case.masses >= 0.0)
    assert np.all(np.isfinite(case.concentration))
    assert np.all(case.concentration >= 0.0)
    assert np.all(np.isfinite(case.charge))
    assert np.all(np.isfinite(case.density_kg_m3))
    assert np.all(case.density_kg_m3 > 0.0)
    assert np.all(np.isfinite(case.volume_fractions))
    assert np.all(case.volume_fractions >= 0.0)
    assert np.isclose(np.sum(case.volume_fractions), 1.0)
    assert np.all(np.isfinite(case.volume))
    assert np.all(case.volume > 0.0)


@pytest.mark.parametrize(
    "case",
    _build_mass_precision_cases(),
    ids=lambda case: case.case_name,
)
def test_mass_precision_cases_construct_particle_data(
    case: _MassPrecisionCase,
) -> None:
    """Each case can initialize ParticleData without shape changes."""
    particle_data = ParticleData(
        masses=case.masses,
        concentration=case.concentration,
        charge=case.charge,
        density=case.density_kg_m3,
        volume=case.volume,
    )

    assert particle_data.masses.shape == case.masses.shape
    assert particle_data.concentration.shape == case.concentration.shape
    assert particle_data.charge.shape == case.charge.shape
    assert particle_data.density.shape == case.density_kg_m3.shape
    assert particle_data.volume.shape == case.volume.shape


def test_mass_precision_case_radii_match_expected_scale_order(
    mass_precision_cases: list[_MassPrecisionCase],
) -> None:
    """Derived radii preserve both exact targets and ascending scale bands."""
    mean_radii = []

    for case in mass_precision_cases:
        particle_data = ParticleData(
            masses=case.masses,
            concentration=case.concentration,
            charge=case.charge,
            density=case.density_kg_m3,
            volume=case.volume,
        )
        derived_radii = particle_data.radii
        assert np.all(np.isfinite(derived_radii))
        assert np.all(derived_radii >= 0.0)

        expected_radii = _expected_case_radii(case)
        npt.assert_allclose(derived_radii, expected_radii, rtol=1e-12)

        mean_radii.append(np.mean(derived_radii))

    assert mean_radii == sorted(mean_radii)


def test_mass_precision_case_rejects_malformed_particle_data_shape() -> None:
    """Malformed shapes still trigger ParticleData shape validation errors."""
    case = _build_mass_precision_cases()[0]

    with pytest.raises(
        ValueError,
        match=r"masses must be 3D \(n_boxes, n_particles, n_species\)",
    ):
        ParticleData(
            masses=case.masses[0],
            concentration=case.concentration,
            charge=case.charge,
            density=case.density_kg_m3,
            volume=case.volume,
        )

    with pytest.raises(
        ValueError,
        match=r"concentration shape \(2, 2\) doesn't match expected \(2, 3\)",
    ):
        ParticleData(
            masses=case.masses,
            concentration=case.concentration[:, :-1],
            charge=case.charge,
            density=case.density_kg_m3,
            volume=case.volume,
        )


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda case: _copy_case(
                case,
                masses=np.where(
                    np.indices(case.masses.shape).sum(axis=0) == 0,
                    -1.0,
                    case.masses,
                ),
            ),
            "baseline masses must be nonnegative",
        ),
        (
            lambda case: _copy_case(
                case,
                masses=np.where(
                    np.indices(case.masses.shape).sum(axis=0) == 0,
                    np.nan,
                    case.masses,
                ),
            ),
            "baseline masses must be finite",
        ),
    ],
)
def test_mass_precision_case_validator_rejects_invalid_masses(
    mutator,
    message: str,
) -> None:
    """Test-side baseline validation rejects malformed mass arrays."""
    invalid_case = mutator(_build_mass_precision_cases()[0])

    with pytest.raises(ValueError, match=message):
        _validate_case(invalid_case)


@pytest.mark.parametrize(
    "case",
    [
        _build_mass_precision_cases()[0],
        _build_mass_precision_cases()[2],
        _build_mass_precision_cases()[3],
    ],
    ids=lambda case: case.case_name,
)
def test_mass_precision_baseline_policy_preserves_fp64_on_cpu_and_warp(
    case: _MassPrecisionCase,
) -> None:
    """Baseline policy remains absolute per-species fp64 on CPU and Warp."""
    wp = pytest.importorskip("warp")

    particle_data = ParticleData(
        masses=case.masses,
        concentration=case.concentration,
        charge=case.charge,
        density=case.density_kg_m3,
        volume=case.volume,
    )
    assert case.masses.dtype == np.float64
    assert particle_data.masses.dtype == np.float64
    assert particle_data.concentration.dtype == np.float64
    assert particle_data.charge.dtype == np.float64
    assert particle_data.density.dtype == np.float64
    assert particle_data.volume.dtype == np.float64

    gpu_data = to_warp_particle_data(particle_data, device="cpu")
    assert gpu_data.masses.dtype == wp.float64
    assert gpu_data.concentration.dtype == wp.float64
    assert gpu_data.charge.dtype == wp.float64
    assert gpu_data.density.dtype == wp.float64
    assert gpu_data.volume.dtype == wp.float64


@pytest.mark.parametrize(
    "case",
    [
        _build_mass_precision_cases()[0],
        _build_mass_precision_cases()[2],
        _build_mass_precision_cases()[3],
    ],
    ids=lambda case: case.case_name,
)
def test_mass_precision_case_warp_round_trip_preserves_values_exactly(
    case: _MassPrecisionCase,
) -> None:
    """Warp CPU-device round trip preserves deterministic baseline values."""
    pytest.importorskip("warp")
    particle_data = ParticleData(
        masses=case.masses,
        concentration=case.concentration,
        charge=case.charge,
        density=case.density_kg_m3,
        volume=case.volume,
    )

    restored = from_warp_particle_data(
        to_warp_particle_data(particle_data, device="cpu")
    )

    npt.assert_array_equal(restored.masses, particle_data.masses)
    npt.assert_array_equal(
        restored.concentration,
        particle_data.concentration,
    )
    npt.assert_array_equal(restored.charge, particle_data.charge)
    npt.assert_array_equal(restored.density, particle_data.density)
    npt.assert_array_equal(restored.volume, particle_data.volume)
    assert restored.masses.dtype == np.float64
    assert restored.concentration.dtype == np.float64
    assert restored.charge.dtype == np.float64
    assert restored.density.dtype == np.float64
    assert restored.volume.dtype == np.float64
