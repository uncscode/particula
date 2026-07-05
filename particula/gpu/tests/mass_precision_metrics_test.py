"""Study-only fidelity checks for candidate mass-precision projections."""

from typing import Any

import numpy as np
import numpy.testing as npt
import pytest

from particula.gpu.conversion import to_warp_particle_data
from particula.gpu.tests.mass_precision_cases_test import (
    _build_mass_precision_cases,
    _MassPrecisionCase,
)
from particula.particles.particle_data_builder import ParticleDataBuilder

_SUPPORTED_CANDIDATES = (
    "fp32_absolute_mass",
    "mixed_precision_mass_plus_density",
    "fp32_total_mass_fp32_mass_fraction",
)
_UNSUPPORTED_CANDIDATES = (
    "candidate_requiring_runtime_schema_expansion",
)
_MASS_RTOL = 5e-7
_MASS_ATOL = 1e-30
_RADIUS_RTOL = 2e-7
_RADIUS_ATOL = 1e-16
_MASS_PRECISION_CASES = _build_mass_precision_cases()
_CASE_CANDIDATE_PARAMS = [
    (case, candidate_id)
    for case in _MASS_PRECISION_CASES
    for candidate_id in _SUPPORTED_CANDIDATES
]
_CASE_CANDIDATE_IDS = [
    f"{case.case_name}-{candidate_id}"
    for case, candidate_id in _CASE_CANDIDATE_PARAMS
]


def _candidate_ids() -> list[str]:
    """Return the supported executable candidate ids."""
    return list(_SUPPORTED_CANDIDATES)


def _reconstruct_radii(
    masses: np.ndarray,
    density: np.ndarray,
) -> np.ndarray:
    """Reconstruct radii using the production particle-data relationship."""
    return np.cbrt(
        3.0 * np.sum(masses / density[np.newaxis, np.newaxis, :], axis=-1)
        / (4.0 * np.pi)
    )


def _project_candidate(
    case: _MassPrecisionCase,
    candidate_id: str,
) -> dict[str, Any]:
    """Project one study case into a candidate representation."""
    if candidate_id == "fp32_absolute_mass":
        projected_masses = case.masses.astype(np.float32)
        return {
            "candidate_id": candidate_id,
            "projected": {"masses": projected_masses},
            "reconstructed_masses": projected_masses.astype(np.float64),
        }

    if candidate_id == "mixed_precision_mass_plus_density":
        projected = {
            "masses": case.masses.astype(np.float32),
            "concentration": case.concentration.astype(np.float32),
            "charge": case.charge.astype(np.float32),
            "volume": case.volume.astype(np.float32),
            "density": case.density_kg_m3.astype(np.float64),
        }
        return {
            "candidate_id": candidate_id,
            "projected": projected,
            "reconstructed_masses": projected["masses"].astype(np.float64),
        }

    if candidate_id == "fp32_total_mass_fp32_mass_fraction":
        total_mass = np.sum(case.masses, axis=-1, dtype=np.float64)
        projected_total_mass = total_mass.astype(np.float32)
        projected_mass_fractions = np.divide(
            case.masses,
            total_mass[..., np.newaxis],
            where=total_mass[..., np.newaxis] > 0.0,
            out=np.zeros_like(case.masses),
        ).astype(np.float32)
        reconstructed_masses = (
            projected_total_mass.astype(np.float64)[..., np.newaxis]
            * projected_mass_fractions.astype(np.float64)
        )
        return {
            "candidate_id": candidate_id,
            "projected": {
                "total_mass": projected_total_mass,
                "mass_fractions": projected_mass_fractions,
            },
            "reconstructed_masses": reconstructed_masses,
        }

    raise ValueError(f"Unsupported candidate id: {candidate_id}")


@pytest.mark.parametrize(
    ("case", "candidate_id"),
    _CASE_CANDIDATE_PARAMS,
    ids=_CASE_CANDIDATE_IDS,
)
def test_candidate_reconstructed_masses_match_fp64_baseline(
    case: _MassPrecisionCase,
    candidate_id: str,
) -> None:
    """Each candidate reconstructs masses close to the baseline."""
    candidate = _project_candidate(case, candidate_id)
    reconstructed_masses = candidate["reconstructed_masses"]

    npt.assert_allclose(
        reconstructed_masses,
        case.masses,
        rtol=_MASS_RTOL,
        atol=_MASS_ATOL,
        err_msg=(
            "candidate reconstruction exceeded mass tolerance for "
            f"case={case.case_name}, candidate={candidate_id}"
        ),
    )


@pytest.mark.parametrize(
    ("case", "candidate_id"),
    _CASE_CANDIDATE_PARAMS,
    ids=_CASE_CANDIDATE_IDS,
)
def test_candidate_reconstructed_radii_match_fp64_baseline(
    case: _MassPrecisionCase,
    candidate_id: str,
) -> None:
    """Each candidate reconstructs radii close to the baseline."""
    candidate = _project_candidate(case, candidate_id)
    reconstructed_masses = candidate["reconstructed_masses"]
    reconstructed_radii = _reconstruct_radii(
        reconstructed_masses,
        case.density_kg_m3,
    )
    baseline_radii = _reconstruct_radii(case.masses, case.density_kg_m3)

    npt.assert_allclose(
        reconstructed_radii,
        baseline_radii,
        rtol=_RADIUS_RTOL,
        atol=_RADIUS_ATOL,
        err_msg=(
            "candidate reconstruction exceeded radius tolerance for "
            f"case={case.case_name}, candidate={candidate_id}"
        ),
    )


def test_projection_helper_invalid_candidate_raises_value_error() -> None:
    """Invalid candidate ids fail deterministically with a clear error."""
    case = _MASS_PRECISION_CASES[0]

    with pytest.raises(
        ValueError,
        match="Unsupported candidate id: definitely_not_supported",
    ):
        _project_candidate(case, "definitely_not_supported")


def test_candidate_ids_return_supported_executable_candidates() -> None:
    """Supported executable candidates are returned in stable order."""
    assert _candidate_ids() == list(_SUPPORTED_CANDIDATES)


def test_total_mass_fraction_candidate_handles_zero_total_mass_without_warnings(
) -> None:
    """Zero-total-mass particles reconstruct zeros cleanly."""
    case = _MASS_PRECISION_CASES[2]
    zero_mass_case = _MassPrecisionCase(
        case_name=f"{case.case_name}_zero_total_mass",
        size_band=case.size_band,
        radius_unit=case.radius_unit,
        density_unit=case.density_unit,
        volume_fraction_unit=case.volume_fraction_unit,
        target_radius_m=case.target_radius_m,
        density_kg_m3=case.density_kg_m3.copy(),
        volume_fractions=case.volume_fractions.copy(),
        masses=np.zeros_like(case.masses),
        concentration=case.concentration.copy(),
        charge=case.charge.copy(),
        volume=case.volume.copy(),
    )

    candidate = _project_candidate(
        zero_mass_case,
        "fp32_total_mass_fp32_mass_fraction",
    )

    npt.assert_array_equal(
        candidate["projected"]["total_mass"],
        np.zeros_like(candidate["projected"]["total_mass"]),
    )
    npt.assert_array_equal(
        candidate["projected"]["mass_fractions"],
        np.zeros_like(candidate["projected"]["mass_fractions"]),
    )
    npt.assert_array_equal(
        candidate["reconstructed_masses"],
        np.zeros_like(candidate["reconstructed_masses"]),
    )


def test_unsupported_candidates_remain_doc_only() -> None:
    """Unsupported candidates remain documented but not executable."""
    assert _UNSUPPORTED_CANDIDATES == (
        "candidate_requiring_runtime_schema_expansion",
    )
    assert not set(_UNSUPPORTED_CANDIDATES) & set(_SUPPORTED_CANDIDATES)


def test_study_helpers_leave_cpu_defaults_at_float64() -> None:
    """The study stays isolated from production CPU dtype defaults."""
    case = _MASS_PRECISION_CASES[2]
    data = (
        ParticleDataBuilder()
        .set_masses(case.masses, units="kg")
        .set_density(case.density_kg_m3, units="kg/m^3")
        .set_concentration(case.concentration, units="1/m^3")
        .set_charge(case.charge)
        .set_volume(case.volume, units="m^3")
        .build()
    )

    assert data.masses.dtype == np.float64
    assert data.concentration.dtype == np.float64
    assert data.charge.dtype == np.float64
    assert data.density.dtype == np.float64
    assert data.volume.dtype == np.float64


def test_study_helpers_leave_warp_defaults_at_float64() -> None:
    """The study stays isolated from production Warp dtype defaults."""
    wp = pytest.importorskip("warp")
    case = _MASS_PRECISION_CASES[2]
    data = (
        ParticleDataBuilder()
        .set_masses(case.masses, units="kg")
        .set_density(case.density_kg_m3, units="kg/m^3")
        .set_concentration(case.concentration, units="1/m^3")
        .set_charge(case.charge)
        .set_volume(case.volume, units="m^3")
        .build()
    )

    gpu_data = to_warp_particle_data(data, device="cpu")

    assert gpu_data.masses.dtype == wp.float64
    assert gpu_data.concentration.dtype == wp.float64
    assert gpu_data.charge.dtype == wp.float64
    assert gpu_data.density.dtype == wp.float64
    assert gpu_data.volume.dtype == wp.float64
