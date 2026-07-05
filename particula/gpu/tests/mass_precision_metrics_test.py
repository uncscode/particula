"""Study-only fidelity checks for candidate mass-precision projections."""

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.testing as npt
import pytest

from particula.dynamics.condensation.mass_transfer import get_mass_transfer
from particula.gpu.conversion import to_warp_particle_data
from particula.gpu.tests.mass_precision_study_support import (
    _build_mass_precision_cases,
    _MassPrecisionCase,
    _project_candidate,
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
_MIXED_SCALE_MASS_RTOL = 6e-7
_MIXED_SCALE_RADIUS_RTOL = 2.5e-7
_AGGREGATE_DELTA_RTOL = 5e-7
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


@dataclass(frozen=True)
class _CandidateStudyResult:
    """Cached reconstruction metrics for one study case and candidate."""

    case: _MassPrecisionCase
    candidate_id: str
    projected: dict[str, np.ndarray]
    reconstructed_masses: np.ndarray
    baseline_radii: np.ndarray
    reconstructed_radii: np.ndarray
    absolute_mass_error: np.ndarray
    relative_mass_error: np.ndarray
    absolute_radius_error: np.ndarray
    relative_radius_error: np.ndarray
    aggregate_mass_delta: np.ndarray


@dataclass(frozen=True)
class _MassTransferReferenceResult:
    """Cached CPU-reference mass-transfer comparison for one candidate."""

    case: _MassPrecisionCase
    candidate_id: str
    baseline_mass_transfer: np.ndarray
    candidate_mass_transfer: np.ndarray
    per_particle_delta: np.ndarray
    aggregate_species_delta: np.ndarray


@dataclass(frozen=True)
class _ClampMetrics:
    """Raw, clamped, and aggregate metrics for clamp accounting."""

    raw_updated_mass: np.ndarray
    post_clamp_mass: np.ndarray
    clamp_delta: np.ndarray
    clamp_frequency: int
    aggregate_clamp_delta: np.ndarray


def _assert_projected_schema(
    projected: dict[str, np.ndarray],
    case: _MassPrecisionCase,
    candidate_id: str,
) -> None:
    """Assert each candidate publishes the documented projected schema."""
    if candidate_id == "fp32_absolute_mass":
        assert tuple(projected) == ("masses",)
        assert projected["masses"].shape == case.masses.shape
        assert projected["masses"].dtype == np.float32
        return

    if candidate_id == "mixed_precision_mass_plus_density":
        assert tuple(projected) == (
            "masses",
            "concentration",
            "charge",
            "volume",
            "density",
        )
        assert projected["masses"].shape == case.masses.shape
        assert projected["masses"].dtype == np.float32
        assert projected["concentration"].shape == case.concentration.shape
        assert projected["concentration"].dtype == np.float32
        assert projected["charge"].shape == case.charge.shape
        assert projected["charge"].dtype == np.float32
        assert projected["volume"].shape == case.volume.shape
        assert projected["volume"].dtype == np.float32
        assert projected["density"].shape == case.density_kg_m3.shape
        assert projected["density"].dtype == np.float64
        return

    if candidate_id == "fp32_total_mass_fp32_mass_fraction":
        assert tuple(projected) == ("total_mass", "mass_fractions")
        assert projected["total_mass"].shape == case.masses.shape[:-1]
        assert projected["total_mass"].dtype == np.float32
        assert projected["mass_fractions"].shape == case.masses.shape
        assert projected["mass_fractions"].dtype == np.float32
        return

    raise AssertionError(
        f"candidate schema assertion missing for candidate={candidate_id}"
    )


def _candidate_ids() -> list[str]:
    """Return the supported executable candidate ids."""
    return list(_SUPPORTED_CANDIDATES)


def _safe_relative_error(
    observed: np.ndarray,
    baseline: np.ndarray,
) -> np.ndarray:
    """Return warning-clean relative error with deterministic zero handling."""
    absolute_error = np.abs(observed - baseline)
    relative_error = np.zeros_like(absolute_error, dtype=np.float64)
    np.divide(
        absolute_error,
        np.abs(baseline),
        out=relative_error,
        where=np.abs(baseline) > 0.0,
    )
    return relative_error


def _reconstruct_radii(
    masses: np.ndarray,
    density: np.ndarray,
) -> np.ndarray:
    """Reconstruct radii using the production particle-data relationship."""
    total_volume = np.sum(
        np.divide(
            masses,
            density[np.newaxis, np.newaxis, :],
            out=np.zeros_like(masses),
            where=density[np.newaxis, np.newaxis, :] > 0.0,
        ),
        axis=-1,
    )
    return np.cbrt(np.maximum(total_volume, 0.0) * 3.0 / (4.0 * np.pi))


def _candidate_projection_items() -> list[tuple[_MassPrecisionCase, str]]:
    """Return the deterministic case-candidate matrix."""
    return list(_CASE_CANDIDATE_PARAMS)


def _build_mixed_scale_case() -> _MassPrecisionCase:
    """Build a deterministic case with nanometer and droplet masses together."""
    density = np.array([1000.0], dtype=np.float64)
    radii = np.array(
        [
            [1.5e-9, 1.0e-5, 1.2e-5],
            [2.0e-9, 8.0e-6, 1.5e-5],
        ],
        dtype=np.float64,
    )
    total_volume = (4.0 / 3.0) * np.pi * radii**3
    masses = (total_volume * density[0])[..., np.newaxis]
    return _MassPrecisionCase(
        case_name="mixed_scale_small_and_droplet",
        size_band="mixed nanometer and droplet scale",
        radius_unit="m",
        density_unit="kg/m^3",
        volume_fraction_unit="fraction",
        target_radius_m=1.5e-9,
        density_kg_m3=density,
        volume_fractions=np.array([1.0], dtype=np.float64),
        masses=masses,
        concentration=np.array(
            [[150.0, 1.0, 0.8], [200.0, 1.2, 0.6]],
            dtype=np.float64,
        ),
        charge=np.array([[0.0, 1.0, -1.0], [0.0, -1.0, 2.0]], dtype=np.float64),
        volume=np.array([2.0e-6, 2.5e-6], dtype=np.float64),
    )


def _build_candidate_study_result(
    case: _MassPrecisionCase,
    candidate_id: str,
) -> _CandidateStudyResult:
    """Build cached reconstruction metrics for one case-candidate pair."""
    candidate = _project_candidate(case, candidate_id)
    reconstructed_masses = candidate["reconstructed_masses"]
    baseline_radii = _reconstruct_radii(case.masses, case.density_kg_m3)
    reconstructed_radii = _reconstruct_radii(
        reconstructed_masses,
        case.density_kg_m3,
    )
    absolute_mass_error = np.abs(reconstructed_masses - case.masses)
    absolute_radius_error = np.abs(reconstructed_radii - baseline_radii)
    return _CandidateStudyResult(
        case=case,
        candidate_id=candidate_id,
        projected=candidate["projected"],
        reconstructed_masses=reconstructed_masses,
        baseline_radii=baseline_radii,
        reconstructed_radii=reconstructed_radii,
        absolute_mass_error=absolute_mass_error,
        relative_mass_error=_safe_relative_error(
            reconstructed_masses,
            case.masses,
        ),
        absolute_radius_error=absolute_radius_error,
        relative_radius_error=_safe_relative_error(
            reconstructed_radii,
            baseline_radii,
        ),
        aggregate_mass_delta=np.sum(
            reconstructed_masses - case.masses,
            axis=(0, 1),
            dtype=np.float64,
        ),
    )


def _build_mass_transfer_reference_result(
    case: _MassPrecisionCase,
    candidate_id: str,
    candidate: dict[str, Any] | None = None,
) -> _MassTransferReferenceResult:
    """Build cached CPU-reference mass-transfer deltas for one candidate."""
    if candidate is None:
        candidate = _project_candidate(case, candidate_id)
    candidate_projected = candidate["projected"]
    candidate_reconstructed_masses = candidate["reconstructed_masses"]
    candidate_concentration = candidate_projected.get("concentration")
    if candidate_concentration is None:
        candidate_concentration = case.concentration
    candidate_concentration = candidate_concentration.astype(np.float64)
    time_step = 0.75
    mass_rate_scale = 2.5e-2
    baseline_mass_rate = (
        case.masses
        * mass_rate_scale
        / time_step
        / case.concentration[..., np.newaxis]
    )
    candidate_mass_rate = (
        candidate_reconstructed_masses
        * mass_rate_scale
        / time_step
        / candidate_concentration[..., np.newaxis]
    )
    flat_baseline_mass_rate = baseline_mass_rate.reshape(
        -1,
        case.masses.shape[-1],
    )
    flat_candidate_mass_rate = candidate_mass_rate.reshape(
        -1,
        case.masses.shape[-1],
    )
    flat_baseline_mass = case.masses.reshape(-1, case.masses.shape[-1])
    flat_candidate_mass = candidate_reconstructed_masses.reshape(
        -1,
        case.masses.shape[-1],
    )
    flat_concentration = case.concentration.reshape(-1)
    flat_candidate_concentration = candidate_concentration.reshape(-1)
    gas_mass = np.sum(case.masses, axis=(0, 1), dtype=np.float64) * 2.0
    if case.masses.shape[-1] == 1:
        baseline_mass_transfer = get_mass_transfer(
            mass_rate=flat_baseline_mass_rate[:, 0],
            time_step=time_step,
            gas_mass=gas_mass,
            particle_mass=flat_baseline_mass[:, 0],
            particle_concentration=flat_concentration,
        ).reshape(case.masses.shape[:-1] + (1,))
        candidate_mass_transfer = get_mass_transfer(
            mass_rate=flat_candidate_mass_rate[:, 0],
            time_step=time_step,
            gas_mass=gas_mass,
            particle_mass=flat_candidate_mass[:, 0],
            particle_concentration=flat_candidate_concentration,
        ).reshape(case.masses.shape[:-1] + (1,))
    else:
        baseline_mass_transfer = get_mass_transfer(
            mass_rate=flat_baseline_mass_rate,
            time_step=time_step,
            gas_mass=gas_mass,
            particle_mass=flat_baseline_mass,
            particle_concentration=flat_concentration,
        ).reshape(case.masses.shape)
        candidate_mass_transfer = get_mass_transfer(
            mass_rate=flat_candidate_mass_rate,
            time_step=time_step,
            gas_mass=gas_mass,
            particle_mass=flat_candidate_mass,
            particle_concentration=flat_candidate_concentration,
        ).reshape(case.masses.shape)
    per_particle_delta = candidate_mass_transfer - baseline_mass_transfer
    return _MassTransferReferenceResult(
        case=case,
        candidate_id=candidate_id,
        baseline_mass_transfer=baseline_mass_transfer,
        candidate_mass_transfer=candidate_mass_transfer,
        per_particle_delta=per_particle_delta,
        aggregate_species_delta=np.sum(
            per_particle_delta,
            axis=(0, 1),
            dtype=np.float64,
        ),
    )


def _compute_clamp_metrics(
    initial_mass: np.ndarray,
    raw_mass_transfer: np.ndarray,
) -> _ClampMetrics:
    """Compute warning-clean clamp accounting metrics."""
    raw_updated_mass = initial_mass + raw_mass_transfer
    post_clamp_mass = np.maximum(raw_updated_mass, 0.0)
    clamp_delta = post_clamp_mass - raw_updated_mass
    return _ClampMetrics(
        raw_updated_mass=raw_updated_mass,
        post_clamp_mass=post_clamp_mass,
        clamp_delta=clamp_delta,
        clamp_frequency=int(np.count_nonzero(raw_updated_mass < 0.0)),
        aggregate_clamp_delta=np.sum(clamp_delta, axis=(0, 1), dtype=np.float64),
    )
_MIXED_SCALE_CASE = _build_mixed_scale_case()
_CANDIDATE_STUDY_RESULTS = {
    (case.case_name, candidate_id): _build_candidate_study_result(
        case,
        candidate_id,
    )
    for case, candidate_id in _candidate_projection_items()
}
_MIXED_SCALE_RESULTS = {
    candidate_id: _build_candidate_study_result(_MIXED_SCALE_CASE, candidate_id)
    for candidate_id in _SUPPORTED_CANDIDATES
}
_MASS_TRANSFER_REFERENCE_RESULTS = {
    (case.case_name, candidate_id): _build_mass_transfer_reference_result(
        case,
        candidate_id,
    )
    for case, candidate_id in _candidate_projection_items()
}


def _get_candidate_result(
    case: _MassPrecisionCase,
    candidate_id: str,
) -> _CandidateStudyResult:
    """Return cached candidate reconstruction metrics."""
    return _CANDIDATE_STUDY_RESULTS[(case.case_name, candidate_id)]


def _get_mass_transfer_reference_result(
    case: _MassPrecisionCase,
    candidate_id: str,
) -> _MassTransferReferenceResult:
    """Return cached CPU-reference mass-transfer comparison metrics."""
    return _MASS_TRANSFER_REFERENCE_RESULTS[(case.case_name, candidate_id)]


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
    result = _get_candidate_result(case, candidate_id)

    npt.assert_allclose(
        result.reconstructed_masses,
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
    result = _get_candidate_result(case, candidate_id)

    npt.assert_allclose(
        result.reconstructed_radii,
        result.baseline_radii,
        rtol=_RADIUS_RTOL,
        atol=_RADIUS_ATOL,
        err_msg=(
            "candidate reconstruction exceeded radius tolerance for "
            f"case={case.case_name}, candidate={candidate_id}"
        ),
    )


@pytest.mark.parametrize(
    ("case", "candidate_id"),
    _CASE_CANDIDATE_PARAMS,
    ids=_CASE_CANDIDATE_IDS,
)
def test_candidate_projected_payload_matches_documented_schema(
    case: _MassPrecisionCase,
    candidate_id: str,
) -> None:
    """Each candidate exposes the documented projected payload schema."""
    result = _get_candidate_result(case, candidate_id)

    _assert_projected_schema(result.projected, case, candidate_id)


@pytest.mark.parametrize(
    ("case", "candidate_id"),
    _CASE_CANDIDATE_PARAMS,
    ids=_CASE_CANDIDATE_IDS,
)
def test_candidate_absolute_and_relative_mass_error_metrics_are_bounded(
    case: _MassPrecisionCase,
    candidate_id: str,
) -> None:
    """Pure reconstruction error metrics stay within documented bounds."""
    result = _get_candidate_result(case, candidate_id)

    assert np.max(result.relative_mass_error) <= _MASS_RTOL, (
        "pure reconstruction error exceeded relative mass threshold for "
        f"case={case.case_name}, candidate={candidate_id}"
    )
    assert np.all(result.absolute_mass_error >= 0.0)
    assert result.aggregate_mass_delta.shape == case.density_kg_m3.shape


@pytest.mark.parametrize(
    ("case", "candidate_id"),
    _CASE_CANDIDATE_PARAMS,
    ids=_CASE_CANDIDATE_IDS,
)
def test_cpu_reference_mass_transfer_error_is_bounded_by_case_and_candidate(
    case: _MassPrecisionCase,
    candidate_id: str,
) -> None:
    """CPU-reference mass-transfer error remains bounded per case."""
    result = _get_mass_transfer_reference_result(case, candidate_id)

    npt.assert_allclose(
        result.candidate_mass_transfer,
        result.baseline_mass_transfer,
        rtol=_MASS_RTOL,
        atol=_MASS_ATOL,
        err_msg=(
            "CPU-reference mass-transfer error exceeded tolerance for "
            f"case={case.case_name}, candidate={candidate_id}"
        ),
    )
    assert result.aggregate_species_delta.shape == case.density_kg_m3.shape


@pytest.mark.parametrize(
    ("case", "candidate_id"),
    _CASE_CANDIDATE_PARAMS,
    ids=_CASE_CANDIDATE_IDS,
)
def test_cpu_reference_mass_transfer_delta_metrics_remain_visible_and_bounded(
    case: _MassPrecisionCase,
    candidate_id: str,
) -> None:
    """Per-particle and aggregate CPU-reference deltas stay explicit."""
    result = _get_mass_transfer_reference_result(case, candidate_id)
    per_particle_atol = (
        np.max(np.abs(result.baseline_mass_transfer)) * _MASS_RTOL + _MASS_ATOL
    )
    aggregate_atol = (
        np.max(np.abs(np.sum(result.baseline_mass_transfer, axis=(0, 1))))
        * _AGGREGATE_DELTA_RTOL
        + _MASS_ATOL
    )

    npt.assert_allclose(
        result.per_particle_delta,
        np.zeros_like(result.per_particle_delta),
        rtol=_MASS_RTOL,
        atol=per_particle_atol,
        err_msg=(
            "CPU-reference per-particle delta exceeded tolerance for "
            f"case={case.case_name}, candidate={candidate_id}"
        ),
    )
    npt.assert_allclose(
        result.aggregate_species_delta,
        np.zeros_like(result.aggregate_species_delta),
        rtol=_AGGREGATE_DELTA_RTOL,
        atol=aggregate_atol,
        err_msg=(
            "CPU-reference aggregate species-total delta exceeded tolerance "
            f"for case={case.case_name}, candidate={candidate_id}"
        ),
    )


@pytest.mark.parametrize(
    "candidate_id",
    _SUPPORTED_CANDIDATES,
)
def test_mixed_scale_smallest_particle_mass_error_is_bounded(
    candidate_id: str,
) -> None:
    """Smallest-particle in mixed box keeps bounded mass error."""
    result = _MIXED_SCALE_RESULTS[candidate_id]
    smallest_particle_error = result.relative_mass_error[:, 0, :]

    assert np.max(smallest_particle_error) <= _MIXED_SCALE_MASS_RTOL, (
        "mixed-scale smallest-particle mass error exceeded threshold for "
        f"candidate={candidate_id}"
    )


@pytest.mark.parametrize(
    "candidate_id",
    _SUPPORTED_CANDIDATES,
)
def test_mixed_scale_aggregate_mass_error_is_bounded_separately(
    candidate_id: str,
) -> None:
    """Whole-array mixed-scale totals stay bounded apart from smallest slices."""
    result = _MIXED_SCALE_RESULTS[candidate_id]
    baseline_species_total = np.sum(
        _MIXED_SCALE_CASE.masses,
        axis=(0, 1),
        dtype=np.float64,
    )
    reconstructed_species_total = (
        baseline_species_total + result.aggregate_mass_delta
    )
    aggregate_relative_delta = _safe_relative_error(
        reconstructed_species_total,
        baseline_species_total,
    )

    assert np.max(aggregate_relative_delta) <= _AGGREGATE_DELTA_RTOL, (
        "mixed-scale aggregate species-total mass error exceeded threshold "
        f"for candidate={candidate_id}"
    )


@pytest.mark.parametrize(
    "candidate_id",
    _SUPPORTED_CANDIDATES,
)
def test_mixed_scale_smallest_particle_radius_error_is_bounded(
    candidate_id: str,
) -> None:
    """Smallest-particle in mixed box keeps bounded radius error."""
    result = _MIXED_SCALE_RESULTS[candidate_id]
    smallest_particle_error = result.relative_radius_error[:, 0]

    assert np.max(smallest_particle_error) <= _MIXED_SCALE_RADIUS_RTOL, (
        "mixed-scale smallest-particle radius error exceeded threshold for "
        f"candidate={candidate_id}"
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

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        candidate = _project_candidate(
            zero_mass_case,
            "fp32_total_mass_fp32_mass_fraction",
        )
        result = _build_candidate_study_result(
            zero_mass_case,
            "fp32_total_mass_fp32_mass_fraction",
        )

    assert not caught, "zero-total-mass handling emitted an unexpected warning"
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
    npt.assert_array_equal(
        result.reconstructed_radii,
        np.zeros_like(result.reconstructed_radii),
    )
    npt.assert_array_equal(
        result.relative_mass_error,
        np.zeros_like(result.relative_mass_error),
    )


def test_zero_volume_and_zero_effective_radius_reference_paths_are_warning_clean(
) -> None:
    """Zero-volume and zero-effective-radius paths stay deterministic."""
    zero_particle_mass = np.zeros((2, 3, 1), dtype=np.float64)
    zero_mass_rate = np.zeros_like(zero_particle_mass)
    particle_concentration = np.array([[10.0, 20.0, 30.0], [5.0, 6.0, 7.0]])
    gas_mass = np.array([1.0], dtype=np.float64)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reconstructed_radii = _reconstruct_radii(
            zero_particle_mass,
            np.array([1000.0], dtype=np.float64),
        )
        mass_transfer = get_mass_transfer(
            mass_rate=zero_mass_rate.reshape(-1),
            time_step=0.5,
            gas_mass=gas_mass,
            particle_mass=zero_particle_mass.reshape(-1),
            particle_concentration=particle_concentration.reshape(-1),
        ).reshape(zero_mass_rate.shape)

    assert not caught, "zero-volume handling emitted an unexpected warning"
    npt.assert_array_equal(
        reconstructed_radii,
        np.zeros((2, 3), dtype=np.float64),
    )
    npt.assert_array_equal(mass_transfer, zero_mass_rate)


def test_mixed_precision_candidate_reference_uses_candidate_concentration(
) -> None:
    """Candidate concentration perturbations remain visible in reference output."""
    case = _MASS_PRECISION_CASES[0]
    candidate = _project_candidate(case, "mixed_precision_mass_plus_density")
    perturbed_candidate = {
        **candidate,
        "projected": {
            **candidate["projected"],
            "concentration": (
                candidate["projected"]["concentration"]
                * np.array([[1.25, 0.75, 1.5], [0.5, 1.1, 0.9]], dtype=np.float32)
            ).astype(np.float32),
        },
    }

    baseline_result = _build_mass_transfer_reference_result(
        case,
        "mixed_precision_mass_plus_density",
        candidate=candidate,
    )
    perturbed_result = _build_mass_transfer_reference_result(
        case,
        "mixed_precision_mass_plus_density",
        candidate=perturbed_candidate,
    )

    assert not np.allclose(
        perturbed_result.candidate_mass_transfer,
        baseline_result.candidate_mass_transfer,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("candidate_id", _SUPPORTED_CANDIDATES)
def test_clamp_accounting_distinguishes_raw_transfer_from_clamp_delta(
    candidate_id: str,
) -> None:
    """Clamp metrics separate raw transfer error from clamp-induced change."""
    baseline_case = _MIXED_SCALE_CASE
    candidate_result = _MIXED_SCALE_RESULTS[candidate_id]
    scale = np.array(
        [[1.4, 0.25, 1.2], [0.9, 1.3, 0.2]],
        dtype=np.float64,
    )[..., np.newaxis]
    baseline_raw_transfer = -baseline_case.masses * scale
    candidate_raw_transfer = -candidate_result.reconstructed_masses * scale
    baseline_clamp = _compute_clamp_metrics(
        baseline_case.masses,
        baseline_raw_transfer,
    )
    candidate_clamp = _compute_clamp_metrics(
        candidate_result.reconstructed_masses,
        candidate_raw_transfer,
    )

    expected_frequency = int(
        np.count_nonzero(candidate_clamp.raw_updated_mass < 0.0)
    )
    assert candidate_clamp.clamp_frequency == expected_frequency
    npt.assert_array_equal(
        candidate_clamp.clamp_delta,
        np.where(
            candidate_clamp.raw_updated_mass < 0.0,
            -candidate_clamp.raw_updated_mass,
            0.0,
        ),
    )

    raw_error = (
        candidate_clamp.raw_updated_mass - baseline_clamp.raw_updated_mass
    )
    post_clamp_error = (
        candidate_clamp.post_clamp_mass - baseline_clamp.post_clamp_mass
    )
    clamp_error = candidate_clamp.clamp_delta - baseline_clamp.clamp_delta

    npt.assert_allclose(
        post_clamp_error,
        raw_error + clamp_error,
        rtol=_MASS_RTOL,
        atol=_MASS_ATOL,
        err_msg=(
            "clamp accounting failed to keep raw transfer and clamp delta "
            f"separate for candidate={candidate_id}"
        ),
    )
    assert np.all(candidate_clamp.aggregate_clamp_delta >= 0.0)


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
