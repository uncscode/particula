"""Bounded fresh-seed public-step validation for executable coagulation masks."""

from __future__ import annotations

import numpy as np
import pytest

from particula.gpu.kernels.tests._coagulation_public_step_support import (
    _assert_public_invariants,
    _materialize_public_particles,
    _require_warp,
    _run_public_case,
)
from particula.gpu.kernels.tests._coagulation_validation_support import (
    EXECUTABLE_ROWS,
    STOCHASTIC_CASES,
    StochasticCase,
    active_unordered_pairs,
    enabled_component_rates,
    enabled_pair_rate_sum,
    expected_accepted_count,
    expected_mean,
    scheduler_expectation,
    scheduling_concentration,
    sigma_tolerance,
)


def _seed_for_trial(mask: int, trial: int) -> int:
    """Return a deterministic, uint32-compatible fresh seed for one trial."""
    return 10_000 * mask + trial


def test_stochastic_cases_cover_executable_masks_and_bounded_schedules() -> (
    None
):
    """Every approved mask has a finite, bounded one-proposal experiment."""
    assert {case.row.mask for case in STOCHASTIC_CASES} == {
        row.mask for row in EXECUTABLE_ROWS
    }
    assert len(STOCHASTIC_CASES) == len(EXECUTABLE_ROWS)
    for case in STOCHASTIC_CASES:
        assert case.sample_count == 100
        assert case.max_collisions == 1
        assert np.all(np.isfinite(case.fixture.radii))
        assert np.all(case.fixture.radii > 0.0)
        assert np.all(np.isfinite(case.fixture.density))
        assert np.all(case.fixture.density > 0.0)
        assert np.all(np.isfinite(case.fixture.charges))
        assert np.all(np.isfinite(case.fixture.temperature))
        assert np.all(case.fixture.temperature > 0.0)
        assert np.all(np.isfinite(case.fixture.pressure))
        assert np.all(case.fixture.pressure > 0.0)
        assert np.all(np.isfinite(case.fixture.dissipation))
        assert np.all(case.fixture.dissipation > 0.0)
        assert np.all(np.isfinite(case.fixture.fluid_density))
        assert np.all(case.fixture.fluid_density > 0.0)
        assert np.all(np.isfinite(case.concentration))
        assert np.all(case.concentration >= 0.0)
        assert np.all(case.volume > 0.0)
        assert case.time_step > 0.0
        for box in range(case.fixture.radii.shape[0]):
            assert len(case.fixture.active[box]) >= 2
            assert 0.0 < scheduler_expectation(case, box) <= 1.0
            assert enabled_pair_rate_sum(case, box) > 0.0
        assert expected_mean(case) >= 100.0
        assert sigma_tolerance(case) > 0.0


def test_stochastic_oracle_responds_to_pair_set_time_volume_and_sedimentation() -> (
    None
):
    """The independent expectation retains every initial-state schedule factor."""
    case = next(case for case in STOCHASTIC_CASES if case.row.mask == 4)
    baseline = expected_accepted_count(case)
    assert len(active_unordered_pairs(case, 0)) == 3
    assert scheduling_concentration(case, 0) == 2.0
    fewer_active = type(case)(
        case.row,
        type(case.fixture)(
            case.fixture.name,
            case.fixture.radii,
            case.fixture.density,
            case.fixture.charges,
            case.fixture.temperature,
            case.fixture.pressure,
            case.fixture.dissipation,
            case.fixture.fluid_density,
            ((0, 2),) * case.fixture.radii.shape[0],
        ),
        case.concentration,
        case.volume,
        case.time_step,
    )
    assert expected_accepted_count(fewer_active) < baseline
    assert expected_accepted_count(
        type(case)(
            case.row,
            case.fixture,
            case.concentration,
            case.volume,
            case.time_step * 2,
        )
    ) == pytest.approx(2.0 * baseline)
    assert expected_accepted_count(
        type(case)(
            case.row,
            case.fixture,
            case.concentration,
            case.volume * 2,
            case.time_step,
        )
    ) == pytest.approx(0.5 * baseline)
    altered = case.concentration.copy()
    altered[:, [0, 2, 3]] = 3.0
    assert expected_accepted_count(
        type(case)(case.row, case.fixture, altered, case.volume, case.time_step)
    ) == pytest.approx(1.5 * baseline)


def test_stochastic_oracle_sums_each_enabled_component_for_every_mask() -> None:
    """Enabled additive components each change the independent total rate."""
    for case in STOCHASTIC_CASES:
        for box in range(case.fixture.radii.shape[0]):
            explicit = sum(
                sum(enabled_component_rates(case, box, first, second))
                for first, second in active_unordered_pairs(case, box)
            )
            assert enabled_pair_rate_sum(case, box) == pytest.approx(explicit)
            for flag, enabled in zip(
                (1, 2, 4, 8), case.row.enabled, strict=True
            ):
                if enabled:
                    reduced = sum(
                        sum(
                            component
                            for component_flag, component in zip(
                                (1, 2, 4, 8),
                                enabled_component_rates(
                                    case, box, first, second
                                ),
                                strict=True,
                            )
                            if component_flag != flag
                        )
                        for first, second in active_unordered_pairs(case, box)
                    )
                    assert not np.isclose(
                        reduced, explicit, rtol=1e-12, atol=0.0
                    )


def test_seed_schedule_is_unique_and_uint32_compatible() -> None:
    """Each row has exactly 100 distinct public-step RNG streams."""
    for case in STOCHASTIC_CASES:
        seeds = [_seed_for_trial(case.row.mask, trial) for trial in range(100)]
        assert len(set(seeds)) == 100
        assert all(0 <= seed <= np.iinfo(np.uint32).max for seed in seeds)


def test_public_materializer_rejects_invalid_volume_override_shape() -> None:
    """Volume override shape errors are raised before a Warp transfer."""
    fixture = STOCHASTIC_CASES[0].fixture
    for volume in (np.ones(3), np.ones((4, 1))):
        with pytest.raises(ValueError, match="volume override must have shape"):
            _materialize_public_particles(fixture, n_species=1, volume=volume)


def test_public_materializer_rejects_invalid_concentration_override_shape() -> (
    None
):
    """Concentration override rank and particle width are validated on host."""
    fixture = STOCHASTIC_CASES[0].fixture
    for concentration in (np.ones(4), np.ones((4, 3)), np.ones((3, 4))):
        with pytest.raises(
            ValueError, match="concentration override must have shape"
        ):
            _materialize_public_particles(
                fixture, n_species=1, concentration=concentration
            )


def test_public_materializer_retains_p3_overrides_and_inactive_sentinels() -> (
    None
):
    """Explicit P3 active values do not overwrite inactive sentinel state."""
    case = STOCHASTIC_CASES[0]
    particles = _materialize_public_particles(
        case.fixture,
        n_species=1,
        volume=case.volume,
        concentration=case.concentration,
    )
    np.testing.assert_array_equal(particles.volume, case.volume)
    for box, active in enumerate(case.fixture.active):
        np.testing.assert_array_equal(
            particles.concentration[box, list(active)],
            case.concentration[box, list(active)],
        )
        inactive = sorted(set(range(case.fixture.radii.shape[1])) - set(active))
        np.testing.assert_array_equal(particles.masses[box, inactive], 7.0e-31)
        np.testing.assert_array_equal(
            particles.concentration[box, inactive], 0.0
        )
        np.testing.assert_array_equal(particles.charge[box, inactive], 29.0)


DEVICE_PARAMS = (
    pytest.param(
        "cpu",
        marks=[
            pytest.mark.warp,
            pytest.mark.gpu_parity,
            pytest.mark.stochastic,
        ],
    ),
    pytest.param(
        "cuda",
        marks=[
            pytest.mark.warp,
            pytest.mark.gpu_parity,
            pytest.mark.stochastic,
            pytest.mark.cuda,
        ],
    ),
)


@pytest.mark.parametrize("device", DEVICE_PARAMS)
@pytest.mark.parametrize(
    "case", STOCHASTIC_CASES, ids=lambda case: f"mask_{case.row.mask}"
)
def test_public_stochastic_aggregate_matches_independent_expectation(
    case: StochasticCase, device: str
) -> None:
    """Fresh public calls meet the documented initial-state three-sigma bound."""
    wp = _require_warp()
    if device == "cuda":
        from particula.gpu.tests.cuda_availability import (
            CUDA_SKIP_REASON,
            cuda_available,
        )

        if not cuda_available(wp):
            pytest.skip(CUDA_SKIP_REASON)
    observed = 0
    for trial in range(case.sample_count):
        initial, final, particles, pairs, counts, states = _run_public_case(
            case.row,
            case.fixture,
            n_species=1,
            max_collisions=case.max_collisions,
            device=device,
            time_step=case.time_step,
            volume=case.volume,
            concentration=case.concentration,
            seed=_seed_for_trial(case.row.mask, trial),
            turbulent_arrays=bool(case.row.mask & 8),
            turbulent_dissipation=case.fixture.dissipation,
            fluid_density=case.fixture.fluid_density,
        )
        assert (
            particles is not None and pairs is not None and counts is not None
        )
        assert states is not None and final["states"].dtype == np.uint32
        _assert_public_invariants(
            initial,
            final,
            case.fixture.active,
            charge_transfers=True,
        )
        assert np.all((final["counts"] >= 0) & (final["counts"] <= 1))
        observed += int(final["counts"].sum())
    assert abs(observed - expected_mean(case)) <= sigma_tolerance(case)
