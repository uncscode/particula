"""Validate fixed-mask GPU coagulation dispatch and public-step invariants.

Independent deterministic fp64 expectations remain in the Warp-free support
module. Runtime-guarded Warp tests verify public-step state integrity,
caller-owned sidecar and RNG lifecycles, and preflight atomicity. Host-only
metadata coverage remains collectable without the optional backend.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import numpy.testing as npt
import pytest

from particula.dynamics.coagulation.brownian_kernel import (
    get_brownian_kernel_via_system_state,
)
from particula.gpu.kernels._coagulation_config import (
    CoagulationMechanismConfig,
    resolve_coagulation_mechanism_config,
    validate_coagulation_mechanism_capabilities,
)
from particula.gpu.kernels.tests._coagulation_public_step_support import (
    _assert_public_invariants,
    _materialize_public_particles,
    _public_snapshot,
    _require_warp,
    _run_on_warp_devices,
    _run_public_case,
)
from particula.gpu.kernels.tests._coagulation_validation_support import (
    DEFERRED_ROWS,
    EXECUTABLE_ROWS,
    FIXTURE_NAMES_BY_MASK,
    FIXTURES,
    applicable_fixture_names,
    brownian_rate_from_properties,
    pair_rate,
    properties,
    selector_majorant,
)


@lru_cache(maxsize=1)
def _probe_factory() -> Any:  # noqa: C901
    """Construct private-helper observation kernel after Warp is available."""
    wp = _require_warp()
    from particula.gpu.dynamics.coagulation_funcs import (
        brownian_diffusivity_wp,
        g_collection_term_wp,
        kinematic_viscosity_wp,
        particle_mean_free_path_wp,
    )
    from particula.gpu.kernels.coagulation import (
        _total_pair_rate,
        _turbulent_majorant_from_active_radii,
    )
    from particula.gpu.properties.gas_properties import (
        dynamic_viscosity_wp,
        molecule_mean_free_path_wp,
    )
    from particula.gpu.properties.particle_properties import (
        aerodynamic_mobility_wp,
        cunningham_slip_correction_wp,
        knudsen_number_wp,
        mean_thermal_speed_wp,
        settling_velocity_stokes_from_transport_wp,
    )
    from particula.util import constants

    @wp.kernel
    def observe(
        radii: wp.array2d(dtype=wp.float64),  # type: ignore[valid-type]
        masses: wp.array2d(dtype=wp.float64),  # type: ignore[valid-type]
        density: wp.array2d(dtype=wp.float64),  # type: ignore[valid-type]
        charge: wp.array2d(dtype=wp.float64),  # type: ignore[valid-type]
        active: wp.array2d(dtype=wp.int32),  # type: ignore[valid-type]
        active_indices: wp.array2d(dtype=wp.int32),  # type: ignore[valid-type]
        active_count: wp.array(dtype=wp.int32),  # type: ignore[valid-type]
        temperature: wp.array(dtype=wp.float64),  # type: ignore[valid-type]
        pressure: wp.array(dtype=wp.float64),  # type: ignore[valid-type]
        dissipation: wp.array(dtype=wp.float64),  # type: ignore[valid-type]
        fluid_density: wp.array(dtype=wp.float64),  # type: ignore[valid-type]
        mask: wp.int32,  # type: ignore[name-defined]
        diffusivity: wp.array2d(dtype=wp.float64),  # type: ignore[valid-type]
        g_term: wp.array2d(dtype=wp.float64),  # type: ignore[valid-type]
        speed: wp.array2d(dtype=wp.float64),  # type: ignore[valid-type]
        settling: wp.array2d(dtype=wp.float64),  # type: ignore[valid-type]
        nu: wp.array(dtype=wp.float64),  # type: ignore[valid-type]
        rates: wp.array3d(dtype=wp.float64),  # type: ignore[valid-type]
        majorant: wp.array(dtype=wp.float64),  # type: ignore[valid-type]
    ):
        box = wp.tid()
        mu = dynamic_viscosity_wp(
            temperature[box],
            wp.float64(constants.REF_VISCOSITY_AIR_STP),
            wp.float64(constants.REF_TEMPERATURE_STP),
            wp.float64(constants.SUTHERLAND_CONSTANT),
        )
        mfp = molecule_mean_free_path_wp(
            wp.float64(constants.MOLECULAR_WEIGHT_AIR),
            temperature[box],
            pressure[box],
            mu,
            wp.float64(constants.GAS_CONSTANT),
        )
        nu[box] = kinematic_viscosity_wp(mu, fluid_density[box])
        maximum = wp.float64(0.0)
        for i in range(radii.shape[1]):
            if active[box, i] == 1:
                kn = knudsen_number_wp(mfp, radii[box, i])
                slip = cunningham_slip_correction_wp(kn)
                diffusivity[box, i] = brownian_diffusivity_wp(
                    temperature[box],
                    aerodynamic_mobility_wp(radii[box, i], slip, mu),
                    wp.float64(constants.BOLTZMANN_CONSTANT),
                )
                speed[box, i] = mean_thermal_speed_wp(
                    masses[box, i],
                    temperature[box],
                    wp.float64(constants.BOLTZMANN_CONSTANT),
                )
                g_term[box, i] = g_collection_term_wp(
                    particle_mean_free_path_wp(
                        diffusivity[box, i], speed[box, i]
                    ),
                    radii[box, i],
                )
                settling[box, i] = settling_velocity_stokes_from_transport_wp(
                    radii[box, i], density[box, i], mu, mfp
                )
        for i in range(radii.shape[1]):
            for j in range(radii.shape[1]):
                rates[box, i, j] = wp.float64(0.0)
                if active[box, i] == 1 and active[box, j] == 1 and i != j:
                    value = _total_pair_rate(
                        mask,
                        radii[box, i],
                        radii[box, j],
                        diffusivity[box, i],
                        diffusivity[box, j],
                        g_term[box, i],
                        g_term[box, j],
                        speed[box, i],
                        speed[box, j],
                        settling[box, i],
                        settling[box, j],
                        dissipation[box],
                        nu[box],
                        masses[box, i],
                        masses[box, j],
                        charge[box, i],
                        charge[box, j],
                        temperature[box],
                        pressure[box],
                        wp.float64(constants.BOLTZMANN_CONSTANT),
                        wp.float64(constants.ELEMENTARY_CHARGE_VALUE),
                        wp.float64(constants.ELECTRIC_PERMITTIVITY),
                        wp.float64(constants.GAS_CONSTANT),
                        wp.float64(constants.MOLECULAR_WEIGHT_AIR),
                        wp.float64(constants.REF_VISCOSITY_AIR_STP),
                        wp.float64(constants.REF_TEMPERATURE_STP),
                        wp.float64(constants.SUTHERLAND_CONSTANT),
                    )
                    rates[box, i, j] = value
                    if i < j and value > maximum:
                        maximum = value
        if mask == wp.int32(8):
            # This is the selector's special ST1956 two-largest-radii rule.
            maximum = _turbulent_majorant_from_active_radii(
                active_indices,
                box,
                active_count[box],
                radii,
                dissipation[box],
                nu[box],
            )
        majorant[box] = maximum

    return wp, observe


def _observe(fixture_name: str, mask: int) -> dict[str, np.ndarray]:
    """Upload independent fixture state and return synchronized device probes."""
    wp, kernel = _probe_factory()
    fixture = FIXTURES[fixture_name]
    active = np.zeros_like(fixture.radii, dtype=np.int32)
    active_indices = np.zeros_like(active)
    active_count = np.empty(fixture.radii.shape[0], dtype=np.int32)
    for box, indices in enumerate(fixture.active):
        active[box, list(indices)] = 1
        active_indices[box, : len(indices)] = indices
        active_count[box] = len(indices)
    device = "cpu"
    arrays = [
        wp.array(value, dtype=wp.float64, device=device)
        for value in (
            fixture.radii,
            fixture.masses,
            fixture.density,
            fixture.charges,
        )
    ]
    active_array = wp.array(active, dtype=wp.int32, device=device)
    active_indices_array = wp.array(
        active_indices, dtype=wp.int32, device=device
    )
    active_count_array = wp.array(active_count, dtype=wp.int32, device=device)
    scalar_arrays = [
        wp.array(value, dtype=wp.float64, device=device)
        for value in (
            fixture.temperature,
            fixture.pressure,
            fixture.dissipation,
            fixture.fluid_density,
        )
    ]
    shape = fixture.radii.shape
    diffusivity = wp.zeros(shape, dtype=wp.float64, device=device)
    g_term = wp.zeros(shape, dtype=wp.float64, device=device)
    speed = wp.zeros(shape, dtype=wp.float64, device=device)
    settling = wp.zeros(shape, dtype=wp.float64, device=device)
    nu = wp.zeros(shape[0], dtype=wp.float64, device=device)
    rates = wp.zeros((*shape, shape[1]), dtype=wp.float64, device=device)
    majorant = wp.zeros(shape[0], dtype=wp.float64, device=device)
    wp.launch(
        kernel,
        dim=shape[0],
        inputs=[
            *arrays,
            active_array,
            active_indices_array,
            active_count_array,
            *scalar_arrays,
            np.int32(mask),
            diffusivity,
            g_term,
            speed,
            settling,
            nu,
            rates,
            majorant,
        ],
        device=device,
    )
    wp.synchronize_device(device)
    return {
        "diffusivity": diffusivity.numpy(),
        "g_term": g_term.numpy(),
        "speed": speed.numpy(),
        "settling": settling.numpy(),
        "nu": nu.numpy(),
        "rates": rates.numpy(),
        "majorant": majorant.numpy(),
    }


def test_frozen_mask_sets_are_exact() -> None:
    """Freeze the literal executable and deferred fixed-mask boundaries."""
    assert {row.mask for row in EXECUTABLE_ROWS} == {
        1,
        2,
        3,
        4,
        5,
        6,
        8,
        9,
        10,
        12,
        15,
    }
    assert {row.mask for row in DEFERRED_ROWS} == {7, 11, 13, 14}
    assert len(EXECUTABLE_ROWS) == 11
    assert len(DEFERRED_ROWS) == 4
    for row in (*EXECUTABLE_ROWS, *DEFERRED_ROWS):
        assert row.mask == sum(
            flag << bit for bit, flag in enumerate(row.enabled)
        )


def test_fixture_values_and_active_indices_are_valid() -> None:
    """Fixtures remain explicit fp64 data with finite, distinct active slots."""
    for fixture in FIXTURES.values():
        assert fixture.radii.dtype == np.float64
        assert np.all(np.isfinite(fixture.radii)) and np.all(fixture.radii > 0)
        for box, indices in enumerate(fixture.active):
            assert len(indices) == len(set(indices))
            assert all(0 <= index < fixture.radii.shape[1] for index in indices)
            if 1 not in indices:
                assert fixture.radii[box, 1] > max(
                    fixture.radii[box, list(indices)], default=0.0
                )


def test_independent_physical_zero_cases_are_exact() -> None:
    """Repulsion, equal settling velocities, and zero dissipation are exact."""
    assert pair_rate(FIXTURES["repulsive"], 0, 0, 2, 2) == 0.0
    assert pair_rate(FIXTURES["equal_velocity"], 0, 0, 2, 4) == 0.0
    assert pair_rate(FIXTURES["zero_dissipation"], 0, 0, 2, 8) == 0.0


def test_brownian_public_reference_cross_checks_independent_oracle() -> None:
    """Use the public CPU rate only to cross-check the independent oracle."""
    for fixture_name in ("normal", "two_box"):
        fixture = FIXTURES[fixture_name]
        for box, indices in enumerate(fixture.active):
            for position, i in enumerate(indices):
                for j in indices[position + 1 :]:
                    npt.assert_allclose(
                        brownian_rate_from_properties(fixture, box, i, j),
                        np.asarray(
                            get_brownian_kernel_via_system_state(
                                np.array(
                                    [
                                        fixture.radii[box, i],
                                        fixture.radii[box, j],
                                    ]
                                ),
                                np.array(
                                    [
                                        fixture.masses[box, i],
                                        fixture.masses[box, j],
                                    ]
                                ),
                                float(fixture.temperature[box]),
                                float(fixture.pressure[box]),
                            )
                        )[0, 1],
                        rtol=1e-12,
                        atol=0.0,
                    )


def test_literal_fixture_matrix_covers_each_enabled_edge_case() -> None:
    """Each executable row retains its required explicit fixture coverage."""
    assert set(FIXTURE_NAMES_BY_MASK) == {row.mask for row in EXECUTABLE_ROWS}
    for row in EXECUTABLE_ROWS:
        names = applicable_fixture_names(row.mask)
        assert names == FIXTURE_NAMES_BY_MASK[row.mask]
        assert "normal" in names
        if row.mask != 8:
            assert "two_box" in names
        if row.mask & 2:
            assert "repulsive" in names
        if row.mask & 4:
            assert "equal_velocity" in names
        if row.mask & 8:
            assert "zero_dissipation" in names
        assert set(names) <= set(FIXTURES)


@pytest.mark.parametrize(
    "row", EXECUTABLE_ROWS, ids=lambda row: f"mask_{row.mask}"
)
def test_configuration_rows_are_canonical_and_executable(row: Any) -> None:
    """Resolve each literal executable row through the public config boundary."""
    resolved = resolve_coagulation_mechanism_config(
        CoagulationMechanismConfig(mechanisms=row.mechanisms)
    )
    assert resolved.mechanisms == row.mechanisms
    assert resolved.mask == row.mask
    validate_coagulation_mechanism_capabilities(resolved)


@pytest.mark.parametrize(
    "row", DEFERRED_ROWS, ids=lambda row: f"mask_{row.mask}"
)
def test_configuration_rows_report_stable_deferred_error(row: Any) -> None:
    """Recognized three-way rows retain the explicit deferred boundary."""
    resolved = resolve_coagulation_mechanism_config(
        CoagulationMechanismConfig(mechanisms=row.mechanisms)
    )
    with pytest.raises(
        ValueError, match="^Additive coagulation execution is deferred\\.$"
    ):
        validate_coagulation_mechanism_capabilities(resolved)


def test_host_configuration_defaults_and_canonicalizes_permuted_input() -> None:
    """Host-only resolution defaults to Brownian and uses fixed flag order."""
    default = resolve_coagulation_mechanism_config(CoagulationMechanismConfig())
    permuted = resolve_coagulation_mechanism_config(
        CoagulationMechanismConfig(
            mechanisms=(
                "turbulent_shear_st1956",
                "sedimentation_sp2016",
                "charged_hard_sphere",
                "brownian",
            )
        )
    )

    assert default.mechanisms == ("brownian",)
    assert default.mask == 1
    assert permuted.mechanisms == (
        "brownian",
        "charged_hard_sphere",
        "sedimentation_sp2016",
        "turbulent_shear_st1956",
    )
    assert permuted.mask == 15


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (
            CoagulationMechanismConfig(distribution_type="discrete"),
            "distribution_type must be exactly 'particle_resolved'.",
        ),
        (
            CoagulationMechanismConfig(mechanisms=()),
            "mechanisms must be a non-empty tuple of strings.",
        ),
        (
            CoagulationMechanismConfig(
                mechanisms=("brownian", 1)  # type: ignore[arg-type]
            ),
            "mechanisms must contain only string identifiers.",
        ),
        (
            CoagulationMechanismConfig(mechanisms=("brownian", "brownian")),
            "Duplicate coagulation mechanism 'brownian'.",
        ),
        (
            CoagulationMechanismConfig(mechanisms=("unknown",)),
            "Unknown coagulation mechanism 'unknown'.",
        ),
    ],
)
def test_host_configuration_rejects_malformed_structures(
    config: CoagulationMechanismConfig, message: str
) -> None:
    """Host-only resolution rejects every structural validation boundary."""
    with pytest.raises(ValueError, match=f"^{message}$"):
        resolve_coagulation_mechanism_config(config)


@pytest.mark.warp
@pytest.mark.gpu_parity
@pytest.mark.parametrize(
    "row", EXECUTABLE_ROWS, ids=lambda row: f"mask_{row.mask}"
)
def test_pair_property_and_selector_parity_matrix(row: Any) -> None:
    """Compare private observations with independent properties and pair sums."""
    for name in applicable_fixture_names(row.mask):
        _assert_parity_for_fixture(name, row.mask)


def _assert_parity_for_fixture(fixture_name: str, mask: int) -> None:
    """Check one fixture against independent property and rate expectations."""
    fixture = FIXTURES[fixture_name]
    observed = _observe(fixture_name, mask)
    for box, indices in enumerate(fixture.active):
        _assert_observed_properties(observed, fixture, box, indices)
        _assert_observed_pair_rates(observed, fixture, box, indices, mask)
        expected_majorant = selector_majorant(fixture, box, mask)
        npt.assert_allclose(
            observed["majorant"][box],
            expected_majorant,
            rtol=1e-6,
            atol=0.0,
        )


def _assert_observed_properties(
    observed: dict[str, np.ndarray],
    fixture: Any,
    box: int,
    indices: tuple[int, ...],
) -> None:
    """Compare device-observed properties to the independent fp64 oracle."""
    expected_properties = properties(fixture, box)
    for key in ("diffusivity", "g_term", "speed", "settling"):
        if indices:
            npt.assert_allclose(
                observed[key][box, list(indices)],
                np.asarray(expected_properties[key])[list(indices)],
                rtol=1e-6,
                atol=0.0,
            )
    npt.assert_allclose(
        observed["nu"][box],
        expected_properties["nu"],
        rtol=1e-6,
        atol=0.0,
    )


def _assert_observed_pair_rates(
    observed: dict[str, np.ndarray],
    fixture: Any,
    box: int,
    indices: tuple[int, ...],
    mask: int,
) -> None:
    """Compare each active unordered pair with its independent expected rate."""
    for position, i in enumerate(indices):
        for j in indices[position + 1 :]:
            expected = pair_rate(fixture, box, i, j, mask)
            observed_rate = observed["rates"][box, i, j]
            if fixture.name == "repulsive" and expected == 0.0:
                assert expected == 0.0
                assert observed_rate == 0.0
            else:
                npt.assert_allclose(
                    observed_rate,
                    expected,
                    rtol=1e-7 if mask == 1 else 1e-6,
                    atol=0.0,
                )
            npt.assert_allclose(
                observed["rates"][box, j, i],
                observed_rate,
                rtol=0.0,
                atol=0.0,
            )
            assert np.isfinite(observed["rates"][box, i, j])
            assert observed["rates"][box, i, j] >= 0.0
    inactive = sorted(set(range(fixture.radii.shape[1])) - set(indices))
    assert np.count_nonzero(observed["rates"][box, inactive, :]) == 0
    assert np.count_nonzero(observed["rates"][box, :, inactive]) == 0


@pytest.mark.warp
@pytest.mark.gpu_parity
@pytest.mark.parametrize("name", ("zero_active", "one_active"))
def test_sparse_active_boundaries_have_no_pair_rate_or_majorant(
    name: str,
) -> None:
    """Zero/one-active fixtures produce no rates despite inactive slots."""
    observed = _observe(name, 15)
    assert observed["majorant"][0] == 0.0
    assert np.count_nonzero(observed["rates"][0]) == 0


@pytest.mark.warp
@pytest.mark.gpu_parity
@pytest.mark.parametrize(
    "row", EXECUTABLE_ROWS, ids=lambda row: f"mask_{row.mask}"
)
@pytest.mark.parametrize("fixture_name", ("normal", "two_box"))
@pytest.mark.parametrize("n_species", (1, 2))
def test_public_step_preserves_state_integrity_for_executable_masks(
    row: Any, fixture_name: str, n_species: int
) -> None:
    """Every executable public mask preserves inventory, ownership, and slots."""
    wp = _require_warp()
    fixture = FIXTURES[fixture_name]
    for device in _run_on_warp_devices(wp):
        initial, final, _, _, _, _ = _run_public_case(
            row, fixture, n_species=n_species, max_collisions=1, device=device
        )
        _assert_public_invariants(
            initial, final, fixture.active, charge_transfers=bool(row.mask & 2)
        )


@pytest.mark.warp
@pytest.mark.gpu_parity
@pytest.mark.parametrize(
    "row", EXECUTABLE_ROWS, ids=lambda row: f"mask_{row.mask}"
)
def test_executable_masks_force_one_legal_merge(row: Any) -> None:
    """Each executable mask accepts and applies a finite forced pair."""
    wp = _require_warp()
    fixture = FIXTURES["normal"]
    active_by_box = ((0, 2),)
    for device in _run_on_warp_devices(wp):
        initial, final, _, _, _, _ = _run_public_case(
            row,
            fixture,
            n_species=2,
            max_collisions=1,
            device=device,
            time_step=1.0e12,
            active_by_box=active_by_box,
        )
        npt.assert_array_equal(final["counts"], [1])
        npt.assert_array_equal(final["pairs"][0, 0], [0, 2])
        _assert_public_invariants(
            initial, final, active_by_box, charge_transfers=True
        )


@pytest.mark.warp
@pytest.mark.gpu_parity
def test_public_step_preserves_multi_pair_ordering_and_bookkeeping() -> None:
    """Two capacity slots produce two sorted, disjoint applied merges."""
    wp = _require_warp()
    row = next(row for row in EXECUTABLE_ROWS if row.mask == 1)
    fixture = FIXTURES["normal"]
    active_by_box = ((0, 1, 2, 3),)
    for device in _run_on_warp_devices(wp):
        initial, final, _, _, _, _ = _run_public_case(
            row,
            fixture,
            n_species=2,
            max_collisions=2,
            device=device,
            time_step=1.0e12,
            active_by_box=active_by_box,
        )
        npt.assert_array_equal(final["counts"], [2])
        _assert_public_invariants(
            initial, final, active_by_box, charge_transfers=True
        )


@pytest.mark.warp
@pytest.mark.gpu_parity
@pytest.mark.parametrize("name", ("zero_active", "one_active"))
def test_public_step_sparse_inputs_are_exact_noops(name: str) -> None:
    """Zero and one active slots leave all caller-owned particle state intact."""
    wp = _require_warp()
    row = EXECUTABLE_ROWS[0]
    fixture = FIXTURES[name]
    for device in _run_on_warp_devices(wp):
        initial, final, _, _, _, _ = _run_public_case(
            row, fixture, n_species=2, max_collisions=1, device=device
        )
        npt.assert_array_equal(final["masses"], initial["masses"])
        npt.assert_array_equal(final["concentration"], initial["concentration"])
        npt.assert_array_equal(final["charge"], initial["charge"])
        npt.assert_array_equal(final["counts"], [0])


@pytest.mark.warp
@pytest.mark.gpu_parity
def test_zero_collision_capacity_is_rejected_at_public_preflight() -> None:
    """The public API rejects zero capacity rather than silently truncating."""
    wp = _require_warp()
    row = next(row for row in EXECUTABLE_ROWS if row.mask == 1)
    fixture = FIXTURES["normal"]
    for device in _run_on_warp_devices(wp):
        with pytest.raises(
            ValueError, match="^max_collisions must be a positive integer"
        ):
            _run_public_case(
                row, fixture, n_species=2, max_collisions=0, device=device
            )


@pytest.mark.warp
@pytest.mark.gpu_parity
def test_two_active_slots_only_merge_the_single_local_pair() -> None:
    """A two-slot case may only select its sole in-box active pair."""
    wp = _require_warp()
    row = next(row for row in EXECUTABLE_ROWS if row.mask == 1)
    fixture = FIXTURES["normal"]
    active_by_box = ((0, 2),)
    for device in _run_on_warp_devices(wp):
        initial, final, _, _, _, _ = _run_public_case(
            row,
            fixture,
            n_species=2,
            max_collisions=1,
            device=device,
            active_by_box=active_by_box,
        )
        _assert_public_invariants(
            initial, final, active_by_box, charge_transfers=True
        )
        if final["counts"][0]:
            npt.assert_array_equal(final["pairs"][0, 0], [0, 2])


@pytest.mark.warp
@pytest.mark.gpu_parity
@pytest.mark.parametrize(
    ("mask", "fixture_name"),
    ((2, "repulsive"), (4, "equal_velocity")),
)
def test_zero_effective_rate_cases_do_not_mutate_particles(
    mask: int, fixture_name: str
) -> None:
    """Full enabled zero-rate mechanisms cleanly return without a merge."""
    wp = _require_warp()
    row = next(row for row in EXECUTABLE_ROWS if row.mask == mask)
    fixture = FIXTURES[fixture_name]
    for device in _run_on_warp_devices(wp):
        initial, final, _, _, _, _ = _run_public_case(
            row, fixture, n_species=1, max_collisions=1, device=device
        )
        npt.assert_array_equal(final["masses"], initial["masses"])
        npt.assert_array_equal(final["concentration"], initial["concentration"])
        npt.assert_array_equal(final["charge"], initial["charge"])
        npt.assert_array_equal(final["counts"], 0)


@pytest.mark.warp
@pytest.mark.gpu_parity
@pytest.mark.parametrize("mask", (8, 10))
def test_turbulent_public_inputs_accept_scalars_and_device_arrays(
    mask: int,
) -> None:
    """Turbulent scalar and caller-owned fp64 arrays preserve invariants."""
    wp = _require_warp()
    row = next(row for row in EXECUTABLE_ROWS if row.mask == mask)
    fixture = FIXTURES["two_box"]
    for device in _run_on_warp_devices(wp):
        for turbulent_arrays in (False, True):
            initial, final, _, _, _, _ = _run_public_case(
                row,
                fixture,
                n_species=2,
                max_collisions=1,
                device=device,
                turbulent_arrays=turbulent_arrays,
            )
            _assert_public_invariants(
                initial, final, fixture.active, charge_transfers=bool(mask & 2)
            )


@pytest.mark.warp
@pytest.mark.gpu_parity
def test_persistent_rng_advances_only_after_explicit_initialization() -> None:
    """Caller-owned RNG state is reset only through initialize_rng=True."""
    wp = _require_warp()
    from particula.gpu.conversion import to_warp_particle_data
    from particula.gpu.kernels.coagulation import coagulation_step_gpu

    fixture = FIXTURES["normal"]
    row = next(row for row in EXECUTABLE_ROWS if row.mask == 1)
    device = "cpu"
    states = wp.zeros((1,), dtype=wp.uint32, device=device)
    active_by_box = ((0, 1, 2, 3),)
    particles = to_warp_particle_data(
        _materialize_public_particles(
            fixture, n_species=1, active_by_box=active_by_box
        ),
        device=device,
    )
    kwargs = dict(
        temperature=298.15,
        pressure=101325.0,
        time_step=1.0,
        volume=particles.volume,
        max_collisions=1,
        rng_seed=73,
        rng_states=states,
        mechanism_config=CoagulationMechanismConfig(mechanisms=row.mechanisms),
    )
    coagulation_step_gpu(particles, initialize_rng=True, **kwargs)
    initialized = states.numpy().copy()
    coagulation_step_gpu(particles, initialize_rng=False, **kwargs)
    advanced = states.numpy().copy()
    assert not np.array_equal(advanced, initialized)
    reset_particles = to_warp_particle_data(
        _materialize_public_particles(
            fixture, n_species=1, active_by_box=active_by_box
        ),
        device=device,
    )
    coagulation_step_gpu(reset_particles, initialize_rng=True, **kwargs)
    reset = states.numpy().copy()
    control = wp.zeros((1,), dtype=wp.uint32, device=device)
    control_particles = to_warp_particle_data(
        _materialize_public_particles(
            fixture, n_species=1, active_by_box=active_by_box
        ),
        device=device,
    )
    coagulation_step_gpu(
        control_particles,
        **{**kwargs, "rng_states": control, "initialize_rng": True},
    )
    npt.assert_array_equal(control.numpy(), initialized)
    npt.assert_array_equal(reset, control.numpy())


def _assert_snapshot_unchanged(
    before: dict[str, np.ndarray], after: dict[str, np.ndarray]
) -> None:
    """Require a rejected public preflight to leave every owned buffer intact."""
    for key in before:
        npt.assert_array_equal(after[key], before[key])


@pytest.mark.warp
@pytest.mark.gpu_parity
@pytest.mark.parametrize("mask", (7, 11, 13, 14))
def test_deferred_masks_are_atomic_before_caller_owned_state_changes(
    mask: int,
) -> None:
    """Deferred configurations leave particles, outputs, and RNG reusable."""
    wp = _require_warp()
    from particula.gpu.conversion import to_warp_particle_data
    from particula.gpu.kernels.coagulation import coagulation_step_gpu

    row = next(row for row in DEFERRED_ROWS if row.mask == mask)
    fixture = FIXTURES["normal"]
    particles = to_warp_particle_data(
        _materialize_public_particles(fixture, n_species=2), device="cpu"
    )
    pairs = wp.full((1, 1, 2), -7, dtype=wp.int32, device="cpu")
    counts = wp.full((1,), -7, dtype=wp.int32, device="cpu")
    states = wp.full((1,), 17, dtype=wp.uint32, device="cpu")
    before = _public_snapshot(particles, pairs, counts, states)
    kwargs: dict[str, Any] = {}
    if mask & 8:
        kwargs = {"turbulent_dissipation": 2.0e-4, "fluid_density": 1.2}
    with pytest.raises(
        ValueError, match="^Additive coagulation execution is deferred\\.$"
    ):
        coagulation_step_gpu(
            particles,
            temperature=298.15,
            pressure=101325.0,
            time_step=1.0,
            collision_pairs=pairs,
            n_collisions=counts,
            rng_states=states,
            initialize_rng=True,
            mechanism_config=CoagulationMechanismConfig(
                mechanisms=row.mechanisms
            ),
            **kwargs,
        )
    _assert_snapshot_unchanged(
        before, _public_snapshot(particles, pairs, counts, states)
    )


@pytest.mark.warp
@pytest.mark.gpu_parity
@pytest.mark.parametrize(
    "failure",
    (
        "charge",
        "sedimentation_mass",
        "sedimentation_concentration",
        "sedimentation_density",
        "turbulent_missing",
        "turbulent_nonpositive",
    ),
)
def test_selected_public_preflight_failures_are_atomic(failure: str) -> None:
    """Enabled-term validation rejects invalid state before output or RNG work."""
    wp = _require_warp()
    from particula.gpu.conversion import to_warp_particle_data
    from particula.gpu.kernels.coagulation import coagulation_step_gpu

    fixture = FIXTURES["normal"]
    cpu_particles = _materialize_public_particles(fixture, n_species=1)
    if failure == "charge":
        cpu_particles.charge[0, 0] = np.nan
        mechanisms = ("charged_hard_sphere",)
        kwargs: dict[str, Any] = {}
        message = "particle charge"
    elif failure == "sedimentation_mass":
        cpu_particles.masses[0, 0, 0] = np.nan
        mechanisms = ("sedimentation_sp2016",)
        kwargs = {}
        message = "sedimentation particle"
    elif failure == "sedimentation_concentration":
        cpu_particles.concentration[0, 0] = np.nan
        mechanisms = ("sedimentation_sp2016",)
        kwargs = {}
        message = "sedimentation particle concentration"
    elif failure == "sedimentation_density":
        cpu_particles.density[0] = 0.0
        mechanisms = ("sedimentation_sp2016",)
        kwargs = {}
        message = "sedimentation particle density"
    else:
        mechanisms = ("turbulent_shear_st1956",)
        kwargs = {
            "turbulent_dissipation": None
            if failure == "turbulent_missing"
            else -1.0,
            "fluid_density": 1.2,
        }
        message = "turbulent_dissipation"
    particles = to_warp_particle_data(cpu_particles, device="cpu")
    pairs = wp.full((1, 1, 2), -7, dtype=wp.int32, device="cpu")
    counts = wp.full((1,), -7, dtype=wp.int32, device="cpu")
    states = wp.full((1,), 17, dtype=wp.uint32, device="cpu")
    before = _public_snapshot(particles, pairs, counts, states)
    with pytest.raises(ValueError, match=message):
        coagulation_step_gpu(
            particles,
            temperature=298.15,
            pressure=101325.0,
            time_step=1.0,
            collision_pairs=pairs,
            n_collisions=counts,
            rng_states=states,
            initialize_rng=True,
            mechanism_config=CoagulationMechanismConfig(mechanisms=mechanisms),
            **kwargs,
        )
    _assert_snapshot_unchanged(
        before, _public_snapshot(particles, pairs, counts, states)
    )
