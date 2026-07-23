"""End-to-end tests for GPU coagulation kernels.

This module covers coagulation execution, contract validation, and test-local
deterministic probes for additive pair rates, majorants, and selector safety.
The probes measure attempted versus accepted collision counts without changing
the public GPU API.
"""

# pyright: reportGeneralTypeIssues=false
# pyright: reportOperatorIssue=false
# pyright: reportArgumentType=false
# pyright: reportAssignmentType=false
# pyright: reportIndexIssue=false

from __future__ import annotations

import ast
import inspect
import itertools
import pickle
import re
from dataclasses import dataclass
from types import SimpleNamespace

# pyright: reportArgumentType=false
# pyright: reportAssignmentType=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportOperatorIssue=false
from typing import Any, cast

import numpy as np
import numpy.testing as npt
import pytest

wp = pytest.importorskip("warp", reason="Warp not installed")
pytestmark = pytest.mark.warp

if wp is not None:
    import particula.gpu.kernels.coagulation as coagulation_module  # noqa: E402
    from particula.dynamics.coagulation.brownian_kernel import (  # noqa: E402
        get_brownian_kernel_via_system_state,
    )
    from particula.gas.environment_data import EnvironmentData  # noqa: E402
    from particula.gas.gas_data import GasData  # noqa: E402
    from particula.gpu.conversion import (  # noqa: E402
        from_warp_particle_data,
        to_warp_environment_data,
        to_warp_particle_data,
    )
    from particula.gpu.dynamics.coagulation_funcs import (  # noqa: E402
        brownian_diffusivity_wp,
        brownian_kernel_pair_wp,
        g_collection_term_wp,
        particle_mean_free_path_wp,
        sedimentation_sp2016_pair_rate_wp,
        turbulent_shear_st1956_pair_rate_wp,
    )
    from particula.gpu.kernels.coagulation import (  # noqa: E402
        BROWNIAN_MECHANISM,
        BROWNIAN_MECHANISM_FLAG,
        CANONICAL_COAGULATION_MECHANISMS,
        CHARGED_HARD_SPHERE_MECHANISM,
        CHARGED_HARD_SPHERE_MECHANISM_FLAG,
        MAX_CHARGED_ACTIVE_PARTICLES_PER_BOX,
        MAX_SCHEDULED_TRIALS_PER_BOX,
        SEDIMENTATION_SP2016_MECHANISM,
        SEDIMENTATION_SP2016_MECHANISM_FLAG,
        TURBULENT_SHEAR_ST1956_MECHANISM,
        TURBULENT_SHEAR_ST1956_MECHANISM_FLAG,
        CoagulationMechanismConfig,
        _bound_scheduled_trials,
        _charged_majorant_from_active_pairs,
        _checked_positive_finite_add,
        _compact_applied_collision_pairs_kernel,
        _ensure_turbulent_input_array,
        _ensure_volume_array,
        _initialize_rng_states,
        _remove_active_pair_by_rank_swap_pop,
        _resolve_active_pair_by_rank,
        _resolve_collision_capacity,
        _safe_acceptance_probability,
        _sanitize_positive_finite,
        _sedimentation_majorant_from_active_pairs,
        _select_active_pair_by_rank,
        _total_majorant,
        _total_pair_rate,
        _turbulent_majorant_from_active_radii,
        _validate_charge_finite_kernel,
        _validate_collision_counts,
        _validate_collision_pairs,
        _validate_device_arrays,
        _validate_device_match,
        _validate_max_collisions,
        _validate_particle_arrays,
        _validate_rng_states,
        _validate_time_step,
        apply_coagulation_kernel,
        brownian_coagulation_kernel,
        coagulation_step_gpu,
        initialize_coagulation_rng_states,
        resolve_coagulation_mechanism_config,
        validate_coagulation_mechanism_capabilities,
    )
    from particula.gpu.properties.gas_properties import (  # noqa: E402
        dynamic_viscosity_wp,
        molecule_mean_free_path_wp,
    )
    from particula.gpu.properties.particle_properties import (  # noqa: E402
        aerodynamic_mobility_wp,
        cunningham_slip_correction_wp,
        knudsen_number_wp,
        mean_thermal_speed_wp,
        particle_radius_from_volume_wp,
    )
    from particula.gpu.tests.cuda_availability import (  # noqa: E402
        cuda_available,
        warp_devices,
    )

    # pyright: reportGeneralTypeIssues=false
    # pyright: reportOperatorIssue=false
    # pyright: reportArgumentType=false
    # pyright: reportAssignmentType=false
    from particula.particles.particle_data import ParticleData  # noqa: E402
    from particula.util import constants  # noqa: E402


BROWNIAN_COAGULATION_MASK_INPUT_INDEX = 28


def test_mechanism_default_matches_explicit_brownian() -> None:
    """Default mechanism configuration matches explicit Brownian selection."""
    default = resolve_coagulation_mechanism_config(CoagulationMechanismConfig())
    explicit = resolve_coagulation_mechanism_config(
        CoagulationMechanismConfig(mechanisms=(BROWNIAN_MECHANISM,))
    )

    assert default == explicit
    assert default.mechanisms == (BROWNIAN_MECHANISM,)
    assert default.distribution_type == "particle_resolved"
    assert default.mask == BROWNIAN_MECHANISM_FLAG


@pytest.mark.parametrize(
    "public_api",
    (
        CoagulationMechanismConfig,
        resolve_coagulation_mechanism_config,
        validate_coagulation_mechanism_capabilities,
    ),
)
def test_configuration_public_apis_retain_public_module_provenance(
    public_api: object,
) -> None:
    """Public configuration APIs retain introspection and pickle compatibility."""
    assert public_api.__module__ == "particula.gpu.kernels.coagulation"  # type: ignore[union-attr]
    assert pickle.loads(pickle.dumps(public_api)) is public_api  # noqa: S301


def test_mechanism_flags_have_documented_numeric_values() -> None:
    """Mechanism flags retain their fixed dispatch bit assignments."""
    assert BROWNIAN_MECHANISM_FLAG == 1
    assert CHARGED_HARD_SPHERE_MECHANISM_FLAG == 2
    assert SEDIMENTATION_SP2016_MECHANISM_FLAG == 4
    assert TURBULENT_SHEAR_ST1956_MECHANISM_FLAG == 8


def test_mechanism_flag_mapping_is_immutable() -> None:
    """Mechanism flag mapping cannot be changed after module initialization."""
    flag_mapping = coagulation_module._COAGULATION_MECHANISM_FLAGS

    assert dict(flag_mapping) == {
        BROWNIAN_MECHANISM: 1,
        CHARGED_HARD_SPHERE_MECHANISM: 2,
        SEDIMENTATION_SP2016_MECHANISM: 4,
        TURBULENT_SHEAR_ST1956_MECHANISM: 8,
    }
    mutable_flag_mapping = cast(Any, flag_mapping)
    with pytest.raises(TypeError):
        mutable_flag_mapping[BROWNIAN_MECHANISM] = 16

    resolved = resolve_coagulation_mechanism_config(
        CoagulationMechanismConfig(mechanisms=(BROWNIAN_MECHANISM,))
    )

    assert resolved.mask == 1


def test_mechanism_canonicalizes_order_and_mask() -> None:
    """Mechanism resolution uses canonical order and fixed flags."""
    resolved = resolve_coagulation_mechanism_config(
        CoagulationMechanismConfig(
            mechanisms=(
                TURBULENT_SHEAR_ST1956_MECHANISM,
                BROWNIAN_MECHANISM,
                SEDIMENTATION_SP2016_MECHANISM,
                CHARGED_HARD_SPHERE_MECHANISM,
            )
        )
    )

    assert resolved.mechanisms == CANONICAL_COAGULATION_MECHANISMS
    assert resolved.mask == 15


@pytest.mark.parametrize(
    ("mechanisms", "expected_calls"),
    [
        ((BROWNIAN_MECHANISM,), ()),
        ((CHARGED_HARD_SPHERE_MECHANISM,), ("charge", "charged")),
        ((SEDIMENTATION_SP2016_MECHANISM,), ("sedimentation",)),
        (
            (TURBULENT_SHEAR_ST1956_MECHANISM,),
            ("turbulent", "turbulent", "turbulent_particle"),
        ),
        (
            (BROWNIAN_MECHANISM, CHARGED_HARD_SPHERE_MECHANISM),
            ("charge", "charged"),
        ),
        (
            (BROWNIAN_MECHANISM, SEDIMENTATION_SP2016_MECHANISM),
            ("sedimentation",),
        ),
        (
            (BROWNIAN_MECHANISM, TURBULENT_SHEAR_ST1956_MECHANISM),
            ("turbulent", "turbulent", "turbulent_particle"),
        ),
        (
            (CHARGED_HARD_SPHERE_MECHANISM, SEDIMENTATION_SP2016_MECHANISM),
            ("charged", "sedimentation"),
        ),
        (
            (CHARGED_HARD_SPHERE_MECHANISM, TURBULENT_SHEAR_ST1956_MECHANISM),
            (
                "turbulent",
                "turbulent",
                "turbulent_particle",
                "charge",
                "charged",
            ),
        ),
        (
            (SEDIMENTATION_SP2016_MECHANISM, TURBULENT_SHEAR_ST1956_MECHANISM),
            ("turbulent", "turbulent", "turbulent_particle", "sedimentation"),
        ),
        (
            CANONICAL_COAGULATION_MECHANISMS,
            (
                "turbulent",
                "turbulent",
                "turbulent_particle",
                "charged",
                "sedimentation",
            ),
        ),
    ],
)
def test_enabled_term_preflight_uses_registered_mask_bits(
    monkeypatch: pytest.MonkeyPatch,
    mechanisms: tuple[str, ...],
    expected_calls: tuple[str, ...],
) -> None:
    """Enabled terms select only their ordered read-only preflight checks."""
    resolved = resolve_coagulation_mechanism_config(
        CoagulationMechanismConfig(mechanisms=mechanisms)
    )
    calls: list[str] = []

    def _record(name: str) -> Any:
        def _validator(*args: Any, **kwargs: Any) -> None:
            calls.append(name)

        return _validator

    monkeypatch.setattr(
        coagulation_module, "_validate_turbulent_input", _record("turbulent")
    )
    monkeypatch.setattr(
        coagulation_module,
        "_validate_turbulent_particle_physics",
        _record("turbulent_particle"),
    )
    monkeypatch.setattr(
        coagulation_module, "_validate_charge_finite", _record("charge")
    )
    monkeypatch.setattr(
        coagulation_module,
        "_validate_charged_particle_physics",
        _record("charged"),
    )
    monkeypatch.setattr(
        coagulation_module,
        "_validate_sedimentation_particle_physics",
        _record("sedimentation"),
    )

    coagulation_module._preflight_enabled_coagulation_terms(
        resolved,
        SimpleNamespace(charge=object()),
        n_boxes=1,
        n_particles=1,
        device=object(),
        turbulent_dissipation=object(),
        fluid_density=object(),
        validate_charge_finite=False,
    )

    assert calls == list(expected_calls)


def test_brownian_preflight_honors_opt_in_finite_charge_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Brownian-only preflight scans charge only when explicitly requested."""
    resolved = resolve_coagulation_mechanism_config(
        CoagulationMechanismConfig()
    )
    calls: list[str] = []
    monkeypatch.setattr(
        coagulation_module,
        "_validate_charge_finite",
        lambda *_: calls.append("charge"),
    )

    coagulation_module._preflight_enabled_coagulation_terms(
        resolved,
        SimpleNamespace(charge=object()),
        n_boxes=1,
        n_particles=1,
        device=object(),
        turbulent_dissipation=object(),
        fluid_density=object(),
        validate_charge_finite=True,
    )

    assert calls == ["charge"]


@pytest.mark.parametrize(
    ("mechanisms", "message"),
    [
        ((), "mechanisms"),
        ("brownian", "mechanisms"),
        (["brownian"], "mechanisms"),
        ((BROWNIAN_MECHANISM, BROWNIAN_MECHANISM), BROWNIAN_MECHANISM),
        (("unknown",), "unknown"),
        ((BROWNIAN_MECHANISM, 1), "mechanisms"),
    ],
)
def test_mechanism_rejects_structural_failures(
    mechanisms: object,
    message: str,
) -> None:
    """Malformed mechanism selections fail structural validation."""
    config = CoagulationMechanismConfig(mechanisms=mechanisms)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=message):
        resolve_coagulation_mechanism_config(config)


def test_mechanism_duplicate_error_uses_first_repeated_identifier() -> None:
    """Duplicate validation reports the first duplicate reached in input order."""
    config = CoagulationMechanismConfig(
        mechanisms=(
            CHARGED_HARD_SPHERE_MECHANISM,
            BROWNIAN_MECHANISM,
            CHARGED_HARD_SPHERE_MECHANISM,
            BROWNIAN_MECHANISM,
        )
    )

    with pytest.raises(
        ValueError,
        match="Duplicate coagulation mechanism 'charged_hard_sphere'",
    ):
        resolve_coagulation_mechanism_config(config)


@pytest.mark.parametrize(
    "distribution_type",
    ["discrete", "continuous_pdf", None, "Particle_resolved"],
)
def test_mechanism_rejects_invalid_distribution_type(
    distribution_type: object,
) -> None:
    """Only exact particle-resolved distribution type is structurally valid."""
    config = CoagulationMechanismConfig(
        distribution_type=cast(str, distribution_type)
    )

    with pytest.raises(ValueError, match="distribution_type"):
        resolve_coagulation_mechanism_config(config)


def test_sedimentation_only_capability_is_accepted() -> None:
    """Exact particle-resolved SP2016 is an executable public mechanism."""
    resolved = resolve_coagulation_mechanism_config(
        CoagulationMechanismConfig(mechanisms=(SEDIMENTATION_SP2016_MECHANISM,))
    )

    assert resolved.mask == SEDIMENTATION_SP2016_MECHANISM_FLAG
    validate_coagulation_mechanism_capabilities(resolved)


def test_mechanism_support_accepts_turbulent_shear_singleton() -> None:
    """The exact turbulent-shear singleton is executable."""
    resolved = resolve_coagulation_mechanism_config(
        CoagulationMechanismConfig(
            mechanisms=(TURBULENT_SHEAR_ST1956_MECHANISM,)
        )
    )

    assert resolved.mask == TURBULENT_SHEAR_ST1956_MECHANISM_FLAG
    validate_coagulation_mechanism_capabilities(resolved)


def test_mechanism_support_accepts_brownian() -> None:
    """Brownian remains an executable particle-resolved mechanism."""
    resolved = resolve_coagulation_mechanism_config(
        CoagulationMechanismConfig(mechanisms=(BROWNIAN_MECHANISM,))
    )

    validate_coagulation_mechanism_capabilities(resolved)


def test_mechanism_support_accepts_charged_only() -> None:
    """Charged hard-sphere is executable only as its own resolved mask."""
    resolved = resolve_coagulation_mechanism_config(
        CoagulationMechanismConfig(mechanisms=(CHARGED_HARD_SPHERE_MECHANISM,))
    )

    validate_coagulation_mechanism_capabilities(resolved)


def test_charged_only_step_runs_and_preserves_output_identity(
    device: str,
) -> None:
    """Charged-only selection uses caller charge and supplied output buffers."""
    particles = _make_particle_data(n_boxes=1, n_particles=3, n_species=2)
    particles.volume[:] = 1.0e-18
    particles.masses[0, :, 0] = [1.0e-21, 2.0e-21, 3.0e-21]
    particles.masses[0, :, 1] = [4.0e-21, 5.0e-21, 6.0e-21]
    particles.charge = np.array([[1.0, -1.0, 0.0]], dtype=np.float64)
    gpu_particles = to_warp_particle_data(particles, device=device)
    pairs = wp.zeros((1, 1, 2), dtype=wp.int32, device=device)
    counts = wp.zeros((1,), dtype=wp.int32, device=device)
    states = wp.zeros((1,), dtype=wp.uint32, device=device)

    result_particles, result_pairs, result_counts = coagulation_step_gpu(
        gpu_particles,
        temperature=298.15,
        pressure=101325.0,
        time_step=1.0,
        max_collisions=1,
        collision_pairs=pairs,
        n_collisions=counts,
        rng_states=states,
        initialize_rng=True,
        mechanism_config=CoagulationMechanismConfig(
            mechanisms=(CHARGED_HARD_SPHERE_MECHANISM,)
        ),
    )

    assert result_particles is gpu_particles
    assert result_pairs is pairs
    assert result_counts is counts
    assert 0 <= int(counts.numpy()[0]) <= 1


def test_sedimentation_public_step_preserves_caller_owned_return_identity(
    device: str,
) -> None:
    """SP2016 accepts finite charge and preserves caller-owned outputs."""
    particles = _make_particle_data(n_boxes=1, n_particles=3, n_species=1)
    particles.volume[:] = 1.0e-18
    particles.masses[0, :, 0] = [1.0e-15, 8.0e-15, 2.7e-14]
    gpu_particles = to_warp_particle_data(particles, device=device)
    pairs = wp.array(
        np.full((1, 1, 2), -1, dtype=np.int32),
        dtype=wp.int32,
        device=device,
    )
    counts = wp.zeros((1,), dtype=wp.int32, device=device)
    states = wp.zeros((1,), dtype=wp.uint32, device=device)

    returned = coagulation_step_gpu(
        gpu_particles,
        temperature=298.15,
        pressure=101325.0,
        time_step=1.0e308,
        max_collisions=1,
        collision_pairs=pairs,
        n_collisions=counts,
        rng_states=states,
        initialize_rng=True,
        mechanism_config=CoagulationMechanismConfig(
            mechanisms=(SEDIMENTATION_SP2016_MECHANISM,)
        ),
    )

    assert returned == (gpu_particles, pairs, counts)
    assert 0 <= int(counts.numpy()[0]) <= 1


def test_sedimentation_preflight_rejects_nonfinite_charge_atomically(
    device: str,
) -> None:
    """SP2016 rejects non-finite charge before mutating caller-owned state."""
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=1)
    particles.charge[0, 0] = np.nan
    gpu_particles = to_warp_particle_data(particles, device=device)
    initial_mass = gpu_particles.masses.numpy().copy()
    initial_concentration = gpu_particles.concentration.numpy().copy()
    initial_charge = gpu_particles.charge.numpy().copy()
    pairs = wp.array([[[7, 7]]], dtype=wp.int32, device=device)
    counts = wp.array([7], dtype=wp.int32, device=device)
    states = wp.array([7], dtype=wp.uint32, device=device)

    with pytest.raises(ValueError, match="particle charge"):
        coagulation_step_gpu(
            gpu_particles,
            temperature=298.15,
            pressure=101325.0,
            time_step=1.0e308,
            max_collisions=1,
            collision_pairs=pairs,
            n_collisions=counts,
            rng_states=states,
            initialize_rng=True,
            mechanism_config=CoagulationMechanismConfig(
                mechanisms=(SEDIMENTATION_SP2016_MECHANISM,)
            ),
        )

    npt.assert_equal(gpu_particles.masses.numpy(), initial_mass)
    npt.assert_equal(gpu_particles.concentration.numpy(), initial_concentration)
    npt.assert_equal(gpu_particles.charge.numpy(), initial_charge)
    npt.assert_array_equal(pairs.numpy(), [[[7, 7]]])
    npt.assert_array_equal(counts.numpy(), [7])
    npt.assert_array_equal(states.numpy(), [7])


def test_sedimentation_public_step_merges_sparse_multibox_particles(
    device: str,
) -> None:
    """SP2016 conserves per-box species mass and excludes inactive gaps."""
    particles = _make_particle_data(n_boxes=2, n_particles=3, n_species=2)
    particles.volume[:] = 1.0e-18
    particles.masses[:] = [
        [[1.0e-15, 2.0e-15], [9.0e-15, 9.0e-15], [8.0e-15, 16.0e-15]],
        [[2.0e-15, 3.0e-15], [9.0e-15, 9.0e-15], [16.0e-15, 24.0e-15]],
    ]
    particles.concentration[:] = [[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]]
    initial_mass = np.sum(particles.masses, axis=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    temperature = wp.array([298.15, 310.0], dtype=wp.float64, device=device)
    pressure = wp.array([101325.0, 95000.0], dtype=wp.float64, device=device)
    pairs = wp.full((2, 1, 2), -1, dtype=wp.int32, device=device)
    counts = wp.zeros((2,), dtype=wp.int32, device=device)

    result_particles, result_pairs, result_counts = coagulation_step_gpu(
        gpu_particles,
        temperature=temperature,
        pressure=pressure,
        time_step=1.0e308,
        max_collisions=1,
        rng_seed=41,
        collision_pairs=pairs,
        n_collisions=counts,
        mechanism_config=CoagulationMechanismConfig(
            mechanisms=(SEDIMENTATION_SP2016_MECHANISM,)
        ),
    )
    result = from_warp_particle_data(result_particles, sync=True)

    assert result_pairs is pairs
    assert result_counts is counts
    npt.assert_array_equal(counts.numpy(), [1, 1])
    npt.assert_array_equal(pairs.numpy(), [[[0, 2]], [[0, 2]]])
    npt.assert_allclose(
        np.sum(result.masses, axis=1), initial_mass, rtol=1.0e-12, atol=0.0
    )
    npt.assert_array_equal(result.concentration[:, 1], [0.0, 0.0])
    npt.assert_array_equal(result.concentration[:, 2], [0.0, 0.0])
    npt.assert_array_equal(result.masses[:, 2], np.zeros((2, 2)))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("masses", np.nan, "sedimentation particle masses"),
        ("masses", np.inf, "sedimentation particle masses"),
        ("masses", -np.inf, "sedimentation particle masses"),
        ("masses", -1.0, "sedimentation particle masses"),
        ("concentration", np.nan, "sedimentation particle concentration"),
        ("concentration", np.inf, "sedimentation particle concentration"),
        ("concentration", -np.inf, "sedimentation particle concentration"),
        ("concentration", -1.0, "sedimentation particle concentration"),
        ("density", np.inf, "sedimentation particle density"),
        ("density", -np.inf, "sedimentation particle density"),
        ("density", 0.0, "sedimentation particle density"),
        ("density", -1.0, "sedimentation particle density"),
        ("density", np.nan, "sedimentation particle density"),
    ],
)
def test_sedimentation_preflight_rejects_invalid_physics(
    device: str,
    field: str,
    value: float,
    message: str,
) -> None:
    """SP2016 input-domain failures are field-specific and pre-mutation."""
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=1)
    getattr(particles, field)[...] = value
    gpu_particles = to_warp_particle_data(particles, device=device)
    pairs = wp.array([[[7, 7]]], dtype=wp.int32, device=device)
    counts = wp.array([7], dtype=wp.int32, device=device)
    states = wp.array([7], dtype=wp.uint32, device=device)

    with pytest.raises(ValueError, match=message):
        coagulation_step_gpu(
            gpu_particles,
            temperature=298.15,
            pressure=101325.0,
            time_step=1.0,
            max_collisions=1,
            collision_pairs=pairs,
            n_collisions=counts,
            rng_states=states,
            initialize_rng=True,
            mechanism_config=CoagulationMechanismConfig(
                mechanisms=(SEDIMENTATION_SP2016_MECHANISM,)
            ),
        )

    npt.assert_array_equal(pairs.numpy(), [[[7, 7]]])
    npt.assert_array_equal(counts.numpy(), [7])
    npt.assert_array_equal(states.numpy(), [7])


@pytest.mark.parametrize(
    ("mass", "density"),
    [
        (np.finfo(np.float64).max, 1000.0),
        (np.nextafter(0.0, 1.0), np.finfo(np.float64).max),
    ],
)
def test_sedimentation_preflight_rejects_invalid_derived_totals_atomically(
    device: str,
    mass: float,
    density: float,
) -> None:
    """SP2016 rejects overflowing or underflowing active derived totals."""
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=2)
    particles.masses[:] = mass
    particles.density[:] = density
    initial_particles = particles.copy()
    gpu_particles = to_warp_particle_data(particles, device=device)
    pairs = wp.array([[[7, 7]]], dtype=wp.int32, device=device)
    counts = wp.array([7], dtype=wp.int32, device=device)
    states = wp.array([7], dtype=wp.uint32, device=device)

    with pytest.raises(ValueError, match="total mass and volume"):
        coagulation_step_gpu(
            gpu_particles,
            temperature=298.15,
            pressure=101325.0,
            time_step=1.0,
            max_collisions=1,
            collision_pairs=pairs,
            n_collisions=counts,
            rng_states=states,
            initialize_rng=True,
            mechanism_config=CoagulationMechanismConfig(
                mechanisms=(SEDIMENTATION_SP2016_MECHANISM,)
            ),
        )

    _assert_particles_unchanged(gpu_particles, initial_particles)
    npt.assert_array_equal(pairs.numpy(), [[[7, 7]]])
    npt.assert_array_equal(counts.numpy(), [7])
    npt.assert_array_equal(states.numpy(), [7])


def test_sedimentation_preflight_accepts_zero_mass_and_concentration(
    device: str,
) -> None:
    """SP2016 permits zero-valued inactive particle state."""
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=1)
    particles.masses[:] = 0.0
    particles.concentration[:] = 0.0
    gpu_particles = to_warp_particle_data(particles, device=device)

    returned, pairs, counts = coagulation_step_gpu(
        gpu_particles,
        temperature=298.15,
        pressure=101325.0,
        time_step=1.0,
        max_collisions=1,
        mechanism_config=CoagulationMechanismConfig(
            mechanisms=(SEDIMENTATION_SP2016_MECHANISM,)
        ),
    )

    assert returned is gpu_particles
    assert int(counts.numpy()[0]) == 0
    assert pairs.shape == (1, 1, 2)


@pytest.mark.parametrize("density", [0.0, -1.0, np.nan, np.inf])
def test_sedimentation_empty_particles_reject_invalid_density_atomically(
    device: str,
    density: float,
) -> None:
    """SP2016 validates shared density even when particle capacity is empty."""
    particles = _make_particle_data(n_boxes=1, n_particles=0, n_species=1)
    particles.density[:] = density
    gpu_particles = to_warp_particle_data(particles, device=device)
    pairs = wp.array([[[7, 7]]], dtype=wp.int32, device=device)
    counts = wp.array([7], dtype=wp.int32, device=device)
    states = wp.array([7], dtype=wp.uint32, device=device)

    with pytest.raises(ValueError, match="sedimentation particle density"):
        coagulation_step_gpu(
            gpu_particles,
            temperature=298.15,
            pressure=101325.0,
            time_step=1.0,
            max_collisions=1,
            collision_pairs=pairs,
            n_collisions=counts,
            rng_states=states,
            initialize_rng=True,
            mechanism_config=CoagulationMechanismConfig(
                mechanisms=(SEDIMENTATION_SP2016_MECHANISM,)
            ),
        )

    npt.assert_array_equal(pairs.numpy(), [[[7, 7]]])
    npt.assert_array_equal(counts.numpy(), [7])
    npt.assert_array_equal(states.numpy(), [7])


def test_sedimentation_rejects_unequal_positive_concentrations_atomically(
    device: str,
) -> None:
    """SP2016 rejects unrepresentable concentration-weighted merges."""
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=1)
    particles.concentration[:] = [[1.0, 2.0]]
    initial_particles = particles.copy()
    gpu_particles = to_warp_particle_data(particles, device=device)
    pairs = wp.array([[[7, 7]]], dtype=wp.int32, device=device)
    counts = wp.array([7], dtype=wp.int32, device=device)
    states = wp.array([7], dtype=wp.uint32, device=device)

    with pytest.raises(ValueError, match="concentrations must be equal"):
        coagulation_step_gpu(
            gpu_particles,
            temperature=298.15,
            pressure=101325.0,
            time_step=1.0e308,
            max_collisions=1,
            collision_pairs=pairs,
            n_collisions=counts,
            rng_states=states,
            initialize_rng=True,
            mechanism_config=CoagulationMechanismConfig(
                mechanisms=(SEDIMENTATION_SP2016_MECHANISM,)
            ),
        )

    _assert_particles_unchanged(gpu_particles, initial_particles)
    npt.assert_array_equal(pairs.numpy(), [[[7, 7]]])
    npt.assert_array_equal(counts.numpy(), [7])
    npt.assert_array_equal(states.numpy(), [7])


def test_sedimentation_active_cap_accepts_boundary_and_rejects_overflow(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    """SP2016 caps exact-pair work before caller-owned state changes."""
    monkeypatch.setattr(
        coagulation_module, "MAX_SEDIMENTATION_ACTIVE_PARTICLES_PER_BOX", 2
    )
    boundary = _make_particle_data(n_boxes=1, n_particles=2, n_species=1)
    boundary_gpu = to_warp_particle_data(boundary, device=device)
    _, _, boundary_counts = coagulation_step_gpu(
        boundary_gpu,
        temperature=298.15,
        pressure=101325.0,
        time_step=0.0,
        max_collisions=1,
        mechanism_config=CoagulationMechanismConfig(
            mechanisms=(SEDIMENTATION_SP2016_MECHANISM,)
        ),
    )
    npt.assert_array_equal(boundary_counts.numpy(), [0])

    overflow = _make_particle_data(n_boxes=1, n_particles=3, n_species=1)
    overflow_initial = overflow.copy()
    overflow_gpu = to_warp_particle_data(overflow, device=device)
    pairs = wp.array([[[7, 7]]], dtype=wp.int32, device=device)
    counts = wp.array([7], dtype=wp.int32, device=device)
    states = wp.array([7], dtype=wp.uint32, device=device)
    with pytest.raises(
        ValueError, match="MAX_SEDIMENTATION_ACTIVE_PARTICLES_PER_BOX"
    ):
        coagulation_step_gpu(
            overflow_gpu,
            temperature=298.15,
            pressure=101325.0,
            time_step=1.0,
            max_collisions=1,
            collision_pairs=pairs,
            n_collisions=counts,
            rng_states=states,
            initialize_rng=True,
            mechanism_config=CoagulationMechanismConfig(
                mechanisms=(SEDIMENTATION_SP2016_MECHANISM,)
            ),
        )

    _assert_particles_unchanged(overflow_gpu, overflow_initial)
    npt.assert_array_equal(pairs.numpy(), [[[7, 7]]])
    npt.assert_array_equal(counts.numpy(), [7])
    npt.assert_array_equal(states.numpy(), [7])


@pytest.mark.parametrize(
    "mechanisms",
    [
        (BROWNIAN_MECHANISM, CHARGED_HARD_SPHERE_MECHANISM),
        (CHARGED_HARD_SPHERE_MECHANISM, BROWNIAN_MECHANISM),
    ],
)
def test_combined_step_normalizes_order_and_preserves_output_identity(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    mechanisms: tuple[str, str],
) -> None:
    """Combined execution uses one selector/apply path and caller buffers."""
    particles = _make_particle_data(n_boxes=1, n_particles=3, n_species=2)
    particles.volume[:] = 1.0e-18
    particles.masses[0, :, 0] = [1.0e-21, 2.0e-21, 3.0e-21]
    particles.masses[0, :, 1] = [4.0e-21, 5.0e-21, 6.0e-21]
    particles.charge = np.array([[1.0, -1.0, 0.0]], dtype=np.float64)
    gpu_particles = to_warp_particle_data(particles, device=device)
    pairs = wp.zeros((1, 1, 2), dtype=wp.int32, device=device)
    counts = wp.zeros((1,), dtype=wp.int32, device=device)
    states = wp.zeros((1,), dtype=wp.uint32, device=device)
    launches: list[str] = []
    original_launch = coagulation_module.wp.launch

    def _track_combined_launch(kernel: Any, *args: Any, **kwargs: Any) -> Any:
        key = getattr(kernel, "key", "")
        if key in {"brownian_coagulation_kernel", "apply_coagulation_kernel"}:
            launches.append(key)
        return original_launch(kernel, *args, **kwargs)

    monkeypatch.setattr(coagulation_module.wp, "launch", _track_combined_launch)

    result_particles, result_pairs, result_counts = coagulation_step_gpu(
        gpu_particles,
        temperature=298.15,
        pressure=101325.0,
        time_step=1.0,
        max_collisions=1,
        collision_pairs=pairs,
        n_collisions=counts,
        rng_states=states,
        initialize_rng=True,
        mechanism_config=CoagulationMechanismConfig(mechanisms=mechanisms),
    )

    assert result_particles is gpu_particles
    assert result_pairs is pairs
    assert result_counts is counts
    assert 0 <= int(counts.numpy()[0]) <= 1
    assert launches == [
        "brownian_coagulation_kernel",
        "apply_coagulation_kernel",
    ]


@pytest.mark.parametrize(
    "mechanisms",
    [
        (BROWNIAN_MECHANISM, CHARGED_HARD_SPHERE_MECHANISM),
        (SEDIMENTATION_SP2016_MECHANISM, BROWNIAN_MECHANISM),
        (SEDIMENTATION_SP2016_MECHANISM, CHARGED_HARD_SPHERE_MECHANISM),
        (TURBULENT_SHEAR_ST1956_MECHANISM, BROWNIAN_MECHANISM),
        (TURBULENT_SHEAR_ST1956_MECHANISM, CHARGED_HARD_SPHERE_MECHANISM),
        (TURBULENT_SHEAR_ST1956_MECHANISM, SEDIMENTATION_SP2016_MECHANISM),
        (
            TURBULENT_SHEAR_ST1956_MECHANISM,
            SEDIMENTATION_SP2016_MECHANISM,
            CHARGED_HARD_SPHERE_MECHANISM,
            BROWNIAN_MECHANISM,
        ),
    ],
)
def test_approved_additive_masks_use_public_merge_path(
    device: str,
    mechanisms: tuple[str, ...],
) -> None:
    """Approved masks preserve public output and conservation contracts."""
    particles = _make_particle_data(n_boxes=2, n_particles=3, n_species=2)
    particles.volume[:] = 1.0e-18
    particles.concentration[:] = [[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]]
    particles.masses[:] = [
        [[1.0e-15, 2.0e-15], [9.0e-15, 9.0e-15], [3.0e-15, 4.0e-15]],
        [[2.0e-15, 3.0e-15], [9.0e-15, 9.0e-15], [4.0e-15, 5.0e-15]],
    ]
    particles.charge[:] = [[2.0, 11.0, -5.0], [-3.0, 13.0, 7.0]]
    initial_mass = np.sum(particles.masses, axis=1)
    initial_charge = np.sum(particles.charge, axis=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    pairs = wp.full((2, 1, 2), -1, dtype=wp.int32, device=device)
    counts = wp.zeros((2,), dtype=wp.int32, device=device)
    states = wp.zeros((2,), dtype=wp.uint32, device=device)

    returned = coagulation_step_gpu(
        gpu_particles,
        temperature=298.15,
        pressure=101325.0,
        time_step=1.0e308,
        max_collisions=1,
        collision_pairs=pairs,
        n_collisions=counts,
        rng_states=states,
        initialize_rng=True,
        mechanism_config=CoagulationMechanismConfig(mechanisms=mechanisms),
        turbulent_dissipation=1.0,
        fluid_density=1.2,
    )
    result = from_warp_particle_data(gpu_particles, sync=True)

    assert returned == (gpu_particles, pairs, counts)
    npt.assert_array_equal(counts.numpy(), [1, 1])
    npt.assert_array_equal(pairs.numpy(), [[[0, 2]], [[0, 2]]])
    npt.assert_allclose(
        np.sum(result.masses, axis=1), initial_mass, rtol=1.0e-12, atol=0.0
    )
    npt.assert_allclose(
        np.sum(result.charge, axis=1), initial_charge, rtol=0.0, atol=0.0
    )
    npt.assert_array_equal(result.masses[:, 1], particles.masses[:, 1])
    npt.assert_array_equal(result.charge[:, 1], particles.charge[:, 1])
    npt.assert_array_equal(result.masses[:, 2], np.zeros((2, 2)))
    npt.assert_array_equal(result.charge[:, 2], [0.0, 0.0])


@pytest.mark.stochastic
@pytest.mark.parametrize(
    "mechanisms",
    [
        (BROWNIAN_MECHANISM, SEDIMENTATION_SP2016_MECHANISM),
        (CHARGED_HARD_SPHERE_MECHANISM, SEDIMENTATION_SP2016_MECHANISM),
        (BROWNIAN_MECHANISM, TURBULENT_SHEAR_ST1956_MECHANISM),
        (CHARGED_HARD_SPHERE_MECHANISM, TURBULENT_SHEAR_ST1956_MECHANISM),
        (SEDIMENTATION_SP2016_MECHANISM, TURBULENT_SHEAR_ST1956_MECHANISM),
        CANONICAL_COAGULATION_MECHANISMS,
    ],
)
def test_additive_masks_public_mult_pair_counts_include_every_enabled_rate(
    device: str,
    mechanisms: tuple[str, ...],
) -> None:
    """Public additive masks match independent multi-pair rate statistics.

    One public launch uses independent per-box RNG streams to keep this fast.
    The finite timestep is chosen from the exhaustive initial pair-rate oracle.
    Each enabled component is large enough that dropping it moves the expected
    one-collision probability beyond the documented sampling uncertainty.
    """
    temperature = 298.15
    pressure = 101325.0
    fluid_density = 1000.0
    turbulent_dissipation = 2.0e-4
    observations = 2048
    radii = np.array([0.7e-6, 0.9e-6, 1.1e-6, 1.3e-6])
    masses = ((4.0 / 3.0) * np.pi * radii**3 * 1000.0).astype(np.float64)
    particles = ParticleData(
        masses=np.tile(masses, (observations, 1))[..., np.newaxis],
        concentration=np.ones((observations, radii.size), dtype=np.float64),
        charge=np.tile(np.array([0.0, 1.0, -1.0, 2.0]), (observations, 1)),
        density=np.array([1000.0]),
        volume=np.ones(observations, dtype=np.float64),
    )
    component_rates = _additive_component_rate_matrices(
        radii=radii,
        masses=masses,
        charges=particles.charge[0],
        temperature=temperature,
        pressure=pressure,
        turbulent_dissipation=turbulent_dissipation,
        fluid_density=fluid_density,
    )
    enabled_bits = tuple(
        bit
        for bit, mechanism in zip(
            (1, 2, 4, 8),
            CANONICAL_COAGULATION_MECHANISMS,
            strict=True,
        )
        if mechanism in mechanisms
    )
    total_rates = sum(component_rates[bit] for bit in enabled_bits)
    total_rate = float(np.sum(np.triu(total_rates, k=1)))
    majorant = float(np.max(total_rates))
    assert total_rate > 0.0
    assert majorant > 0.0
    for bit in enabled_bits:
        assert float(np.sum(np.triu(component_rates[bit], k=1))) > 0.0

    # Set 2.5 expected candidate draws before acceptance.  This keeps the
    # event probability finite and makes a missing component discriminating.
    volume = majorant * 6.0 / 2.5
    expected_probability = _one_collision_probability(
        total_rate=total_rate,
        majorant=majorant,
        volume=volume,
    )
    for missing_bit in enabled_bits:
        without_component = total_rates - component_rates[missing_bit]
        missing_probability = _one_collision_probability(
            total_rate=float(np.sum(np.triu(without_component, k=1))),
            majorant=float(np.max(without_component)),
            volume=volume,
        )
        standard_error = np.sqrt(
            expected_probability * (1.0 - expected_probability) / observations
        )
        assert (
            abs(expected_probability - missing_probability)
            > 6.0 * standard_error
        )

    gpu_particles = to_warp_particle_data(particles, device=device)
    initial_mass = np.sum(particles.masses, axis=1)
    initial_charge = np.sum(particles.charge, axis=1)
    collision_pairs = wp.full(
        (observations, 1, 2), -1, dtype=wp.int32, device=device
    )
    collision_counts = wp.zeros((observations,), dtype=wp.int32, device=device)
    _, returned_pairs, returned_counts = coagulation_step_gpu(
        gpu_particles,
        temperature=temperature,
        pressure=pressure,
        time_step=1.0,
        volume=volume,
        max_collisions=1,
        rng_seed=731,
        collision_pairs=collision_pairs,
        n_collisions=collision_counts,
        mechanism_config=CoagulationMechanismConfig(mechanisms=mechanisms),
        turbulent_dissipation=turbulent_dissipation,
        fluid_density=fluid_density,
    )
    result = from_warp_particle_data(gpu_particles, sync=True)
    observed_counts = np.asarray(returned_counts.numpy(), dtype=np.int64)
    observed_pairs = np.asarray(returned_pairs.numpy(), dtype=np.int64)
    observed_probability = float(np.mean(observed_counts))
    sigma = np.sqrt(
        expected_probability * (1.0 - expected_probability) / observations
    )

    assert returned_pairs is collision_pairs
    assert returned_counts is collision_counts
    assert (
        abs(observed_probability - expected_probability) <= 4.0 * sigma + 0.02
    )
    assert np.all((observed_counts == 0) | (observed_counts == 1))
    accepted_pairs = observed_pairs[observed_counts.astype(bool), 0]
    assert np.all(accepted_pairs[:, 0] >= 0)
    assert np.all(accepted_pairs[:, 0] < accepted_pairs[:, 1])
    assert np.all(accepted_pairs[:, 1] < radii.size)
    npt.assert_allclose(np.sum(result.masses, axis=1), initial_mass, rtol=1e-12)
    npt.assert_allclose(np.sum(result.charge, axis=1), initial_charge, rtol=0.0)


@pytest.mark.parametrize(
    "mechanisms",
    [
        (CHARGED_HARD_SPHERE_MECHANISM,),
        (BROWNIAN_MECHANISM, CHARGED_HARD_SPHERE_MECHANISM),
    ],
)
def test_charged_modes_merge_and_conserve_mass_and_charge(
    device: str,
    mechanisms: tuple[str, ...],
) -> None:
    """Charged-containing paths apply a merge without inventory drift."""
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=2)
    particles.volume[:] = 1.0e-18
    particles.masses[0] = [[1.0e-21, 2.0e-21], [3.0e-21, 4.0e-21]]
    particles.charge[:] = [[2.0, -5.0]]
    initial_mass = np.sum(particles.masses, axis=1)
    initial_charge = np.sum(particles.charge, axis=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    pairs = wp.zeros((1, 1, 2), dtype=wp.int32, device=device)
    counts = wp.zeros((1,), dtype=wp.int32, device=device)

    result_particles, result_pairs, result_counts = coagulation_step_gpu(
        gpu_particles,
        temperature=298.15,
        pressure=101325.0,
        time_step=1.0,
        max_collisions=1,
        rng_seed=41,
        collision_pairs=pairs,
        n_collisions=counts,
        mechanism_config=CoagulationMechanismConfig(mechanisms=mechanisms),
    )
    result = from_warp_particle_data(result_particles, sync=True)

    assert result_pairs is pairs
    assert result_counts is counts
    npt.assert_array_equal(counts.numpy(), np.array([1], dtype=np.int32))
    npt.assert_array_equal(pairs.numpy(), np.array([[[0, 1]]], dtype=np.int32))
    npt.assert_allclose(np.sum(result.masses, axis=1), initial_mass, rtol=1e-12)
    npt.assert_allclose(
        np.sum(result.charge, axis=1), initial_charge, rtol=1e-12
    )
    npt.assert_allclose(result.masses[0, 0], [4.0e-21, 6.0e-21], rtol=1e-12)
    npt.assert_array_equal(result.masses[0, 1], np.zeros(2))
    assert result.charge[0, 0] == -3.0
    assert result.charge[0, 1] == 0.0
    assert result.concentration[0, 1] == 0.0


def test_combined_multibox_multispecies_conserves_mass_and_charge(
    device: str,
) -> None:
    """Combined mode conserves per-box species mass and signed charge."""
    particles = _make_particle_data(n_boxes=2, n_particles=3, n_species=2)
    particles.volume[:] = 1.0e-18
    particles.masses[:] = [
        [[1.0e-21, 2.0e-21], [3.0e-21, 4.0e-21], [0.0, 0.0]],
        [[5.0e-21, 6.0e-21], [7.0e-21, 8.0e-21], [0.0, 0.0]],
    ]
    particles.concentration[:, -1] = 1.0
    particles.charge[:] = [[2.0, -5.0, 7.0], [-3.0, 4.0, -8.0]]
    initial_mass = np.sum(particles.masses, axis=1)
    initial_charge = np.sum(particles.charge, axis=1)
    gpu_particles = to_warp_particle_data(particles, device=device)

    result_particles, _, _ = coagulation_step_gpu(
        gpu_particles,
        temperature=298.15,
        pressure=101325.0,
        time_step=1.0,
        max_collisions=1,
        rng_seed=41,
        mechanism_config=CoagulationMechanismConfig(
            mechanisms=(BROWNIAN_MECHANISM, CHARGED_HARD_SPHERE_MECHANISM)
        ),
    )
    result = from_warp_particle_data(result_particles, sync=True)

    npt.assert_allclose(np.sum(result.masses, axis=1), initial_mass, rtol=1e-12)
    npt.assert_allclose(
        np.sum(result.charge, axis=1), initial_charge, rtol=1e-12
    )


def test_combined_requested_orders_produce_equivalent_outcomes(
    device: str,
) -> None:
    """Reversed combined configurations produce the same seeded outcome."""
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=2)
    particles.volume[:] = 1.0e-18
    particles.masses[0] = [[1.0e-21, 2.0e-21], [3.0e-21, 4.0e-21]]
    particles.charge[:] = [[2.0, -5.0]]
    initial_mass = np.sum(particles.masses, axis=1)
    initial_charge = np.sum(particles.charge, axis=1)
    outcomes = []

    for mechanisms in (
        (BROWNIAN_MECHANISM, CHARGED_HARD_SPHERE_MECHANISM),
        (CHARGED_HARD_SPHERE_MECHANISM, BROWNIAN_MECHANISM),
    ):
        gpu_particles = to_warp_particle_data(particles.copy(), device=device)
        pairs = wp.full((1, 1, 2), -7, dtype=wp.int32, device=device)
        counts = wp.full((1,), -7, dtype=wp.int32, device=device)
        states = wp.zeros((1,), dtype=wp.uint32, device=device)
        result, _, _ = coagulation_step_gpu(
            gpu_particles,
            temperature=298.15,
            pressure=101325.0,
            time_step=1.0,
            max_collisions=1,
            collision_pairs=pairs,
            n_collisions=counts,
            rng_states=states,
            initialize_rng=True,
            mechanism_config=CoagulationMechanismConfig(mechanisms=mechanisms),
        )
        outcomes.append(
            (pairs.numpy(), counts.numpy(), from_warp_particle_data(result))
        )

    first_pairs, first_counts, first_particles = outcomes[0]
    second_pairs, second_counts, second_particles = outcomes[1]
    npt.assert_array_equal(first_pairs, second_pairs)
    npt.assert_array_equal(first_counts, second_counts)
    npt.assert_array_equal(first_particles.masses, second_particles.masses)
    npt.assert_array_equal(first_particles.charge, second_particles.charge)
    npt.assert_allclose(np.sum(first_particles.masses, axis=1), initial_mass)
    npt.assert_allclose(np.sum(first_particles.charge, axis=1), initial_charge)


def test_charged_only_multispecies_total_mass_regression(
    device: str,
) -> None:
    """Charged-only selection uses summed species mass, not a component."""
    particles = _make_particle_data(n_boxes=1, n_particles=3, n_species=2)
    particles.volume[:] = 1.0e-18
    particles.masses[0] = [
        [1.0e-21, 9.0e-21],
        [4.0e-21, 6.0e-21],
        [2.0e-21, 1.0e-20],
    ]
    particles.charge[:] = [[1.0, -1.0, 2.0]]
    total_masses = np.sum(particles.masses[0], axis=1)
    radii = np.cbrt(3.0 * total_masses / (4.0 * np.pi * particles.density[0]))

    expected_rate = _charged_hard_sphere_oracle(
        radii[0],
        radii[1],
        total_masses[0],
        total_masses[1],
        particles.charge[0, 0],
        particles.charge[0, 1],
        298.15,
        101325.0,
    )
    assert expected_rate > 0.0

    gpu_particles = to_warp_particle_data(particles, device=device)
    collision_pairs = wp.array(
        np.full((1, 1, 2), -7, dtype=np.int32),
        dtype=wp.int32,
        device=device,
    )
    collision_counts = wp.array(
        np.array([-7], dtype=np.int32), dtype=wp.int32, device=device
    )
    rng_states = wp.array(
        np.array([17], dtype=np.uint32), dtype=wp.uint32, device=device
    )
    total_masses_buffer = wp.zeros((1, 3), dtype=wp.float64, device=device)
    radii_buffer = wp.zeros((1, 3), dtype=wp.float64, device=device)
    diffusivities_buffer = wp.zeros((1, 3), dtype=wp.float64, device=device)
    g_terms_buffer = wp.zeros((1, 3), dtype=wp.float64, device=device)
    speeds_buffer = wp.zeros((1, 3), dtype=wp.float64, device=device)
    settling_velocities_buffer = wp.zeros(
        (1, 3), dtype=wp.float64, device=device
    )
    active_indices = wp.zeros((1, 3), dtype=wp.int32, device=device)

    wp.launch(
        brownian_coagulation_kernel,
        dim=(1,),
        inputs=[
            gpu_particles.masses,
            gpu_particles.concentration,
            gpu_particles.density,
            gpu_particles.volume,
            wp.array([298.15], dtype=wp.float64, device=device),
            wp.array([101325.0], dtype=wp.float64, device=device),
            wp.float64(constants.GAS_CONSTANT),
            wp.float64(constants.BOLTZMANN_CONSTANT),
            wp.float64(constants.MOLECULAR_WEIGHT_AIR),
            wp.float64(constants.REF_VISCOSITY_AIR_STP),
            wp.float64(constants.REF_TEMPERATURE_STP),
            wp.float64(constants.SUTHERLAND_CONSTANT),
            wp.float64(constants.ELEMENTARY_CHARGE_VALUE),
            wp.float64(constants.ELECTRIC_PERMITTIVITY),
            wp.float64(0.0),
            radii_buffer,
            diffusivities_buffer,
            g_terms_buffer,
            speeds_buffer,
            settling_velocities_buffer,
            wp.zeros((1,), dtype=wp.float64, device=device),
            wp.zeros((1,), dtype=wp.float64, device=device),
            total_masses_buffer,
            gpu_particles.charge,
            active_indices,
            collision_pairs,
            collision_counts,
            rng_states,
            wp.int32(CHARGED_HARD_SPHERE_MECHANISM_FLAG),
            wp.int32(1),
        ],
        device=device,
    )
    wp.synchronize()

    npt.assert_allclose(
        np.asarray(total_masses_buffer.numpy())[0],
        total_masses,
        rtol=1.0e-12,
        atol=0.0,
    )

    compact = np.array([[0, 1, 2]], dtype=np.int32)
    direct, dispatched = _run_charged_majorant_probes(
        device,
        compact,
        np.array([3], dtype=np.int32),
        np.asarray(radii_buffer.numpy()),
        np.asarray(total_masses_buffer.numpy()),
        np.asarray(gpu_particles.charge.numpy()),
        np.array([298.15], dtype=np.float64),
        np.array([101325.0], dtype=np.float64),
    )
    expected_majorant = _charged_majorant_oracle(
        compact[0],
        3,
        np.asarray(radii_buffer.numpy())[0],
        np.asarray(total_masses_buffer.numpy())[0],
        np.asarray(gpu_particles.charge.numpy())[0],
        298.15,
        101325.0,
    )
    npt.assert_allclose(direct, [expected_majorant], rtol=1.0e-11, atol=0.0)
    npt.assert_allclose(dispatched, [expected_majorant], rtol=1.0e-11, atol=0.0)

    initial_collision_pairs = np.asarray(collision_pairs.numpy()).copy()

    result_particles, result_pairs, result_counts = coagulation_step_gpu(
        gpu_particles,
        temperature=298.15,
        pressure=101325.0,
        time_step=0.0,
        max_collisions=1,
        collision_pairs=collision_pairs,
        n_collisions=collision_counts,
        rng_states=rng_states,
        initialize_rng=True,
        mechanism_config=CoagulationMechanismConfig(
            mechanisms=(CHARGED_HARD_SPHERE_MECHANISM,)
        ),
    )

    assert result_particles is gpu_particles
    assert result_pairs is collision_pairs
    assert result_counts is collision_counts
    npt.assert_array_equal(np.asarray(collision_counts.numpy()), np.array([0]))
    # A successful no-op call writes the count but leaves unused pair slots
    # caller-owned and unchanged.
    npt.assert_array_equal(
        np.asarray(collision_pairs.numpy()), initial_collision_pairs
    )


@pytest.mark.stochastic
@pytest.mark.parametrize("charges", [(0.0, 0.0), (3.0, 3.0), (3.0, -3.0)])
def test_charged_only_fresh_seed_counts_match_pair_rate_oracle(
    device: str,
    charges: tuple[float, float],
) -> None:
    """Fresh-seed charged counts stay within a four-sigma oracle bound.

    The two-active-particle fixture makes the exhaustive majorant equal the
    independent pair rate. Volume selects a 0.2 scheduled-trial probability,
    so each fresh seed is an independent Bernoulli observation rather than an
    exact stochastic replay assertion.
    """
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=1)
    particles.masses[:] = 1.0e-21
    particles.charge[:] = [charges]
    radius = (3.0 * 1.0e-21 / (4.0 * np.pi * particles.density[0])) ** (
        1.0 / 3.0
    )
    rate = _charged_hard_sphere_oracle(
        radius,
        radius,
        1.0e-21,
        1.0e-21,
        charges[0],
        charges[1],
        298.15,
        101325.0,
    )
    assert rate > 0.0
    trial_probability = 0.2
    volume = rate / trial_probability
    observations = 64
    accepted = 0
    for seed in range(observations):
        gpu_particles = to_warp_particle_data(particles, device=device)
        _, _, counts = coagulation_step_gpu(
            gpu_particles,
            temperature=298.15,
            pressure=101325.0,
            time_step=1.0,
            volume=volume,
            max_collisions=1,
            rng_seed=seed,
            mechanism_config=CoagulationMechanismConfig(
                mechanisms=(CHARGED_HARD_SPHERE_MECHANISM,)
            ),
        )
        accepted += int(counts.numpy()[0])

    expected = observations * trial_probability
    sigma = np.sqrt(
        observations * trial_probability * (1.0 - trial_probability)
    )
    assert abs(accepted - expected) <= 4.0 * sigma + 1.0


@pytest.mark.stochastic
def test_combined_fresh_seed_counts_match_independent_rate_oracle(
    device: str,
) -> None:
    """Combined counts stay within a four-sigma independent-rate bound.

    With two active particles, the production majorant equals the sum of the
    independently calculated Brownian and charged pair rates. The chosen volume
    yields a 0.2 scheduled-trial probability, making fresh seeds independent
    Bernoulli observations rather than exact stochastic replay assertions.
    """
    temperature = 298.15
    pressure = 101325.0
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=1)
    particles.masses[:] = 1.0e-21
    particles.charge[:] = [[0.0, 0.0]]
    radius = (3.0 * 1.0e-21 / (4.0 * np.pi * particles.density[0])) ** (
        1.0 / 3.0
    )
    brownian_rate_matrix = np.asarray(
        get_brownian_kernel_via_system_state(
            particle_radius=np.array([radius, radius]),
            particle_mass=np.array([1.0e-21, 1.0e-21]),
            temperature=temperature,
            pressure=pressure,
        ),
        dtype=np.float64,
    )
    brownian_rate = float(brownian_rate_matrix[0, 1])
    charged_rate = _charged_hard_sphere_oracle(
        radius, radius, 1.0e-21, 1.0e-21, 0.0, 0.0, temperature, pressure
    )
    combined_rate = brownian_rate + charged_rate
    assert combined_rate > 0.0
    trial_probability = 0.2
    observations = 64
    accepted = 0
    for seed in range(observations):
        gpu_particles = to_warp_particle_data(particles, device=device)
        _, _, counts = coagulation_step_gpu(
            gpu_particles,
            temperature=temperature,
            pressure=pressure,
            time_step=1.0,
            volume=combined_rate / trial_probability,
            max_collisions=1,
            rng_seed=seed,
            mechanism_config=CoagulationMechanismConfig(
                mechanisms=(BROWNIAN_MECHANISM, CHARGED_HARD_SPHERE_MECHANISM)
            ),
        )
        accepted += int(counts.numpy()[0])

    expected = observations * trial_probability
    sigma = np.sqrt(
        observations * trial_probability * (1.0 - trial_probability)
    )
    assert abs(accepted - expected) <= 4.0 * sigma + 1.0, (
        f"accepted={accepted}, expected={expected}, sigma={sigma}"
    )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("masses", np.nan),
        ("masses", -1.0e-21),
        ("density", np.nan),
        ("density", 0.0),
        ("concentration", np.nan),
        ("concentration", -1.0),
    ],
)
@pytest.mark.parametrize(
    "mechanisms",
    [
        (CHARGED_HARD_SPHERE_MECHANISM,),
        (BROWNIAN_MECHANISM, CHARGED_HARD_SPHERE_MECHANISM),
    ],
)
def test_charged_modes_reject_invalid_particle_physics_before_mutation(
    device: str,
    field: str,
    invalid_value: float,
    mechanisms: tuple[str, ...],
) -> None:
    """Charged-containing modes reject hostile inputs before side effects."""
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=1)
    if field == "density":
        particles.density[0] = invalid_value
    else:
        getattr(particles, field)[0, 0] = invalid_value
    initial_particles = particles.copy()
    gpu_particles = to_warp_particle_data(particles, device=device)
    pairs = wp.array(np.full((1, 1, 2), -7), dtype=wp.int32, device=device)
    counts = wp.array(np.array([-7]), dtype=wp.int32, device=device)
    states = wp.array(np.array([17]), dtype=wp.uint32, device=device)

    with pytest.raises(ValueError, match="charged particle"):
        coagulation_step_gpu(
            gpu_particles,
            temperature=298.15,
            pressure=101325.0,
            time_step=1.0,
            max_collisions=1,
            collision_pairs=pairs,
            n_collisions=counts,
            rng_states=states,
            initialize_rng=True,
            mechanism_config=CoagulationMechanismConfig(mechanisms=mechanisms),
        )

    _assert_particles_unchanged(gpu_particles, initial_particles)
    npt.assert_array_equal(pairs.numpy(), np.full((1, 1, 2), -7))
    npt.assert_array_equal(counts.numpy(), np.array([-7], dtype=np.int32))
    npt.assert_array_equal(states.numpy(), np.array([17], dtype=np.uint32))


@pytest.mark.parametrize(
    "mechanisms",
    [
        (CHARGED_HARD_SPHERE_MECHANISM,),
        (BROWNIAN_MECHANISM, CHARGED_HARD_SPHERE_MECHANISM),
    ],
)
def test_charged_preflight_excludes_volume_underflow_slots(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    mechanisms: tuple[str, ...],
) -> None:
    """Selector-ineligible subnormal slots do not consume the charged cap."""
    monkeypatch.setattr(
        coagulation_module, "MAX_CHARGED_ACTIVE_PARTICLES_PER_BOX", 2
    )
    particles = _make_particle_data(n_boxes=1, n_particles=3, n_species=1)
    particles.masses[0, :, 0] = [1.0e-21, 1.0e-21, np.nextafter(0.0, 1.0)]
    gpu_particles = to_warp_particle_data(particles, device=device)

    _, _, counts = coagulation_step_gpu(
        gpu_particles,
        temperature=298.15,
        pressure=101325.0,
        time_step=0.0,
        max_collisions=1,
        mechanism_config=CoagulationMechanismConfig(mechanisms=mechanisms),
    )

    npt.assert_array_equal(counts.numpy(), np.array([0], dtype=np.int32))


@pytest.mark.parametrize(
    "mechanisms",
    [
        (CHARGED_HARD_SPHERE_MECHANISM,),
        (BROWNIAN_MECHANISM, CHARGED_HARD_SPHERE_MECHANISM),
    ],
)
def test_charged_modes_reject_over_limit_active_particles_before_mutation(
    device: str,
    mechanisms: tuple[str, ...],
) -> None:
    """Charged modes bound exact-majorant work before output or RNG mutation."""
    particles = _make_particle_data(
        n_boxes=1,
        n_particles=MAX_CHARGED_ACTIVE_PARTICLES_PER_BOX + 1,
        n_species=1,
    )
    initial_particles = particles.copy()
    gpu_particles = to_warp_particle_data(particles, device=device)
    pairs = wp.array(np.full((1, 1, 2), -7), dtype=wp.int32, device=device)
    counts = wp.array(np.array([-7]), dtype=wp.int32, device=device)
    states = wp.array(np.array([17]), dtype=wp.uint32, device=device)

    with pytest.raises(
        ValueError, match="MAX_CHARGED_ACTIVE_PARTICLES_PER_BOX"
    ):
        coagulation_step_gpu(
            gpu_particles,
            temperature=298.15,
            pressure=101325.0,
            time_step=1.0,
            max_collisions=1,
            collision_pairs=pairs,
            n_collisions=counts,
            rng_states=states,
            initialize_rng=True,
            mechanism_config=CoagulationMechanismConfig(mechanisms=mechanisms),
        )

    _assert_particles_unchanged(gpu_particles, initial_particles)
    npt.assert_array_equal(pairs.numpy(), np.full((1, 1, 2), -7))
    npt.assert_array_equal(counts.numpy(), np.array([-7], dtype=np.int32))
    npt.assert_array_equal(states.numpy(), np.array([17], dtype=np.uint32))


@pytest.mark.parametrize(
    "mechanisms",
    [
        (CHARGED_HARD_SPHERE_MECHANISM,),
        (BROWNIAN_MECHANISM, CHARGED_HARD_SPHERE_MECHANISM),
    ],
)
def test_charged_cap_ignores_positive_concentration_ineligible_slots(
    device: str,
    mechanisms: tuple[str, ...],
) -> None:
    """The charged cap counts only particles the selector can choose."""
    particles = _make_particle_data(
        n_boxes=1,
        n_particles=MAX_CHARGED_ACTIVE_PARTICLES_PER_BOX + 1,
        n_species=1,
    )
    particles.masses[0, -1, :] = 0.0
    gpu_particles = to_warp_particle_data(particles, device=device)
    pairs = wp.array(np.full((1, 1, 2), -7), dtype=wp.int32, device=device)
    counts = wp.array(np.array([-7]), dtype=wp.int32, device=device)
    states = wp.array(np.array([17]), dtype=wp.uint32, device=device)

    _, result_pairs, result_counts = coagulation_step_gpu(
        gpu_particles,
        temperature=298.15,
        pressure=101325.0,
        time_step=0.0,
        max_collisions=1,
        collision_pairs=pairs,
        n_collisions=counts,
        rng_states=states,
        initialize_rng=True,
        mechanism_config=CoagulationMechanismConfig(mechanisms=mechanisms),
    )

    assert result_pairs is pairs
    assert result_counts is counts
    npt.assert_array_equal(counts.numpy(), np.array([0], dtype=np.int32))


def test_mechanism_support_accepts_combined_brownian_and_charged() -> None:
    """Canonical Brownian-plus-charged configuration is executable."""
    resolved = resolve_coagulation_mechanism_config(
        CoagulationMechanismConfig(
            mechanisms=(BROWNIAN_MECHANISM, CHARGED_HARD_SPHERE_MECHANISM)
        )
    )

    validate_coagulation_mechanism_capabilities(resolved)


def _warp_kernel(function):
    """Decorate kernels only when Warp is available."""
    if wp is None:
        return function
    return wp.kernel(function)


def _available_warp_devices() -> list[str]:
    """Return collection-safe Warp device params."""
    if wp is None:
        return ["cpu"]
    return warp_devices(wp)


def _charged_hard_sphere_oracle(
    radius_i: float,
    radius_j: float,
    mass_i: float,
    mass_j: float,
    charge_i: float,
    charge_j: float,
    temperature: float,
    pressure: float,
) -> float:
    """Independently reproduce the guarded charged hard-sphere pair rate."""
    values = np.array(
        [
            radius_i,
            radius_j,
            mass_i,
            mass_j,
            charge_i,
            charge_j,
            temperature,
            pressure,
            constants.BOLTZMANN_CONSTANT,
            constants.ELEMENTARY_CHARGE_VALUE,
            constants.ELECTRIC_PERMITTIVITY,
            constants.GAS_CONSTANT,
            constants.MOLECULAR_WEIGHT_AIR,
            constants.REF_VISCOSITY_AIR_STP,
            constants.REF_TEMPERATURE_STP,
            constants.SUTHERLAND_CONSTANT,
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)) or np.any(
        values[[0, 1, 2, 3, 6, 7]] <= 0
    ):
        return 0.0
    pi_value = np.pi
    with np.errstate(all="ignore"):
        viscosity = (
            constants.REF_VISCOSITY_AIR_STP
            * (temperature / constants.REF_TEMPERATURE_STP) ** 1.5
            * (constants.REF_TEMPERATURE_STP + constants.SUTHERLAND_CONSTANT)
            / (temperature + constants.SUTHERLAND_CONSTANT)
        )
        mean_free_path = (2.0 * viscosity / pressure) / np.sqrt(
            8.0
            * constants.MOLECULAR_WEIGHT_AIR
            / (pi_value * constants.GAS_CONSTANT * temperature)
        )
        kn_i, kn_j = mean_free_path / radius_i, mean_free_path / radius_j
        slip_i = 1.0 + kn_i * (1.257 + 0.4 * np.exp(-1.1 / kn_i))
        slip_j = 1.0 + kn_j * (1.257 + 0.4 * np.exp(-1.1 / kn_j))
        friction_i = 6.0 * pi_value * viscosity * radius_i / slip_i
        friction_j = 6.0 * pi_value * viscosity * radius_j / slip_j
        radius_sum = radius_i + radius_j
        potential = -(
            charge_i * charge_j * constants.ELEMENTARY_CHARGE_VALUE**2
        ) / (
            4.0
            * pi_value
            * constants.ELECTRIC_PERMITTIVITY
            * radius_sum
            * constants.BOLTZMANN_CONSTANT
            * temperature
        )
        potential = np.clip(potential, -200.0, 200.0)
        kinetic = 1.0 + potential if potential >= 0.0 else np.exp(potential)
        continuum = (
            1.0 if potential == 0.0 else potential / (1.0 - np.exp(-potential))
        )
        if abs(potential) < 1.0e-5 and potential != 0.0:
            continuum = 1.0 + potential / 2.0 + potential**2 / 12.0
        reduced_mass = mass_i * mass_j / (mass_i + mass_j)
        reduced_friction = friction_i * friction_j / (friction_i + friction_j)
        knudsen = (
            np.sqrt(constants.BOLTZMANN_CONSTANT * temperature * reduced_mass)
            / reduced_friction
        ) / (radius_sum * continuum / kinetic)
        numerator = (
            4.0 * pi_value * knudsen**2
            + 25.836 * knudsen**3
            + np.sqrt(8.0 * pi_value) * 11.211 * knudsen**4
        )
        denominator = (
            1.0 + 3.502 * knudsen + 7.211 * knudsen**2 + 11.211 * knudsen**3
        )
        result = (
            numerator
            / denominator
            * reduced_friction
            * radius_sum**3
            * kinetic**2
            / (reduced_mass * continuum)
        )
    return float(result) if np.isfinite(result) and result > 0.0 else 0.0


def _additive_component_rate_matrices(
    *,
    radii: np.ndarray,
    masses: np.ndarray,
    charges: np.ndarray,
    temperature: float,
    pressure: float,
    turbulent_dissipation: float,
    fluid_density: float,
) -> dict[int, np.ndarray]:
    """Calculate independent rate matrices for every public component."""
    brownian = np.asarray(
        get_brownian_kernel_via_system_state(
            particle_radius=radii,
            particle_mass=masses,
            temperature=temperature,
            pressure=pressure,
        ),
        dtype=np.float64,
    )
    charged = np.zeros_like(brownian)
    sedimentation = np.zeros_like(brownian)
    turbulent = np.zeros_like(brownian)
    viscosity = (
        constants.REF_VISCOSITY_AIR_STP
        * (temperature / constants.REF_TEMPERATURE_STP) ** 1.5
        * (constants.REF_TEMPERATURE_STP + constants.SUTHERLAND_CONSTANT)
        / (temperature + constants.SUTHERLAND_CONSTANT)
    )
    kinematic_viscosity = viscosity / fluid_density
    velocities = np.array(
        [_settling_velocity_oracle(radius, 1000.0) for radius in radii]
    )
    for first, second in itertools.combinations(range(radii.size), 2):
        charged[first, second] = _charged_hard_sphere_oracle(
            radii[first],
            radii[second],
            masses[first],
            masses[second],
            charges[first],
            charges[second],
            temperature,
            pressure,
        )
        sedimentation[first, second] = _sedimentation_sp2016_oracle(
            radii[first],
            radii[second],
            velocities[first],
            velocities[second],
        )
        turbulent[first, second] = (
            np.sqrt(
                np.pi * turbulent_dissipation / (120.0 * kinematic_viscosity)
            )
            * (2.0 * (radii[first] + radii[second])) ** 3
        )
    return {
        BROWNIAN_MECHANISM_FLAG: brownian,
        CHARGED_HARD_SPHERE_MECHANISM_FLAG: charged,
        SEDIMENTATION_SP2016_MECHANISM_FLAG: sedimentation,
        TURBULENT_SHEAR_ST1956_MECHANISM_FLAG: turbulent,
    }


def _one_collision_probability(
    *,
    total_rate: float,
    majorant: float,
    volume: float,
) -> float:
    """Return the exact pre-merge chance of one accepted public trial."""
    expected_trials = majorant * 6.0 / volume
    scheduled_trials = int(np.floor(expected_trials))
    remainder = expected_trials - scheduled_trials
    acceptance_probability = total_rate / (6.0 * majorant)
    no_collision = (1.0 - acceptance_probability) ** scheduled_trials
    no_collision *= 1.0 - remainder * acceptance_probability
    return 1.0 - no_collision


def _charged_majorant_oracle(
    active_indices: np.ndarray,
    active_count: int,
    radii: np.ndarray,
    masses: np.ndarray,
    charges: np.ndarray,
    temperature: float,
    pressure: float,
) -> float:
    """Return the independent exhaustive compact-active charged maximum."""
    candidates = (
        _charged_hard_sphere_oracle(
            radii[first],
            radii[second],
            masses[first],
            masses[second],
            charges[first],
            charges[second],
            temperature,
            pressure,
        )
        for first, second in itertools.combinations(
            active_indices[:active_count], 2
        )
    )
    return max(candidates, default=0.0)


def _run_charged_majorant_probes(
    device: str,
    compact: np.ndarray,
    active_counts: np.ndarray,
    radii: np.ndarray,
    masses: np.ndarray,
    charges: np.ndarray,
    temperature: np.ndarray,
    pressure: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Upload explicit fp64 fixtures and return synchronized probe results."""
    n_boxes = compact.shape[0]
    direct = wp.zeros((n_boxes,), dtype=wp.float64, device=device)
    dispatched = wp.zeros((n_boxes,), dtype=wp.float64, device=device)
    wp.launch(
        _charged_majorant_probe_kernel,
        dim=n_boxes,
        inputs=[
            wp.array(compact, dtype=wp.int32, device=device),
            wp.array(active_counts, dtype=wp.int32, device=device),
            wp.array(radii, dtype=wp.float64, device=device),
            wp.array(masses, dtype=wp.float64, device=device),
            wp.array(charges, dtype=wp.float64, device=device),
            wp.array(temperature, dtype=wp.float64, device=device),
            wp.array(pressure, dtype=wp.float64, device=device),
            wp.int32(CHARGED_HARD_SPHERE_MECHANISM_FLAG),
            direct,
            dispatched,
        ],
        device=device,
    )
    wp.synchronize_device(device)
    return np.asarray(direct.numpy()), np.asarray(dispatched.numpy())


def _sedimentation_sp2016_oracle(
    radius_i: float,
    radius_j: float,
    velocity_i: float,
    velocity_j: float,
) -> float:
    """Return an independent guarded SP2016 sedimentation pair rate."""
    values = np.array(
        [radius_i, radius_j, velocity_i, velocity_j], dtype=np.float64
    )
    if (
        not np.all(np.isfinite(values))
        or np.any(values[:2] <= 0.0)
        or np.any(values[2:] < 0.0)
    ):
        return 0.0
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        rate = np.pi * (radius_i + radius_j) ** 2 * abs(velocity_i - velocity_j)
    return float(rate) if np.isfinite(rate) and rate > 0.0 else 0.0


def _sedimentation_majorant_oracle(
    active_indices: np.ndarray,
    active_count: int,
    radii: np.ndarray,
    velocities: np.ndarray,
) -> float:
    """Return the independent exhaustive compact-active SP2016 maximum."""
    candidates = (
        _sedimentation_sp2016_oracle(
            radii[first], radii[second], velocities[first], velocities[second]
        )
        for first, second in itertools.combinations(
            active_indices[:active_count], 2
        )
    )
    return max(candidates, default=0.0)


def _settling_velocity_oracle(radius: float, density: float) -> float:
    """Return an independent Stokes/Cunningham velocity for the test fixture."""
    temperature = 298.15
    pressure = 101325.0
    with np.errstate(all="ignore"):
        viscosity = (
            constants.REF_VISCOSITY_AIR_STP
            * (temperature / constants.REF_TEMPERATURE_STP) ** 1.5
            * (constants.REF_TEMPERATURE_STP + constants.SUTHERLAND_CONSTANT)
            / (temperature + constants.SUTHERLAND_CONSTANT)
        )
        mean_free_path = (2.0 * viscosity / pressure) / np.sqrt(
            8.0
            * constants.MOLECULAR_WEIGHT_AIR
            / (np.pi * constants.GAS_CONSTANT * temperature)
        )
        knudsen = mean_free_path / radius
        slip = 1.0 + knudsen * (1.257 + 0.4 * np.exp(-1.1 / knudsen))
        velocity = (
            2.0
            * radius**2
            * density
            * slip
            * constants.STANDARD_GRAVITY
            / (9.0 * viscosity)
        )
    return float(velocity) if np.isfinite(velocity) and velocity > 0.0 else 0.0


@_warp_kernel
def _sedimentation_pair_rate_probe_kernel(
    radii: Any,
    settling_velocities: Any,
    direct: Any,
    dispatched: Any,
) -> None:
    """Probe the P1 rate and fixed-mask pair-rate dispatcher."""
    direct[0] = sedimentation_sp2016_pair_rate_wp(
        radii[0], radii[1], settling_velocities[0], settling_velocities[1]
    )
    dispatched[0] = _total_pair_rate(
        wp.int32(SEDIMENTATION_SP2016_MECHANISM_FLAG),
        radii[0],
        radii[1],
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        settling_velocities[0],
        settling_velocities[1],
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
    )


@_warp_kernel
def _sedimentation_majorant_probe_kernel(
    active_indices: Any,
    active_counts: Any,
    radii: Any,
    settling_velocities: Any,
    total_masses: Any,
    charges: Any,
    temperature: Any,
    pressure: Any,
    direct: Any,
    dispatched: Any,
) -> None:
    """Probe direct and fixed-mask dispatched sedimentation majorants."""
    box_idx = wp.tid()
    active_count = active_counts[box_idx]
    direct[box_idx] = _sedimentation_majorant_from_active_pairs(
        active_indices,
        box_idx,
        active_count,
        radii,
        settling_velocities,
    )
    dispatched[box_idx] = _total_majorant(
        wp.int32(SEDIMENTATION_SP2016_MECHANISM_FLAG),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        active_indices,
        box_idx,
        active_count,
        radii,
        settling_velocities,
        wp.float64(0.0),
        wp.float64(0.0),
        total_masses,
        charges,
        temperature,
        pressure,
        wp.float64(1.0),
        wp.float64(1.0),
        wp.float64(1.0),
        wp.float64(1.0),
        wp.float64(1.0),
        wp.float64(1.0),
        wp.float64(1.0),
        wp.float64(1.0),
    )


def _run_sedimentation_majorant_probes(
    device: str,
    compact: np.ndarray,
    active_counts: np.ndarray,
    radii: np.ndarray,
    velocities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run direct and dispatched SP2016 majorant probes on explicit fixtures."""
    direct = wp.zeros((compact.shape[0],), dtype=wp.float64, device=device)
    dispatched = wp.zeros((compact.shape[0],), dtype=wp.float64, device=device)
    wp.launch(
        _sedimentation_majorant_probe_kernel,
        dim=compact.shape[0],
        inputs=[
            wp.array(compact, dtype=wp.int32, device=device),
            wp.array(active_counts, dtype=wp.int32, device=device),
            wp.array(radii, dtype=wp.float64, device=device),
            wp.array(velocities, dtype=wp.float64, device=device),
            wp.zeros(radii.shape, dtype=wp.float64, device=device),
            wp.zeros(radii.shape, dtype=wp.float64, device=device),
            wp.ones((compact.shape[0],), dtype=wp.float64, device=device),
            wp.ones((compact.shape[0],), dtype=wp.float64, device=device),
            direct,
            dispatched,
        ],
        device=device,
    )
    wp.synchronize_device(device)
    return np.asarray(direct.numpy()), np.asarray(dispatched.numpy())


_INT32_MAX = np.iinfo(np.int32).max


@dataclass(frozen=True)
class _AttemptDiagnostics:
    """Seeded mirrored-versus-production coagulation diagnostics."""

    scheduled_trial_counts: np.ndarray
    executed_trial_counts: np.ndarray
    accepted_counts: np.ndarray
    production_counts: np.ndarray
    diagnostic_pairs: np.ndarray
    production_pairs: np.ndarray
    diagnostic_masses: np.ndarray
    production_masses: np.ndarray
    diagnostic_concentration: np.ndarray
    production_concentration: np.ndarray
    diagnostic_rng_states: np.ndarray
    production_rng_states: np.ndarray


@dataclass(frozen=True)
class _CoagulationStepResult:
    """Snapshot of one seeded coagulation step from a fresh particle state."""

    collision_counts: np.ndarray
    collision_pairs: np.ndarray
    result_particles: ParticleData
    rng_states: np.ndarray | None


@dataclass(frozen=True)
class _ExpectedCollisionStatistics:
    """Brownian repeated-run reference statistics for active particle pairs."""

    expected_mean: float
    expected_sigma: float
    active_pair_count: int


@_warp_kernel
def _selector_guard_regression_kernel(
    active_indices: Any,
    resolved_pairs: Any,
    unresolved_attempts: Any,
) -> None:
    """Verify unresolved rank selection does not truncate later attempts."""
    box_idx = wp.tid()
    invalid_pair = _resolve_active_pair_by_rank(
        active_indices,
        box_idx,
        active_indices.shape[1],
        wp.int32(0),
        wp.int32(5),
    )
    if invalid_pair[0] < wp.int32(0) or invalid_pair[1] < wp.int32(0):
        unresolved_attempts[box_idx] = wp.int32(1)

    valid_pair = _resolve_active_pair_by_rank(
        active_indices,
        box_idx,
        active_indices.shape[1],
        wp.int32(0),
        wp.int32(2),
    )
    resolved_pairs[box_idx, 0] = valid_pair[0]
    resolved_pairs[box_idx, 1] = valid_pair[1]


@_warp_kernel
def _bound_scheduled_trials_probe_kernel(
    expected_trials: Any,
    bounded_trials: Any,
) -> None:
    """Probe the scheduled-trial clamp helper from a Warp kernel."""
    box_idx = wp.tid()
    bounded_trials[box_idx] = _bound_scheduled_trials(expected_trials[box_idx])


@_warp_kernel
def _additive_helper_probe_kernel(
    mechanism_mask: Any,
    values: Any,
    pair_rate: Any,
    majorant: Any,
    sanitized: Any,
    active_indices: Any,
    radii: Any,
    total_masses: Any,
    charges: Any,
    temperature: Any,
    pressure: Any,
) -> None:
    """Probe fixed-mask additive helpers and finite-term sanitization."""
    pair_rate[0] = _total_pair_rate(
        mechanism_mask,
        values[0],
        values[1],
        values[2],
        values[3],
        values[4],
        values[5],
        values[6],
        values[7],
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        total_masses[0, 0],
        total_masses[0, 1],
        charges[0, 0],
        charges[0, 1],
        temperature[0],
        pressure[0],
        wp.float64(constants.BOLTZMANN_CONSTANT),
        wp.float64(constants.ELEMENTARY_CHARGE_VALUE),
        wp.float64(constants.ELECTRIC_PERMITTIVITY),
        wp.float64(constants.GAS_CONSTANT),
        wp.float64(constants.MOLECULAR_WEIGHT_AIR),
        wp.float64(constants.REF_VISCOSITY_AIR_STP),
        wp.float64(constants.REF_TEMPERATURE_STP),
        wp.float64(constants.SUTHERLAND_CONSTANT),
    )
    majorant[0] = _total_majorant(
        mechanism_mask,
        values[0],
        values[1],
        values[2],
        values[3],
        values[4],
        values[5],
        values[6],
        values[7],
        active_indices,
        wp.int32(0),
        wp.int32(0),
        radii,
        radii,
        wp.float64(0.0),
        wp.float64(0.0),
        total_masses,
        charges,
        temperature,
        pressure,
        wp.float64(constants.BOLTZMANN_CONSTANT),
        wp.float64(constants.ELEMENTARY_CHARGE_VALUE),
        wp.float64(constants.ELECTRIC_PERMITTIVITY),
        wp.float64(constants.GAS_CONSTANT),
        wp.float64(constants.MOLECULAR_WEIGHT_AIR),
        wp.float64(constants.REF_VISCOSITY_AIR_STP),
        wp.float64(constants.REF_TEMPERATURE_STP),
        wp.float64(constants.SUTHERLAND_CONSTANT),
    )
    sanitized[0] = _sanitize_positive_finite(values[8])


@_warp_kernel
def _all_mask_additive_probe_kernel(
    mechanism_mask: Any,
    values: Any,
    velocities: Any,
    active_indices: Any,
    radii: Any,
    total_masses: Any,
    charges: Any,
    temperature: Any,
    pressure: Any,
    turbulent_dissipation: Any,
    kinematic_viscosity: Any,
    pair_rate: Any,
    majorant: Any,
) -> None:
    """Probe all recognized additive masks against one active pair and majorant."""
    pair_rate[0] = _total_pair_rate(
        mechanism_mask,
        radii[0, 0],
        radii[0, 1],
        values[0],
        values[1],
        values[2],
        values[3],
        values[4],
        values[5],
        velocities[0, 0],
        velocities[0, 1],
        turbulent_dissipation[0],
        kinematic_viscosity[0],
        total_masses[0, 0],
        total_masses[0, 1],
        charges[0, 0],
        charges[0, 1],
        temperature[0],
        pressure[0],
        wp.float64(constants.BOLTZMANN_CONSTANT),
        wp.float64(constants.ELEMENTARY_CHARGE_VALUE),
        wp.float64(constants.ELECTRIC_PERMITTIVITY),
        wp.float64(constants.GAS_CONSTANT),
        wp.float64(constants.MOLECULAR_WEIGHT_AIR),
        wp.float64(constants.REF_VISCOSITY_AIR_STP),
        wp.float64(constants.REF_TEMPERATURE_STP),
        wp.float64(constants.SUTHERLAND_CONSTANT),
    )
    majorant[0] = _total_majorant(
        mechanism_mask,
        radii[0, 0],
        radii[0, 1],
        values[0],
        values[1],
        values[2],
        values[3],
        values[4],
        values[5],
        active_indices,
        wp.int32(0),
        wp.int32(2),
        radii,
        velocities,
        turbulent_dissipation[0],
        kinematic_viscosity[0],
        total_masses,
        charges,
        temperature,
        pressure,
        wp.float64(constants.BOLTZMANN_CONSTANT),
        wp.float64(constants.ELEMENTARY_CHARGE_VALUE),
        wp.float64(constants.ELECTRIC_PERMITTIVITY),
        wp.float64(constants.GAS_CONSTANT),
        wp.float64(constants.MOLECULAR_WEIGHT_AIR),
        wp.float64(constants.REF_VISCOSITY_AIR_STP),
        wp.float64(constants.REF_TEMPERATURE_STP),
        wp.float64(constants.SUTHERLAND_CONSTANT),
    )


@_warp_kernel
def _charged_majorant_probe_kernel(
    active_indices: Any,
    active_counts: Any,
    radii: Any,
    total_masses: Any,
    charges: Any,
    temperature: Any,
    pressure: Any,
    mechanism_mask: Any,
    majorants: Any,
    dispatched_majorants: Any,
) -> None:
    """Probe direct and dispatched charged majorants one box per thread."""
    box_idx = wp.tid()
    active_count = active_counts[box_idx]
    direct = _charged_majorant_from_active_pairs(
        active_indices,
        box_idx,
        active_count,
        radii,
        total_masses,
        charges,
        temperature,
        pressure,
        wp.float64(constants.BOLTZMANN_CONSTANT),
        wp.float64(constants.ELEMENTARY_CHARGE_VALUE),
        wp.float64(constants.ELECTRIC_PERMITTIVITY),
        wp.float64(constants.GAS_CONSTANT),
        wp.float64(constants.MOLECULAR_WEIGHT_AIR),
        wp.float64(constants.REF_VISCOSITY_AIR_STP),
        wp.float64(constants.REF_TEMPERATURE_STP),
        wp.float64(constants.SUTHERLAND_CONSTANT),
    )
    majorants[box_idx] = direct
    dispatched_majorants[box_idx] = _total_majorant(
        mechanism_mask,
        wp.float64(1.0e-8),
        wp.float64(5.0e-8),
        wp.float64(1.0e-9),
        wp.float64(2.0e-10),
        wp.float64(2.0e-8),
        wp.float64(4.0e-8),
        wp.float64(0.2),
        wp.float64(0.1),
        active_indices,
        box_idx,
        active_count,
        radii,
        radii,
        wp.float64(0.0),
        wp.float64(0.0),
        total_masses,
        charges,
        temperature,
        pressure,
        wp.float64(constants.BOLTZMANN_CONSTANT),
        wp.float64(constants.ELEMENTARY_CHARGE_VALUE),
        wp.float64(constants.ELECTRIC_PERMITTIVITY),
        wp.float64(constants.GAS_CONSTANT),
        wp.float64(constants.MOLECULAR_WEIGHT_AIR),
        wp.float64(constants.REF_VISCOSITY_AIR_STP),
        wp.float64(constants.REF_TEMPERATURE_STP),
        wp.float64(constants.SUTHERLAND_CONSTANT),
    )


@_warp_kernel
def _synthetic_total_sampling_probe_kernel(
    rate_terms: Any,
    majorant_terms: Any,
    pair_sentinel: Any,
    totals: Any,
    candidate_count: Any,
    acceptance_draw_count: Any,
    accepted_count: Any,
) -> None:
    """Exercise total-rate sampling guards with test-only synthetic terms."""
    total_rate = _checked_positive_finite_add(rate_terms[0], rate_terms[1])
    total_majorant = _checked_positive_finite_add(
        majorant_terms[0], majorant_terms[1]
    )
    totals[0] = total_rate
    totals[1] = total_majorant

    if not (wp.isfinite(total_majorant) and total_majorant > wp.float64(0.0)):
        return

    candidate_count[0] = wp.int32(1)
    acceptance_probability = _safe_acceptance_probability(
        total_rate, total_majorant
    )
    if acceptance_probability <= wp.float64(0.0):
        return

    # Exactly one acceptance draw occurs for each otherwise valid candidate.
    acceptance_draw_count[0] = wp.int32(1)
    state = wp.rand_init(wp.int32(17))
    if wp.randf(state) < acceptance_probability:
        pair_sentinel[0] = wp.int32(0)
        pair_sentinel[1] = wp.int32(1)
        accepted_count[0] = wp.int32(1)


@_warp_kernel
def _resolve_active_pair_probe_kernel(
    active_flags: Any,
    resolved_pairs: Any,
    rank_i: Any,
    adjusted_rank_j: Any,
) -> None:
    """Probe active-rank resolution for one-box deterministic checks."""
    box_idx = wp.tid()
    pair = _resolve_active_pair_by_rank(
        active_flags,
        box_idx,
        active_flags.shape[1],
        rank_i,
        adjusted_rank_j,
    )
    resolved_pairs[box_idx, 0] = pair[0]
    resolved_pairs[box_idx, 1] = pair[1]


@_warp_kernel
def _select_active_pair_probe_kernel(
    active_indices: Any,
    resolved_pairs: Any,
    rank_i: Any,
    adjusted_rank_j: Any,
) -> None:
    """Probe direct active-pair lookup from a compact active-index buffer."""
    box_idx = wp.tid()
    pair = _select_active_pair_by_rank(
        active_indices,
        box_idx,
        rank_i,
        adjusted_rank_j,
    )
    resolved_pairs[box_idx, 0] = pair[0]
    resolved_pairs[box_idx, 1] = pair[1]


@_warp_kernel
def _remove_active_pair_probe_kernel(
    active_indices: Any,
    updated_counts: Any,
    rank_i: Any,
    adjusted_rank_j: Any,
) -> None:
    """Probe swap-pop active-pair removal for deterministic checks."""
    box_idx = wp.tid()
    active_count = wp.int32(0)
    for particle_idx in range(active_indices.shape[1]):
        if active_indices[box_idx, particle_idx] >= wp.int32(0):
            active_count += wp.int32(1)

    updated_counts[box_idx] = _remove_active_pair_by_rank_swap_pop(
        active_indices,
        box_idx,
        active_count,
        rank_i,
        adjusted_rank_j,
    )


@pytest.fixture(params=_available_warp_devices())
def device(request) -> str:
    """Provide available Warp devices for testing."""
    return request.param


@pytest.mark.parametrize(
    "mechanism_mask",
    [
        0,
        CHARGED_HARD_SPHERE_MECHANISM_FLAG,
        SEDIMENTATION_SP2016_MECHANISM_FLAG,
        TURBULENT_SHEAR_ST1956_MECHANISM_FLAG,
        BROWNIAN_MECHANISM_FLAG,
        BROWNIAN_MECHANISM_FLAG | CHARGED_HARD_SPHERE_MECHANISM_FLAG,
        BROWNIAN_MECHANISM_FLAG | SEDIMENTATION_SP2016_MECHANISM_FLAG,
        BROWNIAN_MECHANISM_FLAG | TURBULENT_SHEAR_ST1956_MECHANISM_FLAG,
    ],
)
def test_additive_helpers_ignore_reserved_bits(
    device: str,
    mechanism_mask: int,
) -> None:
    """Fixed-mask helpers include only enabled, finite component contributions."""
    values = np.array(
        [1.0e-8, 5.0e-8, 1.0e-9, 2.0e-10, 2.0e-8, 4.0e-8, 0.2, 0.1, 3.0],
        dtype=np.float64,
    )
    values_wp = wp.array(values, dtype=wp.float64, device=device)
    pair_rate = wp.zeros((1,), dtype=wp.float64, device=device)
    majorant = wp.zeros((1,), dtype=wp.float64, device=device)
    sanitized = wp.zeros((1,), dtype=wp.float64, device=device)
    active_indices = wp.array(
        np.array([[-1]], dtype=np.int32), dtype=wp.int32, device=device
    )
    radii = wp.zeros((1, 1), dtype=wp.float64, device=device)
    total_masses = wp.array(
        np.array([[1.0e-23, 3.0e-22]], dtype=np.float64),
        dtype=wp.float64,
        device=device,
    )
    charges = wp.array(
        np.array([[1.0, -2.0]], dtype=np.float64),
        dtype=wp.float64,
        device=device,
    )
    temperature = wp.array(
        np.array([298.15], dtype=np.float64), dtype=wp.float64, device=device
    )
    pressure = wp.array(
        np.array([101325.0], dtype=np.float64), dtype=wp.float64, device=device
    )

    wp.launch(
        _additive_helper_probe_kernel,
        dim=1,
        inputs=[
            wp.int32(mechanism_mask),
            values_wp,
            pair_rate,
            majorant,
            sanitized,
            active_indices,
            radii,
            total_masses,
            charges,
            temperature,
            pressure,
        ],
        device=device,
    )
    wp.synchronize()

    pair_value = float(np.asarray(pair_rate.numpy()).item())
    majorant_value = float(np.asarray(majorant.numpy()).item())
    expected_brownian = 0.0
    if mechanism_mask & BROWNIAN_MECHANISM_FLAG:
        sum_radius = values[0] + values[1]
        sum_diffusivity = values[2] + values[3]
        g_sqrt = np.sqrt(values[4] ** 2 + values[5] ** 2)
        speed_sqrt = np.sqrt(values[6] ** 2 + values[7] ** 2)
        expected_brownian = (
            4.0
            * np.pi
            * sum_diffusivity
            * sum_radius
            / (
                sum_radius / (sum_radius + g_sqrt)
                + 4.0 * sum_diffusivity / (sum_radius * speed_sqrt)
            )
        )
    expected_charged = 0.0
    if mechanism_mask & CHARGED_HARD_SPHERE_MECHANISM_FLAG:
        expected_charged = _charged_hard_sphere_oracle(
            values[0],
            values[1],
            1.0e-23,
            3.0e-22,
            1.0,
            -2.0,
            298.15,
            101325.0,
        )
    expected = expected_brownian + expected_charged
    npt.assert_allclose(pair_value, expected, rtol=1.0e-11, atol=0.0)
    if mechanism_mask & BROWNIAN_MECHANISM_FLAG:
        npt.assert_allclose(
            majorant_value,
            expected_brownian,
            rtol=1.0e-12,
            atol=0.0,
        )
    else:
        assert majorant_value == 0.0
    if mechanism_mask & (
        BROWNIAN_MECHANISM_FLAG | CHARGED_HARD_SPHERE_MECHANISM_FLAG
    ):
        assert pair_value > 0.0
    assert float(np.asarray(sanitized.numpy()).item()) == 3.0


@pytest.mark.parametrize(
    "invalid_value",
    [0.0, -1.0, float("nan"), float("inf"), float("-inf")],
)
def test_additive_sanitizer_rejects_invalid_terms(
    device: str,
    invalid_value: float,
) -> None:
    """Nonpositive and nonfinite terms make no additive contribution."""
    values = np.array(
        [
            1.0e-8,
            5.0e-8,
            1.0e-9,
            2.0e-10,
            2.0e-8,
            4.0e-8,
            0.2,
            0.1,
            invalid_value,
        ],
        dtype=np.float64,
    )
    values_wp = wp.array(values, dtype=wp.float64, device=device)
    pair_rate = wp.zeros((1,), dtype=wp.float64, device=device)
    majorant = wp.zeros((1,), dtype=wp.float64, device=device)
    sanitized = wp.zeros((1,), dtype=wp.float64, device=device)
    active_indices = wp.array(
        np.array([[-1]], dtype=np.int32), dtype=wp.int32, device=device
    )
    radii = wp.zeros((1, 1), dtype=wp.float64, device=device)
    total_masses = wp.zeros((1, 1), dtype=wp.float64, device=device)
    charges = wp.zeros((1, 1), dtype=wp.float64, device=device)
    temperature = wp.array(
        np.array([298.15], dtype=np.float64), dtype=wp.float64, device=device
    )
    pressure = wp.array(
        np.array([101325.0], dtype=np.float64), dtype=wp.float64, device=device
    )
    wp.launch(
        _additive_helper_probe_kernel,
        dim=1,
        inputs=[
            wp.int32(0),
            values_wp,
            pair_rate,
            majorant,
            sanitized,
            active_indices,
            radii,
            total_masses,
            charges,
            temperature,
            pressure,
        ],
        device=device,
    )
    wp.synchronize()

    assert float(np.asarray(sanitized.numpy()).item()) == 0.0


@pytest.mark.gpu_parity
@pytest.mark.parametrize("mechanism_mask", [3, 5, 6, 9, 10, 12, 15])
def test_all_recognized_additive_masks_against_independent_pair_oracles(
    device: str, mechanism_mask: int
) -> None:
    """Every recognized additive mask sums finite independent component rates."""
    radii = np.array([[1.0e-8, 5.0e-8]], dtype=np.float64)
    transport = np.array(
        [1.0e-9, 2.0e-10, 2.0e-8, 4.0e-8, 0.2, 0.1], dtype=np.float64
    )
    velocities = np.array([[0.5, 3.0]], dtype=np.float64)
    masses = np.array([[1.0e-23, 3.0e-22]], dtype=np.float64)
    charges = np.array([[1.0, -2.0]], dtype=np.float64)
    pair_rate = wp.zeros((1,), dtype=wp.float64, device=device)
    majorant = wp.zeros((1,), dtype=wp.float64, device=device)
    wp.launch(
        _all_mask_additive_probe_kernel,
        dim=1,
        inputs=[
            wp.int32(mechanism_mask),
            wp.array(transport, dtype=wp.float64, device=device),
            wp.array(velocities, dtype=wp.float64, device=device),
            wp.array([[0, 1]], dtype=wp.int32, device=device),
            wp.array(radii, dtype=wp.float64, device=device),
            wp.array(masses, dtype=wp.float64, device=device),
            wp.array(charges, dtype=wp.float64, device=device),
            wp.array([298.15], dtype=wp.float64, device=device),
            wp.array([101325.0], dtype=wp.float64, device=device),
            wp.array([2.0], dtype=wp.float64, device=device),
            wp.array([1.5e-5], dtype=wp.float64, device=device),
            pair_rate,
            majorant,
        ],
        device=device,
    )
    wp.synchronize_device(device)

    brownian = (
        4.0
        * np.pi
        * (transport[0] + transport[1])
        * (radii[0, 0] + radii[0, 1])
        / (
            (radii[0, 0] + radii[0, 1])
            / (radii[0, 0] + radii[0, 1] + np.hypot(transport[2], transport[3]))
            + 4.0
            * (transport[0] + transport[1])
            / (
                (radii[0, 0] + radii[0, 1])
                * np.hypot(transport[4], transport[5])
            )
        )
    )
    components = [
        brownian,
        _charged_hard_sphere_oracle(
            radii[0, 0],
            radii[0, 1],
            masses[0, 0],
            masses[0, 1],
            charges[0, 0],
            charges[0, 1],
            298.15,
            101325.0,
        ),
        _sedimentation_sp2016_oracle(*radii[0], *velocities[0]),
        np.sqrt(np.pi * 2.0 / (120.0 * 1.5e-5)) * (2.0 * np.sum(radii[0])) ** 3,
    ]
    expected = sum(
        component
        for bit, component in zip((1, 2, 4, 8), components, strict=True)
        if mechanism_mask & bit
    )
    observed_rate = float(pair_rate.numpy()[0])
    observed_majorant = float(majorant.numpy()[0])
    assert np.isfinite(observed_rate) and observed_rate > 0.0
    assert np.isfinite(observed_majorant) and observed_majorant >= observed_rate
    # The two-active fixture makes every component maximum the selected pair.
    npt.assert_allclose(observed_rate, expected, rtol=2.0e-10, atol=0.0)


@pytest.mark.gpu_parity
@pytest.mark.parametrize(
    ("charges", "temperature", "pressure"),
    [
        (np.array([0.0, 0.0, 0.0, 0.0]), 298.15, 101325.0),
        (np.array([1.0, 2.0, 3.0, 4.0]), 280.0, 90000.0),
        (np.array([-1.0, 2.0, -3.0, 4.0]), 310.0, 110000.0),
    ],
)
def test_charged_majorant_matches_independent_pair_maximum(
    device: str,
    charges: np.ndarray,
    temperature: float,
    pressure: float,
) -> None:
    """Direct and dispatched bounds equal the independent pair maximum."""
    radii = np.array([[2.0e-9, 9.0e-9, 2.0e-7, 1.0e-5]], dtype=np.float64)
    masses = np.array([[1.0e-23, 4.0e-22, 3.0e-17, 4.0e-10]], dtype=np.float64)
    compact = np.array([[2, 0, 3, 1]], dtype=np.int32)
    expected = _charged_majorant_oracle(
        compact[0], 4, radii[0], masses[0], charges, temperature, pressure
    )
    direct, dispatched = _run_charged_majorant_probes(
        device,
        compact,
        np.array([4], dtype=np.int32),
        radii,
        masses,
        charges[np.newaxis, :],
        np.array([temperature]),
        np.array([pressure]),
    )
    # fp64 Warp arithmetic follows the independent scalar oracle to roundoff.
    npt.assert_allclose(direct, [expected], rtol=1.0e-11, atol=0.0)
    npt.assert_allclose(dispatched, [expected], rtol=1.0e-11, atol=0.0)
    for first, second in itertools.combinations(compact[0], 2):
        rate = _charged_hard_sphere_oracle(
            radii[0, first],
            radii[0, second],
            masses[0, first],
            masses[0, second],
            charges[first],
            charges[second],
            temperature,
            pressure,
        )
        assert np.isfinite(rate) and rate >= 0.0
        assert rate <= np.nextafter(direct[0], np.inf)
        assert rate <= np.nextafter(dispatched[0], np.inf)


@pytest.mark.gpu_parity
def test_charged_majorant_bounds_equal_radius_active_pair(
    device: str,
) -> None:
    """Direct and dispatched bounds cover an equal-radius active pair."""
    radii = np.array([[4.0e-9, 4.0e-9, 2.0e-7]], dtype=np.float64)
    masses = np.array([[3.0e-23, 3.0e-23, 4.0e-17]], dtype=np.float64)
    charges = np.array([[2.0, -3.0, 1.0]], dtype=np.float64)
    compact = np.array([[2, 1, 0]], dtype=np.int32)
    temperature = np.float64(298.15)
    pressure = np.float64(101325.0)
    expected = _charged_majorant_oracle(
        compact[0], 3, radii[0], masses[0], charges[0], temperature, pressure
    )
    direct, dispatched = _run_charged_majorant_probes(
        device,
        compact,
        np.array([3], dtype=np.int32),
        radii,
        masses,
        charges,
        np.array([temperature], dtype=np.float64),
        np.array([pressure], dtype=np.float64),
    )

    npt.assert_allclose(direct, [expected], rtol=1.0e-11, atol=0.0)
    npt.assert_allclose(dispatched, [expected], rtol=1.0e-11, atol=0.0)
    for first, second in itertools.combinations(compact[0], 2):
        rate = _charged_hard_sphere_oracle(
            radii[0, first],
            radii[0, second],
            masses[0, first],
            masses[0, second],
            charges[0, first],
            charges[0, second],
            temperature,
            pressure,
        )
        assert np.isfinite(rate) and rate >= 0.0
        # Permit one ULP between independent scalar and Warp fp64 arithmetic.
        assert rate <= np.nextafter(direct[0], np.inf)
        assert rate <= np.nextafter(dispatched[0], np.inf)


@pytest.mark.gpu_parity
def test_charged_majorant_ignores_slots_outside_compact_active_list(
    device: str,
) -> None:
    """Only compact ranks contribute, even when excluded slots have high rates."""
    radii = np.array([[2.0e-9, 1.0e-4, 8.0e-9, 2.0e-4]], dtype=np.float64)
    masses = np.array([[1.0e-23, 1.0e-7, 3.0e-22, 2.0e-7]], dtype=np.float64)
    charges = np.array([[1.0, -100.0, -2.0, 100.0]], dtype=np.float64)
    compact = np.array([[2, 0, -1, -1]], dtype=np.int32)
    expected = _charged_majorant_oracle(
        compact[0], 2, radii[0], masses[0], charges[0], 298.15, 101325.0
    )
    direct, dispatched = _run_charged_majorant_probes(
        device,
        compact,
        np.array([2], dtype=np.int32),
        radii,
        masses,
        charges,
        np.array([298.15]),
        np.array([101325.0]),
    )
    npt.assert_allclose(direct, [expected], rtol=1.0e-11, atol=0.0)
    npt.assert_allclose(dispatched, [expected], rtol=1.0e-11, atol=0.0)


@pytest.mark.parametrize("active_count", [0, 1])
def test_charged_majorant_returns_zero_for_fewer_than_two_active_particles(
    device: str, active_count: int
) -> None:
    """The helper returns exact safe zero before compact-list indexing."""
    direct, dispatched = _run_charged_majorant_probes(
        device,
        np.array([[0, 1]], dtype=np.int32),
        np.array([active_count], dtype=np.int32),
        np.array([[2.0e-9, 3.0e-9]], dtype=np.float64),
        np.array([[1.0e-22, 2.0e-22]], dtype=np.float64),
        np.zeros((1, 2), dtype=np.float64),
        np.array([298.15]),
        np.array([101325.0]),
    )
    npt.assert_array_equal(direct, np.zeros(1, dtype=np.float64))
    npt.assert_array_equal(dispatched, np.zeros(1, dtype=np.float64))


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf, 0.0, -1.0])
@pytest.mark.parametrize(
    "invalid_field", ["radius", "mass", "temperature", "pressure"]
)
def test_charged_majorant_sanitizes_invalid_selected_pair_candidates(
    device: str, invalid_field: str, invalid_value: float
) -> None:
    """Invalid selected charged inputs make the exact two-particle bound zero."""
    radii = np.array([[2.0e-9, 3.0e-9]], dtype=np.float64)
    masses = np.array([[1.0e-22, 2.0e-22]], dtype=np.float64)
    charges = np.array([[1.0, -1.0]], dtype=np.float64)
    temperature = np.array([298.15], dtype=np.float64)
    pressure = np.array([101325.0], dtype=np.float64)
    if invalid_field == "radius":
        radii[0, 1] = invalid_value
    elif invalid_field == "mass":
        masses[0, 1] = invalid_value
    elif invalid_field == "temperature":
        temperature[0] = invalid_value
    else:
        pressure[0] = invalid_value
    direct, dispatched = _run_charged_majorant_probes(
        device,
        np.array([[0, 1]], dtype=np.int32),
        np.array([2], dtype=np.int32),
        radii,
        masses,
        charges,
        temperature,
        pressure,
    )
    npt.assert_array_equal(direct, np.zeros(1, dtype=np.float64))
    npt.assert_array_equal(dispatched, np.zeros(1, dtype=np.float64))


def test_charged_majorant_returns_zero_when_all_selected_rates_are_zero(
    device: str,
) -> None:
    """All zero-radius selected pairs make no charged majorant contribution."""
    direct, dispatched = _run_charged_majorant_probes(
        device,
        np.array([[2, 0, 1]], dtype=np.int32),
        np.array([3], dtype=np.int32),
        np.zeros((1, 3), dtype=np.float64),
        np.array([[1.0e-22, 2.0e-22, 3.0e-22]], dtype=np.float64),
        np.array([[1.0, -1.0, 2.0]], dtype=np.float64),
        np.array([298.15], dtype=np.float64),
        np.array([101325.0], dtype=np.float64),
    )

    npt.assert_array_equal(direct, np.zeros(1, dtype=np.float64))
    npt.assert_array_equal(dispatched, np.zeros(1, dtype=np.float64))


@pytest.mark.gpu_parity
def test_charged_majorant_matches_each_box_of_nonuniform_fixture(
    device: str,
) -> None:
    """Each box uses its own compact set and thermodynamic state."""
    compact = np.array([[2, 0, 1], [1, 2, 0]], dtype=np.int32)
    active_counts = np.array([3, 2], dtype=np.int32)
    radii = np.array(
        [[2.0e-9, 8.0e-9, 3.0e-7], [4.0e-8, 2.0e-6, 8.0e-6]],
        dtype=np.float64,
    )
    masses = np.array(
        [[1.0e-23, 2.0e-22, 5.0e-18], [3.0e-19, 7.0e-14, 2.0e-12]],
        dtype=np.float64,
    )
    charges = np.array([[1.0, -2.0, 3.0], [-3.0, 1.0, 2.0]], dtype=np.float64)
    temperature = np.array([280.0, 315.0], dtype=np.float64)
    pressure = np.array([85000.0, 110000.0], dtype=np.float64)
    expected = np.array(
        [
            _charged_majorant_oracle(
                compact[box],
                active_counts[box],
                radii[box],
                masses[box],
                charges[box],
                temperature[box],
                pressure[box],
            )
            for box in range(2)
        ],
        dtype=np.float64,
    )

    direct, dispatched = _run_charged_majorant_probes(
        device,
        compact,
        active_counts,
        radii,
        masses,
        charges,
        temperature,
        pressure,
    )

    npt.assert_allclose(direct, expected, rtol=1.0e-11, atol=0.0)
    npt.assert_allclose(dispatched, expected, rtol=1.0e-11, atol=0.0)


@pytest.mark.gpu_parity
def test_charged_majorant_scans_each_unique_sparse_active_pair(
    device: str,
) -> None:
    """Sparse, non-sorted compact ranks enumerate all and only six pairs."""
    compact = np.array([[5, 1, 4, 2, -1, -1]], dtype=np.int32)
    radii = np.array(
        [[1.0e-4, 2.0e-9, 8.0e-8, 5.0e-4, 3.0e-7, 2.0e-6]],
        dtype=np.float64,
    )
    masses = np.array(
        [[1.0e-8, 1.0e-23, 2.0e-18, 1.0e-6, 7.0e-17, 3.0e-13]],
        dtype=np.float64,
    )
    charges = np.array(
        [[100.0, 100.0, 1.0, -2.0, -100.0, 100.0]],
        dtype=np.float64,
    )
    selected_pairs = list(itertools.combinations(compact[0, :4], 2))
    rates = np.array(
        [
            _charged_hard_sphere_oracle(
                radii[0, first],
                radii[0, second],
                masses[0, first],
                masses[0, second],
                charges[0, first],
                charges[0, second],
                298.15,
                101325.0,
            )
            for first, second in selected_pairs
        ],
        dtype=np.float64,
    )
    expected = float(np.max(rates))
    assert len(selected_pairs) == 6
    assert int(np.argmax(rates)) not in (0, len(rates) - 1)

    direct, dispatched = _run_charged_majorant_probes(
        device,
        compact,
        np.array([4], dtype=np.int32),
        radii,
        masses,
        charges,
        np.array([298.15], dtype=np.float64),
        np.array([101325.0], dtype=np.float64),
    )

    npt.assert_allclose(direct, [expected], rtol=1.0e-11, atol=0.0)
    npt.assert_allclose(dispatched, [expected], rtol=1.0e-11, atol=0.0)


@pytest.mark.gpu_parity
def test_charged_majorant_dispatcher_adds_brownian_term(device: str) -> None:
    """Private dispatcher adds independent terms for its direct unit probe."""
    compact = np.array([[0, 1]], dtype=np.int32)
    radii = np.array([[2.0e-9, 8.0e-9]], dtype=np.float64)
    masses = np.array([[1.0e-23, 3.0e-22]], dtype=np.float64)
    charges = np.array([[1.0, -2.0]], dtype=np.float64)
    temperature = np.array([298.15], dtype=np.float64)
    pressure = np.array([101325.0], dtype=np.float64)
    direct, _ = _run_charged_majorant_probes(
        device,
        compact,
        np.array([2], dtype=np.int32),
        radii,
        masses,
        charges,
        temperature,
        pressure,
    )
    brownian = (
        4.0
        * np.pi
        * (1.0e-9 + 2.0e-10)
        * (1.0e-8 + 5.0e-8)
        / (
            (1.0e-8 + 5.0e-8) / ((1.0e-8 + 5.0e-8) + np.hypot(2.0e-8, 4.0e-8))
            + 4.0
            * (1.0e-9 + 2.0e-10)
            / ((1.0e-8 + 5.0e-8) * np.hypot(0.2, 0.1))
        )
    )
    active = wp.array(compact, dtype=wp.int32, device=device)
    counts = wp.array(
        np.array([2], dtype=np.int32), dtype=wp.int32, device=device
    )
    output = wp.zeros((1,), dtype=wp.float64, device=device)
    wp.launch(
        _charged_majorant_probe_kernel,
        dim=1,
        inputs=[
            active,
            counts,
            wp.array(radii, dtype=wp.float64, device=device),
            wp.array(masses, dtype=wp.float64, device=device),
            wp.array(charges, dtype=wp.float64, device=device),
            wp.array(temperature, dtype=wp.float64, device=device),
            wp.array(pressure, dtype=wp.float64, device=device),
            wp.int32(
                BROWNIAN_MECHANISM_FLAG | CHARGED_HARD_SPHERE_MECHANISM_FLAG
            ),
            wp.zeros((1,), dtype=wp.float64, device=device),
            output,
        ],
        device=device,
    )
    wp.synchronize_device(device)
    npt.assert_allclose(
        output.numpy(), direct + brownian, rtol=1.0e-11, atol=0.0
    )


@pytest.mark.gpu_parity
def test_sedimentation_majorant_matches_independent_sparse_pair_oracle(
    device: str,
) -> None:
    """Direct and dispatched SP2016 bounds scan all compact sparse pairs."""
    compact = np.array([[5, 1, 4, 2, -1, -1]], dtype=np.int32)
    radii = np.array(
        [[2.0e-6, 1.0e-6, 4.0e-6, 8.0e-6, 3.0e-6, 6.0e-6]],
        dtype=np.float64,
    )
    velocities = np.array([[3.0, 0.5, 8.0, 2.0, 4.0, 6.0]], dtype=np.float64)
    expected = _sedimentation_majorant_oracle(
        compact[0], 4, radii[0], velocities[0]
    )
    direct, dispatched = _run_sedimentation_majorant_probes(
        device, compact, np.array([4], dtype=np.int32), radii, velocities
    )

    assert expected > 0.0
    npt.assert_allclose(direct, [expected], rtol=1.0e-12, atol=0.0)
    npt.assert_allclose(dispatched, [expected], rtol=1.0e-12, atol=0.0)
    for first, second in itertools.combinations(compact[0, :4], 2):
        rate = _sedimentation_sp2016_oracle(
            radii[0, first],
            radii[0, second],
            velocities[0, first],
            velocities[0, second],
        )
        assert rate <= np.nextafter(direct[0], np.inf)


@pytest.mark.parametrize("active_count", [0, 1])
def test_sedimentation_majorant_returns_zero_before_low_active_indexing(
    device: str, active_count: int
) -> None:
    """SP2016 majorants return exact zero for zero or one compact active slot."""
    direct, dispatched = _run_sedimentation_majorant_probes(
        device,
        np.array([[0, 1]], dtype=np.int32),
        np.array([active_count], dtype=np.int32),
        np.array([[2.0e-6, 4.0e-6]], dtype=np.float64),
        np.array([[1.0, 4.0]], dtype=np.float64),
    )
    npt.assert_array_equal(direct, np.zeros(1, dtype=np.float64))
    npt.assert_array_equal(dispatched, np.zeros(1, dtype=np.float64))


@pytest.mark.parametrize(
    "velocities",
    [
        np.array([[2.0, 2.0]], dtype=np.float64),
        np.array([[np.nan, 2.0]], dtype=np.float64),
        np.array([[np.inf, 2.0]], dtype=np.float64),
        np.array([[-1.0, 2.0]], dtype=np.float64),
    ],
)
def test_sedimentation_majorant_excludes_equal_and_invalid_velocities(
    device: str, velocities: np.ndarray
) -> None:
    """Equal, non-finite, and negative velocities make no SP2016 bound."""
    direct, dispatched = _run_sedimentation_majorant_probes(
        device,
        np.array([[0, 1]], dtype=np.int32),
        np.array([2], dtype=np.int32),
        np.array([[2.0e-6, 4.0e-6]], dtype=np.float64),
        velocities,
    )
    npt.assert_array_equal(direct, np.zeros(1, dtype=np.float64))
    npt.assert_array_equal(dispatched, np.zeros(1, dtype=np.float64))


def test_sedimentation_pair_rate_contributes_through_fixed_dispatch(
    device: str,
) -> None:
    """The sedimentation-only mask preserves the P1 pair rate in dispatch."""
    radii = np.array([2.0e-6, 6.0e-6], dtype=np.float64)
    velocities = np.array([0.5, 3.0], dtype=np.float64)
    direct = wp.zeros((1,), dtype=wp.float64, device=device)
    dispatched = wp.zeros((1,), dtype=wp.float64, device=device)
    wp.launch(
        _sedimentation_pair_rate_probe_kernel,
        dim=1,
        inputs=[
            wp.array(radii, dtype=wp.float64, device=device),
            wp.array(velocities, dtype=wp.float64, device=device),
            direct,
            dispatched,
        ],
        device=device,
    )
    wp.synchronize_device(device)

    expected = _sedimentation_sp2016_oracle(*radii, *velocities)
    assert expected > 0.0
    npt.assert_allclose(direct.numpy(), [expected], rtol=1.0e-12, atol=0.0)
    npt.assert_allclose(dispatched.numpy(), [expected], rtol=1.0e-12, atol=0.0)


def _run_synthetic_total_sampling_probe(
    device: str,
    rate_terms: tuple[float, float],
    majorant_terms: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the test-only total-sampling probe with sentinel output state."""
    rates = wp.array(
        np.array(rate_terms, dtype=np.float64), dtype=wp.float64, device=device
    )
    majorants = wp.array(
        np.array(majorant_terms, dtype=np.float64),
        dtype=wp.float64,
        device=device,
    )
    pair_sentinel = wp.array(
        np.array([-7, -7], dtype=np.int32), dtype=wp.int32, device=device
    )
    totals = wp.zeros((2,), dtype=wp.float64, device=device)
    candidate_count = wp.zeros((1,), dtype=wp.int32, device=device)
    acceptance_draw_count = wp.zeros((1,), dtype=wp.int32, device=device)
    accepted_count = wp.zeros((1,), dtype=wp.int32, device=device)
    wp.launch(
        _synthetic_total_sampling_probe_kernel,
        dim=1,
        inputs=[
            rates,
            majorants,
            pair_sentinel,
            totals,
            candidate_count,
            acceptance_draw_count,
            accepted_count,
        ],
        device=device,
    )
    wp.synchronize()
    return (
        np.asarray(pair_sentinel.numpy()),
        np.asarray(totals.numpy()),
        np.asarray(candidate_count.numpy()),
        np.asarray(acceptance_draw_count.numpy()),
        np.asarray(accepted_count.numpy()),
    )


def test_synthetic_additive_terms_use_one_total_and_acceptance_draw(
    device: str,
) -> None:
    """Two finite synthetic terms sum once and make one accepted draw."""
    pair, totals, candidates, draws, accepted = (
        _run_synthetic_total_sampling_probe(
            device,
            rate_terms=(2.0, 3.0),
            majorant_terms=(4.0, 1.0),
        )
    )

    npt.assert_allclose(totals, np.array([5.0, 5.0], dtype=np.float64))
    npt.assert_array_equal(candidates, np.array([1], dtype=np.int32))
    npt.assert_array_equal(draws, np.array([1], dtype=np.int32))
    npt.assert_array_equal(accepted, np.array([1], dtype=np.int32))
    npt.assert_array_equal(pair, np.array([0, 1], dtype=np.int32))


@pytest.mark.parametrize(
    ("rate_terms", "majorant_terms", "expected_candidates"),
    [
        ((1.0, 1.0), (0.0, 0.0), 0),
        ((1.0, 1.0), (-1.0, 0.0), 0),
        ((1.0, 1.0), (float("nan"), 0.0), 0),
        ((1.0, 1.0), (float("inf"), 0.0), 0),
        ((3.0, 0.0), (2.0, 0.0), 1),
        ((0.0, 0.0), (2.0, 0.0), 1),
        ((float("nan"), 0.0), (2.0, 0.0), 1),
    ],
)
def test_synthetic_invalid_totals_preserve_sampling_output_state(
    device: str,
    rate_terms: tuple[float, float],
    majorant_terms: tuple[float, float],
    expected_candidates: int,
) -> None:
    """Invalid, zero, and underestimated terms cannot mutate pair output."""
    pair, _, candidates, draws, accepted = _run_synthetic_total_sampling_probe(
        device,
        rate_terms=rate_terms,
        majorant_terms=majorant_terms,
    )

    npt.assert_array_equal(
        candidates, np.array([expected_candidates], dtype=np.int32)
    )
    npt.assert_array_equal(draws, np.array([0], dtype=np.int32))
    npt.assert_array_equal(accepted, np.array([0], dtype=np.int32))
    npt.assert_array_equal(pair, np.array([-7, -7], dtype=np.int32))


@pytest.mark.parametrize(
    ("rate_terms", "majorant_terms"),
    [
        ((np.finfo(np.float64).max, np.finfo(np.float64).max), (1.0, 1.0)),
        ((1.0, 1.0), (np.finfo(np.float64).max, np.finfo(np.float64).max)),
    ],
)
def test_synthetic_overflowed_totals_fail_closed(
    device: str,
    rate_terms: tuple[float, float],
    majorant_terms: tuple[float, float],
) -> None:
    """Overflowed additive totals become safe zero before selector work."""
    pair, totals, _, draws, accepted = _run_synthetic_total_sampling_probe(
        device, rate_terms, majorant_terms
    )

    assert np.any(totals == 0.0)
    npt.assert_array_equal(draws, np.array([0], dtype=np.int32))
    npt.assert_array_equal(accepted, np.array([0], dtype=np.int32))
    npt.assert_array_equal(pair, np.array([-7, -7], dtype=np.int32))


@pytest.mark.parametrize(("ulps", "expected_draws"), [(8, 1), (9, 0)])
def test_selector_ratio_allows_only_eight_ulp_majorant_overshoot(
    device: str, ulps: int, expected_draws: int
) -> None:
    """Only the bounded fp64 majorant overshoot can advance acceptance RNG."""
    majorant = 1.0
    rate = majorant
    for _ in range(ulps):
        rate = np.nextafter(rate, np.inf)
    _, _, candidates, draws, _ = _run_synthetic_total_sampling_probe(
        device, (rate, 0.0), (majorant, 0.0)
    )

    npt.assert_array_equal(candidates, np.array([1], dtype=np.int32))
    npt.assert_array_equal(draws, np.array([expected_draws], dtype=np.int32))


def _make_particle_data(
    n_boxes: int,
    n_particles: int,
    n_species: int,
    concentration_scale: float = 1.0,
) -> Any:
    """Create deterministic particle data for coagulation tests."""
    base_masses = np.linspace(1.0e-18, 2.0e-18, n_species, dtype=np.float64)
    masses = np.empty((n_boxes, n_particles, n_species), dtype=np.float64)
    for box_idx in range(n_boxes):
        for particle_idx in range(n_particles):
            scale = 1.0 + 0.2 * particle_idx + 0.05 * box_idx
            masses[box_idx, particle_idx, :] = base_masses * scale
    concentration = np.full(
        (n_boxes, n_particles),
        concentration_scale,
        dtype=np.float64,
    )
    charge = np.zeros((n_boxes, n_particles), dtype=np.float64)
    density = np.linspace(1000.0, 1400.0, n_species, dtype=np.float64)
    volume = np.full((n_boxes,), 1.0e-6, dtype=np.float64)
    return ParticleData(
        masses=masses,
        concentration=concentration,
        charge=charge,
        density=density,
        volume=volume,
    )


def _make_gas_data(n_boxes: int, n_species: int) -> GasData:
    """Create gas data for Brownian kernel reference calculations."""
    molar_mass = np.linspace(0.018, 0.05, n_species, dtype=np.float64)
    concentration = np.full((n_boxes, n_species), 1.0e-6, dtype=np.float64)
    partitioning = np.ones((n_species,), dtype=bool)
    names = [f"species_{idx}" for idx in range(n_species)]
    return GasData(
        name=names,
        molar_mass=molar_mass,
        concentration=concentration,
        partitioning=partitioning,
    )


def _make_environment_data(
    n_boxes: int,
    n_species: int,
    temperature: float = 298.15,
    pressure: float = 101325.0,
) -> EnvironmentData:
    """Create deterministic environment data for contract tests."""
    return EnvironmentData(
        temperature=np.full((n_boxes,), temperature, dtype=np.float64),
        pressure=np.full((n_boxes,), pressure, dtype=np.float64),
        saturation_ratio=np.ones((n_boxes, n_species), dtype=np.float64),
    )


def _assert_particles_unchanged(
    gpu_particles: Any,
    initial_particles: ParticleData,
) -> None:
    """Assert GPU particle arrays still match a CPU snapshot."""
    result_particles = from_warp_particle_data(gpu_particles, sync=True)
    npt.assert_allclose(result_particles.masses, initial_particles.masses)
    npt.assert_allclose(
        result_particles.concentration,
        initial_particles.concentration,
    )
    npt.assert_allclose(result_particles.charge, initial_particles.charge)


def _accumulate_collision_counts(
    *,
    particles: ParticleData,
    device: str,
    seeds: range,
    time_step: float,
    max_collisions: int,
    temperature: float | Any | None,
    pressure: float | Any | None,
    environment: Any | None = None,
    volume: float | Any | None = None,
    mechanism_config: CoagulationMechanismConfig | None = None,
) -> np.ndarray:
    """Accumulate per-box collision counts across a fixed seed set.

    Args:
        particles: CPU particle fixture to convert for each seeded run.
        device: Warp device used for the repeated coagulation launches.
        seeds: Deterministic seed range used to replay fresh runs.
        time_step: Coagulation step size in seconds.
        max_collisions: Requested per-box collision capacity.
        temperature: Scalar or Warp-array temperature input passed through to
            ``coagulation_step_gpu``.
        pressure: Scalar or Warp-array pressure input passed through to
            ``coagulation_step_gpu``.
        environment: Optional explicit environment input for the launches.
        volume: Optional scalar or device volume override in m^3.
        mechanism_config: Optional coagulation configuration forwarded to each
            entry-point call.

    Returns:
        Per-box summed collision counts accumulated across all requested seeds.
    """
    total_counts = np.zeros(particles.masses.shape[0], dtype=np.int64)

    for seed in seeds:
        gpu_particles = to_warp_particle_data(particles, device=device)
        _, _, collision_counts = coagulation_step_gpu(
            gpu_particles,
            temperature=temperature,
            pressure=pressure,
            time_step=time_step,
            volume=volume,
            max_collisions=max_collisions,
            rng_seed=seed,
            environment=environment,
            mechanism_config=mechanism_config,
        )
        wp.synchronize()
        total_counts += np.asarray(collision_counts.numpy(), dtype=np.int64)

    return total_counts


def _run_seeded_coagulation_step(
    particles: ParticleData,
    device: str,
    *,
    time_step: float,
    max_collisions: int,
    rng_seed: int,
    temperature: float = 298.15,
    pressure: float = 101325.0,
    volume: float | np.ndarray | None = None,
    rng_states: Any | None = None,
    initialize_rng: bool = False,
    mechanism_config: CoagulationMechanismConfig | None = None,
) -> _CoagulationStepResult:
    """Run one seeded coagulation step from a fresh particle fixture.

    Args:
        particles: CPU particle fixture to convert for the requested device.
        device: Warp device used for the seeded launch.
        time_step: Coagulation step size in seconds.
        max_collisions: Requested per-box collision capacity.
        rng_seed: Seed supplied to ``coagulation_step_gpu``.
        temperature: Gas temperature in K.
        pressure: Gas pressure in Pa.
        volume: Optional scalar or array box volume override in m^3.
        rng_states: Optional caller-owned RNG state buffer to reuse or reset.
        initialize_rng: Whether to reseed a provided ``rng_states`` buffer.
        mechanism_config: Optional coagulation configuration forwarded to the
            public entry point.

    Returns:
        Collision counts, accepted pairs, restored particle state, and an
        optional snapshot of caller-owned RNG state after the launch.
    """
    gpu_particles = to_warp_particle_data(particles, device=device)
    n_boxes, n_particles, _ = particles.masses.shape
    collision_capacity = _resolve_collision_capacity(
        max_collisions=max_collisions,
        n_boxes=n_boxes,
        n_particles=n_particles,
    )
    collision_pairs = wp.zeros(
        (n_boxes, collision_capacity, 2), dtype=wp.int32, device=device
    )
    collision_counts = wp.zeros((n_boxes,), dtype=wp.int32, device=device)

    _, _, returned_counts = coagulation_step_gpu(
        gpu_particles,
        temperature=temperature,
        pressure=pressure,
        time_step=time_step,
        volume=volume,
        max_collisions=max_collisions,
        rng_seed=rng_seed,
        collision_pairs=collision_pairs,
        n_collisions=collision_counts,
        rng_states=rng_states,
        initialize_rng=initialize_rng,
        mechanism_config=mechanism_config,
    )
    wp.synchronize()

    rng_state_snapshot = None
    if rng_states is not None:
        rng_state_snapshot = np.asarray(
            rng_states.numpy(), dtype=np.uint32
        ).copy()

    return _CoagulationStepResult(
        collision_counts=np.asarray(returned_counts.numpy(), dtype=np.int64),
        collision_pairs=np.asarray(collision_pairs.numpy(), dtype=np.int64),
        result_particles=from_warp_particle_data(gpu_particles, sync=True),
        rng_states=rng_state_snapshot,
    )


def _get_expected_collision_statistics(
    particles: ParticleData,
    *,
    time_step: float,
    n_trials: int,
    temperature: float = 298.15,
    pressure: float = 101325.0,
    volume: float | None = None,
) -> _ExpectedCollisionStatistics:
    """Return Brownian expected mean and sigma for seeded repeated runs.

    Args:
        particles: CPU particle fixture used to derive radii and masses.
        time_step: Coagulation step size in seconds.
        n_trials: Number of independent seeded fresh runs being summarized.
        temperature: Gas temperature in K.
        pressure: Gas pressure in Pa.
        volume: Optional scalar box volume override in m^3.

    The reference assumes repeated fresh runs from the same initial particle
    state, so every trial uses the same active-particle mask and the same
    unique upper-triangle Brownian pairs.

    Returns:
        Expected total-collision mean, Poisson-style sigma, and the active
        upper-triangle pair count used by the repeated-run reference.
    """
    density_value = float(np.asarray(particles.density).ravel().item(0))
    masses_slice = np.ravel(np.asarray(particles.masses[0], dtype=np.float64))
    concentration_slice = np.ravel(
        np.asarray(particles.concentration[0], dtype=np.float64)
    )
    active_mask = concentration_slice > 0.0
    radii = np.cbrt(3.0 * masses_slice / (4.0 * np.pi * density_value))
    kernel_matrix = get_brownian_kernel_via_system_state(
        particle_radius=radii,
        particle_mass=masses_slice,
        temperature=temperature,
        pressure=pressure,
    )
    active_kernel_matrix = np.asarray(kernel_matrix, dtype=np.float64)[
        np.ix_(active_mask, active_mask)
    ]
    kernel_values = active_kernel_matrix[
        np.triu_indices(active_kernel_matrix.shape[0], k=1)
    ]
    active_pair_count = int(kernel_values.size)
    volume_value = (
        float(np.asarray(particles.volume).ravel().item(0))
        if volume is None
        else float(volume)
    )
    expected_mean = float(
        np.sum(kernel_values) * time_step * n_trials / volume_value
    )
    expected_sigma = float(np.sqrt(expected_mean))
    return _ExpectedCollisionStatistics(
        expected_mean=expected_mean,
        expected_sigma=expected_sigma,
        active_pair_count=active_pair_count,
    )


def _coagulation_outcomes_match(
    left: _CoagulationStepResult,
    right: _CoagulationStepResult,
) -> bool:
    """Return ``True`` when two seeded coagulation outcomes are identical."""
    return (
        np.array_equal(left.collision_counts, right.collision_counts)
        and np.array_equal(left.collision_pairs, right.collision_pairs)
        and np.allclose(
            left.result_particles.masses,
            right.result_particles.masses,
        )
        and np.allclose(
            left.result_particles.concentration,
            right.result_particles.concentration,
        )
    )


def _make_mixed_npf_droplet_particle_data() -> ParticleData:
    """Build deterministic mixed NPF and droplet particle data.

    Returns:
        Particle data with one box containing nanometer-scale and droplet-scale
        particles, explicit ``np.float64`` masses, and concentrations chosen to
        exercise mixed-scale coagulation diagnostics.
    """
    density = np.array([1000.0], dtype=np.float64)
    radii = np.array([[1.5e-9, 2.0e-9, 1.0e-5, 1.5e-5]], dtype=np.float64)
    total_volume = (4.0 / 3.0) * np.pi * radii**3
    masses = (total_volume * density[0])[..., np.newaxis]
    return ParticleData(
        masses=masses,
        concentration=np.array([[150.0, 200.0, 1.0, 0.8]], dtype=np.float64),
        charge=np.array([[0.0, 0.0, 1.0, -1.0]], dtype=np.float64),
        density=density,
        volume=np.array([2.0e-6], dtype=np.float64),
    )


@_warp_kernel
def _brownian_coagulation_attempt_diagnostic_kernel(  # noqa: C901
    masses: Any,
    concentration: Any,
    density: Any,
    volume: Any,
    temperature: Any,
    pressure: Any,
    gas_constant: Any,
    boltzmann_constant: Any,
    molecular_weight_air: Any,
    ref_viscosity: Any,
    ref_temperature: Any,
    sutherland_constant: Any,
    time_step: Any,
    radii: Any,
    diffusivities: Any,
    g_terms: Any,
    speeds: Any,
    active_indices: Any,
    collision_pairs: Any,
    n_collisions: Any,
    scheduled_trial_counts: Any,
    executed_trial_counts: Any,
    rng_states: Any,
    collision_capacity: Any,
) -> None:
    """Mirror the production sampler and expose bounded trial diagnostics.

    This test-local kernel reproduces the production Brownian rejection
    sampler's rounded ``expected_trials`` logic and accepted-collision
    bookkeeping while storing both bounded scheduled trials and true executed
    trials for diagnostics.
    """
    box_idx = wp.tid()
    n_particles = masses.shape[1]
    n_species = masses.shape[2]

    temperature_value = temperature[box_idx]
    pressure_value = pressure[box_idx]

    dynamic_viscosity = dynamic_viscosity_wp(
        temperature_value,
        ref_viscosity,
        ref_temperature,
        sutherland_constant,
    )
    mean_free_path = molecule_mean_free_path_wp(
        molecular_weight_air,
        temperature_value,
        pressure_value,
        dynamic_viscosity,
        gas_constant,
    )

    active_count = wp.int32(0)
    scheduled_trial_counts[box_idx] = wp.int32(0)
    executed_trial_counts[box_idx] = wp.int32(0)
    for particle_idx in range(n_particles):
        if concentration[box_idx, particle_idx] <= wp.float64(0.0):
            active_indices[box_idx, particle_idx] = wp.int32(-1)
            radii[box_idx, particle_idx] = wp.float64(0.0)
            diffusivities[box_idx, particle_idx] = wp.float64(0.0)
            g_terms[box_idx, particle_idx] = wp.float64(0.0)
            speeds[box_idx, particle_idx] = wp.float64(0.0)
            continue

        total_mass = wp.float64(0.0)
        total_volume = wp.float64(0.0)
        for species_idx in range(n_species):
            species_mass = masses[box_idx, particle_idx, species_idx]
            total_mass += species_mass
            total_volume += species_mass / density[species_idx]

        if total_volume <= wp.float64(0.0) or total_mass <= wp.float64(0.0):
            active_indices[box_idx, particle_idx] = wp.int32(-1)
            radii[box_idx, particle_idx] = wp.float64(0.0)
            diffusivities[box_idx, particle_idx] = wp.float64(0.0)
            g_terms[box_idx, particle_idx] = wp.float64(0.0)
            speeds[box_idx, particle_idx] = wp.float64(0.0)
            continue

        radius = particle_radius_from_volume_wp(total_volume)
        knudsen = knudsen_number_wp(mean_free_path, radius)
        slip = cunningham_slip_correction_wp(knudsen)
        mobility = aerodynamic_mobility_wp(radius, slip, dynamic_viscosity)
        diffusivity = brownian_diffusivity_wp(
            temperature_value, mobility, boltzmann_constant
        )
        speed = mean_thermal_speed_wp(
            total_mass, temperature_value, boltzmann_constant
        )
        particle_mean_free_path = particle_mean_free_path_wp(diffusivity, speed)
        g_term = g_collection_term_wp(particle_mean_free_path, radius)

        active_indices[box_idx, active_count] = wp.int32(particle_idx)
        radii[box_idx, particle_idx] = radius
        diffusivities[box_idx, particle_idx] = diffusivity
        g_terms[box_idx, particle_idx] = g_term
        speeds[box_idx, particle_idx] = speed
        active_count += wp.int32(1)

    if active_count < wp.int32(2):
        n_collisions[box_idx] = wp.int32(0)
        return

    majorant_total = wp.float64(0.0)
    for first_rank in range(active_count - wp.int32(1)):
        first_idx = wp.int32(active_indices[box_idx, first_rank])
        for second_rank in range(first_rank + wp.int32(1), active_count):
            second_idx = wp.int32(active_indices[box_idx, second_rank])
            pair_rate = _total_pair_rate(
                wp.int32(BROWNIAN_MECHANISM_FLAG),
                radii[box_idx, first_idx],
                radii[box_idx, second_idx],
                diffusivities[box_idx, first_idx],
                diffusivities[box_idx, second_idx],
                g_terms[box_idx, first_idx],
                g_terms[box_idx, second_idx],
                speeds[box_idx, first_idx],
                speeds[box_idx, second_idx],
                wp.float64(0.0),
                wp.float64(0.0),
                wp.float64(0.0),
                wp.float64(0.0),
                wp.float64(1.0),
                wp.float64(1.0),
                wp.float64(0.0),
                wp.float64(0.0),
                temperature_value,
                pressure_value,
                boltzmann_constant,
                wp.float64(constants.ELEMENTARY_CHARGE_VALUE),
                wp.float64(constants.ELECTRIC_PERMITTIVITY),
                gas_constant,
                molecular_weight_air,
                ref_viscosity,
                ref_temperature,
                sutherland_constant,
            )
            if pair_rate > majorant_total:
                majorant_total = pair_rate

    if not (wp.isfinite(majorant_total) and majorant_total > wp.float64(0.0)):
        n_collisions[box_idx] = wp.int32(0)
        return

    possible_pairs = (
        wp.float64(active_count)
        * wp.float64(active_count - 1)
        / wp.float64(2.0)
    )
    expected_trials = (
        majorant_total * possible_pairs * time_step / volume[box_idx]
    )
    if not (wp.isfinite(expected_trials) and expected_trials > wp.float64(0.0)):
        n_collisions[box_idx] = wp.int32(0)
        return

    collision_count = wp.int32(0)
    state = rng_states[box_idx]

    bounded_trials = _bound_scheduled_trials(expected_trials)

    tests = wp.int32(bounded_trials)
    remainder = bounded_trials - wp.float64(tests)
    if wp.randf(state) < remainder:
        tests += wp.int32(1)
    scheduled_trial_counts[box_idx] = tests
    rng_states[box_idx] = state

    if tests <= wp.int32(0):
        n_collisions[box_idx] = wp.int32(0)
        return

    for _ in range(tests):
        if collision_count >= collision_capacity or active_count < wp.int32(2):
            break

        executed_trial_counts[box_idx] += wp.int32(1)

        rank_i = wp.randi(state, wp.int32(0), active_count)
        rank_j = wp.randi(state, wp.int32(0), active_count - wp.int32(1))
        adjusted_rank_j = rank_j
        if adjusted_rank_j >= rank_i:
            adjusted_rank_j += wp.int32(1)

        selected_pair = _select_active_pair_by_rank(
            active_indices,
            box_idx,
            rank_i,
            adjusted_rank_j,
        )
        selected_i = selected_pair[0]
        selected_j = selected_pair[1]

        if selected_i < wp.int32(0) or selected_j < wp.int32(0):
            continue

        if selected_j < selected_i:
            temp_idx = selected_i
            selected_i = selected_j
            selected_j = temp_idx

        total_rate = _total_pair_rate(
            wp.int32(BROWNIAN_MECHANISM_FLAG),
            radii[box_idx, selected_i],
            radii[box_idx, selected_j],
            diffusivities[box_idx, selected_i],
            diffusivities[box_idx, selected_j],
            g_terms[box_idx, selected_i],
            g_terms[box_idx, selected_j],
            speeds[box_idx, selected_i],
            speeds[box_idx, selected_j],
            wp.float64(0.0),
            wp.float64(0.0),
            wp.float64(0.0),
            wp.float64(0.0),
            wp.float64(1.0),
            wp.float64(1.0),
            wp.float64(0.0),
            wp.float64(0.0),
            temperature_value,
            pressure_value,
            boltzmann_constant,
            wp.float64(constants.ELEMENTARY_CHARGE_VALUE),
            wp.float64(constants.ELECTRIC_PERMITTIVITY),
            gas_constant,
            molecular_weight_air,
            ref_viscosity,
            ref_temperature,
            sutherland_constant,
        )
        if not (wp.isfinite(total_rate) and total_rate > wp.float64(0.0)):
            continue
        if wp.randf(state) < total_rate / majorant_total:
            collision_pairs[box_idx, collision_count, 0] = selected_i
            collision_pairs[box_idx, collision_count, 1] = selected_j
            collision_count += wp.int32(1)
            active_count = _remove_active_pair_by_rank_swap_pop(
                active_indices,
                box_idx,
                active_count,
                rank_i,
                adjusted_rank_j,
            )

    n_collisions[box_idx] = collision_count
    rng_states[box_idx] = state


def _validate_test_local_diagnostic_inputs(
    *,
    time_step: float,
    temperature: float,
    pressure: float,
) -> tuple[float, float, float]:
    """Validate local mirrored-diagnostic scalar inputs before launch."""
    time_step_value = _validate_time_step(time_step)

    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and > 0")
    if not np.isfinite(pressure) or pressure <= 0.0:
        raise ValueError("pressure must be finite and > 0")

    return time_step_value, float(temperature), float(pressure)


def _trim_collision_pairs(
    collision_pairs: np.ndarray,
    collision_counts: np.ndarray,
) -> list[np.ndarray]:
    """Trim padded collision-pair buffers down to accepted pairs only."""
    trimmed_pairs: list[np.ndarray] = []
    for box_idx, count in enumerate(collision_counts.tolist()):
        trimmed_pairs.append(collision_pairs[box_idx, :count, :].copy())
    return trimmed_pairs


def _assert_trimmed_collision_pairs_equal(
    diagnostic_pairs: np.ndarray,
    diagnostic_counts: np.ndarray,
    production_pairs: np.ndarray,
    production_counts: np.ndarray,
) -> None:
    """Assert accepted collision-pair prefixes match box by box."""
    diagnostic_trimmed = _trim_collision_pairs(
        diagnostic_pairs,
        diagnostic_counts,
    )
    production_trimmed = _trim_collision_pairs(
        production_pairs,
        production_counts,
    )

    assert len(diagnostic_trimmed) == len(production_trimmed)
    for diagnostic_box_pairs, production_box_pairs in zip(
        diagnostic_trimmed,
        production_trimmed,
        strict=True,
    ):
        npt.assert_array_equal(diagnostic_box_pairs, production_box_pairs)


def _collect_test_local_attempt_diagnostics(
    particles: ParticleData,
    device: str,
    *,
    time_step: float,
    max_collisions: int,
    rng_seed: int,
    temperature: float = 298.15,
    pressure: float = 101325.0,
    volume: float | np.ndarray | None = None,
) -> _AttemptDiagnostics:
    """Collect mirrored and production diagnostics for one seeded run.

    Args:
        particles: CPU particle fixture to convert for the requested device.
        device: Warp device name used for the mirrored and production launches.
        time_step: Coagulation step size in seconds.
        max_collisions: Requested per-box collision capacity.
        rng_seed: Seed used for both the mirrored and production sampler.
        temperature: Gas temperature in K.
        pressure: Gas pressure in Pa.
        volume: Optional box volume override in m^3.

    Returns:
        Mirrored and production trial-count, pair, particle-state, and RNG-state
        diagnostics for seeded parity assertions.
    """
    time_step, temperature, pressure = _validate_test_local_diagnostic_inputs(
        time_step=time_step,
        temperature=temperature,
        pressure=pressure,
    )
    n_boxes, n_particles, _ = particles.masses.shape
    collision_capacity = _resolve_collision_capacity(
        max_collisions=max_collisions,
        n_boxes=n_boxes,
        n_particles=n_particles,
    )

    diagnostic_particles = to_warp_particle_data(particles, device=device)
    temperature_array = wp.array(
        np.full((n_boxes,), temperature, dtype=np.float64),
        dtype=wp.float64,
        device=device,
    )
    pressure_array = wp.array(
        np.full((n_boxes,), pressure, dtype=np.float64),
        dtype=wp.float64,
        device=device,
    )
    volume_source = particles.volume if volume is None else volume
    volume_array = _ensure_volume_array(volume_source, n_boxes, device)
    radii = wp.zeros((n_boxes, n_particles), dtype=wp.float64, device=device)
    diffusivities = wp.zeros(
        (n_boxes, n_particles), dtype=wp.float64, device=device
    )
    g_terms = wp.zeros((n_boxes, n_particles), dtype=wp.float64, device=device)
    speeds = wp.zeros((n_boxes, n_particles), dtype=wp.float64, device=device)
    active_indices = wp.zeros(
        (n_boxes, n_particles), dtype=wp.int32, device=device
    )
    collision_pairs = wp.zeros(
        (n_boxes, collision_capacity, 2), dtype=wp.int32, device=device
    )
    n_collisions = wp.zeros((n_boxes,), dtype=wp.int32, device=device)
    scheduled_trial_counts = wp.zeros((n_boxes,), dtype=wp.int32, device=device)
    executed_trial_counts = wp.zeros((n_boxes,), dtype=wp.int32, device=device)
    diagnostic_rng_states = wp.zeros((n_boxes,), dtype=wp.uint32, device=device)
    initialize_coagulation_rng_states(
        rng_seed, diagnostic_rng_states, device=device
    )

    wp.launch(
        _brownian_coagulation_attempt_diagnostic_kernel,
        dim=(n_boxes,),
        inputs=[
            diagnostic_particles.masses,
            diagnostic_particles.concentration,
            diagnostic_particles.density,
            volume_array,
            temperature_array,
            pressure_array,
            wp.float64(constants.GAS_CONSTANT),
            wp.float64(constants.BOLTZMANN_CONSTANT),
            wp.float64(constants.MOLECULAR_WEIGHT_AIR),
            wp.float64(constants.REF_VISCOSITY_AIR_STP),
            wp.float64(constants.REF_TEMPERATURE_STP),
            wp.float64(constants.SUTHERLAND_CONSTANT),
            wp.float64(time_step),
            radii,
            diffusivities,
            g_terms,
            speeds,
            active_indices,
            collision_pairs,
            n_collisions,
            scheduled_trial_counts,
            executed_trial_counts,
            diagnostic_rng_states,
            wp.int32(collision_capacity),
        ],
        device=device,
    )
    wp.launch(
        apply_coagulation_kernel,
        dim=(n_boxes, collision_capacity),
        inputs=[
            diagnostic_particles.masses,
            diagnostic_particles.concentration,
            diagnostic_particles.charge,
            collision_pairs,
            n_collisions,
        ],
        device=device,
    )
    wp.launch(
        _compact_applied_collision_pairs_kernel,
        dim=1,
        inputs=[collision_pairs, n_collisions],
        device=device,
    )
    wp.synchronize()
    diagnostic_result = from_warp_particle_data(diagnostic_particles, sync=True)

    production_particles = to_warp_particle_data(particles, device=device)
    production_collision_pairs = wp.zeros(
        (n_boxes, collision_capacity, 2), dtype=wp.int32, device=device
    )
    production_counts = wp.zeros((n_boxes,), dtype=wp.int32, device=device)
    production_rng_states = wp.zeros((n_boxes,), dtype=wp.uint32, device=device)
    initialize_coagulation_rng_states(
        rng_seed, production_rng_states, device=device
    )
    _, _, production_counts = coagulation_step_gpu(
        production_particles,
        temperature=temperature,
        pressure=pressure,
        time_step=time_step,
        volume=volume_source,
        max_collisions=max_collisions,
        rng_seed=rng_seed,
        collision_pairs=production_collision_pairs,
        n_collisions=production_counts,
        rng_states=production_rng_states,
    )
    wp.synchronize()
    production_result = from_warp_particle_data(production_particles, sync=True)

    return _AttemptDiagnostics(
        scheduled_trial_counts=np.asarray(
            scheduled_trial_counts.numpy(), dtype=np.int64
        ),
        executed_trial_counts=np.asarray(
            executed_trial_counts.numpy(), dtype=np.int64
        ),
        accepted_counts=np.asarray(n_collisions.numpy(), dtype=np.int64),
        production_counts=np.asarray(production_counts.numpy(), dtype=np.int64),
        diagnostic_pairs=np.asarray(collision_pairs.numpy(), dtype=np.int64),
        production_pairs=np.asarray(
            production_collision_pairs.numpy(), dtype=np.int64
        ),
        diagnostic_masses=np.asarray(
            diagnostic_result.masses, dtype=np.float64
        ),
        production_masses=np.asarray(
            production_result.masses, dtype=np.float64
        ),
        diagnostic_concentration=np.asarray(
            diagnostic_result.concentration, dtype=np.float64
        ),
        production_concentration=np.asarray(
            production_result.concentration, dtype=np.float64
        ),
        diagnostic_rng_states=np.asarray(
            diagnostic_rng_states.numpy(), dtype=np.uint32
        ),
        production_rng_states=np.asarray(
            production_rng_states.numpy(), dtype=np.uint32
        ),
    )


@_warp_kernel
def _draw_single_random_kernel(rng_states: Any, draws: Any) -> None:
    """Draw one random float from each RNG state for deterministic probing."""
    box_idx = wp.tid()
    state = rng_states[box_idx]
    draws[box_idx] = wp.float64(wp.randf(state))


def test_coagulation_step_gpu_signature_keeps_runtime_options_keyword_only() -> (
    None
):
    """Configuration, RNG reset, and environment inputs stay keyword-only."""
    parameters = inspect.signature(coagulation_step_gpu).parameters

    assert parameters["mechanism_config"].default is None
    assert parameters["mechanism_config"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["initialize_rng"].default is False
    assert parameters["initialize_rng"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["environment"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["turbulent_dissipation"].default is None
    assert (
        parameters["turbulent_dissipation"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )
    assert parameters["fluid_density"].default is None
    assert parameters["fluid_density"].kind is inspect.Parameter.KEYWORD_ONLY


class _ParticlesAccessSentinel:
    """Raise if configuration preflight touches particle runtime state."""

    @property
    def masses(self) -> Any:
        """Reject any runtime particle access."""
        raise AssertionError(
            "configuration validation accessed particles.masses"
        )


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (object(), "mechanism_config must be a CoagulationMechanismConfig."),
        (
            CoagulationMechanismConfig(mechanisms=()),
            "mechanisms must be a non-empty tuple of strings.",
        ),
        (
            CoagulationMechanismConfig(mechanisms=[BROWNIAN_MECHANISM]),  # type: ignore[arg-type]
            "mechanisms must be a non-empty tuple of strings.",
        ),
        (
            CoagulationMechanismConfig(
                mechanisms=(BROWNIAN_MECHANISM, BROWNIAN_MECHANISM)
            ),
            "Duplicate coagulation mechanism 'brownian'.",
        ),
        (
            CoagulationMechanismConfig(mechanisms=("unknown",)),
            "Unknown coagulation mechanism 'unknown'.",
        ),
        (
            CoagulationMechanismConfig(distribution_type="discrete"),
            "distribution_type must be exactly 'particle_resolved'.",
        ),
        (
            CoagulationMechanismConfig(
                mechanisms=(
                    BROWNIAN_MECHANISM,
                    CHARGED_HARD_SPHERE_MECHANISM,
                    SEDIMENTATION_SP2016_MECHANISM,
                )
            ),
            "Additive coagulation execution is deferred.",
        ),
    ],
)
def test_coagulation_step_gpu_rejects_config_before_runtime_access(
    config: object,
    message: str,
) -> None:
    """Rejected configuration never accesses runtime particle state."""
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        coagulation_step_gpu(
            _ParticlesAccessSentinel(),
            temperature=None,
            pressure=None,
            time_step=object(),
            mechanism_config=config,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("n_boxes", [1, 2])
@pytest.mark.parametrize(
    ("config", "message"),
    [
        (object(), "mechanism_config must be a CoagulationMechanismConfig."),
        (
            CoagulationMechanismConfig(mechanisms=()),
            "mechanisms must be a non-empty tuple of strings.",
        ),
        (
            CoagulationMechanismConfig(
                mechanisms="brownian"  # type: ignore[arg-type]
            ),
            "mechanisms must be a non-empty tuple of strings.",
        ),
        (
            CoagulationMechanismConfig(
                mechanisms=[BROWNIAN_MECHANISM]  # type: ignore[arg-type]
            ),
            "mechanisms must be a non-empty tuple of strings.",
        ),
        (
            CoagulationMechanismConfig(
                mechanisms=(BROWNIAN_MECHANISM, 1)  # type: ignore[arg-type]
            ),
            "mechanisms must contain only string identifiers.",
        ),
        (
            CoagulationMechanismConfig(
                mechanisms=(BROWNIAN_MECHANISM, BROWNIAN_MECHANISM)
            ),
            "Duplicate coagulation mechanism 'brownian'.",
        ),
        (
            CoagulationMechanismConfig(mechanisms=("unknown",)),
            "Unknown coagulation mechanism 'unknown'.",
        ),
        (
            CoagulationMechanismConfig(distribution_type="continuous_pdf"),
            "distribution_type must be exactly 'particle_resolved'.",
        ),
        (
            CoagulationMechanismConfig(distribution_type="discrete"),
            "distribution_type must be exactly 'particle_resolved'.",
        ),
        (
            CoagulationMechanismConfig(
                mechanisms=(
                    BROWNIAN_MECHANISM,
                    CHARGED_HARD_SPHERE_MECHANISM,
                    SEDIMENTATION_SP2016_MECHANISM,
                )
            ),
            "Additive coagulation execution is deferred.",
        ),
    ],
)
def test_coagulation_step_gpu_rejected_config_preserves_caller_state(
    device: str,
    n_boxes: int,
    config: object,
    message: str,
) -> None:
    """Rejected configuration leaves every supplied mutable input unchanged."""
    particles = _make_particle_data(n_boxes=n_boxes, n_particles=4, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    collision_pairs = wp.array(
        np.full((n_boxes, 2, 2), -7, dtype=np.int32),
        dtype=wp.int32,
        device=device,
    )
    collision_counts = wp.array(
        np.arange(1, n_boxes + 1, dtype=np.int32),
        dtype=wp.int32,
        device=device,
    )
    rng_states = wp.array(
        np.arange(11, 11 + n_boxes, dtype=np.uint32),
        dtype=wp.uint32,
        device=device,
    )
    caller_buffers = (collision_pairs, collision_counts, rng_states)
    particle_arrays = (
        gpu_particles.masses,
        gpu_particles.concentration,
        gpu_particles.charge,
    )
    initial_particles = from_warp_particle_data(gpu_particles, sync=True)
    initial_pairs = np.asarray(collision_pairs.numpy()).copy()
    initial_counts = np.asarray(collision_counts.numpy()).copy()
    initial_rng_states = np.asarray(rng_states.numpy()).copy()

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        coagulation_step_gpu(
            gpu_particles,
            temperature=298.15,
            pressure=101325.0,
            time_step=0.1,
            max_collisions=4,
            collision_pairs=collision_pairs,
            n_collisions=collision_counts,
            rng_states=rng_states,
            initialize_rng=True,
            mechanism_config=config,  # type: ignore[arg-type]
        )

    assert particle_arrays[0] is gpu_particles.masses
    assert particle_arrays[1] is gpu_particles.concentration
    assert particle_arrays[2] is gpu_particles.charge
    assert collision_pairs is caller_buffers[0]
    assert collision_counts is caller_buffers[1]
    assert rng_states is caller_buffers[2]
    npt.assert_array_equal(np.asarray(collision_pairs.numpy()), initial_pairs)
    npt.assert_array_equal(np.asarray(collision_counts.numpy()), initial_counts)
    npt.assert_array_equal(np.asarray(rng_states.numpy()), initial_rng_states)
    restored = from_warp_particle_data(gpu_particles, sync=True)
    npt.assert_allclose(restored.masses, initial_particles.masses)
    npt.assert_allclose(restored.concentration, initial_particles.concentration)
    npt.assert_allclose(restored.charge, initial_particles.charge)


@pytest.mark.parametrize(
    ("mechanisms", "message"),
    [
        (
            (
                BROWNIAN_MECHANISM,
                CHARGED_HARD_SPHERE_MECHANISM,
                SEDIMENTATION_SP2016_MECHANISM,
            ),
            "Additive coagulation execution is deferred.",
        ),
        (
            (
                BROWNIAN_MECHANISM,
                CHARGED_HARD_SPHERE_MECHANISM,
                TURBULENT_SHEAR_ST1956_MECHANISM,
            ),
            "Additive coagulation execution is deferred.",
        ),
        (
            (
                BROWNIAN_MECHANISM,
                SEDIMENTATION_SP2016_MECHANISM,
                TURBULENT_SHEAR_ST1956_MECHANISM,
            ),
            "Additive coagulation execution is deferred.",
        ),
        (
            (
                CHARGED_HARD_SPHERE_MECHANISM,
                SEDIMENTATION_SP2016_MECHANISM,
                TURBULENT_SHEAR_ST1956_MECHANISM,
            ),
            "Additive coagulation execution is deferred.",
        ),
    ],
)
def test_deferred_masks_preserve_caller_owned_state(
    device: str, mechanisms: tuple[str, ...], message: str
) -> None:
    """Deferred masks finish read-only preflight without downstream mutation."""
    particles = _make_particle_data(n_boxes=1, n_particles=4, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    collision_pairs = wp.array(
        np.full((1, 2, 2), -7, dtype=np.int32), dtype=wp.int32, device=device
    )
    collision_counts = wp.array([3], dtype=wp.int32, device=device)
    rng_states = wp.array([11], dtype=wp.uint32, device=device)
    initial_particles = from_warp_particle_data(gpu_particles, sync=True)
    initial_pairs = collision_pairs.numpy().copy()
    initial_counts = collision_counts.numpy().copy()
    initial_rng_states = rng_states.numpy().copy()
    turbulent_enabled = TURBULENT_SHEAR_ST1956_MECHANISM in mechanisms

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        coagulation_step_gpu(
            gpu_particles,
            temperature=298.15,
            pressure=101325.0,
            time_step=0.1,
            max_collisions=2,
            collision_pairs=collision_pairs,
            n_collisions=collision_counts,
            rng_states=rng_states,
            initialize_rng=True,
            mechanism_config=CoagulationMechanismConfig(mechanisms=mechanisms),
            turbulent_dissipation=1.0 if turbulent_enabled else None,
            fluid_density=1.2 if turbulent_enabled else None,
        )

    restored = from_warp_particle_data(gpu_particles, sync=True)
    npt.assert_allclose(restored.masses, initial_particles.masses)
    npt.assert_allclose(restored.concentration, initial_particles.concentration)
    npt.assert_allclose(restored.charge, initial_particles.charge)
    npt.assert_array_equal(collision_pairs.numpy(), initial_pairs)
    npt.assert_array_equal(collision_counts.numpy(), initial_counts)
    npt.assert_array_equal(rng_states.numpy(), initial_rng_states)


@pytest.mark.parametrize(
    (
        "mechanisms",
        "invalid_field",
        "invalid_value",
        "turbulent_dissipation",
        "fluid_density",
        "message",
    ),
    [
        (
            (
                BROWNIAN_MECHANISM,
                CHARGED_HARD_SPHERE_MECHANISM,
                SEDIMENTATION_SP2016_MECHANISM,
            ),
            "density",
            0.0,
            None,
            None,
            "Additive coagulation execution is deferred.",
        ),
        (
            (
                BROWNIAN_MECHANISM,
                CHARGED_HARD_SPHERE_MECHANISM,
                TURBULENT_SHEAR_ST1956_MECHANISM,
            ),
            None,
            None,
            0.0,
            1000.0,
            "turbulent_dissipation must be finite and > 0",
        ),
        (
            (
                BROWNIAN_MECHANISM,
                SEDIMENTATION_SP2016_MECHANISM,
                TURBULENT_SHEAR_ST1956_MECHANISM,
            ),
            "masses",
            -1.0,
            1.0,
            1000.0,
            "turbulent particle charge, masses, and concentrations must be "
            "finite; masses and concentrations must be nonnegative; density "
            "must be finite and > 0; active mass and volume must be finite "
            "and > 0",
        ),
    ],
)
def test_deferred_masks_reject_invalid_enabled_terms_atomically(
    device: str,
    mechanisms: tuple[str, ...],
    invalid_field: str | None,
    invalid_value: float | None,
    turbulent_dissipation: float | None,
    fluid_density: float | None,
    message: str,
) -> None:
    """Deferred masks preserve their established validation ordering."""
    particles = _make_particle_data(n_boxes=1, n_particles=4, n_species=1)
    if invalid_field is not None:
        getattr(particles, invalid_field)[...] = invalid_value
    gpu_particles = to_warp_particle_data(particles, device=device)
    collision_pairs = wp.array(
        np.full((1, 2, 2), -7, dtype=np.int32), dtype=wp.int32, device=device
    )
    collision_counts = wp.array([3], dtype=wp.int32, device=device)
    rng_states = wp.array([11], dtype=wp.uint32, device=device)
    particle_arrays = (
        gpu_particles.masses,
        gpu_particles.concentration,
        gpu_particles.charge,
    )
    initial_particles = from_warp_particle_data(gpu_particles, sync=True)
    initial_pairs = np.asarray(collision_pairs.numpy()).copy()
    initial_counts = np.asarray(collision_counts.numpy()).copy()
    initial_rng_states = np.asarray(rng_states.numpy()).copy()

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        coagulation_step_gpu(
            gpu_particles,
            temperature=298.15,
            pressure=101325.0,
            time_step=0.1,
            max_collisions=2,
            collision_pairs=collision_pairs,
            n_collisions=collision_counts,
            rng_states=rng_states,
            initialize_rng=True,
            mechanism_config=CoagulationMechanismConfig(mechanisms=mechanisms),
            turbulent_dissipation=turbulent_dissipation,
            fluid_density=fluid_density,
        )

    assert particle_arrays[0] is gpu_particles.masses
    assert particle_arrays[1] is gpu_particles.concentration
    assert particle_arrays[2] is gpu_particles.charge
    npt.assert_array_equal(np.asarray(collision_pairs.numpy()), initial_pairs)
    npt.assert_array_equal(np.asarray(collision_counts.numpy()), initial_counts)
    npt.assert_array_equal(np.asarray(rng_states.numpy()), initial_rng_states)
    restored = from_warp_particle_data(gpu_particles, sync=True)
    npt.assert_allclose(restored.masses, initial_particles.masses)
    npt.assert_allclose(restored.concentration, initial_particles.concentration)
    npt.assert_allclose(restored.charge, initial_particles.charge)


@pytest.mark.parametrize(
    ("mechanisms", "message"),
    [
        (
            (
                BROWNIAN_MECHANISM,
                CHARGED_HARD_SPHERE_MECHANISM,
                SEDIMENTATION_SP2016_MECHANISM,
            ),
            "Additive coagulation execution is deferred.",
        ),
        (
            (
                BROWNIAN_MECHANISM,
                CHARGED_HARD_SPHERE_MECHANISM,
                TURBULENT_SHEAR_ST1956_MECHANISM,
            ),
            "Additive coagulation execution is deferred.",
        ),
        (
            (
                BROWNIAN_MECHANISM,
                SEDIMENTATION_SP2016_MECHANISM,
                TURBULENT_SHEAR_ST1956_MECHANISM,
            ),
            "Additive coagulation execution is deferred.",
        ),
        (
            (
                CHARGED_HARD_SPHERE_MECHANISM,
                SEDIMENTATION_SP2016_MECHANISM,
                TURBULENT_SHEAR_ST1956_MECHANISM,
            ),
            "Additive coagulation execution is deferred.",
        ),
    ],
)
def test_deferred_masks_bypass_downstream_runtime_helpers(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    mechanisms: tuple[str, ...],
    message: str,
) -> None:
    """Deferred masks retain their capability errors before runtime work."""
    particles = _make_particle_data(n_boxes=1, n_particles=4, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    turbulent_enabled = TURBULENT_SHEAR_ST1956_MECHANISM in mechanisms

    def _unexpected(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("deferred mask reached downstream runtime work")

    for helper_name in (
        "_normalize_turbulent_input_array",
        "_validate_time_step",
        "_ensure_environment_arrays",
        "_ensure_volume_array",
        "_resolve_collision_capacity",
        "_validate_collision_counts",
        "_validate_rng_states",
        "initialize_coagulation_rng_states",
    ):
        monkeypatch.setattr(coagulation_module, helper_name, _unexpected)
    monkeypatch.setattr(
        coagulation_module, "brownian_coagulation_kernel", _unexpected
    )
    monkeypatch.setattr(
        coagulation_module, "apply_coagulation_kernel", _unexpected
    )

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        coagulation_step_gpu(
            gpu_particles,
            temperature=298.15,
            pressure=101325.0,
            time_step=0.1,
            mechanism_config=CoagulationMechanismConfig(mechanisms=mechanisms),
            turbulent_dissipation=1.0 if turbulent_enabled else None,
            fluid_density=1000.0 if turbulent_enabled else None,
        )


@pytest.mark.parametrize("source", ["direct_arrays", "environment"])
def test_coagulation_config_error_precedes_environment_normalization(
    device: str,
    source: str,
) -> None:
    """Configuration rejection wins over invalid direct and sidecar sources."""
    particles = _make_particle_data(n_boxes=2, n_particles=3, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    invalid_config = CoagulationMechanismConfig(mechanisms=())

    if source == "direct_arrays":
        temperature = wp.array([0.0, 0.0], dtype=wp.float64, device=device)
        pressure = wp.array([0.0, 0.0], dtype=wp.float64, device=device)
        environment = None
    else:
        environment = to_warp_environment_data(
            _make_environment_data(n_boxes=2, n_species=1), device=device
        )
        environment.pressure = wp.array(
            [0.0, 0.0], dtype=wp.float64, device=device
        )
        temperature = 298.15
        pressure = 101325.0

    with pytest.raises(
        ValueError,
        match="^mechanisms must be a non-empty tuple of strings\\.$",
    ):
        coagulation_step_gpu(
            gpu_particles,
            temperature=temperature,
            pressure=pressure,
            time_step=0.1,
            environment=environment,
            mechanism_config=invalid_config,
        )


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (object(), "mechanism_config must be a CoagulationMechanismConfig."),
        (
            CoagulationMechanismConfig(mechanisms=()),
            "mechanisms must be a non-empty tuple of strings.",
        ),
        (
            CoagulationMechanismConfig(distribution_type="discrete"),
            "distribution_type must be exactly 'particle_resolved'.",
        ),
    ],
)
def test_coagulation_config_rejection_bypasses_runtime_helpers(
    monkeypatch: pytest.MonkeyPatch,
    config: object,
    message: str,
) -> None:
    """Configuration errors occur before validation, allocation, or launch."""

    def _unexpected_runtime_access(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("configuration rejection accessed runtime state")

    for helper_name in (
        "_validate_particle_arrays",
        "_validate_device_arrays",
        "_validate_time_step",
        "_ensure_environment_arrays",
        "_ensure_volume_array",
        "_resolve_collision_capacity",
        "_validate_rng_states",
        "initialize_coagulation_rng_states",
    ):
        monkeypatch.setattr(
            coagulation_module, helper_name, _unexpected_runtime_access
        )
    monkeypatch.setattr(
        coagulation_module.wp, "zeros", _unexpected_runtime_access
    )
    monkeypatch.setattr(
        coagulation_module.wp, "launch", _unexpected_runtime_access
    )

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        coagulation_step_gpu(
            _ParticlesAccessSentinel(),
            temperature=None,
            pressure=None,
            time_step=object(),
            mechanism_config=config,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("name", ["turbulent_dissipation", "fluid_density"])
@pytest.mark.parametrize("value", [1.5, np.float64(2.5)])
def test_ensure_turbulent_input_array_broadcasts_floating_scalars(
    device: str,
    name: str,
    value: float | np.floating[Any],
) -> None:
    """Turbulent floating scalars broadcast into private per-box storage."""
    result = _ensure_turbulent_input_array(
        name, value, 2, wp.get_device(device)
    )

    assert result.shape == (2,)
    assert result.dtype == wp.float64
    npt.assert_allclose(np.asarray(result.numpy()), float(value))


@pytest.mark.parametrize("name", ["turbulent_dissipation", "fluid_density"])
@pytest.mark.parametrize("dtype", [wp.float64])
def test_ensure_turbulent_input_array_reuses_valid_warp_arrays(
    device: str,
    name: str,
    dtype: Any,
) -> None:
    """Valid heterogeneous per-box Warp inputs retain their identity and dtype."""
    expected = np.array([1.5, 2.5], dtype=np.float64)
    value = wp.array(expected, dtype=dtype, device=device)

    result = _ensure_turbulent_input_array(
        name, value, 2, wp.get_device(device)
    )

    assert result is value
    assert result.dtype == dtype
    npt.assert_allclose(np.asarray(result.numpy()), expected, rtol=1e-6)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (None, "floating scalar"),
        (True, "floating scalar"),
        (1, "floating scalar"),
        ("1.0", "floating scalar"),
        (np.array([1.0]), "shape"),
        (0.0, "finite and > 0"),
        (-1.0, "finite and > 0"),
        (np.nan, "finite and > 0"),
        (np.inf, "finite and > 0"),
    ],
)
def test_ensure_turbulent_input_array_rejects_invalid_host_values(
    device: str,
    value: object,
    message: str,
) -> None:
    """Turbulent scalar contracts reject invalid host values by name."""
    with pytest.raises(
        ValueError,
        match=f"turbulent_dissipation.*{re.escape(message)}",
    ):
        _ensure_turbulent_input_array("turbulent_dissipation", value, 2, device)


@pytest.mark.parametrize(
    ("value_factory", "message"),
    [
        (
            lambda device: wp.array(
                [1.0, 2.0, 3.0], dtype=wp.float64, device=device
            ),
            "shape",
        ),
        (
            lambda device: wp.array([1, 2], dtype=wp.int32, device=device),
            "dtype float64",
        ),
        (
            lambda device: wp.array(
                [1.0, 2.0], dtype=wp.float32, device=device
            ),
            "dtype float64",
        ),
        (
            lambda device: wp.array(
                [1.0, 0.0], dtype=wp.float64, device=device
            ),
            "finite and > 0",
        ),
    ],
)
def test_ensure_turbulent_input_array_rejects_invalid_warp_arrays(
    device: str,
    value_factory: Any,
    message: str,
) -> None:
    """Turbulent Warp inputs reject malformed shape, dtype, and values."""
    value = value_factory(device)

    with pytest.raises(
        ValueError,
        match=f"fluid_density.*{re.escape(message)}",
    ):
        _ensure_turbulent_input_array(
            "fluid_density", value, 2, wp.get_device(device)
        )


def test_ensure_turbulent_input_array_rejects_different_device(
    device: str,
) -> None:
    """Turbulent arrays must reside on the active particle device."""
    alternatives = [
        candidate for candidate in warp_devices(wp) if candidate != device
    ]
    if not alternatives:
        pytest.skip("No second Warp device available")
    value = wp.array([1.0, 2.0], dtype=wp.float64, device=alternatives[0])

    with pytest.raises(ValueError, match="fluid_density.*device"):
        _ensure_turbulent_input_array(
            "fluid_density", value, 2, wp.get_device(device)
        )


@pytest.mark.parametrize(
    ("turbulent_dissipation", "fluid_density", "message"),
    [
        (None, 1000.0, "turbulent_dissipation.*floating scalar"),
        (1.0, None, "fluid_density.*floating scalar"),
        (True, 1000.0, "turbulent_dissipation.*floating scalar"),
        (1.0, 1000, "fluid_density.*floating scalar"),
        ("invalid", 1000.0, "turbulent_dissipation.*floating scalar"),
        (1.0, np.array([1000.0]), "fluid_density.*shape"),
        (0.0, 1000.0, "turbulent_dissipation.*finite and > 0"),
        (1.0, 0.0, "fluid_density.*finite and > 0"),
        (np.nan, 1000.0, "turbulent_dissipation.*finite and > 0"),
        (1.0, np.inf, "fluid_density.*finite and > 0"),
    ],
)
def test_turbulent_p2_preflight_precedes_capability_and_runtime_work(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    turbulent_dissipation: object,
    fluid_density: object,
    message: str,
) -> None:
    """Invalid P2 input stops before downstream allocation, RNG, or launch."""
    particles = _make_particle_data(n_boxes=2, n_particles=3, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    pairs = wp.array(np.full((2, 1, 2), -7), dtype=wp.int32, device=device)
    counts = wp.array([3, 4], dtype=wp.int32, device=device)
    rng_states = wp.array([5, 6], dtype=wp.uint32, device=device)
    initial_particles = from_warp_particle_data(gpu_particles, sync=True)
    initial_pairs = np.asarray(pairs.numpy()).copy()
    initial_counts = np.asarray(counts.numpy()).copy()
    initial_rng = np.asarray(rng_states.numpy()).copy()
    particle_arrays = (
        gpu_particles.masses,
        gpu_particles.concentration,
        gpu_particles.charge,
    )

    def _unexpected(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("invalid P2 input reached downstream work")

    for helper_name in (
        "_broadcast_scalar_array",
        "_ensure_environment_arrays",
        "_ensure_volume_array",
        "initialize_coagulation_rng_states",
    ):
        monkeypatch.setattr(coagulation_module, helper_name, _unexpected)
    monkeypatch.setattr(coagulation_module.wp, "zeros", _unexpected)
    monkeypatch.setattr(coagulation_module.wp, "launch", _unexpected)

    with pytest.raises(
        ValueError,
        match=message,
    ):
        coagulation_step_gpu(
            gpu_particles,
            temperature=298.15,
            pressure=101325.0,
            time_step=0.1,
            collision_pairs=pairs,
            n_collisions=counts,
            rng_states=rng_states,
            initialize_rng=True,
            mechanism_config=CoagulationMechanismConfig(
                mechanisms=(TURBULENT_SHEAR_ST1956_MECHANISM,)
            ),
            turbulent_dissipation=turbulent_dissipation,
            fluid_density=fluid_density,
        )

    restored = from_warp_particle_data(gpu_particles, sync=True)
    assert particle_arrays[0] is gpu_particles.masses
    assert particle_arrays[1] is gpu_particles.concentration
    assert particle_arrays[2] is gpu_particles.charge
    npt.assert_allclose(restored.masses, initial_particles.masses)
    npt.assert_allclose(restored.concentration, initial_particles.concentration)
    npt.assert_allclose(restored.charge, initial_particles.charge)
    npt.assert_array_equal(np.asarray(pairs.numpy()), initial_pairs)
    npt.assert_array_equal(np.asarray(counts.numpy()), initial_counts)
    npt.assert_array_equal(np.asarray(rng_states.numpy()), initial_rng)


def test_brownian_ignores_turbulent_p2_inputs(device: str) -> None:
    """Non-turbulent calls do not validate irrelevant turbulent P2 inputs."""
    particles = _make_particle_data(n_boxes=1, n_particles=3, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)

    result, _, _ = coagulation_step_gpu(
        gpu_particles,
        temperature=298.15,
        pressure=101325.0,
        time_step=0.1,
        turbulent_dissipation="invalid",
        fluid_density=np.array([0.0]),
    )

    assert result is gpu_particles


@pytest.mark.gpu_parity
@pytest.mark.parametrize("use_arrays", [False, True])
def test_turbulent_singleton_executes_and_conserves_mass(
    device: str,
    use_arrays: bool,
) -> None:
    """ST1956 singleton normalizes P2 inputs and preserves species inventory."""
    particles = _make_particle_data(n_boxes=2, n_particles=2, n_species=2)
    particles.volume[:] = 1.0e-18
    particles.masses[:] = [
        [[1.0e-15, 2.0e-15], [8.0e-15, 16.0e-15]],
        [[3.0e-15, 6.0e-15], [2.7e-14, 5.4e-14]],
    ]
    initial_inventory = np.sum(particles.masses, axis=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    pairs = wp.full((2, 1, 2), -1, dtype=wp.int32, device=device)
    counts = wp.zeros((2,), dtype=wp.int32, device=device)
    states = wp.zeros((2,), dtype=wp.uint32, device=device)
    dissipation: Any = 1.0
    density: Any = np.float64(1000.0)
    if use_arrays:
        dissipation = wp.array([1.0, 2.0], dtype=wp.float64, device=device)
        density = wp.array([1000.0, 997.0], dtype=wp.float64, device=device)

    returned_particles, returned_pairs, returned_counts = coagulation_step_gpu(
        gpu_particles,
        temperature=298.15,
        pressure=101325.0,
        time_step=1.0e308,
        max_collisions=1,
        collision_pairs=pairs,
        n_collisions=counts,
        rng_states=states,
        initialize_rng=True,
        mechanism_config=CoagulationMechanismConfig(
            mechanisms=(TURBULENT_SHEAR_ST1956_MECHANISM,)
        ),
        turbulent_dissipation=dissipation,
        fluid_density=density,
    )

    assert returned_particles is gpu_particles
    assert returned_pairs is pairs
    assert returned_counts is counts
    assert np.all((counts.numpy() >= 0) & (counts.numpy() <= 1))
    result = from_warp_particle_data(gpu_particles, sync=True)
    npt.assert_allclose(
        np.sum(result.masses, axis=1),
        initial_inventory,
        rtol=1.0e-12,
        atol=1.0e-30,
    )


@pytest.mark.parametrize(
    ("invalid_field", "invalid_value"),
    [
        ("masses", np.nan),
        ("masses", np.inf),
        ("masses", -1.0),
        ("concentration", np.nan),
        ("concentration", np.inf),
        ("concentration", -1.0),
        ("density", np.nan),
        ("density", np.inf),
        ("density", 0.0),
        ("density", -1.0),
        ("charge", np.nan),
        ("charge", np.inf),
    ],
)
def test_turbulent_particle_preflight_rejects_before_mutable_runtime_work(
    device: str,
    invalid_field: str,
    invalid_value: float,
) -> None:
    """Invalid ST1956 particle state leaves caller-owned resources unchanged."""
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=1)
    setattr(particles, invalid_field, getattr(particles, invalid_field).copy())
    if invalid_field == "masses":
        particles.masses[0, 0, 0] = invalid_value
    elif invalid_field == "concentration":
        particles.concentration[0, 0] = invalid_value
    elif invalid_field == "charge":
        particles.charge[0, 0] = invalid_value
    else:
        particles.density[0] = invalid_value
    gpu_particles = to_warp_particle_data(particles, device=device)
    pairs = wp.array([[[7, 7]]], dtype=wp.int32, device=device)
    counts = wp.array([7], dtype=wp.int32, device=device)
    states = wp.array([7], dtype=wp.uint32, device=device)
    initial_pairs = np.asarray(pairs.numpy()).copy()
    initial_counts = np.asarray(counts.numpy()).copy()
    initial_states = np.asarray(states.numpy()).copy()
    initial_particles = from_warp_particle_data(gpu_particles, sync=True)

    with pytest.raises(ValueError, match="turbulent particle"):
        coagulation_step_gpu(
            gpu_particles,
            temperature=298.15,
            pressure=101325.0,
            time_step=1.0,
            collision_pairs=pairs,
            n_collisions=counts,
            rng_states=states,
            initialize_rng=True,
            mechanism_config=CoagulationMechanismConfig(
                mechanisms=(TURBULENT_SHEAR_ST1956_MECHANISM,)
            ),
            turbulent_dissipation=1.0,
            fluid_density=1000.0,
        )

    restored = from_warp_particle_data(gpu_particles, sync=True)
    npt.assert_allclose(restored.masses, initial_particles.masses)
    npt.assert_allclose(restored.concentration, initial_particles.concentration)
    npt.assert_allclose(restored.density, initial_particles.density)
    npt.assert_allclose(restored.charge, initial_particles.charge)
    npt.assert_array_equal(pairs.numpy(), initial_pairs)
    npt.assert_array_equal(counts.numpy(), initial_counts)
    npt.assert_array_equal(states.numpy(), initial_states)


def test_turbulent_particle_preflight_rejects_overflowed_active_totals(
    device: str,
) -> None:
    """ST1956 rejects finite masses whose derived active total overflows."""
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=2)
    particles.masses[0, 0] = np.finfo(np.float64).max
    gpu_particles = to_warp_particle_data(particles, device=device)
    pairs = wp.array([[[7, 7]]], dtype=wp.int32, device=device)
    counts = wp.array([7], dtype=wp.int32, device=device)
    states = wp.array([7], dtype=wp.uint32, device=device)
    initial_particles = from_warp_particle_data(gpu_particles, sync=True)

    with pytest.raises(ValueError, match="active mass and volume"):
        coagulation_step_gpu(
            gpu_particles,
            temperature=298.15,
            pressure=101325.0,
            time_step=1.0,
            collision_pairs=pairs,
            n_collisions=counts,
            rng_states=states,
            initialize_rng=True,
            mechanism_config=CoagulationMechanismConfig(
                mechanisms=(TURBULENT_SHEAR_ST1956_MECHANISM,)
            ),
            turbulent_dissipation=1.0,
            fluid_density=1000.0,
        )

    _assert_particles_unchanged(gpu_particles, initial_particles)
    npt.assert_array_equal(pairs.numpy(), [[[7, 7]]])
    npt.assert_array_equal(counts.numpy(), [7])
    npt.assert_array_equal(states.numpy(), [7])


@pytest.mark.parametrize(
    "float32_input",
    ["turbulent_dissipation", "fluid_density"],
)
def test_turbulent_singleton_rejects_float32_p2_arrays_without_mutation(
    device: str,
    float32_input: str,
) -> None:
    """ST1956 rejects fp32 P2 arrays before caller-owned state changes."""
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    pairs = wp.array([[[7, 7]]], dtype=wp.int32, device=device)
    counts = wp.array([7], dtype=wp.int32, device=device)
    states = wp.array([7], dtype=wp.uint32, device=device)
    initial_particles = from_warp_particle_data(gpu_particles, sync=True)
    dissipation = wp.array([1.0], dtype=wp.float64, device=device)
    density = wp.array([1000.0], dtype=wp.float64, device=device)
    if float32_input == "turbulent_dissipation":
        dissipation = wp.array([1.0], dtype=wp.float32, device=device)
    else:
        density = wp.array([1000.0], dtype=wp.float32, device=device)

    with pytest.raises(
        ValueError,
        match=f"{float32_input} must use dtype float64",
    ):
        coagulation_step_gpu(
            gpu_particles,
            temperature=298.15,
            pressure=101325.0,
            time_step=1.0,
            collision_pairs=pairs,
            n_collisions=counts,
            rng_states=states,
            initialize_rng=True,
            mechanism_config=CoagulationMechanismConfig(
                mechanisms=(TURBULENT_SHEAR_ST1956_MECHANISM,)
            ),
            turbulent_dissipation=dissipation,
            fluid_density=density,
        )

    _assert_particles_unchanged(gpu_particles, initial_particles)
    npt.assert_array_equal(pairs.numpy(), [[[7, 7]]])
    npt.assert_array_equal(counts.numpy(), [7])
    npt.assert_array_equal(states.numpy(), [7])


@pytest.mark.parametrize(
    ("concentration", "expected_count", "active_slots"),
    [
        (np.zeros((1, 4), dtype=np.float64), 0, set()),
        (np.array([[0.0, 1.0, 0.0, 0.0]]), 0, {1}),
        (np.array([[1.0, 0.0, 0.0, 1.0]]), 1, {0, 3}),
    ],
)
def test_turbulent_singleton_selector_handles_sparse_active_slots(
    device: str,
    concentration: np.ndarray,
    expected_count: int,
    active_slots: set[int],
) -> None:
    """ST1956 public selection handles degenerate and gapped active slots."""
    particles = _make_particle_data(n_boxes=1, n_particles=4, n_species=1)
    particles.concentration = concentration
    particles.volume[:] = 1.0e-18
    gpu_particles = to_warp_particle_data(particles, device=device)
    pairs = wp.full((1, 1, 2), -1, dtype=wp.int32, device=device)
    counts = wp.zeros((1,), dtype=wp.int32, device=device)
    states = wp.zeros((1,), dtype=wp.uint32, device=device)

    _, returned_pairs, returned_counts = coagulation_step_gpu(
        gpu_particles,
        temperature=298.15,
        pressure=101325.0,
        time_step=1.0e308,
        max_collisions=1,
        collision_pairs=pairs,
        n_collisions=counts,
        rng_states=states,
        initialize_rng=True,
        mechanism_config=CoagulationMechanismConfig(
            mechanisms=(TURBULENT_SHEAR_ST1956_MECHANISM,)
        ),
        turbulent_dissipation=1.0,
        fluid_density=1000.0,
    )

    assert returned_pairs is pairs
    assert returned_counts is counts
    count = int(counts.numpy()[0])
    selected = np.asarray(pairs.numpy())[0, :count]
    assert count == expected_count
    if count:
        assert np.all(selected[:, 0] < selected[:, 1])
        assert set(selected.ravel()).issubset(active_slots)


def test_turbulent_selector_keeps_heterogeneous_box_pairs_active_and_sorted(
    device: str,
) -> None:
    """ST1956 selects only initially active, in-bounds pairs per box."""
    particles = _make_particle_data(n_boxes=2, n_particles=4, n_species=1)
    particles.volume[:] = 1.0e-18
    particles.concentration[:] = [
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0, 0.0],
    ]
    initially_active = [{0, 3}, {1, 2}]
    gpu_particles = to_warp_particle_data(particles, device=device)
    pairs = wp.full((2, 1, 2), -1, dtype=wp.int32, device=device)
    counts = wp.zeros((2,), dtype=wp.int32, device=device)
    states = wp.zeros((2,), dtype=wp.uint32, device=device)

    returned_particles, returned_pairs, returned_counts = coagulation_step_gpu(
        gpu_particles,
        temperature=298.15,
        pressure=101325.0,
        time_step=1.0e308,
        max_collisions=1,
        collision_pairs=pairs,
        n_collisions=counts,
        rng_states=states,
        initialize_rng=True,
        mechanism_config=CoagulationMechanismConfig(
            mechanisms=(TURBULENT_SHEAR_ST1956_MECHANISM,)
        ),
        turbulent_dissipation=1.0,
        fluid_density=1000.0,
    )

    assert returned_particles is gpu_particles
    assert returned_pairs is pairs
    assert returned_counts is counts
    npt.assert_array_equal(counts.numpy(), [1, 1])
    for box_idx, pair in enumerate(pairs.numpy()[:, 0]):
        assert 0 <= pair[0] < pair[1] < particles.masses.shape[1]
        assert set(pair).issubset(initially_active[box_idx])


def test_turbulent_persistent_rng_reuses_state_and_explicit_reset_replays(
    device: str,
) -> None:
    """ST1956 advances persistent RNG state unless explicitly reset."""
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=1)
    particles.volume[:] = 1.0e-18
    states = wp.zeros((1,), dtype=wp.uint32, device=device)
    replay_states = wp.zeros((1,), dtype=wp.uint32, device=device)
    initialize_coagulation_rng_states(9, replay_states, device=device)
    pairs = []
    counts = []
    state_snapshots = []
    for rng_states, reset in (
        (states, True),
        (states, False),
        (replay_states, True),
    ):
        gpu_particles = to_warp_particle_data(particles, device=device)
        collision_pairs = wp.full((1, 1, 2), -1, dtype=wp.int32, device=device)
        n_collisions = wp.zeros((1,), dtype=wp.int32, device=device)
        coagulation_step_gpu(
            gpu_particles,
            temperature=298.15,
            pressure=101325.0,
            time_step=1.0e308,
            max_collisions=1,
            rng_seed=41,
            collision_pairs=collision_pairs,
            n_collisions=n_collisions,
            rng_states=rng_states,
            initialize_rng=reset,
            mechanism_config=CoagulationMechanismConfig(
                mechanisms=(TURBULENT_SHEAR_ST1956_MECHANISM,)
            ),
            turbulent_dissipation=1.0,
            fluid_density=1000.0,
        )
        pairs.append(np.asarray(collision_pairs.numpy()).copy())
        counts.append(np.asarray(n_collisions.numpy()).copy())
        state_snapshots.append(np.asarray(rng_states.numpy()).copy())

    assert not np.array_equal(state_snapshots[0], state_snapshots[1])
    npt.assert_array_equal(pairs[0], pairs[2])
    npt.assert_array_equal(counts[0], counts[2])
    npt.assert_array_equal(state_snapshots[0], state_snapshots[2])


@pytest.mark.stochastic
def test_turbulent_fresh_seed_acceptance_matches_rate_to_majorant_ratio(
    device: str,
) -> None:
    """ST1956 fresh-seed acceptance stays within a documented 4-sigma bound."""
    temperature = 298.15
    fluid_density = 1000.0
    dissipation = 1.0
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=1)
    particles.masses[:] = 1.0e-15
    radius = (3.0e-15 / (4.0 * np.pi * particles.density[0])) ** (1.0 / 3.0)
    viscosity = (
        constants.REF_VISCOSITY_AIR_STP
        * (temperature / constants.REF_TEMPERATURE_STP) ** 1.5
        * (constants.REF_TEMPERATURE_STP + constants.SUTHERLAND_CONSTANT)
        / (temperature + constants.SUTHERLAND_CONSTANT)
    )
    rate = np.sqrt(np.pi * dissipation / (120.0 * (viscosity / fluid_density)))
    rate *= (4.0 * radius) ** 3
    trial_probability = 0.2
    observations = 64
    accepted = 0
    for seed in range(observations):
        gpu_particles = to_warp_particle_data(particles, device=device)
        _, _, counts = coagulation_step_gpu(
            gpu_particles,
            temperature=temperature,
            pressure=101325.0,
            time_step=1.0,
            volume=rate / trial_probability,
            max_collisions=1,
            rng_seed=seed,
            mechanism_config=CoagulationMechanismConfig(
                mechanisms=(TURBULENT_SHEAR_ST1956_MECHANISM,)
            ),
            turbulent_dissipation=dissipation,
            fluid_density=fluid_density,
        )
        accepted += int(counts.numpy()[0])

    expected = observations * trial_probability
    sigma = np.sqrt(
        observations * trial_probability * (1.0 - trial_probability)
    )
    assert abs(accepted - expected) <= 4.0 * sigma + 1.0
    assert accepted > 0


def test_turbulent_three_way_mask_rejects_before_runtime_mutation(
    device: str,
) -> None:
    """Deferred turbulent three-way masks reject before runtime mutation."""
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    pairs = wp.array([[[7, 7]]], dtype=wp.int32, device=device)
    counts = wp.array([7], dtype=wp.int32, device=device)
    states = wp.array([7], dtype=wp.uint32, device=device)
    initial_mass = gpu_particles.masses.numpy().copy()

    with pytest.raises(
        ValueError,
        match="^Additive coagulation execution is deferred\\.$",
    ):
        coagulation_step_gpu(
            gpu_particles,
            temperature=298.15,
            pressure=101325.0,
            time_step=1.0,
            collision_pairs=pairs,
            n_collisions=counts,
            rng_states=states,
            initialize_rng=True,
            mechanism_config=CoagulationMechanismConfig(
                mechanisms=(
                    BROWNIAN_MECHANISM,
                    CHARGED_HARD_SPHERE_MECHANISM,
                    TURBULENT_SHEAR_ST1956_MECHANISM,
                )
            ),
            turbulent_dissipation=1.0,
            fluid_density=1000.0,
        )

    npt.assert_equal(gpu_particles.masses.numpy(), initial_mass)
    npt.assert_array_equal(pairs.numpy(), [[[7, 7]]])
    npt.assert_array_equal(counts.numpy(), [7])
    npt.assert_array_equal(states.numpy(), [7])


@_warp_kernel
def _turbulent_rate_and_majorant_probe_kernel(
    active_indices: Any,
    active_counts: Any,
    radii: Any,
    dissipation: Any,
    kinematic_viscosity: Any,
    temperature: Any,
    pressure: Any,
    pair_rates: Any,
    dispatched_pair_rates: Any,
    majorants: Any,
    dispatched_majorants: Any,
) -> None:
    """Probe ST1956 pair-rate dispatch and compact-active majorants."""
    box_idx = wp.tid()
    pair_rates[box_idx] = turbulent_shear_st1956_pair_rate_wp(
        radii[box_idx, 0],
        radii[box_idx, 1],
        dissipation[box_idx],
        kinematic_viscosity[box_idx],
    )
    dispatched_pair_rates[box_idx] = _total_pair_rate(
        wp.int32(TURBULENT_SHEAR_ST1956_MECHANISM_FLAG),
        radii[box_idx, 0],
        radii[box_idx, 1],
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        dissipation[box_idx],
        kinematic_viscosity[box_idx],
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(1.0),
        wp.float64(1.0),
        wp.float64(1.0),
        wp.float64(1.0),
        wp.float64(1.0),
        wp.float64(1.0),
        wp.float64(1.0),
        wp.float64(1.0),
    )
    majorants[box_idx] = _turbulent_majorant_from_active_radii(
        active_indices,
        box_idx,
        active_counts[box_idx],
        radii,
        dissipation[box_idx],
        kinematic_viscosity[box_idx],
    )
    dispatched_majorants[box_idx] = _total_majorant(
        wp.int32(TURBULENT_SHEAR_ST1956_MECHANISM_FLAG),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        active_indices,
        box_idx,
        active_counts[box_idx],
        radii,
        radii,
        dissipation[box_idx],
        kinematic_viscosity[box_idx],
        radii,
        radii,
        temperature,
        pressure,
        wp.float64(1.0),
        wp.float64(1.0),
        wp.float64(1.0),
        wp.float64(1.0),
        wp.float64(1.0),
        wp.float64(1.0),
        wp.float64(1.0),
        wp.float64(1.0),
    )


def test_turbulent_public_dispatch_keeps_linear_majorant_path() -> None:
    """The public singleton dispatch selects the test-local linear diagnostic.

    Warp inlines ``@wp.func`` helpers into the public selector, so wall-clock
    timing cannot reliably distinguish an O(A) scan from an O(A²) scan across
    CPU and CUDA.  This structural diagnostic inspects the public kernel's
    parsed source: the singleton branch must call the two-largest-radii helper,
    while only its alternate branch may contain the exhaustive pair scan.
    """
    source = inspect.getsource(
        coagulation_module.brownian_coagulation_kernel.func
    )
    tree = ast.parse(source)
    singleton_branch = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and "mechanism_mask == wp.int32(TURBULENT_SHEAR_ST1956_MECHANISM_FLAG)"
        in ast.unparse(node.test)
    )
    singleton_calls = [
        node.func.id
        for node in ast.walk(singleton_branch)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]

    assert "_turbulent_majorant_from_active_radii" in singleton_calls
    assert any(isinstance(node, ast.For) for node in singleton_branch.orelse)
    assert not any(isinstance(node, ast.For) for node in singleton_branch.body)


@pytest.mark.gpu_parity
def test_turbulent_pair_rate_and_majorant_match_independent_oracle(
    device: str,
) -> None:
    """ST1956 dispatch and two-largest-radius bounds match the SI equation."""
    radii = np.array(
        [[1.0e-9, 3.0e-9, 2.0e-9, 4.0e-9], [5.0e-9] * 4],
        dtype=np.float64,
    )
    active_indices = np.array([[3, 0, 2, 1], [0, -1, -1, -1]], dtype=np.int32)
    active_counts = np.array([4, 1], dtype=np.int32)
    dissipation = np.array([2.0, 0.5], dtype=np.float64)
    viscosity = np.array([1.5e-5, 1.0e-5], dtype=np.float64)
    pair_rates = wp.zeros((2,), dtype=wp.float64, device=device)
    dispatched_pair_rates = wp.zeros((2,), dtype=wp.float64, device=device)
    majorants = wp.zeros((2,), dtype=wp.float64, device=device)
    dispatched_majorants = wp.zeros((2,), dtype=wp.float64, device=device)
    wp.launch(
        _turbulent_rate_and_majorant_probe_kernel,
        dim=2,
        inputs=[
            wp.array(active_indices, dtype=wp.int32, device=device),
            wp.array(active_counts, dtype=wp.int32, device=device),
            wp.array(radii, dtype=wp.float64, device=device),
            wp.array(dissipation, dtype=wp.float64, device=device),
            wp.array(viscosity, dtype=wp.float64, device=device),
            wp.array([298.15, 298.15], dtype=wp.float64, device=device),
            wp.array([101325.0, 101325.0], dtype=wp.float64, device=device),
            pair_rates,
            dispatched_pair_rates,
            majorants,
            dispatched_majorants,
        ],
        device=device,
    )
    wp.synchronize_device(device)

    def oracle(
        radius_i: float,
        radius_j: float,
        epsilon: float,
        nu: float,
    ) -> float:
        return float(
            np.sqrt(np.pi * epsilon / (120.0 * nu))
            * (2.0 * (radius_i + radius_j)) ** 3
        )

    expected_pair = np.array(
        [
            oracle(
                radii[box, 0], radii[box, 1], dissipation[box], viscosity[box]
            )
            for box in range(2)
        ],
        dtype=np.float64,
    )
    expected_majorant = oracle(4.0e-9, 3.0e-9, dissipation[0], viscosity[0])
    npt.assert_allclose(
        pair_rates.numpy(), expected_pair, rtol=1.0e-12, atol=0.0
    )
    npt.assert_allclose(
        dispatched_pair_rates.numpy(), expected_pair, rtol=1.0e-12, atol=0.0
    )
    npt.assert_allclose(
        majorants.numpy(), [expected_majorant, 0.0], rtol=1.0e-12, atol=0.0
    )
    npt.assert_allclose(
        dispatched_majorants.numpy(),
        [expected_majorant, 0.0],
        rtol=1.0e-12,
        atol=0.0,
    )
    for first, second in itertools.combinations(active_indices[0], 2):
        assert oracle(
            radii[0, first], radii[0, second], dissipation[0], viscosity[0]
        ) <= np.nextafter(expected_majorant, np.inf)


@pytest.mark.parametrize(
    "source",
    [
        "scalar",
        "direct_arrays",
        "scalar_temperature_array_pressure",
        "array_temperature_scalar_pressure",
        "environment",
    ],
)
def test_coagulation_step_gpu_explicit_brownian_matches_default_and_dispatches_mask(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    source: str,
) -> None:
    """Explicit Brownian config preserves default seeded dispatch behavior."""
    particles = _make_particle_data(n_boxes=2, n_particles=4, n_species=1)
    particles.volume[:] = 1.0e-18
    initial_masses = particles.masses.copy()
    initial_concentration = particles.concentration.copy()
    config = CoagulationMechanismConfig(mechanisms=(BROWNIAN_MECHANISM,))
    masks: list[Any] = []
    original_launch = coagulation_module.wp.launch

    def _tracking_launch(kernel: Any, *args: Any, **kwargs: Any) -> Any:
        if getattr(kernel, "key", "") == "brownian_coagulation_kernel":
            masks.append(
                kwargs["inputs"][BROWNIAN_COAGULATION_MASK_INPUT_INDEX]
            )
        return original_launch(kernel, *args, **kwargs)

    monkeypatch.setattr(coagulation_module.wp, "launch", _tracking_launch)

    results = []
    for mechanism_config in (None, config):
        gpu_particles = to_warp_particle_data(particles, device=device)
        pairs = wp.zeros((2, 2, 2), dtype=wp.int32, device=device)
        counts = wp.zeros((2,), dtype=wp.int32, device=device)
        rng_states = wp.zeros((2,), dtype=wp.uint32, device=device)
        if source == "scalar":
            temperature, pressure, environment = 298.15, 101325.0, None
        elif source == "direct_arrays":
            temperature = wp.array(
                [298.15, 299.15], dtype=wp.float64, device=device
            )
            pressure = wp.array(
                [101325.0, 100800.0], dtype=wp.float64, device=device
            )
            environment = None
        elif source == "scalar_temperature_array_pressure":
            temperature = 298.15
            pressure = wp.array(
                [101325.0, 100800.0], dtype=wp.float64, device=device
            )
            environment = None
        elif source == "array_temperature_scalar_pressure":
            temperature = wp.array(
                [298.15, 299.15], dtype=wp.float64, device=device
            )
            pressure = 101325.0
            environment = None
        else:
            temperature, pressure = None, None
            environment = to_warp_environment_data(
                _make_environment_data(n_boxes=2, n_species=1), device=device
            )

        coagulation_step_gpu(
            gpu_particles,
            temperature=temperature,
            pressure=pressure,
            time_step=0.1,
            max_collisions=4,
            rng_seed=37,
            collision_pairs=pairs,
            n_collisions=counts,
            rng_states=rng_states,
            initialize_rng=True,
            environment=environment,
            mechanism_config=mechanism_config,
        )
        wp.synchronize()
        results.append(
            (
                np.asarray(pairs.numpy()).copy(),
                np.asarray(counts.numpy()).copy(),
                np.asarray(rng_states.numpy()).copy(),
                from_warp_particle_data(gpu_particles, sync=True),
            )
        )

    default, explicit = results
    assert int(default[1].sum()) > 0
    npt.assert_array_equal(default[0], explicit[0])
    npt.assert_array_equal(default[1], explicit[1])
    npt.assert_array_equal(default[2], explicit[2])
    npt.assert_allclose(default[3].masses, explicit[3].masses)
    npt.assert_allclose(default[3].concentration, explicit[3].concentration)
    npt.assert_allclose(default[3].charge, explicit[3].charge)
    assert not np.array_equal(default[3].masses, initial_masses)
    assert not np.array_equal(default[3].concentration, initial_concentration)
    assert masks == [wp.int32(BROWNIAN_MECHANISM_FLAG)] * 2


def test_coagulation_step_gpu_scalar_positional_call_remains_valid(
    device: str,
) -> None:
    """Legacy positional scalar callers remain source-compatible."""
    particles = _make_particle_data(n_boxes=1, n_particles=3, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)

    _, collision_pairs, collision_counts = coagulation_step_gpu(
        gpu_particles,
        298.15,
        101325.0,
        0.1,
        max_collisions=4,
        rng_seed=3,
    )
    wp.synchronize()

    assert collision_pairs.shape == (1, 1, 2)
    assert collision_counts.shape == (1,)


def test_coagulation_step_gpu_omitted_rng_states_keeps_legacy_behavior(
    device: str,
) -> None:
    """Omitting ``rng_states`` still allocates usable seeded state."""
    particles = _make_particle_data(n_boxes=1, n_particles=4, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)

    _, collision_pairs, collision_counts = coagulation_step_gpu(
        gpu_particles,
        temperature=298.15,
        pressure=101325.0,
        time_step=0.1,
        rng_seed=19,
        max_collisions=4,
    )
    wp.synchronize()

    assert collision_pairs.shape == (1, 2, 2)
    assert collision_counts.shape == (1,)
    assert np.all(np.asarray(collision_counts.numpy()) >= 0)


def test_mixed_npf_droplet_fixture_returns_float64_particle_data() -> None:
    """The mixed-scale fixture keeps canonical float64 shapes and scales."""
    particles = _make_mixed_npf_droplet_particle_data()

    assert particles.masses.shape == (1, 4, 1)
    assert particles.concentration.shape == (1, 4)
    assert particles.charge.shape == (1, 4)
    assert particles.density.shape == (1,)
    assert particles.volume.shape == (1,)
    assert particles.masses.dtype == np.float64
    assert particles.concentration.dtype == np.float64
    assert particles.charge.dtype == np.float64
    assert particles.density.dtype == np.float64
    assert particles.volume.dtype == np.float64

    density_value = float(particles.density[0])
    radii = np.cbrt(
        3.0 * particles.masses[:, :, 0] / (4.0 * np.pi * density_value)
    )
    assert np.min(radii) < 1.0e-8
    assert np.max(radii) > 1.0e-6
    assert np.min(particles.masses[:, :, 0]) < 1.0e-20
    assert np.max(particles.masses[:, :, 0]) > 1.0e-12


@pytest.mark.parametrize("device", _available_warp_devices())
def test_mixed_npf_droplet_fixture_converts_on_supported_warp_devices(
    device: str,
) -> None:
    """The mixed-scale fixture round-trips cleanly on supported Warp devices."""
    particles = _make_mixed_npf_droplet_particle_data()

    gpu_particles = to_warp_particle_data(particles, device=device)
    restored = from_warp_particle_data(gpu_particles, sync=True)

    npt.assert_allclose(restored.masses, particles.masses)
    npt.assert_allclose(restored.concentration, particles.concentration)
    npt.assert_allclose(restored.charge, particles.charge)
    npt.assert_allclose(restored.density, particles.density)
    npt.assert_allclose(restored.volume, particles.volume)


@pytest.mark.parametrize(
    ("temperature", "pressure"),
    [
        (298.15, 101325.0),
        (298.15, None),
        (None, 101325.0),
    ],
)
def test_coagulation_step_gpu_rejects_mixed_environment_inputs(
    device: str,
    temperature: float | None,
    pressure: float | None,
) -> None:
    """Mixed scalar and environment inputs raise a stable contract error."""
    particles = _make_particle_data(n_boxes=1, n_particles=3, n_species=1)
    environment = to_warp_environment_data(
        _make_environment_data(n_boxes=1, n_species=1),
        device=device,
    )
    gpu_particles = to_warp_particle_data(particles, device=device)

    with pytest.raises(
        ValueError,
        match="direct temperature/pressure inputs with environment",
    ):
        coagulation_step_gpu(
            gpu_particles,
            temperature=temperature,
            pressure=pressure,
            time_step=0.1,
            environment=environment,
        )


def test_coagulation_step_gpu_accepts_explicit_environment(
    device: str,
) -> None:
    """Pure ``environment=...`` execution succeeds when inputs are valid."""
    particles = _make_particle_data(n_boxes=1, n_particles=3, n_species=1)
    environment = to_warp_environment_data(
        _make_environment_data(n_boxes=1, n_species=1),
        device=device,
    )
    gpu_particles = to_warp_particle_data(particles, device=device)

    _, scalar_pairs, scalar_counts = coagulation_step_gpu(
        gpu_particles,
        temperature=298.15,
        pressure=101325.0,
        time_step=0.1,
        max_collisions=4,
        rng_seed=3,
    )
    scalar_pairs_np = np.asarray(scalar_pairs.numpy()).copy()
    scalar_counts_np = np.asarray(scalar_counts.numpy()).copy()

    gpu_particles = to_warp_particle_data(particles, device=device)
    _, env_pairs, env_counts = coagulation_step_gpu(
        gpu_particles,
        temperature=None,
        pressure=None,
        time_step=0.1,
        max_collisions=4,
        rng_seed=3,
        environment=environment,
    )

    npt.assert_array_equal(env_pairs.numpy(), scalar_pairs_np)
    npt.assert_array_equal(env_counts.numpy(), scalar_counts_np)


@pytest.mark.parametrize(
    ("temperature", "pressure"),
    [
        (298.15, None),
        (None, 101325.0),
        (None, None),
    ],
)
def test_coagulation_step_gpu_rejects_missing_scalar_inputs_without_environment(
    device: str,
    temperature: float | None,
    pressure: float | None,
) -> None:
    """Scalar-mode calls require both temperature and pressure."""
    particles = _make_particle_data(n_boxes=1, n_particles=3, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)

    with pytest.raises(
        ValueError,
        match="temperature and pressure must both be provided",
    ):
        coagulation_step_gpu(
            gpu_particles,
            temperature=temperature,
            pressure=pressure,
            time_step=0.1,
        )


@pytest.mark.parametrize(
    ("temperature", "pressure", "message"),
    [
        (
            298.15,
            101325.0,
            "direct temperature/pressure inputs with environment",
        ),
        (
            None,
            None,
            r"\(n_boxes,\)",
        ),
    ],
)
def test_coagulation_step_gpu_contract_errors_short_circuit_before_launch(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    temperature: float | None,
    pressure: float | None,
    message: str,
) -> None:
    """Contract errors fire before volume setup or any Warp launch."""
    particles = _make_particle_data(n_boxes=1, n_particles=3, n_species=1)
    environment_data = _make_environment_data(n_boxes=1, n_species=1)
    if temperature is None and pressure is None:
        environment_data.temperature = np.array([298.15, 299.15])
    environment = to_warp_environment_data(environment_data, device=device)
    gpu_particles = to_warp_particle_data(particles, device=device)
    calls: list[str] = []

    def _unexpected_ensure_volume_array(*args: Any, **kwargs: Any) -> Any:
        calls.append("ensure_volume_array")
        raise AssertionError("_ensure_volume_array should not be called")

    def _unexpected_launch(*args: Any, **kwargs: Any) -> None:
        calls.append("launch")
        raise AssertionError("wp.launch should not be called")

    monkeypatch.setattr(
        coagulation_module,
        "_ensure_volume_array",
        _unexpected_ensure_volume_array,
    )
    monkeypatch.setattr(
        coagulation_module, "_validate_charge_finite", lambda *_: None
    )
    monkeypatch.setattr(coagulation_module.wp, "launch", _unexpected_launch)

    with pytest.raises(ValueError, match=message):
        coagulation_step_gpu(
            gpu_particles,
            temperature=temperature,
            pressure=pressure,
            time_step=0.1,
            environment=environment,
        )

    assert calls == []


@pytest.mark.parametrize(
    ("temperature", "pressure", "message"),
    [
        (0.0, 101325.0, "temperature must be finite and > 0"),
        (298.15, 0.0, "pressure must be finite and > 0"),
        (float("nan"), 101325.0, "temperature must be finite and > 0"),
    ],
)
def test_coagulation_step_gpu_invalid_scalar_domains_short_circuit_before_launch(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    temperature: float,
    pressure: float,
    message: str,
) -> None:
    """Invalid scalar domains fail before any setup or Warp launch work."""
    particles = _make_particle_data(n_boxes=1, n_particles=3, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    calls: list[str] = []

    def _unexpected_ensure_volume_array(*args: Any, **kwargs: Any) -> Any:
        calls.append("ensure_volume_array")
        raise AssertionError("_ensure_volume_array should not be called")

    def _unexpected_launch(*args: Any, **kwargs: Any) -> None:
        calls.append("launch")
        raise AssertionError("wp.launch should not be called")

    monkeypatch.setattr(
        coagulation_module,
        "_ensure_volume_array",
        _unexpected_ensure_volume_array,
    )
    monkeypatch.setattr(
        coagulation_module, "_validate_charge_finite", lambda *_: None
    )
    monkeypatch.setattr(coagulation_module.wp, "launch", _unexpected_launch)

    with pytest.raises(ValueError, match=message):
        coagulation_step_gpu(
            gpu_particles,
            temperature=temperature,
            pressure=pressure,
            time_step=0.1,
        )

    assert calls == []


def test_coagulation_step_gpu_invalid_environment_domains_short_circuit_before_launch(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    """Invalid environment arrays fail before downstream setup or kernels."""
    particles = _make_particle_data(n_boxes=1, n_particles=3, n_species=1)
    environment = to_warp_environment_data(
        _make_environment_data(n_boxes=1, n_species=1), device=device
    )
    environment.pressure = wp.array([0.0], dtype=wp.float64, device=device)
    gpu_particles = to_warp_particle_data(particles, device=device)
    calls: list[str] = []

    def _unexpected_ensure_volume_array(*args: Any, **kwargs: Any) -> Any:
        calls.append("ensure_volume_array")
        raise AssertionError("_ensure_volume_array should not be called")

    original_launch = coagulation_module.wp.launch

    def _tracking_launch(kernel: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(getattr(kernel, "key", str(kernel)))
        return original_launch(kernel, *args, **kwargs)

    monkeypatch.setattr(
        coagulation_module,
        "_ensure_volume_array",
        _unexpected_ensure_volume_array,
    )
    monkeypatch.setattr(
        coagulation_module, "_validate_charge_finite", lambda *_: None
    )
    monkeypatch.setattr(coagulation_module.wp, "launch", _tracking_launch)

    with pytest.raises(
        ValueError,
        match="environment.pressure must be finite and > 0",
    ):
        coagulation_step_gpu(
            gpu_particles,
            temperature=None,
            pressure=None,
            time_step=0.1,
            environment=environment,
        )

    assert calls == []


def test_coagulation_step_gpu_invalid_environment_preserves_buffers_and_particles(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    """Invalid environment data leaves caller-owned buffers unchanged."""
    particles = _make_particle_data(n_boxes=2, n_particles=3, n_species=1)
    environment = to_warp_environment_data(
        _make_environment_data(n_boxes=2, n_species=1), device=device
    )
    environment.pressure = wp.array(
        [101325.0, 0.0],
        dtype=wp.float64,
        device=device,
    )
    gpu_particles = to_warp_particle_data(particles, device=device)
    initial_particles = from_warp_particle_data(gpu_particles, sync=True)
    collision_pairs = wp.array(
        np.arange(16, dtype=np.int32).reshape(2, 4, 2),
        dtype=wp.int32,
        device=device,
    )
    n_collisions = wp.array(
        np.array([1, 2], dtype=np.int32),
        dtype=wp.int32,
        device=device,
    )
    rng_states = wp.array(
        np.array([17, 23], dtype=np.uint32),
        dtype=wp.uint32,
        device=device,
    )
    initial_collision_pairs = np.asarray(collision_pairs.numpy()).copy()
    initial_n_collisions = np.asarray(n_collisions.numpy()).copy()
    initial_rng_states = np.asarray(rng_states.numpy()).copy()
    calls: list[str] = []

    def _unexpected_ensure_volume_array(*args: Any, **kwargs: Any) -> Any:
        calls.append("ensure_volume_array")
        raise AssertionError("_ensure_volume_array should not be called")

    def _unexpected_launch(*args: Any, **kwargs: Any) -> None:
        calls.append("launch")
        raise AssertionError("wp.launch should not be called")

    monkeypatch.setattr(
        coagulation_module,
        "_ensure_volume_array",
        _unexpected_ensure_volume_array,
    )
    monkeypatch.setattr(
        coagulation_module, "_validate_charge_finite", lambda *_: None
    )
    monkeypatch.setattr(coagulation_module.wp, "launch", _unexpected_launch)

    with pytest.raises(
        ValueError,
        match="environment.pressure must be finite and > 0",
    ):
        coagulation_step_gpu(
            gpu_particles,
            temperature=None,
            pressure=None,
            time_step=0.1,
            max_collisions=4,
            collision_pairs=collision_pairs,
            n_collisions=n_collisions,
            rng_states=rng_states,
            environment=environment,
        )

    assert calls == []
    npt.assert_array_equal(
        np.asarray(collision_pairs.numpy()),
        initial_collision_pairs,
    )
    npt.assert_array_equal(
        np.asarray(n_collisions.numpy()),
        initial_n_collisions,
    )
    npt.assert_array_equal(np.asarray(rng_states.numpy()), initial_rng_states)
    _assert_particles_unchanged(gpu_particles, initial_particles)


def test_coagulation_step_gpu_accepts_direct_environment_arrays(
    device: str,
) -> None:
    """Direct ``(n_boxes,)`` Warp-array inputs execute successfully."""
    particles = _make_particle_data(n_boxes=2, n_particles=3, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    temperature = wp.array([298.15, 301.15], dtype=wp.float64, device=device)
    pressure = wp.array([101325.0, 100800.0], dtype=wp.float64, device=device)

    _, collision_pairs, collision_counts = coagulation_step_gpu(
        gpu_particles,
        temperature=temperature,
        pressure=pressure,
        time_step=0.1,
        max_collisions=4,
        rng_seed=3,
    )

    assert collision_pairs.shape == (2, 1, 2)
    assert collision_counts.shape == (2,)


@pytest.mark.parametrize(
    ("temperature", "pressure"),
    [
        (298.15, np.array([101325.0, 100800.0], dtype=np.float64)),
        (np.array([298.15, 301.15], dtype=np.float64), 101325.0),
    ],
)
def test_coagulation_step_gpu_accepts_hybrid_scalar_and_array_inputs(
    device: str,
    temperature: float | np.ndarray,
    pressure: float | np.ndarray,
) -> None:
    """Hybrid direct inputs execute successfully."""
    particles = _make_particle_data(n_boxes=2, n_particles=3, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)

    if isinstance(temperature, np.ndarray):
        temperature = wp.array(temperature, dtype=wp.float64, device=device)
    if isinstance(pressure, np.ndarray):
        pressure = wp.array(pressure, dtype=wp.float64, device=device)

    _, collision_pairs, collision_counts = coagulation_step_gpu(
        gpu_particles,
        temperature=temperature,
        pressure=pressure,
        time_step=0.1,
        max_collisions=4,
        rng_seed=3,
    )

    assert collision_pairs.shape == (2, 1, 2)
    assert collision_counts.shape == (2,)


def test_coagulation_step_gpu_preserves_direct_environment_array_dtypes(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    """Direct Warp arrays are reused without dtype coercion."""
    particles = _make_particle_data(n_boxes=2, n_particles=3, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    original_launch = coagulation_module.wp.launch
    launch_dtypes: list[tuple[Any, Any]] = []

    def _tracking_launch(kernel: Any, *args: Any, **kwargs: Any) -> Any:
        inputs = kwargs.get("inputs", [])
        if getattr(kernel, "key", "") == "brownian_coagulation_kernel":
            launch_dtypes.append((inputs[4].dtype, inputs[5].dtype))
        return original_launch(kernel, *args, **kwargs)

    monkeypatch.setattr(coagulation_module.wp, "launch", _tracking_launch)

    coagulation_step_gpu(
        gpu_particles,
        temperature=wp.array([298.15, 301.15], dtype=wp.float64, device=device),
        pressure=wp.array(
            [101325.0, 100800.0], dtype=wp.float64, device=device
        ),
        time_step=0.1,
        max_collisions=4,
        rng_seed=3,
    )

    assert launch_dtypes == [(wp.float64, wp.float64)]


def test_coagulation_step_gpu_preserves_environment_array_dtypes(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    """Explicit environment arrays are reused without dtype coercion."""
    particles = _make_particle_data(n_boxes=2, n_particles=3, n_species=1)

    class _EnvironmentLike:
        def __init__(self) -> None:
            self.temperature = wp.array(
                [298.15, 301.15], dtype=wp.float64, device=device
            )
            self.pressure = wp.array(
                [101325.0, 100800.0], dtype=wp.float64, device=device
            )

    environment = _EnvironmentLike()
    gpu_particles = to_warp_particle_data(particles, device=device)
    original_launch = coagulation_module.wp.launch
    launch_dtypes: list[tuple[Any, Any]] = []

    def _tracking_launch(kernel: Any, *args: Any, **kwargs: Any) -> Any:
        inputs = kwargs.get("inputs", [])
        if getattr(kernel, "key", "") == "brownian_coagulation_kernel":
            launch_dtypes.append((inputs[4].dtype, inputs[5].dtype))
        return original_launch(kernel, *args, **kwargs)

    monkeypatch.setattr(coagulation_module.wp, "launch", _tracking_launch)

    coagulation_step_gpu(
        gpu_particles,
        temperature=None,
        pressure=None,
        time_step=0.1,
        max_collisions=4,
        rng_seed=3,
        environment=environment,
    )

    assert launch_dtypes == [(wp.float64, wp.float64)]


def test_coagulation_step_gpu_environment_shape_mismatch_raises_value_error(
    device: str,
) -> None:
    """Environment arrays must match ``(n_boxes,)`` before launch work."""
    particles = _make_particle_data(n_boxes=1, n_particles=3, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    environment = to_warp_environment_data(
        _make_environment_data(1, 1), device=device
    )
    environment.temperature = wp.array(
        [298.15, 299.15], dtype=wp.float64, device=device
    )

    with pytest.raises(ValueError, match=r"\(n_boxes,\)"):
        coagulation_step_gpu(
            gpu_particles,
            temperature=None,
            pressure=None,
            time_step=0.1,
            environment=environment,
        )


def test_coagulation_step_gpu_environment_device_mismatch_raises_value_error(
    device: str,
) -> None:
    """Environment arrays on the wrong device fail before launch work."""
    wrong_device = "cpu" if device == "cuda" else "cuda"
    if wrong_device == "cuda" and not cuda_available(wp):
        pytest.skip("CUDA not available for mismatch test")

    particles = _make_particle_data(n_boxes=1, n_particles=3, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    environment = to_warp_environment_data(
        _make_environment_data(1, 1), device=wrong_device
    )

    with pytest.raises(ValueError, match="environment.temperature device"):
        coagulation_step_gpu(
            gpu_particles,
            temperature=None,
            pressure=None,
            time_step=0.1,
            environment=environment,
        )


def test_coagulation_step_gpu_direct_temperature_shape_mismatch_raises(
    device: str,
) -> None:
    """Direct temperature arrays must match ``(n_boxes,)`` before launch."""
    particles = _make_particle_data(n_boxes=2, n_particles=3, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    temperature = wp.array([298.15], dtype=wp.float64, device=device)

    with pytest.raises(ValueError, match=r"temperature shape .*\(n_boxes,\)"):
        coagulation_step_gpu(
            gpu_particles,
            temperature=temperature,
            pressure=101325.0,
            time_step=0.1,
        )


def test_coagulation_step_gpu_direct_pressure_device_mismatch_raises(
    device: str,
) -> None:
    """Direct pressure arrays on the wrong device fail before launch."""
    wrong_device = "cpu" if device == "cuda" else "cuda"
    if wrong_device == "cuda" and not cuda_available(wp):
        pytest.skip("CUDA not available for mismatch test")

    particles = _make_particle_data(n_boxes=1, n_particles=3, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    pressure = wp.array([101325.0], dtype=wp.float64, device=wrong_device)

    with pytest.raises(ValueError, match="pressure device"):
        coagulation_step_gpu(
            gpu_particles,
            temperature=298.15,
            pressure=pressure,
            time_step=0.1,
        )


def test_coagulation_step_gpu_reuses_direct_environment_arrays(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    """Coagulation forwards validated direct arrays without rebuilding them."""
    particles = _make_particle_data(n_boxes=2, n_particles=3, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    temperature = wp.array([298.15, 301.15], dtype=wp.float64, device=device)
    pressure = wp.array([101325.0, 100800.0], dtype=wp.float64, device=device)
    original_launch = coagulation_module.wp.launch
    forwarded_inputs: list[tuple[Any, Any]] = []

    def _tracking_launch(kernel: Any, *args: Any, **kwargs: Any) -> Any:
        inputs = kwargs.get("inputs", [])
        if getattr(kernel, "key", "") == "brownian_coagulation_kernel":
            forwarded_inputs.append((inputs[4], inputs[5]))
        return original_launch(kernel, *args, **kwargs)

    monkeypatch.setattr(coagulation_module.wp, "launch", _tracking_launch)

    coagulation_step_gpu(
        gpu_particles,
        temperature=temperature,
        pressure=pressure,
        time_step=0.1,
        max_collisions=4,
        rng_seed=3,
    )

    assert forwarded_inputs == [(temperature, pressure)]


def test_coagulation_step_gpu_uniform_direct_arrays_match_scalar_results(
    device: str,
) -> None:
    """Uniform direct arrays stay within the established tolerance band."""
    particles = _make_particle_data(n_boxes=2, n_particles=6, n_species=1)
    seeds = range(11, 19)
    time_step = 0.5
    max_collisions = 16

    scalar_counts = _accumulate_collision_counts(
        particles=particles,
        device=device,
        seeds=seeds,
        time_step=time_step,
        max_collisions=max_collisions,
        temperature=298.15,
        pressure=101325.0,
    )
    uniform_counts = _accumulate_collision_counts(
        particles=particles,
        device=device,
        seeds=seeds,
        time_step=time_step,
        max_collisions=max_collisions,
        temperature=wp.array([298.15, 298.15], dtype=wp.float64, device=device),
        pressure=wp.array(
            [101325.0, 101325.0], dtype=wp.float64, device=device
        ),
    )

    diff = np.abs(uniform_counts - scalar_counts)
    tolerance = np.maximum(3.0 * np.sqrt(np.maximum(scalar_counts, 1.0)), 1.0)

    assert np.all(diff <= tolerance), (
        "Uniform direct arrays should preserve scalar coagulation behavior "
        "within the established stochastic tolerance band."
    )


@pytest.mark.parametrize(
    ("temperature", "pressure"),
    [
        (298.15, None),
        (None, 101325.0),
        (None, None),
    ],
)
def test_coagulation_step_gpu_missing_scalar_inputs_short_circuit_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    temperature: float | None,
    pressure: float | None,
) -> None:
    """Missing scalar inputs fail before setup, launch, or input mutation."""
    particles = _make_particle_data(n_boxes=1, n_particles=3, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    rng_states = wp.array(
        np.array([17], dtype=np.uint32), dtype=wp.uint32, device=device
    )
    initial_rng_states = np.asarray(rng_states.numpy()).copy()
    initial_particles = from_warp_particle_data(gpu_particles, sync=True)
    calls: list[str] = []

    def _unexpected_ensure_volume_array(*args: Any, **kwargs: Any) -> Any:
        calls.append("ensure_volume_array")
        raise AssertionError("_ensure_volume_array should not be called")

    def _unexpected_launch(*args: Any, **kwargs: Any) -> None:
        calls.append("launch")
        raise AssertionError("wp.launch should not be called")

    monkeypatch.setattr(
        coagulation_module,
        "_ensure_volume_array",
        _unexpected_ensure_volume_array,
    )
    monkeypatch.setattr(
        coagulation_module, "_validate_charge_finite", lambda *_: None
    )
    monkeypatch.setattr(coagulation_module.wp, "launch", _unexpected_launch)

    with pytest.raises(
        ValueError,
        match="temperature and pressure must both be provided",
    ):
        coagulation_step_gpu(
            gpu_particles,
            temperature=temperature,
            pressure=pressure,
            time_step=0.1,
            rng_states=rng_states,
        )

    assert calls == []
    npt.assert_array_equal(np.asarray(rng_states.numpy()), initial_rng_states)
    result_particles = from_warp_particle_data(gpu_particles, sync=True)
    npt.assert_allclose(result_particles.masses, initial_particles.masses)
    npt.assert_allclose(
        result_particles.concentration,
        initial_particles.concentration,
    )


@_warp_kernel
# type: ignore[misc]
def _brownian_kernel_matrix_kernel(
    radii: Any,
    masses: Any,
    temperature: Any,
    pressure: Any,
    boltzmann_constant: Any,
    gas_constant: Any,
    molecular_weight_air: Any,
    ref_viscosity: Any,
    ref_temperature: Any,
    sutherland_constant: Any,
    kernel_out: Any,
) -> None:
    """Compute Brownian kernel matrix using shared GPU building blocks."""  # type: ignore
    i_idx, j_idx = wp.tid()  # type: ignore[misc]
    dynamic_viscosity = dynamic_viscosity_wp(
        temperature,
        ref_viscosity,
        ref_temperature,
        sutherland_constant,
    )
    mean_free_path = molecule_mean_free_path_wp(
        molecular_weight_air,
        temperature,
        pressure,
        dynamic_viscosity,
        gas_constant,
    )
    knudsen_i = knudsen_number_wp(mean_free_path, radii[i_idx])
    knudsen_j = knudsen_number_wp(mean_free_path, radii[j_idx])
    slip_i = cunningham_slip_correction_wp(knudsen_i)
    slip_j = cunningham_slip_correction_wp(knudsen_j)
    mobility_i = aerodynamic_mobility_wp(
        radii[i_idx],
        slip_i,
        dynamic_viscosity,
    )
    mobility_j = aerodynamic_mobility_wp(
        radii[j_idx],
        slip_j,
        dynamic_viscosity,
    )
    diffusivity_i = brownian_diffusivity_wp(
        temperature,
        mobility_i,
        boltzmann_constant,
    )
    diffusivity_j = brownian_diffusivity_wp(
        temperature,
        mobility_j,
        boltzmann_constant,
    )
    speed_i = mean_thermal_speed_wp(
        masses[i_idx],
        temperature,
        boltzmann_constant,
    )
    speed_j = mean_thermal_speed_wp(
        masses[j_idx],
        temperature,
        boltzmann_constant,
    )
    mean_free_path_i = particle_mean_free_path_wp(diffusivity_i, speed_i)
    mean_free_path_j = particle_mean_free_path_wp(diffusivity_j, speed_j)
    g_term_i = g_collection_term_wp(mean_free_path_i, radii[i_idx])
    g_term_j = g_collection_term_wp(mean_free_path_j, radii[j_idx])
    kernel_out[i_idx, j_idx] = brownian_kernel_pair_wp(
        radii[i_idx],
        radii[j_idx],
        diffusivity_i,
        diffusivity_j,
        g_term_i,
        g_term_j,
        speed_i,
        speed_j,
        wp.float64(1.0),
    )


@pytest.mark.gpu_parity
def test_brownian_kernel_matrix_parity_gpu_cpu(device: str) -> None:
    """GPU Brownian kernel matrix matches CPU reference."""
    temperature = 298.15
    pressure = 101325.0
    radii = np.array([1.0e-8, 5.0e-8, 1.0e-7], dtype=np.float64)
    masses = np.array([1.0e-21, 4.0e-21, 8.0e-21], dtype=np.float64)

    expected = get_brownian_kernel_via_system_state(
        particle_radius=radii,
        particle_mass=masses,
        temperature=temperature,
        pressure=pressure,
    )

    radii_wp = wp.array(radii, dtype=wp.float64, device=device)
    masses_wp = wp.array(masses, dtype=wp.float64, device=device)
    kernel_wp = wp.zeros(
        (len(radii), len(radii)), dtype=wp.float64, device=device
    )

    wp.launch(
        _brownian_kernel_matrix_kernel,
        dim=(len(radii), len(radii)),
        inputs=[
            radii_wp,
            masses_wp,
            wp.float64(temperature),
            wp.float64(pressure),
            wp.float64(constants.BOLTZMANN_CONSTANT),
            wp.float64(constants.GAS_CONSTANT),
            wp.float64(constants.MOLECULAR_WEIGHT_AIR),
            wp.float64(constants.REF_VISCOSITY_AIR_STP),
            wp.float64(constants.REF_TEMPERATURE_STP),
            wp.float64(constants.SUTHERLAND_CONSTANT),
        ],
        outputs=[kernel_wp],
        device=device,
    )
    wp.synchronize()

    npt.assert_allclose(kernel_wp.numpy(), expected, rtol=1.0e-7)


@pytest.mark.stochastic
def test_coagulation_statistical_collision_rate(device: str) -> None:
    """Collision counts follow expected Brownian rate statistics."""
    temperature = 298.15
    pressure = 101325.0
    time_step = 0.5
    n_steps = 60
    particles = _make_particle_data(n_boxes=1, n_particles=6, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)

    total_collisions = 0
    for step_idx in range(n_steps):
        _, _, collision_counts = coagulation_step_gpu(
            gpu_particles,
            temperature=temperature,
            pressure=pressure,
            time_step=time_step,
            rng_seed=42 + step_idx,
            max_collisions=16,
        )
        wp.synchronize()
        collision_array = np.asarray(collision_counts.numpy())
        total_collisions += int(collision_array.sum())

    density_value = float(np.asarray(particles.density).ravel().item(0))
    masses_array: np.ndarray = np.asarray(particles.masses, dtype=np.float64)
    masses_slice = np.ravel(masses_array[0])
    radii = np.cbrt(3.0 * masses_slice / (4.0 * np.pi * density_value))
    masses = masses_slice
    kernel_matrix = get_brownian_kernel_via_system_state(
        particle_radius=radii,
        particle_mass=masses,
        temperature=temperature,
        pressure=pressure,
    )
    kernel_matrix_array = np.asarray(kernel_matrix, dtype=np.float64)
    kernel_values = np.asarray(kernel_matrix_array)[
        np.triu_indices(len(radii), k=1)
    ]
    kernel_values = np.atleast_1d(kernel_values)
    volume = float(np.asarray(particles.volume).ravel().item(0))
    expected_mean = np.sum(kernel_values) * time_step * n_steps / volume
    expected_sigma = np.sqrt(expected_mean)
    assert total_collisions == pytest.approx(
        expected_mean, abs=3.0 * expected_sigma
    )


def test_coagulation_multi_box_independence(device: str) -> None:
    """Collision counts remain isolated per box."""
    time_step = 1.0
    particles = _make_particle_data(n_boxes=3, n_particles=5, n_species=1)
    particles.concentration[1, :] = 0.0
    particles.concentration[2, 0] = 0.0
    temperature = wp.array(
        [300.0, 303.0, 297.0], dtype=wp.float64, device=device
    )
    pressure = wp.array(
        [101325.0, 100500.0, 102000.0],
        dtype=wp.float64,
        device=device,
    )

    gpu_particles = to_warp_particle_data(particles, device=device)
    _, _, collision_counts = coagulation_step_gpu(
        gpu_particles,
        temperature=temperature,
        pressure=pressure,
        time_step=time_step,
        rng_seed=12,
        max_collisions=16,
    )
    wp.synchronize()
    result = np.asarray(collision_counts.numpy())

    assert result.reshape(-1)[1] == 0
    assert result.reshape(-1)[0] >= result.reshape(-1)[2]


@pytest.mark.stochastic
def test_coagulation_step_gpu_nonuniform_environment_changes_collision_trend(
    device: str,
) -> None:
    """Nonuniform environment inputs shift box-local collisions directionally."""
    particles = _make_particle_data(n_boxes=2, n_particles=12, n_species=1)
    particles.masses[1, :, :] = particles.masses[0, :, :]
    temperature = np.array([250.0, 350.0], dtype=np.float64)
    pressure = np.array([150000.0, 50000.0], dtype=np.float64)
    environment = to_warp_environment_data(
        EnvironmentData(
            temperature=temperature,
            pressure=pressure,
            saturation_ratio=np.ones((2, 1), dtype=np.float64),
        ),
        device=device,
    )
    density_value = float(np.asarray(particles.density).item())
    mass_values = np.asarray(particles.masses[:, :, 0], dtype=np.float64)
    expected_rates = np.array(
        [
            np.sum(
                np.asarray(
                    get_brownian_kernel_via_system_state(
                        particle_radius=np.cbrt(
                            3.0
                            * box_mass_values
                            / (4.0 * np.pi * density_value)
                        ),
                        particle_mass=box_mass_values,
                        temperature=temp_value,
                        pressure=pressure_value,
                    ),
                    dtype=np.float64,
                )[np.triu_indices(len(box_mass_values), k=1)]
            )
            for box_mass_values, temp_value, pressure_value in zip(
                mass_values,
                temperature,
                pressure,
                strict=True,
            )
        ],
        dtype=np.float64,
    )
    collision_totals = _accumulate_collision_counts(
        particles=particles,
        device=device,
        seeds=range(31, 51),
        time_step=0.5,
        max_collisions=64,
        temperature=None,
        pressure=None,
        environment=environment,
        volume=1.0e-14,
    )

    higher_rate_idx = int(np.argmax(expected_rates))
    lower_rate_idx = int(np.argmin(expected_rates))

    assert expected_rates[higher_rate_idx] > expected_rates[lower_rate_idx]
    assert collision_totals.shape == (2,)
    assert (
        collision_totals[higher_rate_idx] > collision_totals[lower_rate_idx]
    ), (
        "Nonuniform environment coverage is directional/statistical: "
        "the box with the larger CPU Brownian-rate reference should accumulate "
        "more collisions across the fixed seed set, without requiring exact "
        "per-seed counts."
    )


def test_coagulation_step_gpu_nonuniform_arrays_keep_single_active_box_idle(
    device: str,
) -> None:
    """A box with fewer than two active particles still records no collisions."""
    particles = _make_particle_data(n_boxes=2, n_particles=4, n_species=1)
    particles.concentration[1, 1:] = 0.0
    gpu_particles = to_warp_particle_data(particles, device=device)

    _, _, collision_counts = coagulation_step_gpu(
        gpu_particles,
        temperature=wp.array([298.15, 308.15], dtype=wp.float64, device=device),
        pressure=wp.array(
            [101325.0, 98000.0],
            dtype=wp.float64,
            device=device,
        ),
        time_step=0.5,
        rng_seed=29,
        max_collisions=8,
    )
    wp.synchronize()

    result = np.asarray(collision_counts.numpy())

    assert result.shape == (2,)
    assert result[1] == 0


def test_heterogeneous_majorant_retains_non_radius_extrema_collision(
    device: str,
) -> None:
    """All-pair majorant retains a faster equal-radius composition pair."""
    density = np.array([1000.0, 3000.0], dtype=np.float64)
    masses = np.array(
        [
            [0.00e-18, 3.00e-18],
            [0.45e-18, 1.65e-18],
            [0.50e-18, 1.50e-18],
        ],
        dtype=np.float64,
    )
    particles = ParticleData(
        masses=masses[np.newaxis, :, :],
        concentration=np.ones((1, 3), dtype=np.float64),
        charge=np.zeros((1, 3), dtype=np.float64),
        density=density,
        volume=np.array([1.0e-18], dtype=np.float64),
    )
    total_volume = np.sum(masses / density, axis=1)
    radii = np.cbrt(3.0 * total_volume / (4.0 * np.pi))
    pair_rates = np.asarray(
        get_brownian_kernel_via_system_state(
            particle_radius=radii,
            particle_mass=np.sum(masses, axis=1),
            temperature=298.15,
            pressure=101325.0,
        ),
        dtype=np.float64,
    )

    # The former equal-radius tie handling selected (0, 1), although the
    # mass/composition-dependent (1, 2) rate is larger.
    assert pair_rates[1, 2] > pair_rates[0, 1]
    initial_mass = float(np.sum(particles.masses))
    retained_high_rate_pair = False
    for seed in range(1, 101):
        gpu_particles = to_warp_particle_data(particles, device=device)
        pairs = wp.zeros((1, 1, 2), dtype=wp.int32, device=device)
        counts = wp.zeros((1,), dtype=wp.int32, device=device)
        rng_states = wp.zeros((1,), dtype=wp.uint32, device=device)
        coagulation_step_gpu(
            gpu_particles,
            temperature=298.15,
            pressure=101325.0,
            time_step=0.5,
            max_collisions=1,
            rng_seed=seed,
            collision_pairs=pairs,
            n_collisions=counts,
            rng_states=rng_states,
            initialize_rng=True,
            validate_charge_finite=True,
        )
        result = from_warp_particle_data(gpu_particles, sync=True)
        npt.assert_allclose(np.sum(result.masses), initial_mass, rtol=1.0e-12)
        if int(np.asarray(counts.numpy())[0]) == 1:
            selected_pair = np.asarray(pairs.numpy())[0, 0]
            if np.array_equal(selected_pair, np.array([1, 2], dtype=np.int32)):
                retained_high_rate_pair = True
                break

    assert retained_high_rate_pair


def test_mixed_scale_diagnostic_reports_attempted_and_accepted_counts(
    device: str,
) -> None:
    """Mixed-scale diagnostics report bounded scheduled and executed trials.

    The private mirrored sampler should expose finite integer-like attempted
    trial counts, preserve the invariant ``scheduled >= executed >= accepted``,
    and stay in lockstep with production-observable results for the same seeded
    run.
    """
    particles = _make_mixed_npf_droplet_particle_data()
    diagnostics = _collect_test_local_attempt_diagnostics(
        particles,
        device,
        time_step=0.5,
        max_collisions=8,
        rng_seed=41,
        volume=1.0e-14,
    )

    assert diagnostics.scheduled_trial_counts.shape == (1,)
    assert diagnostics.executed_trial_counts.shape == (1,)
    assert diagnostics.accepted_counts.shape == (1,)
    assert diagnostics.production_counts.shape == (1,)
    assert np.issubdtype(diagnostics.scheduled_trial_counts.dtype, np.integer)
    assert np.issubdtype(diagnostics.executed_trial_counts.dtype, np.integer)
    assert np.issubdtype(diagnostics.accepted_counts.dtype, np.integer)
    assert np.all(np.isfinite(diagnostics.scheduled_trial_counts))
    assert np.all(np.isfinite(diagnostics.executed_trial_counts))
    assert np.all(np.isfinite(diagnostics.accepted_counts))
    assert np.all(diagnostics.scheduled_trial_counts >= 0)
    assert np.all(diagnostics.executed_trial_counts >= 0)
    assert np.all(
        diagnostics.scheduled_trial_counts >= diagnostics.executed_trial_counts
    )
    assert np.all(
        diagnostics.executed_trial_counts >= diagnostics.accepted_counts
    )
    for count_buffer, pair_buffer in (
        (diagnostics.accepted_counts, diagnostics.diagnostic_pairs),
        (diagnostics.production_counts, diagnostics.production_pairs),
    ):
        assert np.all(count_buffer <= pair_buffer.shape[1])
        assert np.all(count_buffer <= 8)
        assert np.all(count_buffer <= particles.masses.shape[1] // 2)
    npt.assert_array_equal(
        diagnostics.accepted_counts,
        diagnostics.production_counts,
    )
    _assert_trimmed_collision_pairs_equal(
        diagnostics.diagnostic_pairs,
        diagnostics.accepted_counts,
        diagnostics.production_pairs,
        diagnostics.production_counts,
    )
    npt.assert_allclose(
        diagnostics.diagnostic_masses,
        diagnostics.production_masses,
    )
    npt.assert_allclose(
        diagnostics.diagnostic_concentration,
        diagnostics.production_concentration,
    )
    npt.assert_array_equal(
        diagnostics.diagnostic_rng_states,
        diagnostics.production_rng_states,
    )


@pytest.mark.stochastic
def test_mixed_scale_brownian_collision_totals_match_expected_mean_within_sigma_tolerance(  # noqa: E501
    device: str,
) -> None:
    """Mixed-scale bounded-selector totals stay within Brownian sigma.

    This checks the shipped E3-F2-P2 bounded selector path against the CPU
    Brownian expected mean across repeated fresh seeded runs.
    """
    particles = _make_mixed_npf_droplet_particle_data()
    seed_range = range(101, 201)
    trial_count = len(seed_range)
    time_step = 0.5
    max_collisions = 8
    volume = 1.0e-10
    evaluated_path = "shipped E3-F2-P2 bounded selector path"

    collision_totals = _accumulate_collision_counts(
        particles=particles,
        device=device,
        seeds=seed_range,
        time_step=time_step,
        max_collisions=max_collisions,
        temperature=298.15,
        pressure=101325.0,
        volume=volume,
    )
    observed_total = int(collision_totals.sum())
    statistics = _get_expected_collision_statistics(
        particles,
        time_step=time_step,
        n_trials=trial_count,
        volume=volume,
    )
    sigma_multiplier = 3.0
    tolerance = sigma_multiplier * statistics.expected_sigma
    message = (
        f"{evaluated_path}: observed_total={observed_total}, "
        f"expected_mean={statistics.expected_mean:.6f}, "
        f"sigma={statistics.expected_sigma:.6f}, "
        f"tolerance={tolerance:.6f}, trials={trial_count}, "
        f"active_pairs={statistics.active_pair_count}"
    )

    assert np.isfinite(observed_total), message
    assert observed_total >= 0, message
    assert observed_total == pytest.approx(
        statistics.expected_mean,
        abs=tolerance,
    ), message


def test_mixed_scale_expected_collision_statistics_use_active_pairs_only() -> (
    None
):
    """Mixed-scale Brownian reference excludes inactive particles."""
    particles = _make_mixed_npf_droplet_particle_data()
    particles.concentration[0, -1] = 0.0
    time_step = 0.5
    trial_count = 100
    volume = 1.0e-10

    statistics = _get_expected_collision_statistics(
        particles,
        time_step=time_step,
        n_trials=trial_count,
        volume=volume,
    )

    density_value = float(np.asarray(particles.density).ravel().item(0))
    masses_slice = np.ravel(np.asarray(particles.masses[0], dtype=np.float64))
    active_masses = masses_slice[:3]
    active_radii = np.cbrt(3.0 * active_masses / (4.0 * np.pi * density_value))
    active_kernel = np.asarray(
        get_brownian_kernel_via_system_state(
            particle_radius=active_radii,
            particle_mass=active_masses,
            temperature=298.15,
            pressure=101325.0,
        ),
        dtype=np.float64,
    )
    expected_mean = float(
        np.sum(active_kernel[np.triu_indices(len(active_radii), k=1)])
        * time_step
        * trial_count
        / volume
    )

    assert statistics.active_pair_count == 3
    assert statistics.expected_mean == pytest.approx(expected_mean)
    assert statistics.expected_sigma == pytest.approx(np.sqrt(expected_mean))


@pytest.mark.stochastic
def test_mixed_scale_acceptance_fraction_is_finite_and_nonnegative(
    device: str,
) -> None:
    """Mixed-scale acceptance fractions stay finite and non-negative.

    The seeded mixed NPF/droplet scenario should produce at least one attempted
    trial so the derived acceptance fraction remains finite on CPU and CUDA.
    """
    particles = _make_mixed_npf_droplet_particle_data()
    diagnostics = _collect_test_local_attempt_diagnostics(
        particles,
        device,
        time_step=0.5,
        max_collisions=8,
        rng_seed=41,
        volume=1.0e-14,
    )

    acceptance_fraction = np.divide(
        diagnostics.accepted_counts,
        diagnostics.executed_trial_counts,
        out=np.zeros_like(diagnostics.accepted_counts, dtype=np.float64),
        where=diagnostics.executed_trial_counts > 0,
    )

    assert np.all(diagnostics.executed_trial_counts > 0)
    assert np.all(np.isfinite(acceptance_fraction))
    assert np.all(acceptance_fraction >= 0.0)


@pytest.mark.stochastic
def test_mixed_scale_selector_only_emits_sorted_active_in_bounds_pairs(
    device: str,
) -> None:
    """Accepted mixed-scale pairs stay sorted, in bounds, and initially active."""
    particles = _make_mixed_npf_droplet_particle_data()
    diagnostics = _collect_test_local_attempt_diagnostics(
        particles,
        device,
        time_step=0.5,
        max_collisions=8,
        rng_seed=41,
        volume=1.0e-14,
    )
    initially_active = particles.concentration[0] > 0.0
    n_particles = particles.masses.shape[1]

    for pair_buffer, count_buffer in (
        (diagnostics.diagnostic_pairs, diagnostics.accepted_counts),
        (diagnostics.production_pairs, diagnostics.production_counts),
    ):
        count = int(count_buffer[0])
        accepted_pairs = pair_buffer[0, :count, :]

        assert np.all(accepted_pairs[:, 0] >= 0)
        assert np.all(accepted_pairs[:, 1] >= 0)
        assert np.all(accepted_pairs[:, 0] < accepted_pairs[:, 1])
        assert np.all(accepted_pairs[:, 1] < n_particles)
        assert np.all(initially_active[accepted_pairs[:, 0]])
        assert np.all(initially_active[accepted_pairs[:, 1]])


@pytest.mark.parametrize(
    ("active_indices",),
    [
        ([],),
        ([0],),
    ],
)
@pytest.mark.stochastic
def test_mixed_scale_sparse_or_degenerate_active_sets_return_zero_collisions(
    device: str,
    active_indices: list[int],
) -> None:
    """Zero- and one-active mixed-scale boxes keep diagnostics finite."""
    particles = _make_mixed_npf_droplet_particle_data()
    particles.concentration[0, :] = 0.0
    if active_indices:
        particles.concentration[0, active_indices] = np.array(
            [150.0] * len(active_indices),
            dtype=np.float64,
        )
    diagnostics = _collect_test_local_attempt_diagnostics(
        particles,
        device,
        time_step=0.5,
        max_collisions=8,
        rng_seed=41,
        volume=1.0e-14,
    )
    acceptance_fraction = np.divide(
        diagnostics.accepted_counts,
        diagnostics.executed_trial_counts,
        out=np.zeros_like(diagnostics.accepted_counts, dtype=np.float64),
        where=diagnostics.executed_trial_counts > 0,
    )

    npt.assert_array_equal(
        diagnostics.accepted_counts,
        np.array([0], dtype=np.int64),
    )
    npt.assert_array_equal(
        diagnostics.production_counts,
        np.array([0], dtype=np.int64),
    )
    assert np.all(np.isfinite(diagnostics.scheduled_trial_counts))
    assert np.all(diagnostics.scheduled_trial_counts >= 0)
    assert np.all(np.isfinite(diagnostics.executed_trial_counts))
    assert np.all(diagnostics.executed_trial_counts >= 0)
    assert np.all(np.isfinite(acceptance_fraction))
    assert np.all(acceptance_fraction >= 0.0)


def test_mixed_scale_diagnostic_tracks_executed_trials_under_early_exit(
    device: str,
) -> None:
    """Executed trials stop at the real early-exit point, not the schedule."""
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=1)
    diagnostics = _collect_test_local_attempt_diagnostics(
        particles,
        device,
        time_step=1.0,
        max_collisions=8,
        rng_seed=7,
        volume=1.0e-18,
    )

    assert diagnostics.scheduled_trial_counts[0] > 1
    assert diagnostics.executed_trial_counts[0] == 1
    assert diagnostics.accepted_counts[0] == 1
    assert diagnostics.production_counts[0] == 1


def test_mixed_scale_two_active_particles_accept_the_only_valid_pair(
    device: str,
) -> None:
    """Exactly two non-adjacent active particles accept their only valid pair."""
    particles = _make_mixed_npf_droplet_particle_data()
    particles.concentration[0, :] = 0.0
    surviving_indices = np.array([0, 3], dtype=np.int64)
    particles.concentration[0, surviving_indices] = np.array(
        [150.0, 0.8],
        dtype=np.float64,
    )
    diagnostics = _collect_test_local_attempt_diagnostics(
        particles,
        device,
        time_step=0.5,
        max_collisions=8,
        rng_seed=41,
        volume=1.0e-14,
    )
    expected_pair = surviving_indices.reshape(1, 2)

    npt.assert_array_equal(
        diagnostics.accepted_counts,
        np.array([1], dtype=np.int64),
    )
    npt.assert_array_equal(
        diagnostics.production_counts,
        np.array([1], dtype=np.int64),
    )
    npt.assert_array_equal(
        diagnostics.diagnostic_pairs[0, :1, :],
        expected_pair,
    )
    npt.assert_array_equal(
        diagnostics.production_pairs[0, :1, :],
        expected_pair,
    )
    assert np.all(diagnostics.diagnostic_pairs[0, 0, 0] >= 0)
    assert (
        diagnostics.diagnostic_pairs[0, 0, 0]
        < diagnostics.diagnostic_pairs[0, 0, 1]
    )
    assert diagnostics.diagnostic_pairs[0, 0, 1] < particles.masses.shape[1]
    assert np.all(np.isfinite(diagnostics.scheduled_trial_counts))
    assert np.all(np.isfinite(diagnostics.executed_trial_counts))
    assert np.all(diagnostics.scheduled_trial_counts >= 0)
    assert np.all(diagnostics.executed_trial_counts >= 0)
    acceptance_fraction = np.divide(
        diagnostics.accepted_counts,
        diagnostics.executed_trial_counts,
        out=np.zeros_like(diagnostics.accepted_counts, dtype=np.float64),
        where=diagnostics.executed_trial_counts > 0,
    )
    assert np.all(np.isfinite(acceptance_fraction))
    assert np.all(acceptance_fraction >= 0.0)


def test_mixed_scale_diagnostic_caps_scheduled_trials_to_operational_budget(
    device: str,
) -> None:
    """Diagnostic scheduled trials are capped at the shared budget."""
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=1)
    diagnostics = _collect_test_local_attempt_diagnostics(
        particles,
        device,
        time_step=1.0,
        max_collisions=8,
        rng_seed=7,
        volume=1.0e-300,
    )

    assert diagnostics.scheduled_trial_counts[0] == MAX_SCHEDULED_TRIALS_PER_BOX
    assert diagnostics.executed_trial_counts[0] == 1
    npt.assert_array_equal(
        diagnostics.accepted_counts,
        diagnostics.production_counts,
    )
    _assert_trimmed_collision_pairs_equal(
        diagnostics.diagnostic_pairs,
        diagnostics.accepted_counts,
        diagnostics.production_pairs,
        diagnostics.production_counts,
    )
    npt.assert_allclose(
        diagnostics.diagnostic_masses,
        diagnostics.production_masses,
    )
    npt.assert_allclose(
        diagnostics.diagnostic_concentration,
        diagnostics.production_concentration,
    )
    npt.assert_array_equal(
        diagnostics.diagnostic_rng_states,
        diagnostics.production_rng_states,
    )


def test_selector_guard_keeps_rank_resolution_available_for_later_attempts(
    device: str,
) -> None:
    """Unresolved rank selection does not truncate later valid resolutions."""
    active_flags = wp.array([[1, 0, 1, 1]], dtype=wp.int32, device=device)
    resolved_pairs = wp.zeros((1, 2), dtype=wp.int32, device=device)
    unresolved_attempts = wp.zeros((1,), dtype=wp.int32, device=device)

    wp.launch(
        _selector_guard_regression_kernel,
        dim=(1,),
        inputs=[active_flags, resolved_pairs, unresolved_attempts],
        device=device,
    )
    wp.synchronize()

    npt.assert_array_equal(
        np.asarray(unresolved_attempts.numpy()),
        np.array([1], dtype=np.int32),
    )
    npt.assert_array_equal(
        np.asarray(resolved_pairs.numpy()),
        np.array([[0, 3]], dtype=np.int32),
    )


@pytest.mark.parametrize(
    ("temperature", "pressure", "time_step", "message"),
    [
        (0.0, 101325.0, 0.5, "temperature must be finite and > 0"),
        (298.15, 0.0, 0.5, "pressure must be finite and > 0"),
        (
            298.15,
            101325.0,
            float("nan"),
            "time_step must be finite and nonnegative",
        ),
    ],
)
def test_mixed_scale_diagnostic_rejects_invalid_physical_inputs(
    device: str,
    temperature: float,
    pressure: float,
    time_step: float,
    message: str,
) -> None:
    """Private mixed-scale diagnostics fail clearly on invalid inputs."""
    particles = _make_mixed_npf_droplet_particle_data()

    with pytest.raises(ValueError, match=message):
        _collect_test_local_attempt_diagnostics(
            particles,
            device,
            temperature=temperature,
            pressure=pressure,
            time_step=time_step,
            max_collisions=8,
            rng_seed=41,
            volume=1.0e-14,
        )


def test_coagulation_conserves_mass_and_signed_charge_per_box(
    device: str,
) -> None:
    """Coagulation conserves per-box mass and signed charge inventories."""
    temperature = 298.15
    pressure = 101325.0
    time_step = 1.0
    particles = _make_particle_data(n_boxes=2, n_particles=4, n_species=2)
    particles.charge = np.array(
        [[-1.5, 2.5, -3.5, 4.5], [5.5, -6.5, 7.5, -8.5]],
        dtype=np.float64,
    )
    gpu_particles = to_warp_particle_data(particles, device=device)

    initial_mass = np.sum(particles.masses, axis=1)
    initial_charge = np.sum(particles.charge, axis=1)
    coagulation_step_gpu(
        gpu_particles,
        temperature=temperature,
        pressure=pressure,
        time_step=time_step,
        rng_seed=7,
        max_collisions=8,
    )
    result = from_warp_particle_data(gpu_particles, sync=True)

    final_mass = np.sum(result.masses, axis=1)
    final_charge = np.sum(result.charge, axis=1)
    npt.assert_allclose(final_mass, initial_mass, rtol=1e-12, atol=1e-30)
    npt.assert_allclose(final_charge, initial_charge, rtol=1e-12, atol=1e-30)


def test_mixed_scale_coagulation_conserves_total_mass(device: str) -> None:
    """Mixed-scale coagulation preserves total mass."""
    particles = _make_mixed_npf_droplet_particle_data()
    gpu_particles = to_warp_particle_data(particles, device=device)
    initial_mass = np.sum(particles.masses)

    coagulation_step_gpu(
        gpu_particles,
        temperature=298.15,
        pressure=101325.0,
        time_step=0.5,
        rng_seed=41,
        max_collisions=8,
        volume=1.0e-14,
    )
    result = from_warp_particle_data(gpu_particles, sync=True)

    final_mass = np.sum(result.masses)
    npt.assert_allclose(final_mass, initial_mass, rtol=1.0e-12)


def test_mixed_scale_repeated_seeded_runs_conserve_total_mass_even_with_zero_acceptance_trials(  # noqa: E501
    device: str,
) -> None:
    """Repeated mixed-scale seeded runs conserve total mass.

    Zero-acceptance trials remain valid outcomes and must preserve the same
    total mass as trials that accept one or more collisions.
    """
    particles = _make_mixed_npf_droplet_particle_data()
    initial_mass = float(np.sum(particles.masses))
    seed_range = range(101, 201)
    zero_acceptance_trials = 0
    accepted_collision_trials = 0

    for seed in seed_range:
        run_result = _run_seeded_coagulation_step(
            particles,
            device,
            temperature=298.15,
            pressure=101325.0,
            time_step=0.5,
            volume=1.0e-10,
            max_collisions=8,
            rng_seed=seed,
        )
        final_mass = float(np.sum(run_result.result_particles.masses))
        npt.assert_allclose(final_mass, initial_mass, rtol=1.0e-12)
        total_collisions = int(run_result.collision_counts.sum())
        assert total_collisions >= 0
        if total_collisions == 0:
            zero_acceptance_trials += 1
            npt.assert_allclose(
                run_result.result_particles.masses,
                particles.masses,
                rtol=1.0e-12,
            )
            npt.assert_allclose(
                run_result.result_particles.concentration,
                particles.concentration,
                rtol=1.0e-12,
            )
        else:
            accepted_collision_trials += 1

    assert zero_acceptance_trials > 0
    assert accepted_collision_trials > 0


def test_coagulation_step_gpu_reuses_preallocated_buffers(
    device: str,
) -> None:
    """Preallocated coagulation buffers are reused without reallocation."""
    particles = _make_particle_data(n_boxes=1, n_particles=4, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    volume = wp.array(np.array([1.0e-6]), dtype=wp.float64, device=device)
    collision_pairs = wp.zeros((1, 4, 2), dtype=wp.int32, device=device)
    n_collisions = wp.zeros((1,), dtype=wp.int32, device=device)
    rng_states = wp.zeros((1,), dtype=wp.uint32, device=device)

    _, returned_pairs, returned_counts = coagulation_step_gpu(
        gpu_particles,
        temperature=298.15,
        pressure=101325.0,
        time_step=0.1,
        volume=volume,
        max_collisions=4,
        rng_seed=11,
        collision_pairs=collision_pairs,
        n_collisions=n_collisions,
        rng_states=rng_states,
    )
    wp.synchronize()

    assert returned_pairs is collision_pairs
    assert returned_counts is n_collisions
    assert np.all(np.asarray(n_collisions.numpy()) >= 0)


def test_coagulation_step_gpu_clamps_auto_allocated_collision_capacity(
    device: str,
) -> None:
    """Auto-allocated collision buffers clamp to the useful per-box limit."""
    particles = _make_particle_data(n_boxes=1, n_particles=6, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)

    _, collision_pairs, collision_counts = coagulation_step_gpu(
        gpu_particles,
        temperature=298.15,
        pressure=101325.0,
        time_step=0.1,
        rng_seed=11,
        max_collisions=256,
    )
    wp.synchronize()

    assert collision_pairs.shape == (1, 3, 2)
    assert collision_counts.shape == (1,)


def test_coagulation_step_gpu_persisted_rng_states_advance_across_repeated_valid_calls(
    device: str,
) -> None:
    """Repeated valid calls advance persisted caller-owned RNG state.

    Reusing the same ``rng_seed`` with the same caller-owned ``rng_states``
    buffer must not restore the original seed-derived state between valid
    calls.
    """
    particles = _make_particle_data(n_boxes=1, n_particles=6, n_species=1)
    rng_states = wp.zeros((1,), dtype=wp.uint32, device=device)
    rng_seed = 37
    repeated_rng_states = rng_states

    wp.launch(
        _initialize_rng_states,
        dim=1,
        inputs=[wp.uint32(rng_seed), rng_states],
        device=device,
    )
    wp.synchronize()
    initial_state = np.asarray(rng_states.numpy()).copy()

    assert repeated_rng_states is rng_states

    first_particles = to_warp_particle_data(particles, device=device)
    coagulation_step_gpu(
        first_particles,
        temperature=298.15,
        pressure=101325.0,
        time_step=1.0,
        volume=1.0e-18,
        rng_seed=rng_seed,
        max_collisions=8,
        rng_states=repeated_rng_states,
    )
    wp.synchronize()
    state_after_first_call = np.asarray(rng_states.numpy()).copy()

    second_particles = to_warp_particle_data(particles, device=device)
    coagulation_step_gpu(
        second_particles,
        temperature=298.15,
        pressure=101325.0,
        time_step=1.0,
        volume=1.0e-18,
        rng_seed=rng_seed,
        max_collisions=8,
        rng_states=repeated_rng_states,
    )
    wp.synchronize()
    state_after_second_call = np.asarray(rng_states.numpy()).copy()

    assert not np.array_equal(state_after_first_call, initial_state)
    assert not np.array_equal(state_after_second_call, state_after_first_call)
    assert not np.array_equal(state_after_second_call, initial_state)


def test_coagulation_step_gpu_multibox_persisted_rng_states_advance(
    device: str,
) -> None:
    """Repeated valid calls advance persisted RNG state for every box."""
    particles = _make_particle_data(n_boxes=2, n_particles=6, n_species=1)
    rng_states = wp.zeros((2,), dtype=wp.uint32, device=device)

    wp.launch(
        _initialize_rng_states,
        dim=2,
        inputs=[wp.uint32(37), rng_states],
        device=device,
    )
    wp.synchronize()
    initial_state = np.asarray(rng_states.numpy()).copy()

    first_particles = to_warp_particle_data(particles, device=device)
    coagulation_step_gpu(
        first_particles,
        temperature=298.15,
        pressure=101325.0,
        time_step=1.0,
        volume=1.0e-18,
        rng_seed=37,
        max_collisions=8,
        rng_states=rng_states,
    )
    wp.synchronize()
    state_after_first_call = np.asarray(rng_states.numpy()).copy()

    second_particles = to_warp_particle_data(particles, device=device)
    coagulation_step_gpu(
        second_particles,
        temperature=298.15,
        pressure=101325.0,
        time_step=1.0,
        volume=1.0e-18,
        rng_seed=37,
        max_collisions=8,
        rng_states=rng_states,
    )
    wp.synchronize()
    state_after_second_call = np.asarray(rng_states.numpy()).copy()

    assert not np.array_equal(state_after_first_call, initial_state)
    assert not np.array_equal(state_after_second_call, state_after_first_call)
    assert not np.array_equal(state_after_second_call, initial_state)


def test_coagulation_step_gpu_initialize_rng_false_reuses_caller_owned_state(
    device: str,
) -> None:
    """Default ``initialize_rng=False`` reuses caller-owned RNG state."""
    particles = _make_particle_data(n_boxes=1, n_particles=6, n_species=1)
    rng_states = wp.zeros((1,), dtype=wp.uint32, device=device)

    wp.launch(
        _initialize_rng_states,
        dim=1,
        inputs=[wp.uint32(37), rng_states],
        device=device,
    )
    wp.synchronize()
    initialized_state = np.asarray(rng_states.numpy()).copy()

    first_particles = to_warp_particle_data(particles, device=device)
    coagulation_step_gpu(
        first_particles,
        temperature=298.15,
        pressure=101325.0,
        time_step=1.0,
        volume=1.0e-18,
        rng_seed=37,
        max_collisions=8,
        rng_states=rng_states,
        initialize_rng=False,
    )
    wp.synchronize()
    state_after_first_call = np.asarray(rng_states.numpy()).copy()

    second_particles = to_warp_particle_data(particles, device=device)
    coagulation_step_gpu(
        second_particles,
        temperature=298.15,
        pressure=101325.0,
        time_step=1.0,
        volume=1.0e-18,
        rng_seed=37,
        max_collisions=8,
        rng_states=rng_states,
        initialize_rng=False,
    )
    wp.synchronize()

    state_after_second_call = np.asarray(rng_states.numpy()).copy()

    assert not np.array_equal(state_after_first_call, initialized_state)
    assert not np.array_equal(state_after_second_call, initialized_state)
    assert not np.array_equal(state_after_second_call, state_after_first_call)


def test_coagulation_step_gpu_initialize_rng_true_resets_caller_owned_state(
    device: str,
) -> None:
    """Explicit ``True`` still forces reset from the provided seed."""
    particles = _make_particle_data(n_boxes=1, n_particles=6, n_species=1)
    rng_states_a = wp.zeros((1,), dtype=wp.uint32, device=device)
    rng_states_b = wp.zeros((1,), dtype=wp.uint32, device=device)

    wp.launch(
        _initialize_rng_states,
        dim=1,
        inputs=[wp.uint32(5), rng_states_a],
        device=device,
    )
    wp.launch(
        _initialize_rng_states,
        dim=1,
        inputs=[wp.uint32(9), rng_states_b],
        device=device,
    )
    wp.synchronize()
    pre_reset_state = np.asarray(rng_states_a.numpy()).copy()

    first_particles = to_warp_particle_data(particles, device=device)
    coagulation_step_gpu(
        first_particles,
        temperature=298.15,
        pressure=101325.0,
        time_step=1.0,
        volume=1.0e-18,
        rng_seed=41,
        max_collisions=8,
        rng_states=rng_states_a,
        initialize_rng=True,
    )

    second_particles = to_warp_particle_data(particles, device=device)
    coagulation_step_gpu(
        second_particles,
        temperature=298.15,
        pressure=101325.0,
        time_step=1.0,
        volume=1.0e-18,
        rng_seed=41,
        max_collisions=8,
        rng_states=rng_states_b,
        initialize_rng=True,
    )
    wp.synchronize()

    post_reset_state_a = np.asarray(rng_states_a.numpy()).copy()
    post_reset_state_b = np.asarray(rng_states_b.numpy()).copy()

    assert not np.array_equal(post_reset_state_a, pre_reset_state)
    npt.assert_array_equal(post_reset_state_a, post_reset_state_b)


def test_mixed_scale_caller_owned_rng_states_advance_without_hidden_reseed(
    device: str,
) -> None:
    """Mixed-scale caller-owned RNG state advances across repeated reuse.

    Reusing the same ``rng_seed`` with a persistent mixed-scale ``rng_states``
    buffer must advance the stream instead of silently reseeding it.
    """
    particles = _make_mixed_npf_droplet_particle_data()
    rng_seed = 41
    replay_volume = 1.0e-14
    rng_states = wp.zeros((1,), dtype=wp.uint32, device=device)
    initialize_coagulation_rng_states(rng_seed, rng_states, device=device)
    initial_rng_state = np.asarray(rng_states.numpy(), dtype=np.uint32).copy()

    first_result = _run_seeded_coagulation_step(
        particles,
        device,
        temperature=298.15,
        pressure=101325.0,
        time_step=0.5,
        volume=replay_volume,
        max_collisions=8,
        rng_seed=rng_seed,
        rng_states=rng_states,
    )
    second_result = _run_seeded_coagulation_step(
        particles,
        device,
        temperature=298.15,
        pressure=101325.0,
        time_step=0.5,
        volume=replay_volume,
        max_collisions=8,
        rng_seed=rng_seed,
        rng_states=rng_states,
    )
    replay_result = _run_seeded_coagulation_step(
        particles,
        device,
        temperature=298.15,
        pressure=101325.0,
        time_step=0.5,
        volume=replay_volume,
        max_collisions=8,
        rng_seed=rng_seed,
        rng_states=wp.zeros((1,), dtype=wp.uint32, device=device),
        initialize_rng=True,
    )

    assert first_result.rng_states is not None
    assert second_result.rng_states is not None
    assert not np.array_equal(first_result.rng_states, initial_rng_state)
    assert not np.array_equal(second_result.rng_states, first_result.rng_states)
    assert _coagulation_outcomes_match(first_result, replay_result)
    assert not _coagulation_outcomes_match(second_result, replay_result)


def test_mixed_scale_initialize_rng_true_replays_seeded_state_and_outcome(
    device: str,
) -> None:
    """Mixed-scale explicit RNG reset reproduces seeded state and outcomes.

    Separate caller-owned buffers should converge to the same post-reset state,
    accepted pairs, and particle outcomes when ``initialize_rng=True`` is used
    with the same seed.
    """
    particles = _make_mixed_npf_droplet_particle_data()
    rng_seed = 41
    rng_states_a = wp.zeros((1,), dtype=wp.uint32, device=device)
    rng_states_b = wp.zeros((1,), dtype=wp.uint32, device=device)

    initialize_coagulation_rng_states(5, rng_states_a, device=device)
    initialize_coagulation_rng_states(9, rng_states_b, device=device)
    pre_reset_state_a = np.asarray(rng_states_a.numpy(), dtype=np.uint32).copy()
    pre_reset_state_b = np.asarray(rng_states_b.numpy(), dtype=np.uint32).copy()

    first_result = _run_seeded_coagulation_step(
        particles,
        device,
        temperature=298.15,
        pressure=101325.0,
        time_step=0.5,
        volume=1.0e-10,
        max_collisions=8,
        rng_seed=rng_seed,
        rng_states=rng_states_a,
        initialize_rng=True,
    )
    second_result = _run_seeded_coagulation_step(
        particles,
        device,
        temperature=298.15,
        pressure=101325.0,
        time_step=0.5,
        volume=1.0e-10,
        max_collisions=8,
        rng_seed=rng_seed,
        rng_states=rng_states_b,
        initialize_rng=True,
    )

    assert first_result.rng_states is not None
    assert second_result.rng_states is not None
    assert not np.array_equal(pre_reset_state_a, pre_reset_state_b)
    assert not np.array_equal(first_result.rng_states, pre_reset_state_a)
    assert not np.array_equal(second_result.rng_states, pre_reset_state_b)
    npt.assert_array_equal(first_result.rng_states, second_result.rng_states)
    npt.assert_array_equal(
        first_result.collision_counts,
        second_result.collision_counts,
    )
    npt.assert_array_equal(
        first_result.collision_pairs,
        second_result.collision_pairs,
    )
    npt.assert_allclose(
        first_result.result_particles.masses,
        second_result.result_particles.masses,
    )
    npt.assert_allclose(
        first_result.result_particles.concentration,
        second_result.result_particles.concentration,
    )


def test_coagulation_step_gpu_multibox_initialize_rng_true_resets_state(
    device: str,
) -> None:
    """Explicit multibox reset reproduces the same seed-derived state."""
    particles = _make_particle_data(n_boxes=2, n_particles=6, n_species=1)
    rng_states_a = wp.zeros((2,), dtype=wp.uint32, device=device)
    rng_states_b = wp.zeros((2,), dtype=wp.uint32, device=device)

    wp.launch(
        _initialize_rng_states,
        dim=2,
        inputs=[wp.uint32(5), rng_states_a],
        device=device,
    )
    wp.launch(
        _initialize_rng_states,
        dim=2,
        inputs=[wp.uint32(9), rng_states_b],
        device=device,
    )
    wp.synchronize()

    first_particles = to_warp_particle_data(particles, device=device)
    coagulation_step_gpu(
        first_particles,
        temperature=298.15,
        pressure=101325.0,
        time_step=1.0,
        volume=1.0e-18,
        rng_seed=41,
        max_collisions=8,
        rng_states=rng_states_a,
        initialize_rng=True,
    )

    second_particles = to_warp_particle_data(particles, device=device)
    coagulation_step_gpu(
        second_particles,
        temperature=298.15,
        pressure=101325.0,
        time_step=1.0,
        volume=1.0e-18,
        rng_seed=41,
        max_collisions=8,
        rng_states=rng_states_b,
        initialize_rng=True,
    )
    wp.synchronize()

    post_reset_state_a = np.asarray(rng_states_a.numpy()).copy()
    post_reset_state_b = np.asarray(rng_states_b.numpy()).copy()
    npt.assert_array_equal(post_reset_state_a, post_reset_state_b)


def test_coagulation_step_gpu_omitted_rng_states_repeat_seed_replays_results(
    device: str,
) -> None:
    """Omitted RNG state allocation is reseeded independently per call."""
    particles = _make_particle_data(n_boxes=1, n_particles=6, n_species=1)

    first_particles = to_warp_particle_data(particles, device=device)
    _, first_pairs, first_counts = coagulation_step_gpu(
        first_particles,
        temperature=298.15,
        pressure=101325.0,
        time_step=1.0,
        volume=1.0e-18,
        rng_seed=37,
        max_collisions=8,
    )
    wp.synchronize()

    second_particles = to_warp_particle_data(particles, device=device)
    _, second_pairs, second_counts = coagulation_step_gpu(
        second_particles,
        temperature=298.15,
        pressure=101325.0,
        time_step=1.0,
        volume=1.0e-18,
        rng_seed=37,
        max_collisions=8,
    )
    wp.synchronize()

    npt.assert_array_equal(
        np.asarray(first_pairs.numpy()),
        second_pairs.numpy(),
    )
    npt.assert_array_equal(
        np.asarray(first_counts.numpy()),
        second_counts.numpy(),
    )


def test_coagulation_step_gpu_multibox_omitted_rng_states_replay_results(
    device: str,
) -> None:
    """Omitted multibox RNG allocation still replays repeated seeded calls."""
    particles = _make_particle_data(n_boxes=2, n_particles=6, n_species=1)

    first_particles = to_warp_particle_data(particles, device=device)
    _, first_pairs, first_counts = coagulation_step_gpu(
        first_particles,
        temperature=298.15,
        pressure=101325.0,
        time_step=1.0,
        volume=1.0e-18,
        rng_seed=37,
        max_collisions=8,
    )
    first_result = from_warp_particle_data(first_particles, sync=True)
    wp.synchronize()

    second_particles = to_warp_particle_data(particles, device=device)
    _, second_pairs, second_counts = coagulation_step_gpu(
        second_particles,
        temperature=298.15,
        pressure=101325.0,
        time_step=1.0,
        volume=1.0e-18,
        rng_seed=37,
        max_collisions=8,
    )
    second_result = from_warp_particle_data(second_particles, sync=True)
    wp.synchronize()

    npt.assert_array_equal(
        np.asarray(first_pairs.numpy()),
        second_pairs.numpy(),
    )
    npt.assert_array_equal(
        np.asarray(first_counts.numpy()),
        second_counts.numpy(),
    )
    npt.assert_allclose(first_result.masses, second_result.masses)
    npt.assert_allclose(
        first_result.concentration,
        second_result.concentration,
    )


def test_coagulation_marks_inactive_particles(device: str) -> None:
    """Merged particles are marked inactive and mass is transferred."""
    temperature = 298.15
    pressure = 101325.0
    time_step = 1.0
    particles = _make_particle_data(n_boxes=1, n_particles=4, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)

    _, _, collision_counts = coagulation_step_gpu(
        gpu_particles,
        temperature=temperature,
        pressure=pressure,
        time_step=time_step,
        volume=1.0e-18,
        rng_seed=3,
        max_collisions=8,
    )
    result = from_warp_particle_data(gpu_particles, sync=True)

    assert np.asarray(collision_counts.numpy()).sum() > 0
    assert np.any(result.concentration == 0.0)
    assert np.max(result.masses) >= np.max(particles.masses)


def _launch_private_coagulation_sampler(
    device: str,
    masses: np.ndarray,
    concentration: np.ndarray,
    mechanism_mask: int,
    *,
    time_step: float,
    volume: float = 1.0,
    rng_state: int = 17,
    states: Any | None = None,
) -> dict[str, Any]:
    """Launch the private selector with caller-visible scratch for regression tests."""
    n_boxes, n_particles, _ = masses.shape
    masses_wp = wp.array(masses, dtype=wp.float64, device=device)
    concentration_wp = wp.array(concentration, dtype=wp.float64, device=device)
    radii = wp.full(
        (n_boxes, n_particles), 9.0, dtype=wp.float64, device=device
    )
    diffusivities = wp.zeros_like(radii)
    g_terms = wp.zeros_like(radii)
    speeds = wp.zeros_like(radii)
    settling_velocities = wp.zeros(
        (n_boxes, n_particles), dtype=wp.float64, device=device
    )
    total_masses = wp.zeros_like(radii)
    active_indices = wp.zeros(
        (n_boxes, n_particles), dtype=wp.int32, device=device
    )
    pairs = wp.full(
        (n_boxes, n_particles, 2), -7, dtype=wp.int32, device=device
    )
    counts = wp.full((n_boxes,), -7, dtype=wp.int32, device=device)
    if states is None:
        states = wp.full((n_boxes,), rng_state, dtype=wp.uint32, device=device)
    wp.launch(
        brownian_coagulation_kernel,
        dim=n_boxes,
        inputs=[
            masses_wp,
            concentration_wp,
            wp.array([1000.0], dtype=wp.float64, device=device),
            wp.full((n_boxes,), volume, dtype=wp.float64, device=device),
            wp.full((n_boxes,), 298.15, dtype=wp.float64, device=device),
            wp.full((n_boxes,), 101325.0, dtype=wp.float64, device=device),
            wp.float64(constants.GAS_CONSTANT),
            wp.float64(constants.BOLTZMANN_CONSTANT),
            wp.float64(constants.MOLECULAR_WEIGHT_AIR),
            wp.float64(constants.REF_VISCOSITY_AIR_STP),
            wp.float64(constants.REF_TEMPERATURE_STP),
            wp.float64(constants.SUTHERLAND_CONSTANT),
            wp.float64(constants.ELEMENTARY_CHARGE_VALUE),
            wp.float64(constants.ELECTRIC_PERMITTIVITY),
            wp.float64(time_step),
            radii,
            diffusivities,
            g_terms,
            speeds,
            settling_velocities,
            wp.zeros((n_boxes,), dtype=wp.float64, device=device),
            wp.zeros((n_boxes,), dtype=wp.float64, device=device),
            total_masses,
            wp.zeros_like(radii),
            active_indices,
            pairs,
            counts,
            states,
            wp.int32(mechanism_mask),
            wp.int32(n_particles),
        ],
        device=device,
    )
    wp.synchronize_device(device)
    return {
        "masses": masses_wp,
        "concentration": concentration_wp,
        "radii": radii,
        "settling_velocities": settling_velocities,
        "active_indices": active_indices,
        "pairs": pairs,
        "counts": counts,
        "states": states,
    }


@pytest.mark.stochastic
def test_private_sedimentation_sampler_bounds_pairs_and_advances_rng(
    device: str,
) -> None:
    """Exact SP2016 dispatch shares the bounded selector and persistent RNG path."""
    masses = np.array([[[1.0e-15], [8.0e-15], [2.7e-14], [6.4e-14]]])
    concentration = np.ones((1, 4), dtype=np.float64)
    first = _launch_private_coagulation_sampler(
        device,
        masses,
        concentration,
        SEDIMENTATION_SP2016_MECHANISM_FLAG,
        time_step=1.0e308,
    )
    initial_states = np.asarray(first["states"].numpy()).copy()
    second = _launch_private_coagulation_sampler(
        device,
        masses,
        concentration,
        SEDIMENTATION_SP2016_MECHANISM_FLAG,
        time_step=1.0e308,
        states=first["states"],
    )
    count = int(first["counts"].numpy()[0])
    pairs = np.asarray(first["pairs"].numpy())[0, :count]

    assert 0 <= count <= 2 <= MAX_SCHEDULED_TRIALS_PER_BOX
    assert np.all(pairs[:, 0] < pairs[:, 1])
    assert set(pairs.ravel()).issubset({0, 1, 2, 3})
    assert len(np.unique(pairs)) == 2 * count
    assert int(initial_states[0]) != 17
    assert int(second["states"].numpy()[0]) != int(initial_states[0])


@pytest.mark.stochastic
def test_private_sedimentation_sampler_scales_trials_by_slot_concentration(
    device: str,
) -> None:
    """Equal non-unit slot weights scale non-saturating SP2016 scheduling."""
    masses = np.array([[[1.0e-15], [8.0e-15], [2.7e-14], [6.4e-14]]])
    trial_counts: list[int] = []
    for concentration_value in (0.25, 4.0):
        count = 0
        for seed in range(64):
            result = _launch_private_coagulation_sampler(
                device,
                masses,
                np.full((1, 4), concentration_value, dtype=np.float64),
                SEDIMENTATION_SP2016_MECHANISM_FLAG,
                time_step=1.0e12,
                rng_state=seed + 1,
            )
            count += int(result["counts"].numpy()[0])
        trial_counts.append(count)

    # The 16x concentration change must increase the aggregate event count;
    # use a conservative bound rather than exact per-seed stochastic replay.
    assert trial_counts[1] >= 2 * trial_counts[0]


@pytest.mark.parametrize(
    ("concentration", "expected_count", "active_slots"),
    [
        (np.zeros((1, 4), dtype=np.float64), 0, set()),
        (np.array([[0.0, 1.0, 0.0, 0.0]]), 0, {1}),
        (np.array([[1.0, 0.0, 0.0, 1.0]]), 1, {0, 3}),
    ],
)
def test_private_sedimentation_sampler_sparse_active_slots(
    device: str,
    concentration: np.ndarray,
    expected_count: int,
    active_slots: set[int],
) -> None:
    """Sparse exact-private launches only select initially active slots."""
    result = _launch_private_coagulation_sampler(
        device,
        np.array([[[1.0e-15], [8.0e-15], [2.7e-14], [6.4e-14]]]),
        concentration,
        SEDIMENTATION_SP2016_MECHANISM_FLAG,
        time_step=1.0e308,
    )
    count = int(result["counts"].numpy()[0])
    pairs = np.asarray(result["pairs"].numpy())[0, :count]

    assert count == expected_count
    assert count <= MAX_SCHEDULED_TRIALS_PER_BOX
    assert set(pairs.ravel()).issubset(active_slots)
    if count:
        assert np.all(pairs[:, 0] < pairs[:, 1])


def test_private_three_way_mask_rejects_without_rng_or_pair_work(
    device: str,
) -> None:
    """Deferred direct three-way masks perform no schedule work."""
    masses = np.array([[[1.0e-15], [8.0e-15]]])
    result = _launch_private_coagulation_sampler(
        device,
        masses,
        np.ones((1, 2), dtype=np.float64),
        BROWNIAN_MECHANISM_FLAG
        | CHARGED_HARD_SPHERE_MECHANISM_FLAG
        | SEDIMENTATION_SP2016_MECHANISM_FLAG,
        time_step=1.0e308,
    )

    npt.assert_array_equal(result["counts"].numpy(), [-7])
    npt.assert_array_equal(result["pairs"].numpy(), [[[-7, -7], [-7, -7]]])
    npt.assert_array_equal(result["states"].numpy(), [17])
    npt.assert_array_equal(result["masses"].numpy(), masses)
    npt.assert_array_equal(result["concentration"].numpy(), [[1.0, 1.0]])
    npt.assert_array_equal(result["radii"].numpy(), [[9.0, 9.0]])
    npt.assert_array_equal(result["settling_velocities"].numpy(), [[0.0, 0.0]])


def test_private_sampler_populates_sedimentation_velocity_scratch(
    device: str,
) -> None:
    """Settling scratch is private to exact SP2016 launches and valid slots."""
    masses = np.array([[[1.0e-15], [8.0e-15], [0.0]]])
    concentration = np.array([[1.0, 1.0, 1.0]], dtype=np.float64)
    brownian = _launch_private_coagulation_sampler(
        device, masses, concentration, BROWNIAN_MECHANISM_FLAG, time_step=0.0
    )
    sedimentation = _launch_private_coagulation_sampler(
        device,
        masses,
        concentration,
        SEDIMENTATION_SP2016_MECHANISM_FLAG,
        time_step=0.0,
    )

    npt.assert_array_equal(brownian["settling_velocities"].numpy(), [[0.0] * 3])
    velocities = np.asarray(sedimentation["settling_velocities"].numpy())[0]
    expected = np.array(
        [
            _settling_velocity_oracle(
                (3.0 * mass / (4.0 * np.pi * 1000.0)) ** (1.0 / 3.0),
                1000.0,
            )
            for mass in masses[0, :2, 0]
        ]
    )
    npt.assert_allclose(velocities[:2], expected, rtol=1.0e-12, atol=0.0)
    assert velocities[2] == 0.0


def test_private_sedimentation_invalid_particles_clear_scratch_and_rng(
    device: str,
) -> None:
    """Invalid private sedimentation slots cannot schedule or advance RNG."""
    result = _launch_private_coagulation_sampler(
        device,
        np.array([[[0.0], [np.nan], [-1.0]]], dtype=np.float64),
        np.ones((1, 3), dtype=np.float64),
        SEDIMENTATION_SP2016_MECHANISM_FLAG,
        time_step=1.0e308,
    )

    npt.assert_array_equal(result["settling_velocities"].numpy(), [[0.0] * 3])
    npt.assert_array_equal(result["counts"].numpy(), [0])
    npt.assert_array_equal(result["states"].numpy(), [17])


def test_brownian_coagulation_kernel_inactive_particles(
    device: str,
) -> None:
    """Brownian kernel returns no collisions when particles are inactive."""
    n_boxes = 1
    n_particles = 2
    n_species = 1
    masses = wp.array(
        np.full((n_boxes, n_particles, n_species), 1.0e-18, dtype=np.float64),
        dtype=wp.float64,
        device=device,
    )
    concentration = wp.zeros(
        (n_boxes, n_particles),
        dtype=wp.float64,
        device=device,
    )
    density = wp.array(
        np.array([1000.0], dtype=np.float64),
        dtype=wp.float64,
        device=device,
    )
    volume = wp.array(
        np.array([1.0e-6], dtype=np.float64),
        dtype=wp.float64,
        device=device,
    )
    temperature = wp.array(
        np.array([298.15], dtype=np.float64),
        dtype=wp.float64,
        device=device,
    )
    pressure = wp.array(
        np.array([101325.0], dtype=np.float64),
        dtype=wp.float64,
        device=device,
    )

    radii = wp.zeros((n_boxes, n_particles), dtype=wp.float64, device=device)
    diffusivities = wp.zeros(
        (n_boxes, n_particles), dtype=wp.float64, device=device
    )
    g_terms = wp.zeros((n_boxes, n_particles), dtype=wp.float64, device=device)
    speeds = wp.zeros((n_boxes, n_particles), dtype=wp.float64, device=device)
    settling_velocities = wp.zeros_like(speeds)
    total_masses = wp.zeros_like(speeds)
    charge = wp.zeros_like(speeds)
    active_flags = wp.zeros(
        (n_boxes, n_particles), dtype=wp.int32, device=device
    )
    collision_pairs = wp.zeros((n_boxes, 4, 2), dtype=wp.int32, device=device)
    n_collisions = wp.zeros((n_boxes,), dtype=wp.int32, device=device)
    rng_states = wp.zeros((n_boxes,), dtype=wp.uint32, device=device)

    wp.launch(
        brownian_coagulation_kernel,
        dim=(n_boxes,),
        inputs=[
            masses,
            concentration,
            density,
            volume,
            temperature,
            pressure,
            wp.float64(constants.GAS_CONSTANT),
            wp.float64(constants.BOLTZMANN_CONSTANT),
            wp.float64(constants.MOLECULAR_WEIGHT_AIR),
            wp.float64(constants.REF_VISCOSITY_AIR_STP),
            wp.float64(constants.REF_TEMPERATURE_STP),
            wp.float64(constants.SUTHERLAND_CONSTANT),
            wp.float64(constants.ELEMENTARY_CHARGE_VALUE),
            wp.float64(constants.ELECTRIC_PERMITTIVITY),
            wp.float64(1.0),
            radii,
            diffusivities,
            g_terms,
            speeds,
            settling_velocities,
            wp.zeros((n_boxes,), dtype=wp.float64, device=device),
            wp.zeros((n_boxes,), dtype=wp.float64, device=device),
            total_masses,
            charge,
            active_flags,
            collision_pairs,
            n_collisions,
            rng_states,
            wp.int32(BROWNIAN_MECHANISM_FLAG),
            wp.int32(4),
        ],
        device=device,
    )
    wp.synchronize()

    assert np.asarray(n_collisions.numpy()).item() == 0
    npt.assert_allclose(np.asarray(radii.numpy()), 0.0)
    npt.assert_allclose(np.asarray(diffusivities.numpy()), 0.0)
    npt.assert_allclose(np.asarray(g_terms.numpy()), 0.0)
    npt.assert_allclose(np.asarray(speeds.numpy()), 0.0)


def test_brownian_kernel_zero_mask_rejects_work_and_caps_overflow_schedule(
    device: str,
) -> None:
    """Private dispatch preserves no-mutation rejection and capped scheduling."""
    n_boxes = 1
    n_particles = 2
    masses = wp.array(
        np.full((n_boxes, n_particles, 1), 1.0e-18, dtype=np.float64),
        dtype=wp.float64,
        device=device,
    )
    concentration = wp.array(
        np.ones((n_boxes, n_particles), dtype=np.float64),
        dtype=wp.float64,
        device=device,
    )
    density = wp.array([1000.0], dtype=wp.float64, device=device)
    volume = wp.array([1.0], dtype=wp.float64, device=device)
    temperature = wp.array([298.15], dtype=wp.float64, device=device)
    pressure = wp.array([101325.0], dtype=wp.float64, device=device)
    radii = wp.zeros((n_boxes, n_particles), dtype=wp.float64, device=device)
    diffusivities = wp.zeros_like(radii)
    g_terms = wp.zeros_like(radii)
    speeds = wp.zeros_like(radii)
    settling_velocities = wp.zeros_like(radii)
    total_masses = wp.zeros_like(radii)
    charge = wp.zeros_like(radii)
    active_indices = wp.zeros(
        (n_boxes, n_particles), dtype=wp.int32, device=device
    )
    collision_pairs = wp.array([[[91, 92]]], dtype=wp.int32, device=device)
    n_collisions = wp.array([9], dtype=wp.int32, device=device)
    rng_states = wp.array([123], dtype=wp.uint32, device=device)

    def launch(mechanism_mask: int, time_step: float) -> None:
        """Launch the private sampler with its production dispatch inputs."""
        wp.launch(
            brownian_coagulation_kernel,
            dim=n_boxes,
            inputs=[
                masses,
                concentration,
                density,
                volume,
                temperature,
                pressure,
                wp.float64(constants.GAS_CONSTANT),
                wp.float64(constants.BOLTZMANN_CONSTANT),
                wp.float64(constants.MOLECULAR_WEIGHT_AIR),
                wp.float64(constants.REF_VISCOSITY_AIR_STP),
                wp.float64(constants.REF_TEMPERATURE_STP),
                wp.float64(constants.SUTHERLAND_CONSTANT),
                wp.float64(constants.ELEMENTARY_CHARGE_VALUE),
                wp.float64(constants.ELECTRIC_PERMITTIVITY),
                wp.float64(time_step),
                radii,
                diffusivities,
                g_terms,
                speeds,
                settling_velocities,
                wp.zeros((n_boxes,), dtype=wp.float64, device=device),
                wp.zeros((n_boxes,), dtype=wp.float64, device=device),
                total_masses,
                charge,
                active_indices,
                collision_pairs,
                n_collisions,
                rng_states,
                wp.int32(mechanism_mask),
                wp.int32(1),
            ],
            device=device,
        )
        wp.synchronize()

    initial_rng_state = np.asarray(rng_states.numpy()).copy()
    launch(0, 1.0)

    assert np.asarray(n_collisions.numpy()).item() == 9
    npt.assert_array_equal(
        np.asarray(collision_pairs.numpy()),
        np.array([[[91, 92]]], dtype=np.int32),
    )
    npt.assert_array_equal(np.asarray(active_indices.numpy()), [[0, 0]])
    npt.assert_array_equal(np.asarray(rng_states.numpy()), initial_rng_state)

    masses = wp.array(
        np.full((n_boxes, n_particles, 1), np.nan, dtype=np.float64),
        dtype=wp.float64,
        device=device,
    )
    launch(BROWNIAN_MECHANISM_FLAG, 1.0)

    assert np.asarray(n_collisions.numpy()).item() == 0
    npt.assert_array_equal(
        np.asarray(collision_pairs.numpy()),
        np.array([[[91, 92]]], dtype=np.int32),
    )
    npt.assert_array_equal(np.asarray(rng_states.numpy()), initial_rng_state)

    masses = wp.array(
        np.full((n_boxes, n_particles, 1), 1.0e-18, dtype=np.float64),
        dtype=wp.float64,
        device=device,
    )
    collision_pairs = wp.array([[[91, 92]]], dtype=wp.int32, device=device)
    n_collisions = wp.array([9], dtype=wp.int32, device=device)
    rng_states = wp.array([123], dtype=wp.uint32, device=device)
    active_indices = wp.zeros(
        (n_boxes, n_particles), dtype=wp.int32, device=device
    )
    launch(BROWNIAN_MECHANISM_FLAG, 1.0e308)

    assert np.asarray(n_collisions.numpy()).item() == 1
    npt.assert_array_equal(
        np.asarray(collision_pairs.numpy()),
        np.array([[[0, 1]]], dtype=np.int32),
    )
    assert not np.array_equal(np.asarray(rng_states.numpy()), initial_rng_state)


def test_apply_coagulation_kernel_merges_particles(device: str) -> None:
    """Apply kernel merges mass and charge and clears donor state."""
    masses = np.array([[[1.0e-18], [2.0e-18]]], dtype=np.float64)
    concentration = np.array([[1.0, 1.0]], dtype=np.float64)
    charge = np.array([[1.5, -0.5]], dtype=np.float64)
    collision_pairs = np.array([[[0, 1]]], dtype=np.int32)
    n_collisions = np.array([1], dtype=np.int32)

    masses_wp = wp.array(masses, dtype=wp.float64, device=device)
    concentration_wp = wp.array(concentration, dtype=wp.float64, device=device)
    charge_wp = wp.array(charge, dtype=wp.float64, device=device)
    collision_pairs_wp = wp.array(
        collision_pairs, dtype=wp.int32, device=device
    )
    n_collisions_wp = wp.array(n_collisions, dtype=wp.int32, device=device)

    wp.launch(
        apply_coagulation_kernel,
        dim=(1, 1),
        inputs=[
            masses_wp,
            concentration_wp,
            charge_wp,
            collision_pairs_wp,
            n_collisions_wp,
        ],
        device=device,
    )
    wp.synchronize()

    result_masses = np.asarray(masses_wp.numpy())
    result_concentration = np.asarray(concentration_wp.numpy())
    result_charge = np.asarray(charge_wp.numpy())

    npt.assert_allclose(result_masses[0, 0, 0], 3.0e-18)
    npt.assert_allclose(result_masses[0, 1, 0], 0.0)
    npt.assert_allclose(result_concentration[0, 1], 0.0)
    npt.assert_allclose(result_charge[0, 0], 1.0)
    npt.assert_allclose(result_charge[0, 1], 0.0)

    n_collisions_zero = wp.zeros((1,), dtype=wp.int32, device=device)
    wp.launch(
        apply_coagulation_kernel,
        dim=(1, 1),
        inputs=[
            masses_wp,
            concentration_wp,
            charge_wp,
            collision_pairs_wp,
            n_collisions_zero,
        ],
        device=device,
    )
    wp.synchronize()

    npt.assert_array_equal(np.asarray(masses_wp.numpy()), result_masses)
    npt.assert_array_equal(
        np.asarray(concentration_wp.numpy()), result_concentration
    )
    npt.assert_array_equal(np.asarray(charge_wp.numpy()), result_charge)


def test_apply_coagulation_kernel_skips_unrepresentable_charge_merge(
    device: str,
) -> None:
    """Finite same-sign charges cannot overflow into mutated particle state."""
    maximum = np.finfo(np.float64).max
    masses = wp.array([[[1.0], [2.0]]], dtype=wp.float64, device=device)
    concentration = wp.array([[1.0, 1.0]], dtype=wp.float64, device=device)
    charge = wp.array([[maximum, maximum]], dtype=wp.float64, device=device)
    pairs = wp.array([[[0, 1]]], dtype=wp.int32, device=device)
    counts = wp.array([1], dtype=wp.int32, device=device)

    wp.launch(
        apply_coagulation_kernel,
        dim=(1, 1),
        inputs=[masses, concentration, charge, pairs, counts],
        device=device,
    )
    wp.launch(
        _compact_applied_collision_pairs_kernel,
        dim=1,
        inputs=[pairs, counts],
        device=device,
    )
    wp.synchronize()

    npt.assert_array_equal(np.asarray(masses.numpy()), [[[1.0], [2.0]]])
    npt.assert_array_equal(np.asarray(concentration.numpy()), [[1.0, 1.0]])
    result_charge = np.asarray(charge.numpy())
    npt.assert_array_equal(result_charge, [[maximum, maximum]])
    assert np.all(np.isfinite(result_charge))
    npt.assert_array_equal(np.asarray(pairs.numpy()), [[[-1, -1]]])
    npt.assert_array_equal(np.asarray(counts.numpy()), [0])


def test_apply_coagulation_kernel_skips_overflowing_mass_merge(
    device: str,
) -> None:
    """Finite masses cannot overflow into a partially merged particle state."""
    maximum = np.finfo(np.float64).max
    masses = wp.array(
        [[[maximum, 1.0], [maximum, 2.0]]],
        dtype=wp.float64,
        device=device,
    )
    concentration = wp.array([[1.0, 1.0]], dtype=wp.float64, device=device)
    charge = wp.array([[1.0, -1.0]], dtype=wp.float64, device=device)
    pairs = wp.array([[[0, 1]]], dtype=wp.int32, device=device)
    counts = wp.array([1], dtype=wp.int32, device=device)

    wp.launch(
        apply_coagulation_kernel,
        dim=(1, 1),
        inputs=[masses, concentration, charge, pairs, counts],
        device=device,
    )
    wp.launch(
        _compact_applied_collision_pairs_kernel,
        dim=1,
        inputs=[pairs, counts],
        device=device,
    )
    wp.synchronize()

    result_masses = np.asarray(masses.numpy())
    npt.assert_array_equal(result_masses, [[[maximum, 1.0], [maximum, 2.0]]])
    assert np.all(np.isfinite(result_masses))
    npt.assert_array_equal(np.asarray(concentration.numpy()), [[1.0, 1.0]])
    npt.assert_array_equal(np.asarray(charge.numpy()), [[1.0, -1.0]])
    npt.assert_array_equal(np.asarray(pairs.numpy()), [[[-1, -1]]])
    npt.assert_array_equal(np.asarray(counts.numpy()), [0])


def test_compaction_uses_only_populated_oversized_buffer_entries(
    device: str,
) -> None:
    """Compaction retains valid populated entries in an oversized buffer."""
    pairs = np.full((1, 64, 2), -7, dtype=np.int32)
    pairs[0, 0] = [0, 1]
    pairs[0, 1] = [-1, -1]
    pairs_wp = wp.array(pairs, dtype=wp.int32, device=device)
    counts_wp = wp.array([2], dtype=wp.int32, device=device)

    wp.launch(
        _compact_applied_collision_pairs_kernel,
        dim=1,
        inputs=[pairs_wp, counts_wp],
        device=device,
    )
    wp.synchronize()

    npt.assert_array_equal(np.asarray(counts_wp.numpy()), [1])
    npt.assert_array_equal(np.asarray(pairs_wp.numpy())[0, 0], [0, 1])
    npt.assert_array_equal(np.asarray(pairs_wp.numpy())[0, 2:], pairs[0, 2:])


def test_apply_coagulation_kernel_skips_self_pair(device: str) -> None:
    """Apply kernel ignores self-collisions without mutating arrays."""
    masses = np.array([[[1.0e-18], [2.0e-18]]], dtype=np.float64)
    concentration = np.array([[1.0, 1.0]], dtype=np.float64)
    charge = np.array([[1.5, -0.5]], dtype=np.float64)
    collision_pairs = np.array([[[0, 0]]], dtype=np.int32)
    n_collisions = np.array([1], dtype=np.int32)

    masses_wp = wp.array(masses, dtype=wp.float64, device=device)
    concentration_wp = wp.array(concentration, dtype=wp.float64, device=device)
    charge_wp = wp.array(charge, dtype=wp.float64, device=device)
    collision_pairs_wp = wp.array(
        collision_pairs, dtype=wp.int32, device=device
    )
    n_collisions_wp = wp.array(n_collisions, dtype=wp.int32, device=device)

    wp.launch(
        apply_coagulation_kernel,
        dim=(1, 1),
        inputs=[
            masses_wp,
            concentration_wp,
            charge_wp,
            collision_pairs_wp,
            n_collisions_wp,
        ],
        device=device,
    )
    wp.synchronize()

    result_masses = np.asarray(masses_wp.numpy())
    result_concentration = np.asarray(concentration_wp.numpy())
    result_charge = np.asarray(charge_wp.numpy())

    npt.assert_array_equal(result_masses, masses)
    npt.assert_array_equal(result_concentration, concentration)
    npt.assert_array_equal(result_charge, charge)


def test_apply_coagulation_kernel_skips_empty_pair(device: str) -> None:
    """Apply kernel ignores entries when collision index is out of range."""
    masses = np.array([[[1.0e-18], [2.0e-18]]], dtype=np.float64)
    concentration = np.array([[1.0, 1.0]], dtype=np.float64)
    charge = np.array([[1.5, -0.5]], dtype=np.float64)
    collision_pairs = np.array([[[0, 1]]], dtype=np.int32)
    n_collisions = np.array([0], dtype=np.int32)

    masses_wp = wp.array(masses, dtype=wp.float64, device=device)
    concentration_wp = wp.array(concentration, dtype=wp.float64, device=device)
    charge_wp = wp.array(charge, dtype=wp.float64, device=device)
    collision_pairs_wp = wp.array(
        collision_pairs, dtype=wp.int32, device=device
    )
    n_collisions_wp = wp.array(n_collisions, dtype=wp.int32, device=device)

    wp.launch(
        apply_coagulation_kernel,
        dim=(1, 1),
        inputs=[
            masses_wp,
            concentration_wp,
            charge_wp,
            collision_pairs_wp,
            n_collisions_wp,
        ],
        device=device,
    )
    wp.synchronize()

    result_masses = np.asarray(masses_wp.numpy())
    result_concentration = np.asarray(concentration_wp.numpy())
    result_charge = np.asarray(charge_wp.numpy())

    npt.assert_array_equal(result_masses, masses)
    npt.assert_array_equal(result_concentration, concentration)
    npt.assert_array_equal(result_charge, charge)


@pytest.mark.gpu_parity
def test_apply_coagulation_kernel_transfers_signed_charge_and_conserves_inventories(
    device: str,
) -> None:
    """Accepted pairs transfer signed charge while conserving inventories."""
    masses = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0], [13.0, 14.0], [15.0, 16.0]],
        ],
        dtype=np.float64,
    )
    concentration = np.array(
        [[2.0, 3.0, 0.0, 5.0], [7.0, 11.0, 0.0, 13.0]],
        dtype=np.float64,
    )
    charge = np.array(
        [[-1.5, 2.5, -3.5, 4.5], [5.5, -6.5, 7.5, -8.5]],
        dtype=np.float64,
    )
    collision_pairs = np.array([[[0, 1]], [[3, 0]]], dtype=np.int32)
    n_collisions = np.ones(2, dtype=np.int32)

    expected_masses = masses.copy()
    expected_concentration = concentration.copy()
    expected_charge = charge.copy()
    for box_idx, (recipient, donor) in enumerate(collision_pairs[:, 0]):
        expected_masses[box_idx, recipient] += expected_masses[box_idx, donor]
        expected_masses[box_idx, donor] = 0.0
        expected_concentration[box_idx, donor] = 0.0
        expected_charge[box_idx, recipient] += expected_charge[box_idx, donor]
        expected_charge[box_idx, donor] = 0.0

    initial_mass_inventory = np.sum(masses, axis=1)
    initial_charge_inventory = np.sum(charge, axis=1)
    masses_wp = wp.array(masses, dtype=wp.float64, device=device)
    concentration_wp = wp.array(concentration, dtype=wp.float64, device=device)
    charge_wp = wp.array(charge, dtype=wp.float64, device=device)
    collision_pairs_wp = wp.array(
        collision_pairs, dtype=wp.int32, device=device
    )
    n_collisions_wp = wp.array(n_collisions, dtype=wp.int32, device=device)

    wp.launch(
        apply_coagulation_kernel,
        dim=(2, 1),
        inputs=[
            masses_wp,
            concentration_wp,
            charge_wp,
            collision_pairs_wp,
            n_collisions_wp,
        ],
        device=device,
    )
    wp.synchronize()

    final_masses = np.asarray(masses_wp.numpy())
    final_concentration = np.asarray(concentration_wp.numpy())
    final_charge = np.asarray(charge_wp.numpy())
    npt.assert_allclose(final_masses, expected_masses, rtol=1e-12, atol=1e-30)
    npt.assert_allclose(
        final_concentration, expected_concentration, rtol=1e-12, atol=1e-30
    )
    npt.assert_allclose(final_charge, expected_charge, rtol=1e-12, atol=1e-30)
    npt.assert_allclose(
        np.sum(final_masses, axis=1),
        initial_mass_inventory,
        rtol=1e-12,
        atol=1e-30,
    )
    npt.assert_allclose(
        np.sum(final_charge, axis=1),
        initial_charge_inventory,
        rtol=1e-12,
        atol=1e-30,
    )


def test_coagulation_validation_rejects_bad_shapes(device: str) -> None:
    """Validation helpers reject mismatched shapes."""
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=2)
    gpu_particles = to_warp_particle_data(particles, device=device)

    gpu_particles.masses = wp.zeros(
        (1, 2, 3),
        dtype=wp.float64,
        device=device,
    )
    with pytest.raises(ValueError, match="particle masses shape"):
        _validate_particle_arrays(gpu_particles, 1, 2, 2)

    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_particles.concentration = wp.zeros(
        (1, 3),
        dtype=wp.float64,
        device=device,
    )
    with pytest.raises(ValueError, match="particle concentration shape"):
        _validate_particle_arrays(gpu_particles, 1, 2, 2)

    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_particles.charge = wp.zeros((1, 3), dtype=wp.float64, device=device)
    with pytest.raises(
        ValueError,
        match="particle charge shape does not match \\(n_boxes, n_particles\\)",
    ):
        _validate_particle_arrays(gpu_particles, 1, 2, 2)

    gpu_particles = to_warp_particle_data(particles, device=device)
    invalid_charge_particles = SimpleNamespace(
        masses=gpu_particles.masses,
        concentration=gpu_particles.concentration,
        charge=wp.zeros((1, 2), dtype=wp.float32, device=device),
        density=gpu_particles.density,
        volume=gpu_particles.volume,
    )
    with pytest.raises(
        ValueError, match="particle charge must use dtype float64"
    ):
        _validate_particle_arrays(invalid_charge_particles, 1, 2, 2)

    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_particles.volume = wp.zeros(
        (2,),
        dtype=wp.float64,
        device=device,
    )
    with pytest.raises(ValueError, match="particle volume shape"):
        _validate_particle_arrays(gpu_particles, 1, 2, 2)

    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_particles.density = wp.zeros(
        (3,),
        dtype=wp.float64,
        device=device,
    )
    with pytest.raises(ValueError, match="particle density length"):
        _validate_particle_arrays(gpu_particles, 1, 2, 2)

    collision_pairs = wp.zeros(
        (1, 3, 2),
        dtype=wp.int32,
        device=device,
    )
    with pytest.raises(ValueError, match="collision_pairs shape"):
        _validate_collision_pairs(
            collision_pairs, (1, 2, 2), gpu_particles.masses.device
        )

    n_collisions = wp.zeros(
        (2,),
        dtype=wp.int32,
        device=device,
    )
    with pytest.raises(ValueError, match="n_collisions shape"):
        _validate_collision_counts(
            n_collisions, (1,), gpu_particles.masses.device
        )

    rng_states = wp.zeros(
        (2,),
        dtype=wp.uint32,
        device=device,
    )
    with pytest.raises(ValueError, match="rng_states shape"):
        _validate_rng_states(rng_states, (1,), gpu_particles.masses.device)


def test_coagulation_step_gpu_rejects_rng_state_shape_before_mutation(
    device: str,
) -> None:
    """Wrong-shape caller RNG state fails before any mutation."""
    particles = _make_particle_data(n_boxes=1, n_particles=3, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    rng_states = wp.array([7, 11], dtype=wp.uint32, device=device)
    initial_rng_states = np.asarray(rng_states.numpy()).copy()

    with pytest.raises(ValueError, match="rng_states shape"):
        coagulation_step_gpu(
            gpu_particles,
            temperature=298.15,
            pressure=101325.0,
            time_step=0.1,
            rng_seed=23,
            rng_states=rng_states,
        )

    npt.assert_array_equal(np.asarray(rng_states.numpy()), initial_rng_states)


@pytest.mark.parametrize(
    ("charge", "message"),
    [
        (
            np.zeros((1, 4), dtype=np.float64),
            "particle charge shape does not match",
        ),
        (
            np.zeros((1, 3), dtype=np.float32),
            "particle charge must use dtype float64",
        ),
    ],
)
def test_coagulation_step_gpu_rejects_charge_metadata_before_downstream_work(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    charge: np.ndarray,
    message: str,
) -> None:
    """Invalid charge metadata fails before inspection or downstream work."""
    particles = _make_particle_data(n_boxes=1, n_particles=3, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    charge_buffer = wp.array(
        charge,
        dtype=wp.float32 if charge.dtype == np.float32 else wp.float64,
        device=device,
    )
    particle_input: Any
    if charge.dtype == np.float32:
        particle_input = SimpleNamespace(
            masses=gpu_particles.masses,
            concentration=gpu_particles.concentration,
            charge=charge_buffer,
            density=gpu_particles.density,
            volume=gpu_particles.volume,
        )
    else:
        gpu_particles.charge = charge_buffer
        particle_input = gpu_particles
    masses_buffer = particle_input.masses
    concentration_buffer = particle_input.concentration
    initial_masses = np.asarray(masses_buffer.numpy()).copy()
    initial_concentration = np.asarray(concentration_buffer.numpy()).copy()
    initial_charge = np.asarray(charge_buffer.numpy()).copy()
    collision_pairs = wp.array(
        np.arange(8, dtype=np.int32).reshape(1, 4, 2),
        dtype=wp.int32,
        device=device,
    )
    n_collisions = wp.array([3], dtype=wp.int32, device=device)
    rng_states = wp.array([17], dtype=wp.uint32, device=device)
    initial_pairs = np.asarray(collision_pairs.numpy()).copy()
    initial_counts = np.asarray(n_collisions.numpy()).copy()
    initial_rng = np.asarray(rng_states.numpy()).copy()

    def _unexpected(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("downstream validation must not run")

    monkeypatch.setattr(
        coagulation_module, "_validate_charge_finite", _unexpected
    )
    monkeypatch.setattr(coagulation_module, "_ensure_volume_array", _unexpected)
    monkeypatch.setattr(
        coagulation_module, "initialize_coagulation_rng_states", _unexpected
    )
    monkeypatch.setattr(coagulation_module.wp, "launch", _unexpected)

    with pytest.raises(ValueError, match=message):
        coagulation_step_gpu(
            particle_input,
            298.15,
            101325.0,
            0.1,
            max_collisions=4,
            collision_pairs=collision_pairs,
            n_collisions=n_collisions,
            rng_states=rng_states,
            initialize_rng=True,
        )

    assert particle_input.masses is masses_buffer
    assert particle_input.concentration is concentration_buffer
    assert particle_input.charge is charge_buffer
    npt.assert_array_equal(np.asarray(collision_pairs.numpy()), initial_pairs)
    npt.assert_array_equal(np.asarray(n_collisions.numpy()), initial_counts)
    npt.assert_array_equal(np.asarray(rng_states.numpy()), initial_rng)
    npt.assert_array_equal(np.asarray(masses_buffer.numpy()), initial_masses)
    npt.assert_array_equal(
        np.asarray(concentration_buffer.numpy()), initial_concentration
    )
    npt.assert_array_equal(np.asarray(charge_buffer.numpy()), initial_charge)


def test_coagulation_step_gpu_rejects_charge_device_before_downstream_work(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    """Wrong-device charge fails before finite inspection or downstream work."""
    wrong_device = "cuda" if device == "cpu" else "cpu"
    if wrong_device == "cuda" and not cuda_available(wp):
        pytest.skip("CUDA not available for mismatch test")
    particles = _make_particle_data(n_boxes=1, n_particles=3, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_particles.charge = wp.zeros(
        (1, 3), dtype=wp.float64, device=wrong_device
    )
    masses_buffer = gpu_particles.masses
    concentration_buffer = gpu_particles.concentration
    charge_buffer = gpu_particles.charge
    initial_masses = np.asarray(masses_buffer.numpy()).copy()
    initial_concentration = np.asarray(concentration_buffer.numpy()).copy()
    initial_charge = np.asarray(charge_buffer.numpy()).copy()
    collision_pairs = wp.array(
        np.arange(8, dtype=np.int32).reshape(1, 4, 2),
        dtype=wp.int32,
        device=device,
    )
    n_collisions = wp.array([3], dtype=wp.int32, device=device)
    rng_states = wp.array([17], dtype=wp.uint32, device=device)
    initial_pairs = np.asarray(collision_pairs.numpy()).copy()
    initial_counts = np.asarray(n_collisions.numpy()).copy()
    initial_rng = np.asarray(rng_states.numpy()).copy()

    def _unexpected(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("downstream validation must not run")

    monkeypatch.setattr(
        coagulation_module, "_validate_charge_finite", _unexpected
    )
    monkeypatch.setattr(coagulation_module, "_ensure_volume_array", _unexpected)
    monkeypatch.setattr(
        coagulation_module, "initialize_coagulation_rng_states", _unexpected
    )
    monkeypatch.setattr(coagulation_module.wp, "launch", _unexpected)

    with pytest.raises(ValueError, match="particle charge device mismatch"):
        coagulation_step_gpu(
            gpu_particles,
            298.15,
            101325.0,
            0.1,
            max_collisions=4,
            collision_pairs=collision_pairs,
            n_collisions=n_collisions,
            rng_states=rng_states,
            initialize_rng=True,
        )

    assert gpu_particles.masses is masses_buffer
    assert gpu_particles.concentration is concentration_buffer
    assert gpu_particles.charge is charge_buffer
    npt.assert_array_equal(np.asarray(collision_pairs.numpy()), initial_pairs)
    npt.assert_array_equal(np.asarray(n_collisions.numpy()), initial_counts)
    npt.assert_array_equal(np.asarray(rng_states.numpy()), initial_rng)
    npt.assert_array_equal(np.asarray(masses_buffer.numpy()), initial_masses)
    npt.assert_array_equal(
        np.asarray(concentration_buffer.numpy()), initial_concentration
    )
    npt.assert_array_equal(np.asarray(charge_buffer.numpy()), initial_charge)


@pytest.mark.parametrize("nonfinite_charge", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize(
    "mechanisms",
    [
        (CHARGED_HARD_SPHERE_MECHANISM,),
        (BROWNIAN_MECHANISM, CHARGED_HARD_SPHERE_MECHANISM),
    ],
)
def test_charged_modes_reject_nonfinite_charge_before_downstream_work(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    nonfinite_charge: float,
    mechanisms: tuple[str, ...],
) -> None:
    """Charged-containing modes reject non-finite charge before downstream work."""
    particles = _make_particle_data(n_boxes=1, n_particles=3, n_species=1)
    particles.charge[0, 1] = nonfinite_charge
    gpu_particles = to_warp_particle_data(particles, device=device)
    masses_buffer = gpu_particles.masses
    concentration_buffer = gpu_particles.concentration
    charge_buffer = gpu_particles.charge
    initial_particles = from_warp_particle_data(gpu_particles, sync=True)
    collision_pairs = wp.array(
        np.arange(8, dtype=np.int32).reshape(1, 4, 2),
        dtype=wp.int32,
        device=device,
    )
    n_collisions = wp.array([3], dtype=wp.int32, device=device)
    rng_states = wp.array([17], dtype=wp.uint32, device=device)
    initial_pairs = np.asarray(collision_pairs.numpy()).copy()
    initial_counts = np.asarray(n_collisions.numpy()).copy()
    initial_rng = np.asarray(rng_states.numpy()).copy()
    launch = coagulation_module.wp.launch
    kernels: list[Any] = []

    def _only_charge_validation(kernel: Any, *args: Any, **kwargs: Any) -> Any:
        kernels.append(kernel)
        if kernel is not _validate_charge_finite_kernel:
            raise AssertionError("only finite-charge validation may launch")
        return launch(kernel, *args, **kwargs)

    def _unexpected(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("downstream work must not run")

    monkeypatch.setattr(
        coagulation_module, "_ensure_environment_arrays", _unexpected
    )
    monkeypatch.setattr(coagulation_module, "_ensure_volume_array", _unexpected)
    monkeypatch.setattr(
        coagulation_module, "initialize_coagulation_rng_states", _unexpected
    )
    monkeypatch.setattr(
        coagulation_module.wp, "launch", _only_charge_validation
    )

    with pytest.raises(
        ValueError, match="particle charge must contain only finite values"
    ):
        coagulation_step_gpu(
            gpu_particles,
            298.15,
            101325.0,
            0.1,
            max_collisions=4,
            collision_pairs=collision_pairs,
            n_collisions=n_collisions,
            rng_states=rng_states,
            initialize_rng=True,
            mechanism_config=CoagulationMechanismConfig(mechanisms=mechanisms),
        )

    assert kernels == [_validate_charge_finite_kernel]
    assert gpu_particles.masses is masses_buffer
    assert gpu_particles.concentration is concentration_buffer
    assert gpu_particles.charge is charge_buffer
    npt.assert_array_equal(np.asarray(collision_pairs.numpy()), initial_pairs)
    npt.assert_array_equal(np.asarray(n_collisions.numpy()), initial_counts)
    npt.assert_array_equal(np.asarray(rng_states.numpy()), initial_rng)
    _assert_particles_unchanged(gpu_particles, initial_particles)


def test_coagulation_step_gpu_zero_charge_preserves_charge_and_sidecars(
    device: str,
) -> None:
    """Zero charge preserves charge and caller-owned result sidecars."""
    charge = np.zeros((1, 3), dtype=np.float64)
    particles = _make_particle_data(n_boxes=1, n_particles=3, n_species=1)
    particles.charge = charge
    gpu_particles = to_warp_particle_data(particles, device=device)
    charge_buffer = gpu_particles.charge
    collision_pairs = wp.zeros((1, 4, 2), dtype=wp.int32, device=device)
    n_collisions = wp.zeros((1,), dtype=wp.int32, device=device)
    rng_states = wp.zeros((1,), dtype=wp.uint32, device=device)

    result = coagulation_step_gpu(
        gpu_particles,
        298.15,
        101325.0,
        0.1,
        max_collisions=4,
        collision_pairs=collision_pairs,
        n_collisions=n_collisions,
        rng_states=rng_states,
        initialize_rng=True,
    )

    assert len(result) == 3
    assert result[0] is gpu_particles
    assert result[1] is collision_pairs
    assert result[2] is n_collisions
    assert gpu_particles.charge is charge_buffer
    npt.assert_array_equal(np.asarray(gpu_particles.charge.numpy()), charge)


def test_coagulation_step_gpu_signed_charge_conserves_charge_and_sidecars(
    device: str,
) -> None:
    """Signed charge conserves per-box inventory and supplied sidecars."""
    charge = np.array([[-1.0, 0.0, 2.0]], dtype=np.float64)
    particles = _make_particle_data(n_boxes=1, n_particles=3, n_species=1)
    particles.charge = charge
    gpu_particles = to_warp_particle_data(particles, device=device)
    charge_buffer = gpu_particles.charge
    collision_pairs = wp.zeros((1, 4, 2), dtype=wp.int32, device=device)
    n_collisions = wp.zeros((1,), dtype=wp.int32, device=device)
    rng_states = wp.zeros((1,), dtype=wp.uint32, device=device)

    result = coagulation_step_gpu(
        gpu_particles,
        298.15,
        101325.0,
        0.1,
        max_collisions=4,
        collision_pairs=collision_pairs,
        n_collisions=n_collisions,
        rng_states=rng_states,
        initialize_rng=True,
    )

    assert len(result) == 3
    assert result[0] is gpu_particles
    assert result[1] is collision_pairs
    assert result[2] is n_collisions
    assert gpu_particles.charge is charge_buffer
    npt.assert_allclose(
        np.sum(np.asarray(gpu_particles.charge.numpy()), axis=1),
        np.sum(charge, axis=1),
        rtol=1e-12,
        atol=1e-30,
    )


def test_coagulation_step_gpu_forced_collision_transfers_charge_exactly(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    """A forced accepted pair clears its donor and transfers its charge."""

    @_warp_kernel
    def _force_pair(pairs: Any, counts: Any) -> None:
        box_idx = wp.tid()
        pairs[box_idx, 0, 0] = wp.int32(0)
        pairs[box_idx, 0, 1] = wp.int32(2)
        counts[box_idx] = wp.int32(1)

    particles = _make_particle_data(n_boxes=1, n_particles=3, n_species=1)
    particles.charge = np.array([[-1.0, 0.0, 2.0]], dtype=np.float64)
    gpu_particles = to_warp_particle_data(particles, device=device)
    pairs = wp.zeros((1, 1, 2), dtype=wp.int32, device=device)
    counts = wp.zeros((1,), dtype=wp.int32, device=device)
    rng_states = wp.zeros((1,), dtype=wp.uint32, device=device)
    launch = coagulation_module.wp.launch

    def _force_brownian(kernel: Any, *args: Any, **kwargs: Any) -> Any:
        if kernel is coagulation_module.brownian_coagulation_kernel:
            inputs = kwargs["inputs"]
            return launch(
                _force_pair,
                dim=(1,),
                inputs=[inputs[25], inputs[26]],
                device=kwargs["device"],
            )
        return launch(kernel, *args, **kwargs)

    monkeypatch.setattr(coagulation_module.wp, "launch", _force_brownian)
    coagulation_step_gpu(
        gpu_particles,
        298.15,
        101325.0,
        0.1,
        max_collisions=1,
        collision_pairs=pairs,
        n_collisions=counts,
        rng_states=rng_states,
        initialize_rng=True,
    )
    wp.synchronize()

    final_charge = np.asarray(gpu_particles.charge.numpy())
    assert counts.numpy()[0] == 1
    assert final_charge[0, 0] == 1.0
    assert final_charge[0, 2] == 0.0
    npt.assert_allclose(final_charge.sum(), particles.charge.sum(), atol=0.0)


def test_coagulation_step_gpu_validates_caller_rng_state_before_allocation(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    """Invalid caller RNG state fails before any internal buffer allocation."""
    particles = _make_particle_data(n_boxes=1, n_particles=6, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    rng_states = wp.array([7, 11], dtype=wp.uint32, device=device)

    def _unexpected_zeros(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("wp.zeros should not run before rng validation")

    monkeypatch.setattr(
        coagulation_module, "_validate_charge_finite", lambda *_: None
    )
    monkeypatch.setattr(coagulation_module.wp, "zeros", _unexpected_zeros)

    with pytest.raises(ValueError, match="rng_states shape"):
        coagulation_step_gpu(
            gpu_particles,
            temperature=298.15,
            pressure=101325.0,
            time_step=0.1,
            max_collisions=np.iinfo(np.int32).max,
            rng_states=rng_states,
        )


@pytest.mark.parametrize(
    ("buffer_name", "message"),
    [
        ("collision_pairs", "collision_pairs buffer must use dtype int32"),
        ("n_collisions", "n_collisions buffer must use dtype int32"),
        ("rng_states", "rng_states buffer must use dtype uint32"),
    ],
)
def test_coagulation_step_gpu_rejects_wrong_buffer_dtypes_before_mutation(
    device: str,
    buffer_name: str,
    message: str,
) -> None:
    """Wrong preallocated buffer dtypes fail before any mutation."""
    particles = _make_particle_data(n_boxes=1, n_particles=3, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    initial_particles = from_warp_particle_data(gpu_particles, sync=True)

    collision_pairs = wp.array(
        np.arange(8, dtype=np.int32).reshape(1, 4, 2),
        dtype=wp.int32,
        device=device,
    )
    n_collisions = wp.array(
        np.array([3], dtype=np.int32),
        dtype=wp.int32,
        device=device,
    )
    rng_states = wp.array(
        np.array([17], dtype=np.uint32),
        dtype=wp.uint32,
        device=device,
    )

    if buffer_name == "collision_pairs":
        collision_pairs = wp.array(
            np.arange(8, dtype=np.float64).reshape(1, 4, 2),
            dtype=wp.float64,
            device=device,
        )
    elif buffer_name == "n_collisions":
        n_collisions = wp.array(
            np.array([3.0], dtype=np.float64),
            dtype=wp.float64,
            device=device,
        )
    else:
        rng_states = wp.array(
            np.array([17], dtype=np.int32),
            dtype=wp.int32,
            device=device,
        )

    initial_collision_pairs = np.asarray(collision_pairs.numpy()).copy()
    initial_n_collisions = np.asarray(n_collisions.numpy()).copy()
    initial_rng_states = np.asarray(rng_states.numpy()).copy()

    with pytest.raises(ValueError, match=message):
        coagulation_step_gpu(
            gpu_particles,
            temperature=298.15,
            pressure=101325.0,
            time_step=0.1,
            max_collisions=4,
            collision_pairs=collision_pairs,
            n_collisions=n_collisions,
            rng_states=rng_states,
        )

    npt.assert_array_equal(
        np.asarray(collision_pairs.numpy()),
        initial_collision_pairs,
    )
    npt.assert_array_equal(
        np.asarray(n_collisions.numpy()),
        initial_n_collisions,
    )
    npt.assert_array_equal(np.asarray(rng_states.numpy()), initial_rng_states)
    _assert_particles_unchanged(gpu_particles, initial_particles)


@pytest.mark.parametrize(
    "time_step",
    [-0.1, float("nan"), float("inf")],
)
def test_coagulation_step_gpu_invalid_time_step_fails_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    time_step: float,
) -> None:
    """Invalid time steps fail before volume setup, RNG init, or mutation."""
    particles = _make_particle_data(n_boxes=1, n_particles=3, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    initial_particles = from_warp_particle_data(gpu_particles, sync=True)
    collision_pairs = wp.array(
        np.arange(8, dtype=np.int32).reshape(1, 4, 2),
        dtype=wp.int32,
        device=device,
    )
    n_collisions = wp.array(
        np.array([2], dtype=np.int32),
        dtype=wp.int32,
        device=device,
    )
    rng_states = wp.array(
        np.array([19], dtype=np.uint32),
        dtype=wp.uint32,
        device=device,
    )
    initial_collision_pairs = np.asarray(collision_pairs.numpy()).copy()
    initial_n_collisions = np.asarray(n_collisions.numpy()).copy()
    initial_rng_states = np.asarray(rng_states.numpy()).copy()
    calls: list[str] = []

    def _unexpected_ensure_volume_array(*args: Any, **kwargs: Any) -> Any:
        calls.append("ensure_volume_array")
        raise AssertionError("_ensure_volume_array should not be called")

    def _unexpected_launch(*args: Any, **kwargs: Any) -> None:
        calls.append("launch")
        raise AssertionError("wp.launch should not be called")

    monkeypatch.setattr(
        coagulation_module,
        "_ensure_volume_array",
        _unexpected_ensure_volume_array,
    )
    monkeypatch.setattr(
        coagulation_module, "_validate_charge_finite", lambda *_: None
    )
    monkeypatch.setattr(coagulation_module.wp, "launch", _unexpected_launch)

    with pytest.raises(
        ValueError,
        match="time_step must be finite and nonnegative",
    ):
        coagulation_step_gpu(
            gpu_particles,
            temperature=298.15,
            pressure=101325.0,
            time_step=time_step,
            max_collisions=4,
            collision_pairs=collision_pairs,
            n_collisions=n_collisions,
            rng_states=rng_states,
        )

    assert calls == []
    npt.assert_array_equal(
        np.asarray(collision_pairs.numpy()),
        initial_collision_pairs,
    )
    npt.assert_array_equal(
        np.asarray(n_collisions.numpy()),
        initial_n_collisions,
    )
    npt.assert_array_equal(np.asarray(rng_states.numpy()), initial_rng_states)
    _assert_particles_unchanged(gpu_particles, initial_particles)


def test_coagulation_step_gpu_invalid_volume_fails_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    """Invalid volumes fail before RNG init, launch, or particle mutation."""
    particles = _make_particle_data(n_boxes=1, n_particles=3, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    initial_particles = from_warp_particle_data(gpu_particles, sync=True)
    collision_pairs = wp.array(
        np.arange(8, dtype=np.int32).reshape(1, 4, 2),
        dtype=wp.int32,
        device=device,
    )
    n_collisions = wp.array(
        np.array([2], dtype=np.int32),
        dtype=wp.int32,
        device=device,
    )
    rng_states = wp.array(
        np.array([19], dtype=np.uint32),
        dtype=wp.uint32,
        device=device,
    )
    initial_collision_pairs = np.asarray(collision_pairs.numpy()).copy()
    initial_n_collisions = np.asarray(n_collisions.numpy()).copy()
    initial_rng_states = np.asarray(rng_states.numpy()).copy()
    calls: list[str] = []

    def _unexpected_launch(*args: Any, **kwargs: Any) -> None:
        calls.append("launch")
        raise AssertionError("wp.launch should not be called")

    monkeypatch.setattr(
        coagulation_module, "_validate_charge_finite", lambda *_: None
    )
    monkeypatch.setattr(coagulation_module.wp, "launch", _unexpected_launch)

    with pytest.raises(ValueError, match="volume must be finite and > 0"):
        coagulation_step_gpu(
            gpu_particles,
            temperature=298.15,
            pressure=101325.0,
            time_step=0.1,
            volume=-1.0,
            max_collisions=4,
            collision_pairs=collision_pairs,
            n_collisions=n_collisions,
            rng_states=rng_states,
        )

    assert calls == []
    npt.assert_array_equal(
        np.asarray(collision_pairs.numpy()),
        initial_collision_pairs,
    )
    npt.assert_array_equal(
        np.asarray(n_collisions.numpy()),
        initial_n_collisions,
    )
    npt.assert_array_equal(np.asarray(rng_states.numpy()), initial_rng_states)
    _assert_particles_unchanged(gpu_particles, initial_particles)


@pytest.mark.parametrize(
    "max_collisions",
    [0, -1, 1.5, True, np.iinfo(np.int32).max + 1],
)
def test_coagulation_step_gpu_invalid_max_collisions_fails_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    max_collisions: object,
) -> None:
    """Invalid collision limits fail before launch or caller-buffer mutation."""
    particles = _make_particle_data(n_boxes=1, n_particles=3, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    initial_particles = from_warp_particle_data(gpu_particles, sync=True)
    collision_pairs = wp.array(
        np.arange(8, dtype=np.int32).reshape(1, 4, 2),
        dtype=wp.int32,
        device=device,
    )
    n_collisions = wp.array(
        np.array([2], dtype=np.int32),
        dtype=wp.int32,
        device=device,
    )
    rng_states = wp.array(
        np.array([19], dtype=np.uint32),
        dtype=wp.uint32,
        device=device,
    )
    initial_collision_pairs = np.asarray(collision_pairs.numpy()).copy()
    initial_n_collisions = np.asarray(n_collisions.numpy()).copy()
    initial_rng_states = np.asarray(rng_states.numpy()).copy()
    calls: list[str] = []

    def _unexpected_launch(*args: Any, **kwargs: Any) -> None:
        calls.append("launch")
        raise AssertionError("wp.launch should not be called")

    monkeypatch.setattr(
        coagulation_module, "_validate_charge_finite", lambda *_: None
    )
    monkeypatch.setattr(coagulation_module.wp, "launch", _unexpected_launch)

    with pytest.raises(
        ValueError,
        match="max_collisions must be a positive integer",
    ):
        coagulation_step_gpu(
            gpu_particles,
            temperature=298.15,
            pressure=101325.0,
            time_step=0.1,
            max_collisions=max_collisions,
            collision_pairs=collision_pairs,
            n_collisions=n_collisions,
            rng_states=rng_states,
        )

    assert calls == []
    npt.assert_array_equal(
        np.asarray(collision_pairs.numpy()),
        initial_collision_pairs,
    )
    npt.assert_array_equal(
        np.asarray(n_collisions.numpy()),
        initial_n_collisions,
    )
    npt.assert_array_equal(np.asarray(rng_states.numpy()), initial_rng_states)
    _assert_particles_unchanged(gpu_particles, initial_particles)


def test_coagulation_step_gpu_invalid_followup_preserves_advanced_rng_states(
    device: str,
) -> None:
    """Invalid follow-up calls preserve already-advanced caller RNG state.

    After one valid call advances a caller-owned ``rng_states`` buffer, a
    later invalid call must fail before mutating that persisted buffer.
    """
    particles = _make_particle_data(n_boxes=1, n_particles=3, n_species=1)
    first_gpu_particles = to_warp_particle_data(particles, device=device)
    rng_states = wp.zeros((1,), dtype=wp.uint32, device=device)

    wp.launch(
        _initialize_rng_states,
        dim=1,
        inputs=[wp.uint32(41), rng_states],
        device=device,
    )
    wp.synchronize()
    initialized_state = np.asarray(rng_states.numpy()).copy()

    coagulation_step_gpu(
        first_gpu_particles,
        temperature=298.15,
        pressure=101325.0,
        time_step=1.0,
        volume=1.0e-18,
        rng_seed=41,
        max_collisions=8,
        rng_states=rng_states,
    )
    wp.synchronize()
    advanced_state = np.asarray(rng_states.numpy()).copy()

    assert not np.array_equal(advanced_state, initialized_state)

    second_gpu_particles = to_warp_particle_data(particles, device=device)
    with pytest.raises(ValueError, match="volume must be finite and > 0"):
        coagulation_step_gpu(
            second_gpu_particles,
            temperature=298.15,
            pressure=101325.0,
            time_step=1.0,
            volume=-1.0,
            rng_seed=41,
            max_collisions=8,
            rng_states=rng_states,
        )

    npt.assert_array_equal(np.asarray(rng_states.numpy()), advanced_state)


def test_coagulation_step_gpu_fractional_skip_advances_rng_state(
    device: str,
) -> None:
    """Fractional-trial early returns still persist the advanced RNG state."""
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=1)
    temperature = 298.15
    pressure = 101325.0
    rng_seed = 43

    probe_states = wp.zeros((1,), dtype=wp.uint32, device=device)
    initialize_coagulation_rng_states(rng_seed, probe_states, device=device)
    probe_draws = wp.zeros((1,), dtype=wp.float64, device=device)
    wp.launch(
        _draw_single_random_kernel,
        dim=1,
        inputs=[probe_states, probe_draws],
        device=device,
    )
    wp.synchronize()
    first_draw = float(np.asarray(probe_draws.numpy())[0])
    assert 0.0 < first_draw < 1.0

    particle_masses = particles.masses[0, :, 0]
    density = particles.density[0]
    particle_radii = ((3.0 * (particle_masses / density)) / (4.0 * np.pi)) ** (
        1.0 / 3.0
    )
    kernel_matrix = get_brownian_kernel_via_system_state(
        particle_radius=particle_radii,
        particle_mass=particle_masses,
        temperature=temperature,
        pressure=pressure,
    )
    kernel_value = float(np.asarray(kernel_matrix)[0, 1])
    volume = kernel_value / (first_draw * 0.5)

    rng_states = wp.zeros((1,), dtype=wp.uint32, device=device)
    initialize_coagulation_rng_states(rng_seed, rng_states, device=device)
    initial_state = np.asarray(rng_states.numpy()).copy()

    first_particles = to_warp_particle_data(particles, device=device)
    _, _, first_counts = coagulation_step_gpu(
        first_particles,
        temperature=temperature,
        pressure=pressure,
        time_step=1.0,
        volume=volume,
        max_collisions=4,
        rng_seed=rng_seed,
        rng_states=rng_states,
    )
    wp.synchronize()
    first_state = np.asarray(rng_states.numpy()).copy()
    first_collision_counts = np.asarray(first_counts.numpy())

    second_particles = to_warp_particle_data(particles, device=device)
    _, _, second_counts = coagulation_step_gpu(
        second_particles,
        temperature=temperature,
        pressure=pressure,
        time_step=1.0,
        volume=volume,
        max_collisions=4,
        rng_seed=rng_seed,
        rng_states=rng_states,
    )
    wp.synchronize()
    second_state = np.asarray(rng_states.numpy()).copy()
    second_collision_counts = np.asarray(second_counts.numpy())

    npt.assert_array_equal(
        first_collision_counts, np.array([0], dtype=np.int32)
    )
    npt.assert_array_equal(
        second_collision_counts, np.array([0], dtype=np.int32)
    )
    assert not np.array_equal(first_state, initial_state)
    assert not np.array_equal(second_state, first_state)


def test_coagulation_validation_rejects_device_mismatch(device: str) -> None:
    """Validation helpers reject device mismatches."""
    wrong_device = "cpu" if device == "cuda" else "cuda"
    if wrong_device == "cuda" and not cuda_available(wp):
        pytest.skip("CUDA not available for mismatch test")

    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)

    collision_pairs = wp.zeros(
        (1, 2, 2),
        dtype=wp.int32,
        device=wrong_device,
    )
    with pytest.raises(ValueError, match="collision_pairs buffer"):
        _validate_collision_pairs(
            collision_pairs, (1, 2, 2), gpu_particles.masses.device
        )

    n_collisions = wp.zeros(
        (1,),
        dtype=wp.int32,
        device=wrong_device,
    )
    with pytest.raises(ValueError, match="n_collisions"):
        _validate_collision_counts(
            n_collisions, (1,), gpu_particles.masses.device
        )

    rng_states = wp.zeros(
        (1,),
        dtype=wp.uint32,
        device=wrong_device,
    )
    with pytest.raises(ValueError, match="rng_states"):
        _validate_rng_states(rng_states, (1,), gpu_particles.masses.device)


def test_coagulation_step_gpu_rejects_rng_state_device_mismatch(
    device: str,
) -> None:
    """Wrong-device caller RNG state fails at the public entrypoint."""
    wrong_device = "cpu" if device == "cuda" else "cuda"
    if wrong_device == "cuda" and not cuda_available(wp):
        pytest.skip("CUDA not available for mismatch test")

    particles = _make_particle_data(n_boxes=1, n_particles=3, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    rng_states = wp.zeros((1,), dtype=wp.uint32, device=wrong_device)

    with pytest.raises(ValueError, match="rng_states"):
        coagulation_step_gpu(
            gpu_particles,
            temperature=298.15,
            pressure=101325.0,
            time_step=0.1,
            rng_seed=23,
            rng_states=rng_states,
        )


def test_coagulation_validate_device_arrays(device: str) -> None:
    """Device validation passes when devices match and fails otherwise."""
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)

    _validate_device_arrays(gpu_particles, gpu_particles.masses.device)

    wrong_device = "cpu" if device == "cuda" else "cuda"
    if wrong_device == "cuda" and not cuda_available(wp):
        pytest.skip("CUDA not available for mismatch test")

    gpu_particles.volume = wp.zeros((1,), dtype=wp.float64, device=wrong_device)
    with pytest.raises(ValueError, match="particle volume device mismatch"):
        _validate_device_arrays(gpu_particles, gpu_particles.masses.device)

    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_particles.charge = wp.zeros(
        (1, 2), dtype=wp.float64, device=wrong_device
    )
    with pytest.raises(ValueError, match="particle charge device mismatch"):
        _validate_device_arrays(gpu_particles, gpu_particles.masses.device)


def test_coagulation_ensure_volume_array(device: str) -> None:
    """Volume helper returns a device array and validates shapes."""
    volume_array = _ensure_volume_array(1.0e-6, n_boxes=2, device=device)
    assert volume_array.shape == (2,)
    npt.assert_allclose(np.asarray(volume_array.numpy()), [1.0e-6, 1.0e-6])

    bad_volume = wp.zeros((3,), dtype=wp.float64, device=device)
    with pytest.raises(ValueError, match="volume shape does not match"):
        _ensure_volume_array(bad_volume, n_boxes=2, device=device)

    volume_array = _ensure_volume_array(1.5e-6, n_boxes=2, device=device)
    _validate_device_match("volume", volume_array, volume_array.device)


@pytest.mark.parametrize(
    "volume",
    [0.0, -1.0, float("nan"), float("inf")],
)
def test_coagulation_ensure_volume_array_rejects_invalid_scalars(
    device: str,
    volume: float,
) -> None:
    """Scalar volume inputs must be positive finite values."""
    with pytest.raises(ValueError, match="volume must be finite and > 0"):
        _ensure_volume_array(volume, n_boxes=1, device=device)


@pytest.mark.parametrize(
    "values",
    [
        np.array([0.0], dtype=np.float64),
        np.array([-1.0], dtype=np.float64),
        np.array([np.nan], dtype=np.float64),
        np.array([np.inf], dtype=np.float64),
    ],
)
def test_coagulation_ensure_volume_array_rejects_invalid_arrays(
    device: str,
    values: np.ndarray,
) -> None:
    """Warp-array volume inputs must be positive finite values."""
    volume = wp.array(values, dtype=wp.float64, device=device)
    with pytest.raises(ValueError, match="volume must be finite and > 0"):
        _ensure_volume_array(volume, n_boxes=1, device=volume.device)


def test_coagulation_ensure_volume_array_rejects_non_warp_tensor_like() -> None:
    """Tensor-like non-Warp volume inputs fail with a stable type error."""

    class _FakeTensorLike:
        def __init__(self) -> None:
            self.shape = (1,)

    with pytest.raises(
        ValueError,
        match=r"volume must be a Warp array with shape \(n_boxes,\)",
    ):
        _ensure_volume_array(_FakeTensorLike(), n_boxes=1, device="cpu")


def test_coagulation_ensure_volume_array_rejects_integer_scalar(
    device: str,
) -> None:
    """Integer scalar volumes are rejected at the GPU boundary."""
    with pytest.raises(ValueError, match="floating scalar"):
        _ensure_volume_array(1, n_boxes=1, device=device)


def test_coagulation_ensure_volume_array_rejects_integer_dtype_array(
    device: str,
) -> None:
    """Only supported Warp float dtypes are accepted for volume arrays."""
    volume = wp.array([1], dtype=wp.int32, device=device)

    with pytest.raises(ValueError, match="supported Warp float dtype"):
        _ensure_volume_array(volume, n_boxes=1, device=volume.device)


@pytest.mark.cuda
def test_coagulation_ensure_volume_array_skips_cuda_host_readback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CUDA volume validation avoids implicit ``.numpy()`` synchronization."""
    if not cuda_available(wp):
        pytest.skip("CUDA not available for readback guard test")

    volume = wp.array([1.0e-6], dtype=wp.float64, device="cuda")

    def _forbidden_numpy(self: Any) -> np.ndarray:
        raise AssertionError("unexpected host readback")

    monkeypatch.setattr(volume, "numpy", _forbidden_numpy, raising=False)

    returned_volume = _ensure_volume_array(
        volume,
        n_boxes=1,
        device=volume.device,
    )

    assert returned_volume is volume


def test_coagulation_validation_helpers_accept_valid_inputs(
    device: str,
) -> None:
    """Validation helpers accept matching buffers and volume arrays."""
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    collision_pairs = wp.zeros((1, 2, 2), dtype=wp.int32, device=device)
    n_collisions = wp.zeros((1,), dtype=wp.int32, device=device)
    rng_states = wp.zeros((1,), dtype=wp.uint32, device=device)
    volume = wp.array(np.array([1.0e-6]), dtype=wp.float64, device=device)

    _validate_particle_arrays(gpu_particles, 1, 2, 1)
    _validate_collision_pairs(
        collision_pairs,
        (1, 2, 2),
        gpu_particles.masses.device,
    )
    _validate_collision_counts(
        n_collisions,
        (1,),
        gpu_particles.masses.device,
    )
    _validate_rng_states(rng_states, (1,), gpu_particles.masses.device)

    returned_volume = _ensure_volume_array(
        volume,
        n_boxes=1,
        device=volume.device,
    )
    assert returned_volume is volume


def test_validate_time_step_accepts_zero_and_float_like_values() -> None:
    """Time-step helper returns normalized finite nonnegative values."""
    assert _validate_time_step(0.0) == 0.0
    assert _validate_time_step(np.float64(0.5)) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "time_step",
    [True, -1.0, float("nan"), float("inf"), object()],
)
def test_validate_time_step_rejects_invalid_values(time_step: object) -> None:
    """Time-step helper rejects invalid domains and non-real inputs."""
    with pytest.raises(ValueError, match="finite and nonnegative"):
        _validate_time_step(time_step)


def test_validate_max_collisions_accepts_positive_int_like_values() -> None:
    """Collision-limit helper returns normalized positive integer values."""
    assert _validate_max_collisions(1) == 1
    assert _validate_max_collisions(np.int64(7)) == 7


@pytest.mark.parametrize(
    "max_collisions",
    [True, 0, -1, 1.5, np.iinfo(np.int32).max + 1, object()],
)
def test_validate_max_collisions_rejects_invalid_values(
    max_collisions: object,
) -> None:
    """Collision-limit helper rejects unsupported values before allocation."""
    with pytest.raises(ValueError, match="positive integer"):
        _validate_max_collisions(max_collisions)


def test_resolve_collision_capacity_clamps_to_physical_limit() -> None:
    """Collision capacity is bounded by the useful per-box particle limit."""
    assert _resolve_collision_capacity(256, n_boxes=1, n_particles=6) == 3
    assert _resolve_collision_capacity(2, n_boxes=1, n_particles=6) == 2


def test_resolve_collision_capacity_clamps_to_buffer_budget() -> None:
    """Collision capacity is bounded by the shared byte-budget ceiling."""
    assert (
        _resolve_collision_capacity(512, n_boxes=70_000_000, n_particles=1024)
        == 1
    )


def test_bound_scheduled_trials_preserves_values_below_int32_limit(
    device: str,
) -> None:
    """Bounded-trial helper leaves finite in-range schedules unchanged."""
    expected_trials = wp.array(
        [0.0, 1.25, 42.0], dtype=wp.float64, device=device
    )
    bounded_trials = wp.zeros((3,), dtype=wp.float64, device=device)

    wp.launch(
        _bound_scheduled_trials_probe_kernel,
        dim=3,
        inputs=[expected_trials, bounded_trials],
        device=device,
    )
    wp.synchronize()

    npt.assert_allclose(
        np.asarray(bounded_trials.numpy()),
        np.array([0.0, 1.25, 42.0], dtype=np.float64),
    )


def test_bound_scheduled_trials_returns_zero_for_nan_and_nonpositive_inputs(
    device: str,
) -> None:
    """Bounded-trial helper fails closed for NaN and nonpositive values."""
    expected_trials = wp.array(
        [float("nan"), -3.0, 0.0], dtype=wp.float64, device=device
    )
    bounded_trials = wp.zeros((3,), dtype=wp.float64, device=device)

    wp.launch(
        _bound_scheduled_trials_probe_kernel,
        dim=3,
        inputs=[expected_trials, bounded_trials],
        device=device,
    )
    wp.synchronize()

    npt.assert_allclose(
        np.asarray(bounded_trials.numpy()),
        np.array([0.0, 0.0, 0.0], dtype=np.float64),
    )


def test_bound_scheduled_trials_caps_values_above_operational_budget(
    device: str,
) -> None:
    """Bounded-trial helper caps large schedules before int32 conversion."""
    expected_trials = wp.array(
        [float(MAX_SCHEDULED_TRIALS_PER_BOX) * 2.0],
        dtype=wp.float64,
        device=device,
    )
    bounded_trials = wp.zeros((1,), dtype=wp.float64, device=device)

    wp.launch(
        _bound_scheduled_trials_probe_kernel,
        dim=1,
        inputs=[expected_trials, bounded_trials],
        device=device,
    )
    wp.synchronize()

    npt.assert_allclose(
        np.asarray(bounded_trials.numpy()),
        np.array([float(MAX_SCHEDULED_TRIALS_PER_BOX)], dtype=np.float64),
    )


def test_resolve_active_pair_by_rank_maps_sparse_active_ranks(
    device: str,
) -> None:
    """Active-rank resolution skips inactive slots and preserves rank order."""
    active_flags = wp.array([[0, 1, 0, 1, 1]], dtype=wp.int32, device=device)
    resolved_pairs = wp.zeros((1, 2), dtype=wp.int32, device=device)

    wp.launch(
        _resolve_active_pair_probe_kernel,
        dim=1,
        inputs=[
            active_flags,
            resolved_pairs,
            wp.int32(1),
            wp.int32(2),
        ],
        device=device,
    )
    wp.synchronize()

    npt.assert_array_equal(
        np.asarray(resolved_pairs.numpy()),
        np.array([[3, 4]], dtype=np.int32),
    )


def test_select_active_pair_by_rank_reads_compact_active_indices(
    device: str,
) -> None:
    """Direct active-pair lookup returns the compact-buffer entries by rank."""
    active_indices = wp.array(
        [[4, 1, 3, -1, -1]], dtype=wp.int32, device=device
    )
    resolved_pairs = wp.zeros((1, 2), dtype=wp.int32, device=device)

    wp.launch(
        _select_active_pair_probe_kernel,
        dim=1,
        inputs=[
            active_indices,
            resolved_pairs,
            wp.int32(1),
            wp.int32(2),
        ],
        device=device,
    )
    wp.synchronize()

    npt.assert_array_equal(
        np.asarray(resolved_pairs.numpy()),
        np.array([[1, 3]], dtype=np.int32),
    )


def test_remove_active_pair_by_rank_swap_pop_removes_both_selected_ranks(
    device: str,
) -> None:
    """Swap-pop removal deletes both selected ranks and compacts survivors."""
    active_indices = wp.array([[4, 1, 3, 2, -1]], dtype=wp.int32, device=device)
    updated_counts = wp.zeros((1,), dtype=wp.int32, device=device)

    wp.launch(
        _remove_active_pair_probe_kernel,
        dim=1,
        inputs=[
            active_indices,
            updated_counts,
            wp.int32(1),
            wp.int32(2),
        ],
        device=device,
    )
    wp.synchronize()

    npt.assert_array_equal(
        np.asarray(updated_counts.numpy()),
        np.array([2], dtype=np.int32),
    )
    npt.assert_array_equal(
        np.asarray(active_indices.numpy()),
        np.array([[4, 2, -1, -1, -1]], dtype=np.int32),
    )


def test_initialize_coagulation_rng_states_changes_output(device: str) -> None:
    """Public RNG initialization helper writes nonzero state in place."""
    rng_states = wp.zeros((4,), dtype=wp.uint32, device=device)
    returned = initialize_coagulation_rng_states(123, rng_states, device=device)

    assert returned is rng_states
    rng_values = np.asarray(rng_states.numpy())
    assert np.any(rng_values != 0)


def test_initialize_coagulation_rng_states_rejects_invalid_buffers(
    device: str,
) -> None:
    """Public RNG initialization helper rejects invalid shape and dtype."""
    wrong_shape = wp.zeros((2, 1), dtype=wp.uint32, device=device)
    wrong_dtype = wp.zeros((2,), dtype=wp.int32, device=device)

    with pytest.raises(ValueError, match=r"shape \(n_boxes,\)"):
        initialize_coagulation_rng_states(123, wrong_shape, device=device)
    with pytest.raises(ValueError, match="dtype uint32"):
        initialize_coagulation_rng_states(123, wrong_dtype, device=device)


def test_initialize_rng_states_changes_output(device: str) -> None:
    """RNG state initialization writes nonzero data."""
    rng_states = wp.zeros((4,), dtype=wp.uint32, device=device)
    wp.launch(
        _initialize_rng_states,
        dim=4,
        inputs=[wp.uint32(123), rng_states],
        device=device,
    )
    wp.synchronize()

    rng_values = np.asarray(rng_states.numpy())
    assert np.any(rng_values != 0)
