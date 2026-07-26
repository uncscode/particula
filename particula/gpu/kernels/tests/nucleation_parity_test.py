"""Independent direct-Warp nucleation parity and conservation evidence.

The NumPy expectations in this module intentionally do not call nucleation
planning or orchestration helpers from the production implementation.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from particula.util.constants import AVOGADRO_NUMBER

pytestmark = [pytest.mark.warp, pytest.mark.gpu_parity]

_DEVICE_CASES = [
    pytest.param("cpu", id="cpu"),
    pytest.param("cuda:0", marks=pytest.mark.cuda, id="cuda"),
]


def _warp():
    """Import Warp only when a device-dependent test begins."""
    return pytest.importorskip("warp")


def _api():
    """Import concrete nucleation records only after Warp is available."""
    from particula.gpu.kernels.exhaustion import ResamplingBuffers
    from particula.gpu.kernels.nucleation import (
        NucleationConfig,
        NucleationDiagnosticBuffers,
        NucleationExhaustionBuffers,
        NucleationExhaustionControls,
        NucleationFinalizedDemandBuffers,
        NucleationScratchBuffers,
        nucleation_step_gpu,
    )

    return SimpleNamespace(
        ResamplingBuffers=ResamplingBuffers,
        NucleationConfig=NucleationConfig,
        NucleationDiagnosticBuffers=NucleationDiagnosticBuffers,
        NucleationExhaustionBuffers=NucleationExhaustionBuffers,
        NucleationExhaustionControls=NucleationExhaustionControls,
        NucleationFinalizedDemandBuffers=NucleationFinalizedDemandBuffers,
        NucleationScratchBuffers=NucleationScratchBuffers,
        nucleation_step_gpu=nucleation_step_gpu,
    )


@pytest.fixture(params=_DEVICE_CASES)
def device(request):
    """Return an available parity device while cleanly skipping unavailable CUDA."""
    wp = _warp()
    if request.param.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA is not available")
    return request.param


def _config(api, *, rate_law="activation", coefficient=None, **changes):
    """Build a compact valid configuration for an independent event oracle."""
    values = dict(
        rate_law=rate_law,
        coefficient=(
            0.1 / AVOGADRO_NUMBER if coefficient is None else coefficient
        ),
        survival_factor=1.0,
        precursor_index=0,
        molecule_counts=(1, 2),
        formation_diameter=1e-9,
        precursor_number_concentration_lower=0.0,
        precursor_number_concentration_upper=1e30,
        temperature_lower=200.0,
        temperature_upper=400.0,
    )
    values.update(changes)
    return api.NucleationConfig(**values)


def _state(device, boxes=1, particles=3, species=2, *, active=False):
    """Create explicit float64 fixed-capacity particle and gas containers."""
    wp = _warp()
    masses = np.zeros((boxes, particles, species), dtype=np.float64)
    concentration = np.zeros((boxes, particles), dtype=np.float64)
    if active and particles:
        masses[:, 0, :] = np.array([2e-18, 3e-18], dtype=np.float64)
        concentration[:, 0] = 2.0
    particle_data = SimpleNamespace(
        masses=wp.array(masses, dtype=wp.float64, device=device),
        concentration=wp.array(concentration, dtype=wp.float64, device=device),
        charge=wp.zeros((boxes, particles), dtype=wp.float64, device=device),
        density=wp.array([1000.0, 1200.0], dtype=wp.float64, device=device),
        volume=wp.ones(boxes, dtype=wp.float64, device=device),
    )
    gas_data = SimpleNamespace(
        molar_mass=wp.array([0.1, 0.2], dtype=wp.float64, device=device),
        concentration=wp.ones(
            (boxes, species), dtype=wp.float64, device=device
        ),
        partitioning=wp.ones((boxes, species), dtype=wp.int32, device=device),
    )
    return particle_data, gas_data


def _sidecars(api, device, boxes, particles, species):
    """Create distinct same-device P2, P3, and P4 caller-owned sidecars."""
    wp = _warp()
    scratch = api.NucleationScratchBuffers(
        *[wp.zeros(boxes, dtype=wp.float64, device=device) for _ in range(3)]
    )
    finalized = api.NucleationFinalizedDemandBuffers(
        wp.zeros(boxes, dtype=wp.int32, device=device),
        wp.zeros(boxes, dtype=wp.float64, device=device),
        wp.zeros((boxes, species), dtype=wp.float64, device=device),
    )
    diagnostics = api.NucleationDiagnosticBuffers(
        wp.zeros(boxes, dtype=wp.int32, device=device),
        wp.full((boxes, particles), -1, dtype=wp.int32, device=device),
        wp.full((boxes, particles), -1, dtype=wp.int32, device=device),
        wp.zeros(boxes, dtype=wp.int32, device=device),
        wp.zeros(boxes, dtype=wp.int32, device=device),
    )
    resampling = api.ResamplingBuffers(
        retained_counts=wp.zeros(boxes, dtype=wp.int32, device=device),
        released_counts=wp.zeros(boxes, dtype=wp.int32, device=device),
        retained_indices=wp.zeros(
            (boxes, particles), dtype=wp.int32, device=device
        ),
        released_indices=wp.zeros(
            (boxes, particles), dtype=wp.int32, device=device
        ),
        sorted_indices=wp.zeros(
            (boxes, particles), dtype=wp.int32, device=device
        ),
        replacement_masses=wp.zeros(
            (boxes, particles, species), dtype=wp.float64, device=device
        ),
        replacement_concentration=wp.zeros(
            (boxes, particles), dtype=wp.float64, device=device
        ),
        replacement_charge=wp.zeros(
            (boxes, particles), dtype=wp.float64, device=device
        ),
        source_radii=wp.zeros(
            (boxes, particles), dtype=wp.float64, device=device
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
    exhaustion = api.NucleationExhaustionBuffers(
        resampling_buffers=resampling,
        demand_workspace=wp.zeros(boxes, dtype=wp.float64, device=device),
        final_demand=wp.zeros(boxes, dtype=wp.float64, device=device),
        requested_scale=wp.ones(boxes, dtype=wp.float64, device=device),
        minimum_scale=wp.ones(boxes, dtype=wp.float64, device=device),
        minimum_volume=wp.ones(boxes, dtype=wp.float64, device=device),
        resolved_scale=wp.zeros(boxes, dtype=wp.float64, device=device),
        resampling_releasable_counts=wp.zeros(
            boxes, dtype=wp.int32, device=device
        ),
        required_release_counts=wp.zeros(boxes, dtype=wp.int32, device=device),
        scaling_required=wp.zeros(boxes, dtype=wp.int32, device=device),
        final_counts=wp.zeros(boxes, dtype=wp.int32, device=device),
        final_selected_slot_indices=wp.full(
            (boxes, particles), -1, dtype=wp.int32, device=device
        ),
    )
    return scratch, finalized, diagnostics, exhaustion


def _numpy_p2(gas, molar_mass, config, time_step):
    """Independently calculate float64 P2 demand, removal, and event mass."""
    counts = np.asarray(config.molecule_counts, dtype=np.float64)
    event_mass = counts * molar_mass / AVOGADRO_NUMBER
    participating = counts > 0
    precursor = gas[:, config.precursor_index] * AVOGADRO_NUMBER
    precursor /= molar_mass[config.precursor_index]
    rate = config.survival_factor * config.coefficient * precursor
    if config.rate_law == "kinetic":
        rate *= precursor
    potential = rate * time_step
    inventory = np.min(
        gas[:, participating] / event_mass[participating], axis=1
    )
    demand = np.minimum(potential, inventory)
    for _ in range(4):
        unsafe = np.any(demand[:, None] * event_mass > gas, axis=1)
        demand[unsafe] = np.nextafter(demand[unsafe], -np.inf)
    return demand, demand[:, None] * event_mass, event_mass


def _particle_inventory(masses, concentration):
    """Return independent per-box, per-species particle mass inventory."""
    return np.einsum(
        "bn,bns->bs", concentration, masses, dtype=np.float64, optimize=True
    )


def _snapshot(*owners):
    """Capture arrays recursively, including nested P4 resampling sidecars."""
    result = []
    seen = set()

    def visit(value):
        """Record each Warp array once while walking record fields."""
        identifier = id(value)
        if identifier in seen:
            return
        seen.add(identifier)
        if hasattr(value, "numpy"):
            result.append(
                (
                    value,
                    value.numpy().copy(),
                    tuple(value.shape),
                    value.dtype,
                    str(value.device),
                )
            )
        elif hasattr(value, "__dict__"):
            for field in vars(value).values():
                visit(field)

    for owner in owners:
        visit(owner)
    return result


def _assert_unchanged(snapshot):
    """Assert caller-owned arrays preserve identity, schema, and exact values."""
    for value, contents, shape, dtype, device in snapshot:
        assert tuple(value.shape) == shape
        assert value.dtype == dtype
        assert str(value.device) == device
        np.testing.assert_array_equal(value.numpy(), contents, strict=True)


@pytest.mark.parametrize("boxes", [1, 2])
@pytest.mark.parametrize("rate_law", ["activation", "kinetic"])
def test_public_nucleation_step_matches_independent_p2_p3_p5_oracle(
    device, boxes, rate_law
) -> None:
    """Direct steps match independent demand, slot insertion, and gas removal."""
    wp = _warp()
    api = _api()
    particles, gas = _state(device, boxes=boxes)
    scratch, finalized, diagnostics, exhaustion = _sidecars(
        api, device, boxes, 3, 2
    )
    coefficient = 0.1 / AVOGADRO_NUMBER
    if rate_law == "kinetic":
        event_mass = np.array([0.1, 0.4], dtype=np.float64) / AVOGADRO_NUMBER
        gas.concentration = wp.array(
            np.tile(event_mass, (boxes, 1)),
            dtype=wp.float64,
            device=device,
        )
        coefficient = 2.0
    config = _config(api, rate_law=rate_law, coefficient=coefficient)
    initial_gas = gas.concentration.numpy().copy()
    demand, removal, event_mass = _numpy_p2(
        initial_gas, gas.molar_mass.numpy(), config, 1.0
    )
    result = api.nucleation_step_gpu(
        particles,
        gas,
        config,
        1.0,
        scratch=scratch,
        finalized_demand=finalized,
        diagnostics=diagnostics,
        exhaustion_controls=api.NucleationExhaustionControls(False, False),
        exhaustion_buffers=exhaustion,
        temperature=300.0,
    )
    wp.synchronize_device(device)
    assert result[0] is particles
    assert result[1] is gas
    expected_masses = np.zeros((boxes, 3, 2), dtype=np.float64)
    expected_concentration = np.zeros((boxes, 3), dtype=np.float64)
    expected_masses[:, 0, :] = event_mass
    expected_concentration[:, 0] = demand
    np.testing.assert_allclose(
        particles.masses.numpy(), expected_masses, rtol=1e-12, atol=1e-30
    )
    np.testing.assert_allclose(
        particles.concentration.numpy(),
        expected_concentration,
        rtol=1e-12,
        atol=1e-30,
    )
    np.testing.assert_allclose(
        particles.charge.numpy(), 0.0, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        gas.concentration.numpy(), initial_gas - removal, rtol=1e-12, atol=1e-30
    )
    np.testing.assert_array_equal(
        exhaustion.final_counts.numpy(), demand.astype(np.int32)
    )
    np.testing.assert_array_equal(
        exhaustion.final_selected_slot_indices.numpy(),
        np.tile([0, -1, -1], (boxes, 1)),
    )


@pytest.mark.parametrize(
    "kind", ["zero_time", "zero_coefficient", "low_saturation"]
)
def test_public_nucleation_step_gates_are_exact_write_free(
    device, kind
) -> None:
    """Zero-demand public gates preserve particles and gas byte-for-byte."""
    wp = _warp()
    api = _api()
    particles, gas = _state(device)
    scratch, finalized, diagnostics, exhaustion = _sidecars(
        api, device, 1, 3, 2
    )
    kwargs = {"time_step": 1.0, "config": _config(api)}
    if kind == "zero_time":
        kwargs["time_step"] = 0.0
    elif kind == "zero_coefficient":
        kwargs["config"] = _config(api, coefficient=0.0)
    else:
        kwargs["config"] = _config(
            api, saturation_lower=2.0, saturation_upper=3.0
        )
        kwargs["saturation"] = wp.ones((1, 2), dtype=wp.float64, device=device)
    before = _snapshot(particles, gas)
    api.nucleation_step_gpu(
        particles,
        gas,
        kwargs["config"],
        kwargs["time_step"],
        scratch=scratch,
        finalized_demand=finalized,
        diagnostics=diagnostics,
        exhaustion_controls=api.NucleationExhaustionControls(False, False),
        exhaustion_buffers=exhaustion,
        temperature=300.0,
        **(
            {"saturation": kwargs["saturation"]}
            if "saturation" in kwargs
            else {}
        ),
    )
    wp.synchronize_device(device)
    _assert_unchanged(before)


def test_public_nucleation_step_conserves_matrix_inventory(device) -> None:
    """Unscaled commits conserve each box/species particle-plus-gas inventory."""
    wp = _warp()
    api = _api()
    particles, gas = _state(device, boxes=2, active=True)
    scratch, finalized, diagnostics, exhaustion = _sidecars(
        api, device, 2, 3, 2
    )
    initial = _particle_inventory(
        particles.masses.numpy(), particles.concentration.numpy()
    )
    initial += gas.concentration.numpy()
    api.nucleation_step_gpu(
        particles,
        gas,
        _config(api),
        1.0,
        scratch=scratch,
        finalized_demand=finalized,
        diagnostics=diagnostics,
        exhaustion_controls=api.NucleationExhaustionControls(False, False),
        exhaustion_buffers=exhaustion,
        temperature=300.0,
    )
    wp.synchronize_device(device)
    final = _particle_inventory(
        particles.masses.numpy(), particles.concentration.numpy()
    )
    final += gas.concentration.numpy()
    np.testing.assert_allclose(final, initial, rtol=1e-12, atol=1e-30)
    assert np.all(np.isfinite(gas.concentration.numpy()))
    assert np.all(gas.concentration.numpy() >= 0.0)


def test_public_nucleation_step_scaling_fallback_has_separate_inventory(
    device,
) -> None:
    """Scaling halves old particle inventory without scaling gas inventory."""
    wp = _warp()
    api = _api()
    particles, gas = _state(device, particles=4, active=True)
    scratch, finalized, diagnostics, exhaustion = _sidecars(
        api, device, 1, 4, 2
    )
    object.__setattr__(
        exhaustion,
        "requested_scale",
        wp.full(1, 0.5, dtype=wp.float64, device=device),
    )
    object.__setattr__(
        exhaustion,
        "minimum_scale",
        wp.full(1, 0.5, dtype=wp.float64, device=device),
    )
    object.__setattr__(
        exhaustion,
        "minimum_volume",
        wp.full(1, 0.5, dtype=wp.float64, device=device),
    )
    initial_particle = _particle_inventory(
        particles.masses.numpy(), particles.concentration.numpy()
    )
    initial_gas = gas.concentration.numpy().copy()
    config = _config(api, coefficient=0.4 / AVOGADRO_NUMBER)
    _, removal, event_mass = _numpy_p2(
        initial_gas, gas.molar_mass.numpy(), config, 1.0
    )

    api.nucleation_step_gpu(
        particles,
        gas,
        config,
        1.0,
        scratch=scratch,
        finalized_demand=finalized,
        diagnostics=diagnostics,
        exhaustion_controls=api.NucleationExhaustionControls(False, True),
        exhaustion_buffers=exhaustion,
        temperature=300.0,
    )
    wp.synchronize_device(device)

    # P4 scales demand from 4 to 2 #/m^3 before P5 removes source mass.
    source_removal = 2.0 * event_mass
    final_particle = _particle_inventory(
        particles.masses.numpy(), particles.concentration.numpy()
    )
    np.testing.assert_array_equal(exhaustion.scaling_required.numpy(), [1])
    np.testing.assert_allclose(
        exhaustion.resolved_scale.numpy(), [0.5], rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        particles.volume.numpy(), [0.5], rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        gas.concentration.numpy(),
        initial_gas - source_removal,
        rtol=1e-12,
        atol=1e-30,
    )
    np.testing.assert_allclose(
        final_particle,
        0.5 * initial_particle + source_removal,
        rtol=1e-12,
        atol=1e-30,
    )
    np.testing.assert_allclose(removal[0], 4.0 * event_mass, rtol=0.0, atol=0.0)


def test_public_nucleation_step_resampling_precedes_scaling(device) -> None:
    """A fully releasable row chooses resampling and preserves inventory."""
    wp = _warp()
    api = _api()
    particles, gas = _state(device, particles=3, active=True)
    particles.masses = wp.array(
        [[[2e-18, 3e-18], [4e-18, 5e-18], [0.0, 0.0]]],
        dtype=wp.float64,
        device=device,
    )
    particles.concentration = wp.array(
        [[2.0, 3.0, 0.0]], dtype=wp.float64, device=device
    )
    scratch, finalized, diagnostics, exhaustion = _sidecars(
        api, device, 1, 3, 2
    )
    object.__setattr__(
        exhaustion,
        "resampling_releasable_counts",
        wp.ones(1, dtype=wp.int32, device=device),
    )
    initial_particle = _particle_inventory(
        particles.masses.numpy(), particles.concentration.numpy()
    )
    initial_gas = gas.concentration.numpy().copy()
    config = _config(api, coefficient=0.2 / AVOGADRO_NUMBER)
    _, removal, _ = _numpy_p2(initial_gas, gas.molar_mass.numpy(), config, 1.0)
    api.nucleation_step_gpu(
        particles,
        gas,
        config,
        1.0,
        scratch=scratch,
        finalized_demand=finalized,
        diagnostics=diagnostics,
        exhaustion_controls=api.NucleationExhaustionControls(True, True),
        exhaustion_buffers=exhaustion,
        temperature=300.0,
    )
    wp.synchronize_device(device)
    final_particle = _particle_inventory(
        particles.masses.numpy(), particles.concentration.numpy()
    )
    np.testing.assert_array_equal(exhaustion.scaling_required.numpy(), [0])
    np.testing.assert_allclose(
        exhaustion.resolved_scale.numpy(), [1.0], rtol=0.0, atol=0.0
    )
    np.testing.assert_array_equal(exhaustion.final_counts.numpy(), [2])
    np.testing.assert_allclose(
        final_particle,
        initial_particle + removal,
        rtol=1e-12,
        atol=1e-30,
    )
    np.testing.assert_allclose(
        gas.concentration.numpy(), initial_gas - removal, rtol=1e-12, atol=1e-30
    )


def test_public_nucleation_step_repeated_calls_use_current_gas(device) -> None:
    """A second commit follows the first call's gas-coupled state, not stale gas."""
    wp = _warp()
    api = _api()
    particles, gas = _state(device, particles=3)
    config = _config(api)
    first_sidecars = _sidecars(api, device, 1, 3, 2)
    api.nucleation_step_gpu(
        particles,
        gas,
        config,
        1.0,
        scratch=first_sidecars[0],
        finalized_demand=first_sidecars[1],
        diagnostics=first_sidecars[2],
        exhaustion_controls=api.NucleationExhaustionControls(False, False),
        exhaustion_buffers=first_sidecars[3],
        temperature=300.0,
    )
    wp.synchronize_device(device)
    before_particles = particles.masses.numpy().copy()
    before_concentration = particles.concentration.numpy().copy()
    before_gas = gas.concentration.numpy().copy()
    demand, removal, event_mass = _numpy_p2(
        before_gas, gas.molar_mass.numpy(), config, 1.0
    )
    second_sidecars = _sidecars(api, device, 1, 3, 2)
    api.nucleation_step_gpu(
        particles,
        gas,
        config,
        1.0,
        scratch=second_sidecars[0],
        finalized_demand=second_sidecars[1],
        diagnostics=second_sidecars[2],
        exhaustion_controls=api.NucleationExhaustionControls(False, False),
        exhaustion_buffers=second_sidecars[3],
        temperature=300.0,
    )
    wp.synchronize_device(device)
    expected_masses = before_particles.copy()
    expected_concentration = before_concentration.copy()
    expected_masses[:, 1, :] = event_mass
    expected_concentration[:, 1] = demand
    np.testing.assert_allclose(
        particles.masses.numpy(), expected_masses, rtol=1e-12, atol=1e-30
    )
    np.testing.assert_allclose(
        particles.concentration.numpy(),
        expected_concentration,
        rtol=1e-12,
        atol=1e-30,
    )
    np.testing.assert_allclose(
        gas.concentration.numpy(), before_gas - removal, rtol=1e-12, atol=1e-30
    )


@pytest.mark.parametrize(
    ("time_step", "temperature", "message"),
    [
        (np.nan, 300.0, "time_step"),
        (-1.0, 300.0, "time_step"),
        (1.0, -1.0, "temperature"),
    ],
)
def test_public_nucleation_step_rejections_preserve_callers(
    device, time_step, temperature, message
) -> None:
    """Public preflight rejection preserves particle, gas, and sidecar storage."""
    api = _api()
    particles, gas = _state(device)
    scratch, finalized, diagnostics, exhaustion = _sidecars(
        api, device, 1, 3, 2
    )
    before = _snapshot(particles, gas)
    with pytest.raises(ValueError, match=message):
        api.nucleation_step_gpu(
            particles,
            gas,
            _config(api),
            time_step,
            scratch=scratch,
            finalized_demand=finalized,
            diagnostics=diagnostics,
            exhaustion_controls=api.NucleationExhaustionControls(False, False),
            exhaustion_buffers=exhaustion,
            temperature=temperature,
        )
    _assert_unchanged(before)


def test_public_nucleation_step_malformed_sidecar_preserves_callers(
    device,
) -> None:
    """A malformed public diagnostic schema rejects before any caller mutation."""
    wp = _warp()
    api = _api()
    particles, gas = _state(device)
    scratch, finalized, diagnostics, exhaustion = _sidecars(
        api, device, 1, 3, 2
    )
    malformed = api.NucleationDiagnosticBuffers(
        wp.zeros(1, dtype=wp.float64, device=device),
        diagnostics.selected_slot_indices,
        diagnostics.free_slot_indices,
        diagnostics.active_slot_counts,
        diagnostics.free_slot_counts,
    )
    before = _snapshot(
        particles, gas, scratch, finalized, malformed, exhaustion
    )
    with pytest.raises(ValueError, match="required dtype"):
        api.nucleation_step_gpu(
            particles,
            gas,
            _config(api),
            1.0,
            scratch=scratch,
            finalized_demand=finalized,
            diagnostics=malformed,
            exhaustion_controls=api.NucleationExhaustionControls(False, False),
            exhaustion_buffers=exhaustion,
            temperature=300.0,
        )
    _assert_unchanged(before)


@pytest.mark.parametrize("particles", [0, 3])
def test_public_nucleation_step_zero_box_is_write_free(
    device, particles
) -> None:
    """Valid zero-box calls validate sidecars then leave empty state unchanged."""
    wp = _warp()
    api = _api()
    particle_data, gas = _state(device, boxes=0, particles=particles)
    scratch, finalized, diagnostics, exhaustion = _sidecars(
        api, device, 0, particles, 2
    )
    before = _snapshot(
        particle_data, gas, scratch, finalized, diagnostics, exhaustion
    )
    result = api.nucleation_step_gpu(
        particle_data,
        gas,
        _config(api),
        1.0,
        scratch=scratch,
        finalized_demand=finalized,
        diagnostics=diagnostics,
        exhaustion_controls=api.NucleationExhaustionControls(False, False),
        exhaustion_buffers=exhaustion,
        temperature=300.0,
    )
    wp.synchronize_device(device)
    assert result[0] is particle_data
    assert result[1] is gas
    _assert_unchanged(before)


def test_public_nucleation_step_zero_capacity_zero_gate_is_write_free(
    device,
) -> None:
    """A valid zero-capacity zero-demand call preserves empty particle fields."""
    wp = _warp()
    api = _api()
    particles, gas = _state(device, particles=0)
    scratch, finalized, diagnostics, exhaustion = _sidecars(
        api, device, 1, 0, 2
    )
    before = _snapshot(particles, gas)
    result = api.nucleation_step_gpu(
        particles,
        gas,
        _config(api, coefficient=0.0),
        1.0,
        scratch=scratch,
        finalized_demand=finalized,
        diagnostics=diagnostics,
        exhaustion_controls=api.NucleationExhaustionControls(False, False),
        exhaustion_buffers=exhaustion,
        temperature=300.0,
    )
    wp.synchronize_device(device)
    assert result[0] is particles
    assert result[1] is gas
    _assert_unchanged(before)
    np.testing.assert_array_equal(exhaustion.final_counts.numpy(), [0])
    assert exhaustion.final_selected_slot_indices.numpy().shape == (1, 0)
