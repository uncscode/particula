"""Native resident semantics, independent widths and diagnostic inventories."""

from dataclasses import replace

import numpy as np
import pytest

from particula.dynamics.tests.native_runnable_test import state
from particula.execution import Backend, Device
from particula.execution.gpu_session import setup_resident_session

wp = pytest.importorskip("warp")
pytestmark = pytest.mark.warp


@pytest.fixture(params=["cpu", "cuda:0"])
def device(request):
    """Test both installed Warp CPU and optional native CUDA."""
    if request.param != "cpu" and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    return request.param


def session_for(device, kind="particle_resolved"):
    """Upload a native two-material, three-gas nonidentity layout."""
    aerosol = state(0.25 if kind == "particle_resolved" else 1.0, kind)
    return aerosol, setup_resident_session(
        aerosol.particles,
        aerosol.gas,
        aerosol.environment,
        Device(Backend.WARP, device),
    )


@pytest.mark.parametrize("kind", ["discrete", "particle_resolved"])
def test_setup_pins_native_kind_and_independent_species_widths(
    device, kind, monkeypatch
):
    """Setup preserves raw state; repeated metadata validation does no work."""
    from particula.execution.gpu_resources import GPUResourceRegistry

    aerosol, session = session_for(device, kind)
    assert session.dimensions.n_species == 2
    assert session.dimensions.gas_species == 3
    assert session.metadata.distribution_type == kind
    assert session.metadata.gas_names == tuple(aerosol.gas.name)
    np.testing.assert_array_equal(
        session.particles.concentration.numpy(), aerosol.particles.concentration
    )
    registry = GPUResourceRegistry(session)
    scratch = registry.acquire_condensation().scratch_buffers
    assert scratch.work_mass_transfer.shape == (1, 2, 2)
    assert scratch.positive_mass_transfer_demand.shape == (1, 3)

    def forbidden(*args, **kwargs):
        pytest.fail("resident metadata validation performed device work")

    with monkeypatch.context() as patch:
        for name in ("launch", "zeros", "empty", "synchronize"):
            patch.setattr(wp, name, forbidden)
        session.__post_init__()
        registry.validate_pinned_session(session)
    session.particles.distribution_kind = 0
    with pytest.raises(ValueError, match="population metadata"):
        registry.validate_pinned_session(session)


def diagnostics_plan(session):
    """Build six explicit gas-order diagnostics against an independent oracle."""
    from particula.dynamics.condensation.species_mapping import (
        CondensationSpeciesMap,
    )
    from particula.execution.diagnostics import (
        ResidentDiagnosticOperation as Op,
    )
    from particula.execution.diagnostics import (
        ResidentDiagnosticRegistration as Registration,
    )
    from particula.execution.tests.diagnostics_test import _plan

    device = session.particles.masses.device
    p, g = session.particles, session.gas
    expected = g.concentration.numpy() * p.volume.numpy()[:, None]
    weight = p.concentration.numpy()
    if session.metadata.distribution_type == "discrete":
        weight = weight * p.volume.numpy()[:, None]
    mass = np.sum(p.masses.numpy() * weight[..., None], axis=1)
    expected[:, 2] += mass[:, 0]
    expected[:, 1] += mass[:, 1]
    mapping = CondensationSpeciesMap(
        session.metadata.gas_names, ((2, 0), (1, 1))
    )
    outputs = [
        wp.zeros(
            (1,) if op is Op.PARTICLE_NUMBER_CONCENTRATION else (1, 3),
            dtype=wp.float64,
            device=device,
        )
        for op in Op
    ]
    baseline = wp.array(expected, dtype=wp.float64, device=device)
    energy = wp.array([[1.0, 2.0, 3.0]], dtype=wp.float64, device=device)
    ledger = wp.zeros((1, 3), dtype=wp.float64, device=device)
    registrations = (
        Registration(Op.GAS_CONCENTRATION_SNAPSHOT, outputs[0]),
        Registration(Op.SATURATION_RATIO_SNAPSHOT, outputs[1]),
        Registration(Op.TOTAL_SPECIES_MASS, outputs[2], species_map=mapping),
        Registration(Op.PARTICLE_NUMBER_CONCENTRATION, outputs[3]),
        Registration(Op.LATENT_HEAT_ENERGY, outputs[4], energy_transfer=energy),
        Registration(
            Op.CONSERVATION_RESIDUAL,
            outputs[5],
            baseline_total_mass=baseline,
            source_ledger=ledger,
            sink_ledger=ledger,
        ),
    )
    return _plan(session, registrations), outputs, expected


@pytest.mark.parametrize("kind", ["discrete", "particle_resolved"])
def test_native_diagnostics_use_explicit_mass_map_and_number_density(
    device, kind
):
    """Gas order and volume remain authoritative for physical reductions."""
    from particula.execution.diagnostics import ResidentDiagnosticsExecutor

    aerosol, session = session_for(device, kind)
    plan, outputs, expected = diagnostics_plan(session)
    executor = ResidentDiagnosticsExecutor()
    executor.execute(plan)
    executor.execute(plan)
    wp.synchronize()
    np.testing.assert_array_equal(outputs[0].numpy(), aerosol.gas.concentration)
    np.testing.assert_allclose(
        outputs[2].numpy(), expected, rtol=1e-12, atol=1e-30
    )
    np.testing.assert_allclose(
        outputs[3].numpy(), aerosol.particles.number_density, rtol=1e-12, atol=0
    )
    np.testing.assert_allclose(outputs[5].numpy(), 0, rtol=0, atol=1e-30)


@pytest.mark.parametrize("failure", ["missing", "order", "incomplete"])
def test_native_mass_diagnostic_rejects_ambiguous_mapping(device, failure):
    """Total mass never infers same-index species or drops unmapped material."""
    from particula.dynamics.condensation.species_mapping import (
        CondensationSpeciesMap,
    )
    from particula.execution.diagnostics import ResidentDiagnosticsExecutor

    _, session = session_for(device)
    plan, outputs, _ = diagnostics_plan(session)
    mapping = None
    if failure == "order":
        mapping = CondensationSpeciesMap(
            tuple(reversed(session.metadata.gas_names)), ((2, 0), (1, 1))
        )
    elif failure == "incomplete":
        mapping = CondensationSpeciesMap(session.metadata.gas_names, ((2, 0),))
    registrations = list(plan.registrations)
    registrations[2] = replace(registrations[2], species_map=mapping)
    plan = replace(plan, registrations=tuple(registrations))
    with pytest.raises(ValueError):
        ResidentDiagnosticsExecutor().execute(plan)
    for output in outputs:
        np.testing.assert_array_equal(output.numpy(), 0)


@pytest.mark.parametrize("latent", [False, True])
def test_native_condensation_adapter_preserves_material_and_gas_widths(
    device, latent
):
    """Selected Warp execution forwards the pinned nonidentity mapping."""
    from particula.execution.adapters.condensation import (
        CondensationExecutionConfig,
        WarpCondensationExecutionAdapter,
        WarpCondensationExecutionState,
        WarpCondensationState,
    )
    from particula.execution.tests.condensation_adapter_test import (
        _configuration,
    )
    from particula.gas import EnvironmentData
    from particula.gpu import to_warp_environment_data
    from particula.gpu.kernels.tests.condensation_native_test import _state

    source = _state(device, scenario="uptake", latent=latent)
    environment = to_warp_environment_data(
        EnvironmentData(
            temperature=np.full(2, 298.15),
            pressure=np.full(2, 101325.0),
            saturation_ratio=np.zeros((2, 4)),
        ),
        device=device,
    )
    request = WarpCondensationExecutionState(
        WarpCondensationState(
            CondensationExecutionConfig(_configuration(latent_heat=latent)),
            source.p,
            source.g,
            environment,
            source.kwargs["thermodynamics"],
            scratch_buffers=source.kwargs["scratch_buffers"],
            species_binding=source.kwargs["species_binding"],
            latent_heat=source.kwargs.get("latent_heat"),
        ),
        1.0,
    )
    before = source.g.concentration.numpy().copy()
    original_mass = np.sum(
        source.p.masses.numpy() * source.p.concentration.numpy()[..., None],
        axis=1,
    )
    binding = WarpCondensationExecutionAdapter().prepare(request)
    binding.execute()
    binding.execute()
    wp.synchronize()
    after = source.g.concentration.numpy()
    assert np.any(after[:, [1, 2]] < before[:, [1, 2]])
    np.testing.assert_array_equal(after[:, [0, 3]], before[:, [0, 3]])
    final_mass = np.sum(
        source.p.masses.numpy() * source.p.concentration.numpy()[..., None],
        axis=1,
    )
    for gas_lane, particle_lane in source.pairs:
        np.testing.assert_allclose(
            final_mass[:, particle_lane]
            + after[:, gas_lane] * source.p.volume.numpy(),
            original_mass[:, particle_lane]
            + before[:, gas_lane] * source.p.volume.numpy(),
            rtol=1e-12,
            atol=1e-30,
        )
    np.testing.assert_array_equal(final_mass[:, 2], original_mass[:, 2])


def test_native_resident_nucleation_pins_mapping_and_separate_sidecars(device):
    """Resident source dispatch preserves explicit counts and source lanes."""
    from particula.execution.gpu_resources import GPUResourceRegistry
    from particula.execution.gpu_session import (
        ResidentDimensions,
        ResidentLifecycle,
        ResidentMetadata,
        ResidentSession,
    )
    from particula.execution.process_adapters import (
        ResidentNucleationAdapter,
        ResidentNucleationRequest,
    )
    from particula.gas import EnvironmentData
    from particula.gpu import to_warp_environment_data
    from particula.gpu.kernels.tests.nucleation_native_test import (
        density_inventory,
        setup,
    )

    p, g, config, kwargs = setup(device)
    environment = to_warp_environment_data(
        EnvironmentData(
            temperature=np.array([300.0]),
            pressure=np.array([101325.0]),
            saturation_ratio=np.ones((1, 4)),
        ),
        device=device,
    )
    session = ResidentSession(
        p,
        g,
        environment,
        ResidentDimensions(1, 20, 3, 4),
        ResidentMetadata(
            Device(Backend.WARP, device),
            kwargs["species_binding"].gas_names,
            distribution_type="particle_resolved",
        ),
        ResidentLifecycle.ACTIVE,
    )
    registry = GPUResourceRegistry(session)
    resources = registry.acquire_nucleation()
    for name in ("requested_scale", "minimum_scale", "minimum_volume"):
        wp.copy(
            getattr(resources.exhaustion, name),
            getattr(kwargs["exhaustion_buffers"], name),
        )
    request = ResidentNucleationRequest(
        session,
        registry,
        resources,
        config,
        1.0,
        kwargs["exhaustion_controls"],
        kwargs["species_binding"],
    )
    before = density_inventory(p, g)
    binding = ResidentNucleationAdapter().prepare(request)
    binding.execute()
    wp.synchronize()
    assert np.count_nonzero(p.concentration.numpy()) == 2
    np.testing.assert_allclose(
        density_inventory(p, g), before, rtol=1e-12, atol=1e-30
    )
    assert resources.finalized_demand.precursor_mass_change.shape == (1, 4)


def test_setup_rejects_overflowing_count_density_before_upload(monkeypatch):
    """Finite stored counts must also represent a finite number density."""
    from particula.gpu import conversion

    aerosol = state(0.25)
    aerosol.particles.volume[:] = 1e-300
    aerosol.particles.concentration[:] = 1e300
    monkeypatch.setattr(
        conversion,
        "to_warp_particle_data",
        lambda *a, **k: pytest.fail("invalid density reached upload"),
    )
    with pytest.raises(ValueError, match="density must be finite"):
        setup_resident_session(
            aerosol.particles,
            aerosol.gas,
            aerosol.environment,
            Device(Backend.WARP, "cpu"),
        )
