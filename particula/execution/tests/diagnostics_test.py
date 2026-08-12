"""Contract tests for concrete resident diagnostic reductions."""

from typing import Any, cast

import numpy as np
import numpy.testing as npt
import pytest

pytest.importorskip("warp")

from particula.execution import Backend, CapabilityRequirements, Device
from particula.execution.diagnostics import (
    ResidentDiagnosticOperation,
    ResidentDiagnosticRegistration,
    ResidentDiagnosticsExecutor,
    ResidentDiagnosticsPlan,
)
from particula.execution.gpu_resources import GPUResourceRegistry
from particula.execution.gpu_session import (
    ResidentDimensions,
    ResidentLifecycle,
    ResidentMetadata,
    ResidentSession,
)
from particula.execution.process_graph import ProcessNode, TimestepPlan
from particula.execution.scheduler import (
    EnabledNodeSelection,
    NucleationCondensationDirection,
    SchedulerProfile,
    resolve_timestep_schedule,
)


def _session() -> ResidentSession:
    """Create a nonuniform two-box resident fixture."""
    wp = pytest.importorskip("warp")
    from particula.gpu.warp_types import (
        WarpEnvironmentData,
        WarpGasData,
        WarpParticleData,
    )

    particles = WarpParticleData()
    particles.masses = wp.array(
        np.array([[[2.0, 3.0], [0.0, 0.0]], [[5.0, 7.0], [11.0, 13.0]]]),
        dtype=wp.float64,
        device="cpu",
    )
    particles.concentration = wp.array(
        np.array([[4.0, 0.0], [2.0, 3.0]]), dtype=wp.float64, device="cpu"
    )
    particles.charge = wp.zeros((2, 2), dtype=wp.float64, device="cpu")
    particles.density = wp.ones(2, dtype=wp.float64, device="cpu")
    particles.volume = wp.array(
        np.array([2.0, 0.5]), dtype=wp.float64, device="cpu"
    )
    gas = WarpGasData()
    gas.molar_mass = wp.ones(2, dtype=wp.float64, device="cpu")
    gas.concentration = wp.array(
        np.array([[1.0, 0.0], [4.0, 6.0]]), dtype=wp.float64, device="cpu"
    )
    gas.vapor_pressure = wp.zeros((2, 2), dtype=wp.float64, device="cpu")
    gas.partitioning = wp.ones((2, 2), dtype=wp.int32, device="cpu")
    environment = WarpEnvironmentData()
    environment.temperature = wp.ones(2, dtype=wp.float64, device="cpu")
    environment.pressure = wp.ones(2, dtype=wp.float64, device="cpu")
    environment.saturation_ratio = wp.array(
        np.array([[2.0, 3.0], [4.0, 5.0]]), dtype=wp.float64, device="cpu"
    )
    return ResidentSession(
        particles,
        gas,
        environment,
        ResidentDimensions(2, 2, 2),
        ResidentMetadata(Device(Backend.WARP, "cpu"), ("a", "b")),
        ResidentLifecycle.ACTIVE,
    )


def _plan(
    session: ResidentSession,
    registrations: tuple[ResidentDiagnosticRegistration, ...],
) -> ResidentDiagnosticsPlan:
    """Create the smallest resolver-produced final diagnostics plan."""
    from particula.execution import process_graph

    node_ids = (
        "environment_update",
        "gas_update",
        "vapor_pressure_refresh",
        "saturation_refresh",
        "diagnostics",
    )
    nodes = tuple(
        ProcessNode(
            item.node_id,
            item.kind,
            item.process,
            CapabilityRequirements(frozenset()),
            item.resources,
            item.invalidates,
        )
        for item in process_graph._NODE_CATALOGUE
        if item.node_id in node_ids
    )
    schedule = resolve_timestep_schedule(
        TimestepPlan(nodes, ()),
        EnabledNodeSelection(frozenset(node_ids)),
        SchedulerProfile(
            NucleationCondensationDirection.NUCLEATION_THEN_CONDENSATION
        ),
    )
    graph = cast(Any, schedule.source_graph)
    return ResidentDiagnosticsPlan(
        session,
        GPUResourceRegistry(session),
        graph,
        schedule,
        next(node for node in graph.nodes if node.node_id == "diagnostics"),
        registrations,
    )


def _matrix(device: str = "cpu") -> Any:
    """Return a caller-owned two-box, two-species diagnostic output."""
    wp = pytest.importorskip("warp")
    return wp.zeros((2, 2), dtype=wp.float64, device=device)


@pytest.mark.warp
@pytest.mark.gpu_parity
def test_diagnostics_reducers_match_independent_numpy_oracle() -> None:
    """Test all closed reducers on a multi-box, multi-species fixture."""
    wp = pytest.importorskip("warp")
    session = _session()
    gas, saturation, total, number, energy, residual = (
        _matrix(),
        _matrix(),
        _matrix(),
        wp.zeros(2, dtype=wp.float64, device="cpu"),
        _matrix(),
        _matrix(),
    )
    transferred = wp.array(
        np.array([[10.0, -2.0], [3.0, 4.0]]), dtype=wp.float64, device="cpu"
    )
    baseline = wp.array(
        np.array([[18.0, 24.0], [21.0, 30.0]]), dtype=wp.float64, device="cpu"
    )
    source = wp.array(
        np.array([[1.0, 2.0], [3.0, 4.0]]), dtype=wp.float64, device="cpu"
    )
    sink = wp.array(
        np.array([[5.0, 6.0], [7.0, 8.0]]), dtype=wp.float64, device="cpu"
    )
    registrations = (
        ResidentDiagnosticRegistration(
            ResidentDiagnosticOperation.GAS_CONCENTRATION_SNAPSHOT, gas
        ),
        ResidentDiagnosticRegistration(
            ResidentDiagnosticOperation.SATURATION_RATIO_SNAPSHOT, saturation
        ),
        ResidentDiagnosticRegistration(
            ResidentDiagnosticOperation.TOTAL_SPECIES_MASS, total
        ),
        ResidentDiagnosticRegistration(
            ResidentDiagnosticOperation.PARTICLE_NUMBER_CONCENTRATION, number
        ),
        ResidentDiagnosticRegistration(
            ResidentDiagnosticOperation.LATENT_HEAT_ENERGY,
            energy,
            energy_transfer=transferred,
        ),
        ResidentDiagnosticRegistration(
            ResidentDiagnosticOperation.CONSERVATION_RESIDUAL,
            residual,
            baseline_total_mass=baseline,
            source_ledger=source,
            sink_ledger=sink,
        ),
    )
    ResidentDiagnosticsExecutor().execute(_plan(session, registrations))
    wp.synchronize()
    particle_mass = np.array(
        [[[2.0, 3.0], [0.0, 0.0]], [[5.0, 7.0], [11.0, 13.0]]]
    )
    concentration = np.array([[4.0, 0.0], [2.0, 3.0]])
    expected_total = np.array([2.0, 0.5])[:, None] * (
        np.sum(particle_mass * concentration[:, :, None], axis=1)
        + np.array([[1.0, 0.0], [4.0, 6.0]])
    )
    npt.assert_allclose(total.numpy(), expected_total, rtol=1e-12, atol=1e-30)
    npt.assert_allclose(
        number.numpy(), concentration.sum(axis=1), rtol=1e-12, atol=1e-30
    )
    npt.assert_array_equal(energy.numpy(), transferred.numpy())
    npt.assert_allclose(
        residual.numpy(),
        expected_total - baseline.numpy() - source.numpy() + sink.numpy(),
        rtol=1e-12,
        atol=1e-30,
    )
    npt.assert_array_equal(
        gas.numpy(), cast(Any, session.gas).concentration.numpy()
    )
    npt.assert_array_equal(
        saturation.numpy(),
        cast(Any, session.environment).saturation_ratio.numpy(),
    )


@pytest.mark.warp
def test_legacy_two_snapshot_plan_remains_supported() -> None:
    """Former two-operation callers retain their exact ordered protocol."""
    wp = pytest.importorskip("warp")
    session = _session()
    gas = _matrix()
    saturation = _matrix()
    registrations = (
        ResidentDiagnosticRegistration(
            ResidentDiagnosticOperation.GAS_CONCENTRATION_SNAPSHOT, gas
        ),
        ResidentDiagnosticRegistration(
            ResidentDiagnosticOperation.SATURATION_RATIO_SNAPSHOT, saturation
        ),
    )

    ResidentDiagnosticsExecutor().execute(_plan(session, registrations))
    wp.synchronize()

    npt.assert_array_equal(
        gas.numpy(), cast(Any, session.gas).concentration.numpy()
    )
    npt.assert_array_equal(
        saturation.numpy(),
        cast(Any, session.environment).saturation_ratio.numpy(),
    )


@pytest.mark.warp
@pytest.mark.gpu_parity
def test_parallel_reductions_match_large_non_power_of_two_oracle() -> None:
    """Particle-lane stages handle substantial non-power-of-two capacity."""
    wp = pytest.importorskip("warp")
    from particula.gpu.warp_types import (
        WarpEnvironmentData,
        WarpGasData,
        WarpParticleData,
    )

    n_boxes, n_particles, n_species = 2, 257, 2
    masses = (
        np.arange(n_boxes * n_particles * n_species, dtype=np.float64).reshape(
            n_boxes, n_particles, n_species
        )
        + 1.0
    ) * 1e-21
    concentration = (
        np.arange(n_boxes * n_particles, dtype=np.float64).reshape(
            n_boxes, n_particles
        )
        + 1.0
    )
    volume = np.array([1.5, 2.5], dtype=np.float64)
    gas_concentration = np.array([[3.0, 5.0], [7.0, 11.0]]) * 1e-18
    particles = WarpParticleData()
    particles.masses = wp.array(masses, dtype=wp.float64, device="cpu")
    particles.concentration = wp.array(
        concentration, dtype=wp.float64, device="cpu"
    )
    particles.charge = wp.zeros(
        (n_boxes, n_particles), dtype=wp.float64, device="cpu"
    )
    particles.density = wp.ones(n_species, dtype=wp.float64, device="cpu")
    particles.volume = wp.array(volume, dtype=wp.float64, device="cpu")
    gas = WarpGasData()
    gas.molar_mass = wp.ones(n_species, dtype=wp.float64, device="cpu")
    gas.concentration = wp.array(
        gas_concentration, dtype=wp.float64, device="cpu"
    )
    gas.vapor_pressure = wp.zeros(
        (n_boxes, n_species), dtype=wp.float64, device="cpu"
    )
    gas.partitioning = wp.ones(
        (n_boxes, n_species), dtype=wp.int32, device="cpu"
    )
    environment = WarpEnvironmentData()
    environment.temperature = wp.ones(n_boxes, dtype=wp.float64, device="cpu")
    environment.pressure = wp.ones(n_boxes, dtype=wp.float64, device="cpu")
    environment.saturation_ratio = wp.ones(
        (n_boxes, n_species), dtype=wp.float64, device="cpu"
    )
    session = ResidentSession(
        particles,
        gas,
        environment,
        ResidentDimensions(n_boxes, n_particles, n_species),
        ResidentMetadata(Device(Backend.WARP, "cpu"), ("a", "b")),
        ResidentLifecycle.ACTIVE,
    )
    total = wp.zeros((n_boxes, n_species), dtype=wp.float64, device="cpu")
    number = wp.zeros(n_boxes, dtype=wp.float64, device="cpu")

    def zero_matrix() -> Any:
        """Allocate one caller-owned matrix for the large fixture."""
        return wp.zeros((n_boxes, n_species), dtype=wp.float64, device="cpu")

    registrations = (
        ResidentDiagnosticRegistration(
            ResidentDiagnosticOperation.GAS_CONCENTRATION_SNAPSHOT,
            zero_matrix(),
        ),
        ResidentDiagnosticRegistration(
            ResidentDiagnosticOperation.SATURATION_RATIO_SNAPSHOT,
            zero_matrix(),
        ),
        ResidentDiagnosticRegistration(
            ResidentDiagnosticOperation.TOTAL_SPECIES_MASS, total
        ),
        ResidentDiagnosticRegistration(
            ResidentDiagnosticOperation.PARTICLE_NUMBER_CONCENTRATION, number
        ),
        ResidentDiagnosticRegistration(
            ResidentDiagnosticOperation.LATENT_HEAT_ENERGY,
            zero_matrix(),
            energy_transfer=zero_matrix(),
        ),
        ResidentDiagnosticRegistration(
            ResidentDiagnosticOperation.CONSERVATION_RESIDUAL,
            zero_matrix(),
            baseline_total_mass=zero_matrix(),
            source_ledger=zero_matrix(),
            sink_ledger=zero_matrix(),
        ),
    )

    ResidentDiagnosticsExecutor().execute(_plan(session, registrations))
    wp.synchronize()

    expected_total = volume[:, None] * (
        np.sum(masses * concentration[:, :, None], axis=1) + gas_concentration
    )
    npt.assert_allclose(total.numpy(), expected_total, rtol=1e-12, atol=1e-30)
    npt.assert_allclose(
        number.numpy(), concentration.sum(axis=1), rtol=1e-12, atol=1e-30
    )


def test_diagnostic_registration_requires_exact_accounting_bindings() -> None:
    """Test protocol construction rejects missing and forbidden inputs."""
    output = object()
    with pytest.raises(ValueError, match="requires only"):
        ResidentDiagnosticRegistration(
            ResidentDiagnosticOperation.LATENT_HEAT_ENERGY, output
        )
    with pytest.raises(ValueError, match="requires baseline"):
        ResidentDiagnosticRegistration(
            ResidentDiagnosticOperation.CONSERVATION_RESIDUAL, output
        )
    with pytest.raises(ValueError, match="forbids"):
        ResidentDiagnosticRegistration(
            ResidentDiagnosticOperation.TOTAL_SPECIES_MASS,
            output,
            energy_transfer=object(),
        )


@pytest.mark.parametrize(
    ("operation", "bindings", "match"),
    (
        (
            ResidentDiagnosticOperation.LATENT_HEAT_ENERGY,
            {"energy_transfer": object(), "baseline_total_mass": object()},
            "requires only",
        ),
        (
            ResidentDiagnosticOperation.CONSERVATION_RESIDUAL,
            {
                "energy_transfer": object(),
                "baseline_total_mass": object(),
                "source_ledger": object(),
                "sink_ledger": object(),
            },
            "requires baseline",
        ),
    ),
)
def test_diagnostic_registration_rejects_forbidden_accounting_combinations(
    operation: ResidentDiagnosticOperation,
    bindings: dict[str, object],
    match: str,
) -> None:
    """Test each accounting-only reducer rejects a conflicting ledger binding."""
    with pytest.raises(ValueError, match=match):
        ResidentDiagnosticRegistration(operation, object(), **bindings)


def test_diagnostic_registration_retains_valid_bindings_by_identity() -> None:
    """Test valid immutable registrations retain every supplied object."""
    output = object()
    energy = object()
    baseline = object()
    source = object()
    sink = object()

    registrations = (
        ResidentDiagnosticRegistration(
            ResidentDiagnosticOperation.GAS_CONCENTRATION_SNAPSHOT, output
        ),
        ResidentDiagnosticRegistration(
            ResidentDiagnosticOperation.SATURATION_RATIO_SNAPSHOT, output
        ),
        ResidentDiagnosticRegistration(
            ResidentDiagnosticOperation.TOTAL_SPECIES_MASS, output
        ),
        ResidentDiagnosticRegistration(
            ResidentDiagnosticOperation.PARTICLE_NUMBER_CONCENTRATION, output
        ),
        ResidentDiagnosticRegistration(
            ResidentDiagnosticOperation.LATENT_HEAT_ENERGY,
            output,
            energy_transfer=energy,
        ),
        ResidentDiagnosticRegistration(
            ResidentDiagnosticOperation.CONSERVATION_RESIDUAL,
            output,
            baseline_total_mass=baseline,
            source_ledger=source,
            sink_ledger=sink,
        ),
    )

    assert tuple(item.operation for item in registrations) == tuple(
        ResidentDiagnosticOperation
    )
    assert registrations[4].output is output
    assert registrations[4].energy_transfer is energy
    assert registrations[5].baseline_total_mass is baseline
    assert registrations[5].source_ledger is source
    assert registrations[5].sink_ledger is sink
    with pytest.raises(AttributeError):
        registrations[0].output = object()  # type: ignore[misc]


def test_diagnostic_registration_rejects_non_enum_operation() -> None:
    """Test protocol construction requires an exact operation enum member."""
    with pytest.raises(TypeError, match="exact ResidentDiagnosticOperation"):
        ResidentDiagnosticRegistration("total_species_mass", object())  # type: ignore[arg-type]


@pytest.mark.warp
def test_diagnostics_launches_parallel_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test particle lanes dispatch through parallel staged accumulation."""
    wp = pytest.importorskip("warp")
    import particula.execution.diagnostics as diagnostics

    session = _session()
    registrations = (
        ResidentDiagnosticRegistration(
            ResidentDiagnosticOperation.GAS_CONCENTRATION_SNAPSHOT, _matrix()
        ),
        ResidentDiagnosticRegistration(
            ResidentDiagnosticOperation.SATURATION_RATIO_SNAPSHOT, _matrix()
        ),
        ResidentDiagnosticRegistration(
            ResidentDiagnosticOperation.TOTAL_SPECIES_MASS, _matrix()
        ),
        ResidentDiagnosticRegistration(
            ResidentDiagnosticOperation.PARTICLE_NUMBER_CONCENTRATION,
            wp.zeros(2, dtype=wp.float64, device="cpu"),
        ),
        ResidentDiagnosticRegistration(
            ResidentDiagnosticOperation.LATENT_HEAT_ENERGY,
            _matrix(),
            energy_transfer=_matrix(),
        ),
        ResidentDiagnosticRegistration(
            ResidentDiagnosticOperation.CONSERVATION_RESIDUAL,
            _matrix(),
            baseline_total_mass=_matrix(),
            source_ledger=_matrix(),
            sink_ledger=_matrix(),
        ),
    )
    launches: list[tuple[str, list[object]]] = []
    original_launch = diagnostics.wp.launch

    def record_launch(
        kernel: object, *args: object, **kwargs: object
    ) -> object:
        """Record the kernel name while preserving normal Warp dispatch."""
        launches.append(
            (cast(Any, kernel).key, cast(list[object], kwargs["inputs"]))
        )
        return cast(Any, original_launch)(kernel, *args, **kwargs)

    monkeypatch.setattr(diagnostics.wp, "launch", record_launch)

    ResidentDiagnosticsExecutor().execute(_plan(session, registrations))

    writer_launches = [
        launch
        for launch in launches
        if launch[0] != "_scan_diagnostic_accounting"
    ]
    assert [key for key, _ in writer_launches] == [
        "_copy_snapshot",
        "_copy_snapshot",
        "_initialize_total_species_mass",
        "_accumulate_particle_species_mass",
        "_clear_particle_number",
        "_accumulate_particle_number",
        "_copy_snapshot",
        "_conservation_residual",
    ]
    assert writer_launches[-1][1][0] is registrations[2].output


@pytest.mark.warp
def test_valid_diagnostics_dispatch_avoids_executor_host_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test valid dispatch does not synchronize or invoke conversion helpers."""
    wp = pytest.importorskip("warp")
    import particula.execution.diagnostics as diagnostics

    session = _session()
    matrices = tuple(_matrix() for _ in range(5))
    registrations = (
        ResidentDiagnosticRegistration(
            ResidentDiagnosticOperation.GAS_CONCENTRATION_SNAPSHOT, matrices[0]
        ),
        ResidentDiagnosticRegistration(
            ResidentDiagnosticOperation.SATURATION_RATIO_SNAPSHOT, matrices[1]
        ),
        ResidentDiagnosticRegistration(
            ResidentDiagnosticOperation.TOTAL_SPECIES_MASS, matrices[2]
        ),
        ResidentDiagnosticRegistration(
            ResidentDiagnosticOperation.PARTICLE_NUMBER_CONCENTRATION,
            wp.zeros(2, dtype=wp.float64, device="cpu"),
        ),
        ResidentDiagnosticRegistration(
            ResidentDiagnosticOperation.LATENT_HEAT_ENERGY,
            matrices[3],
            energy_transfer=_matrix(),
        ),
        ResidentDiagnosticRegistration(
            ResidentDiagnosticOperation.CONSERVATION_RESIDUAL,
            matrices[4],
            baseline_total_mass=_matrix(),
            source_ledger=_matrix(),
            sink_ledger=_matrix(),
        ),
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        """Fail if dispatch crosses the host synchronization boundary."""
        raise AssertionError("diagnostics dispatch must not synchronize")

    monkeypatch.setattr(diagnostics.wp, "synchronize", forbidden)
    ResidentDiagnosticsExecutor().execute(_plan(session, registrations))
