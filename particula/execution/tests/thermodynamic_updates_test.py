"""Test the concrete resident thermodynamic freshness coordinator."""

from typing import Any, cast

import numpy as np
import numpy.testing as npt
import pytest

from particula.execution import (
    CONDENSATION_CAPABILITY_MATRIX,
    CONDENSATION_PROCESS,
)
from particula.execution.gpu_resources import GPUResourceRegistry
from particula.execution.process_graph import (
    DependencyEdge,
    NodeKind,
    ProcessNode,
    TimestepPlan,
)
from particula.execution.scheduler import (
    EnabledNodeSelection,
    NucleationCondensationDirection,
    SchedulerProfile,
    resolve_timestep_schedule,
)
from particula.execution.thermodynamic_updates import (
    ResidentThermodynamicUpdateCoordinator,
    ResidentThermodynamicUpdateRequest,
)
from particula.gpu.kernels.thermodynamics import ThermodynamicsConfig
from particula.util.constants import GAS_CONSTANT


def _node(node_id: str) -> ProcessNode:
    """Build one catalogue node suitable for a resolved test graph."""
    from particula.execution import CapabilityRequirements, process_graph

    schema = next(
        entry
        for entry in process_graph._NODE_CATALOGUE
        if entry.node_id == node_id
    )
    requirements = CapabilityRequirements(frozenset())
    if node_id == "condensation":
        requirements = next(
            entry.requirements
            for entry in CONDENSATION_CAPABILITY_MATRIX.declarations
            if entry.process == CONDENSATION_PROCESS
        )
    return ProcessNode(
        schema.node_id,
        schema.kind,
        schema.process,
        requirements,
        schema.resources,
        schema.invalidates,
    )


def _coordinator(
    node_ids: frozenset[str],
    boxes: int = 1,
    species: int = 1,
    dependencies: tuple[DependencyEdge, ...] = (),
) -> tuple[ResidentThermodynamicUpdateCoordinator, dict[str, ProcessNode], Any]:
    """Build a coordinator with a resolver-produced graph and CPU Warp state."""
    wp = pytest.importorskip("warp")
    from particula.execution.tests.gpu_resources_test import _session

    session = _session(boxes=boxes, species=species)
    gas = cast(Any, session.gas)
    environment = cast(Any, session.environment)
    gas.molar_mass = wp.ones(species, dtype=wp.float64, device="cpu")
    gas.concentration = wp.ones(
        (boxes, species), dtype=wp.float64, device="cpu"
    )
    environment.temperature = wp.full(
        (boxes,), 300.0, dtype=wp.float64, device="cpu"
    )
    registry = GPUResourceRegistry(session)
    nodes = tuple(_node(node_id) for node_id in node_ids)
    plan = TimestepPlan(nodes, dependencies)
    schedule = resolve_timestep_schedule(
        plan,
        EnabledNodeSelection(node_ids),
        SchedulerProfile(
            NucleationCondensationDirection.NUCLEATION_THEN_CONDENSATION
        ),
    )
    graph = cast(Any, schedule.source_graph)
    config = ThermodynamicsConfig(
        wp.zeros(species, dtype=wp.int32, device="cpu"),
        wp.full((species, 4), 2.0, dtype=wp.float64, device="cpu"),
        gas.molar_mass,
    )
    request = ResidentThermodynamicUpdateRequest(
        session, registry, graph, schedule, config
    )
    by_id = {node.node_id: node for node in graph.nodes}
    return ResidentThermodynamicUpdateCoordinator(request), by_id, session


@pytest.mark.warp
def test_environment_refreshes_vapor_then_saturation_before_condensation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test an environment report performs both required writers in order."""
    import particula.execution.thermodynamic_updates as updates

    ids = frozenset(
        {
            "environment_update",
            "gas_update",
            "vapor_pressure_refresh",
            "saturation_refresh",
            "condensation",
        }
    )
    coordinator, nodes, _ = _coordinator(ids)
    calls: list[str] = []
    original_vapor = updates.refresh_vapor_pressure_gpu
    original_saturation = coordinator._refresh_saturation_ratio

    def vapor(
        config: ThermodynamicsConfig, gas_data: Any, temperature: Any
    ) -> None:
        calls.append("vapor")
        original_vapor(config, gas_data, temperature)

    def saturation() -> None:
        calls.append("saturation")
        original_saturation()

    monkeypatch.setattr(updates, "refresh_vapor_pressure_gpu", vapor)
    monkeypatch.setattr(coordinator, "_refresh_saturation_ratio", saturation)
    coordinator.record_completed(nodes["environment_update"])
    coordinator.record_completed(nodes["gas_update"])
    assert (
        coordinator.execute_consumer(
            nodes["condensation"], lambda: calls.append("condensation")
        )
        is None
    )
    assert calls == ["vapor", "saturation", "condensation"]
    assert coordinator.cursor == 5


@pytest.mark.warp
def test_gas_update_elides_fresh_vapor_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test a gas report refreshes saturation without rewriting vapor pressure."""
    ids = frozenset(
        {
            "environment_update",
            "gas_update",
            "vapor_pressure_refresh",
            "saturation_refresh",
            "diagnostics",
        }
    )
    coordinator, nodes, _ = _coordinator(ids)
    calls: list[str] = []
    import particula.execution.thermodynamic_updates as updates

    def vapor(*_args: Any) -> None:
        calls.append("vapor")

    def saturation() -> None:
        calls.append("saturation")

    monkeypatch.setattr(updates, "refresh_vapor_pressure_gpu", vapor)
    monkeypatch.setattr(coordinator, "_refresh_saturation_ratio", saturation)
    coordinator.record_completed(nodes["environment_update"])
    coordinator._stale.clear()
    coordinator.record_completed(nodes["gas_update"])
    coordinator.execute_consumer(nodes["diagnostics"], lambda: None)
    assert calls == ["saturation"]
    assert coordinator.cursor == 5
    assert not coordinator.stale_states


@pytest.mark.warp
def test_condensation_refreshes_saturation_before_later_diagnostics() -> None:
    """Test condensation invalidation is visible to the diagnostics callback."""
    ids = frozenset(
        {
            "environment_update",
            "gas_update",
            "vapor_pressure_refresh",
            "saturation_refresh",
            "condensation",
            "diagnostics",
        }
    )
    coordinator, nodes, session = _coordinator(ids)
    gas = cast(Any, session.gas)
    wp = pytest.importorskip("warp")
    calls: list[str] = []

    coordinator.record_completed(nodes["environment_update"])
    coordinator.record_completed(nodes["gas_update"])

    def condense() -> None:
        wp.copy(
            gas.concentration,
            wp.full((1, 1), 2.0, dtype=wp.float64, device="cpu"),
        )
        calls.append("condensation")

    coordinator.execute_consumer(nodes["condensation"], condense)
    coordinator.execute_consumer(
        nodes["diagnostics"], lambda: calls.append("diagnostics")
    )

    assert calls == ["condensation", "diagnostics"]
    assert coordinator.cursor == 6
    expected = 2.0 * GAS_CONSTANT * 300.0 / 2.0
    npt.assert_allclose(gas.concentration.numpy(), [[2.0]])
    npt.assert_allclose(
        session.environment.saturation_ratio.numpy(), [[expected]], rtol=1e-12
    )


@pytest.mark.warp
def test_condensation_refresh_survives_wall_loss_before_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test diagnostics refreshes condensation state after an ordinary node."""
    import particula.execution.thermodynamic_updates as updates

    ids = frozenset(
        {
            "environment_update",
            "gas_update",
            "vapor_pressure_refresh",
            "saturation_refresh",
            "condensation",
            "wall_loss",
            "diagnostics",
        }
    )
    coordinator, nodes, session = _coordinator(
        ids,
        dependencies=(DependencyEdge("wall_loss", "diagnostics"),),
    )
    gas = cast(Any, session.gas)
    wp = pytest.importorskip("warp")
    calls: list[str] = []
    original_vapor = updates.refresh_vapor_pressure_gpu
    original_saturation = coordinator._refresh_saturation_ratio

    def vapor(
        config: ThermodynamicsConfig, gas_data: Any, temperature: Any
    ) -> None:
        calls.append("vapor")
        original_vapor(config, gas_data, temperature)

    def saturation() -> None:
        calls.append("saturation")
        original_saturation()

    monkeypatch.setattr(updates, "refresh_vapor_pressure_gpu", vapor)
    monkeypatch.setattr(coordinator, "_refresh_saturation_ratio", saturation)

    def condense() -> None:
        wp.copy(
            gas.concentration,
            wp.full((1, 1), 2.0, dtype=wp.float64, device="cpu"),
        )
        calls.append("condensation")

    coordinator.record_completed(nodes["environment_update"])
    coordinator.record_completed(nodes["gas_update"])
    coordinator.execute_consumer(nodes["condensation"], condense)
    coordinator.record_completed(nodes["wall_loss"])
    coordinator.execute_consumer(
        nodes["diagnostics"], lambda: calls.append("diagnostics")
    )

    assert calls == [
        "vapor",
        "saturation",
        "condensation",
        "saturation",
        "diagnostics",
    ]
    assert coordinator.cursor == 7
    assert not coordinator.stale_states
    expected = 2.0 * GAS_CONSTANT * 300.0 / 2.0
    npt.assert_allclose(
        session.environment.saturation_ratio.numpy(), [[expected]], rtol=1e-12
    )


@pytest.mark.warp
def test_rejected_preflight_preserves_state_and_allows_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test invalid schedule, schema, and binding checks retain coordinator state."""
    ids = frozenset(
        {
            "environment_update",
            "gas_update",
            "vapor_pressure_refresh",
            "saturation_refresh",
            "diagnostics",
        }
    )
    coordinator, nodes, _ = _coordinator(ids)
    request = coordinator._request
    initial_stale = coordinator.stale_states
    schedule = request.schedule
    original_ids = schedule.ordered_node_ids

    object.__setattr__(
        schedule, "ordered_node_ids", tuple(reversed(original_ids))
    )
    with pytest.raises(ValueError, match="ordered schedule"):
        coordinator.execute_consumer(
            nodes["diagnostics"], lambda: pytest.fail()
        )
    object.__setattr__(schedule, "ordered_node_ids", original_ids)

    original_kind = nodes["diagnostics"].kind
    object.__setattr__(nodes["diagnostics"], "kind", NodeKind.PROCESS)
    with pytest.raises(ValueError, match="invalid canonical"):
        coordinator.execute_consumer(
            nodes["diagnostics"], lambda: pytest.fail()
        )
    object.__setattr__(nodes["diagnostics"], "kind", original_kind)

    registry = cast(Any, request.registry)
    original_validation = registry.validate_pinned_session
    monkeypatch.setattr(
        registry,
        "validate_pinned_session",
        lambda _session: (_ for _ in ()).throw(ValueError("binding failure")),
    )
    with pytest.raises(ValueError, match="binding failure"):
        coordinator.execute_consumer(
            nodes["diagnostics"], lambda: pytest.fail()
        )
    monkeypatch.setattr(
        registry, "validate_pinned_session", original_validation
    )

    assert coordinator.cursor == 0
    assert coordinator.stale_states == initial_stale
    coordinator.record_completed(nodes["environment_update"])
    coordinator.record_completed(nodes["gas_update"])
    assert (
        coordinator.execute_consumer(nodes["diagnostics"], lambda: "retry")
        == "retry"
    )
    assert coordinator.cursor == 5
    assert not coordinator.stale_states


@pytest.mark.warp
@pytest.mark.parametrize("shape", [(1, 1), (2, 2)])
def test_saturation_refresh_matches_si_reference(
    shape: tuple[int, int],
) -> None:
    """Test the private writer isolates all box/species lanes on device."""
    boxes, species = shape
    ids = frozenset(
        {
            "environment_update",
            "gas_update",
            "vapor_pressure_refresh",
            "saturation_refresh",
            "diagnostics",
        }
    )
    coordinator, nodes, session = _coordinator(ids, boxes, species)
    gas = cast(Any, session.gas)
    environment = cast(Any, session.environment)
    wp = pytest.importorskip("warp")
    concentration = np.arange(1, boxes * species + 1, dtype=np.float64).reshape(
        boxes, species
    )
    vapor_pressure = concentration + 2.0
    temperature = np.arange(boxes, dtype=np.float64) + 290.0
    wp.copy(
        gas.concentration,
        wp.array(concentration, dtype=wp.float64, device="cpu"),
    )
    wp.copy(
        gas.vapor_pressure,
        wp.array(vapor_pressure, dtype=wp.float64, device="cpu"),
    )
    wp.copy(
        environment.temperature,
        wp.array(temperature, dtype=wp.float64, device="cpu"),
    )
    coordinator.record_completed(nodes["environment_update"])
    coordinator.record_completed(nodes["gas_update"])
    coordinator._stale.clear()
    # Vapor pressure is prepared explicitly; only saturation is stale here.
    from particula.execution.process_graph import InvalidatedState

    coordinator._stale = {InvalidatedState.SATURATION_RATIO}
    coordinator.execute_consumer(nodes["diagnostics"], lambda: None)
    expected = (
        concentration * GAS_CONSTANT * temperature[:, None] / vapor_pressure
    )
    npt.assert_allclose(
        environment.saturation_ratio.numpy(), expected, rtol=1e-12
    )


@pytest.mark.warp
def test_writer_failure_preserves_cursor_and_suppresses_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test a failed vapor writer leaves both derived fields stale."""
    import particula.execution.thermodynamic_updates as updates
    from particula.execution.process_graph import InvalidatedState

    ids = frozenset(
        {
            "environment_update",
            "gas_update",
            "vapor_pressure_refresh",
            "saturation_refresh",
            "diagnostics",
        }
    )
    coordinator, nodes, _ = _coordinator(ids)
    coordinator.record_completed(nodes["environment_update"])
    coordinator.record_completed(nodes["gas_update"])

    def fail(*_args: object) -> None:
        raise RuntimeError("vapor failure")

    monkeypatch.setattr(updates, "refresh_vapor_pressure_gpu", fail)
    with pytest.raises(RuntimeError, match="vapor failure"):
        coordinator.execute_consumer(
            nodes["diagnostics"], lambda: pytest.fail()
        )
    assert coordinator.cursor == 2
    assert coordinator.stale_states == {
        InvalidatedState.VAPOR_PRESSURE,
        InvalidatedState.SATURATION_RATIO,
    }


@pytest.mark.warp
def test_saturation_writer_failure_keeps_vapor_fresh_and_cursor_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test partial writer success retains only the vapor-pressure freshness."""
    from particula.execution.process_graph import InvalidatedState

    ids = frozenset(
        {
            "environment_update",
            "gas_update",
            "vapor_pressure_refresh",
            "saturation_refresh",
            "diagnostics",
        }
    )
    coordinator, nodes, _ = _coordinator(ids)
    coordinator.record_completed(nodes["environment_update"])
    coordinator.record_completed(nodes["gas_update"])
    calls: list[str] = []

    def fail() -> None:
        calls.append("saturation")
        raise RuntimeError("saturation failure")

    monkeypatch.setattr(coordinator, "_refresh_saturation_ratio", fail)
    with pytest.raises(RuntimeError, match="saturation failure"):
        coordinator.execute_consumer(
            nodes["diagnostics"], lambda: calls.append("callback")
        )

    assert calls == ["saturation"]
    assert coordinator.cursor == 2
    assert coordinator.stale_states == {InvalidatedState.SATURATION_RATIO}


@pytest.mark.warp
def test_consumer_failure_does_not_consume_node_or_apply_invalidations() -> (
    None
):
    """Test a failing callback leaves the successful refreshes and cursor intact."""
    ids = frozenset(
        {
            "environment_update",
            "gas_update",
            "vapor_pressure_refresh",
            "saturation_refresh",
            "condensation",
        }
    )
    coordinator, nodes, _ = _coordinator(ids)
    coordinator.record_completed(nodes["environment_update"])
    coordinator.record_completed(nodes["gas_update"])

    with pytest.raises(RuntimeError, match="consumer failure"):
        coordinator.execute_consumer(
            nodes["condensation"],
            lambda: (_ for _ in ()).throw(RuntimeError("consumer failure")),
        )

    assert coordinator.cursor == 2
    assert not coordinator.stale_states


@pytest.mark.warp
def test_invalid_callback_and_out_of_order_report_preserve_coordinator_state() -> (
    None
):
    """Test rejected public calls leave cursor and stale markers unchanged."""
    ids = frozenset(
        {
            "environment_update",
            "gas_update",
            "vapor_pressure_refresh",
            "saturation_refresh",
            "diagnostics",
        }
    )
    coordinator, nodes, _ = _coordinator(ids)
    initial_stale = coordinator.stale_states

    with pytest.raises(ValueError, match="next scheduled"):
        coordinator.record_completed(nodes["gas_update"])
    with pytest.raises(TypeError, match="callback must be callable"):
        coordinator.execute_consumer(nodes["diagnostics"], cast(Any, None))

    assert coordinator.cursor == 0
    assert coordinator.stale_states == initial_stale


@pytest.mark.warp
@pytest.mark.parametrize("boxes, species", [(0, 1), (1, 0)])
def test_empty_saturation_refresh_is_write_free(
    boxes: int,
    species: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test canonical empty dimensions bypass the private Warp launch."""
    ids = frozenset(
        {
            "environment_update",
            "gas_update",
            "vapor_pressure_refresh",
            "saturation_refresh",
            "diagnostics",
        }
    )
    coordinator, _, _ = _coordinator(ids, boxes, species)
    import particula.execution.thermodynamic_updates as updates

    monkeypatch.setattr(
        updates.wp,
        "launch",
        lambda *_args, **_kwargs: pytest.fail(
            "empty refresh launched a kernel"
        ),
    )

    coordinator._refresh_saturation_ratio()
