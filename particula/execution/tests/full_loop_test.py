"""Regression coverage for complete resident-loop dispatch ordering.

The scheduler owns ordering and lifecycle composition.  These tests isolate
that boundary with recorder adapters so they can assert the complete resolved
trace without copying resident device payloads.
"""

from typing import Any, cast

import pytest

_NODE_IDS = (
    "communication",
    "volume_evolution",
    "environment_update",
    "gas_update",
    "vapor_pressure_refresh",
    "saturation_refresh",
    "condensation",
    "brownian_coagulation",
    "dilution",
    "wall_loss",
    "nucleation",
    "diagnostics",
)


def _scheduler_module() -> Any:
    """Import the Warp-dependent concrete scheduler for guarded rows."""
    pytest.importorskip("warp")
    import particula.execution.resident_scheduler as resident_scheduler

    return resident_scheduler


def _request(module: Any) -> tuple[Any, Any]:
    """Build an already-preflighted complete request with stable identities."""
    nodes = tuple(
        type("Node", (), {"node_id": node_id})() for node_id in _NODE_IDS
    )

    class Guard:
        """Record lifecycle closure for each synthetic completed timestep."""

        def __init__(self) -> None:
            self.completed_steps = 0
            self._token = object()

        def begin_step(self, _duration: object) -> object:
            return self._token

        def complete_step(self, token: object) -> None:
            assert token is self._token
            self.completed_steps += 1

    request = object.__new__(module.ResidentSimulationRequest)
    guard = Guard()
    object.__setattr__(request, "graph", type("Graph", (), {"nodes": nodes})())
    object.__setattr__(
        request,
        "schedule",
        type("Schedule", (), {"ordered_node_ids": _NODE_IDS})(),
    )
    object.__setattr__(request, "guard", guard)
    for name in (
        "session",
        "registry",
        "thermodynamics",
        "environment_update",
        "gas_update",
        "condensation",
        "coagulation",
        "dilution",
        "wall_loss",
        "nucleation",
        "diagnostics",
        "communication",
    ):
        object.__setattr__(request, name, object())
    return request, guard


def _install_recorders(  # noqa: C901
    monkeypatch: pytest.MonkeyPatch,
    module: Any,
    request: Any,
    events: list[str],
) -> None:
    """Install composition-only recorders without touching resident payloads."""

    class Updates:
        """Record state-update request identity."""

        def execute(self, item: object) -> None:
            events.append(
                "environment_update"
                if item is request.environment_update
                else "gas_update"
            )

    class Thermal:
        """Expose virtual refreshes only for the two supported consumers."""

        def __init__(self, _request: object) -> None:
            pass

        def record_completed(self, node: object) -> None:
            node_id = cast(Any, node).node_id
            if node_id not in {
                "communication",
                "volume_evolution",
                "environment_update",
                "gas_update",
            }:
                events.append(node_id)

        def execute_consumer(self, node: object, callback: Any) -> None:
            node_id = cast(Any, node).node_id
            assert node_id in {"condensation", "diagnostics"}
            events.extend(("vapor_pressure_refresh", "saturation_refresh"))
            callback()
            events.append(node_id)

    class Communication:
        """Record the two canonical barriers."""

        def __init__(self, _request: object) -> None:
            pass

        def execute_communication(self) -> None:
            events.append("communication")

        def execute_volume_evolution(self) -> None:
            events.append("volume_evolution")

    def adapter(name: str) -> type[Any]:
        """Return one no-payload adapter for an ordinary process node."""

        class Adapter:
            """Leave ordinary-node trace ownership with the coordinator."""

            def execute(self, _item: object) -> None:
                return None

        Adapter.__name__ = name
        return Adapter

    monkeypatch.setattr(module, "ResidentStateUpdateExecutor", Updates)
    monkeypatch.setattr(module, "ResidentCommunicationExecutor", Communication)
    monkeypatch.setattr(
        module, "ResidentThermodynamicUpdateRequest", lambda *args: args
    )
    monkeypatch.setattr(
        module, "ResidentThermodynamicUpdateCoordinator", Thermal
    )
    monkeypatch.setattr(
        module, "WarpCondensationExecutionAdapter", adapter("Condensation")
    )
    monkeypatch.setattr(
        module,
        "ResidentBrownianCoagulationExecutionAdapter",
        adapter("Coagulation"),
    )
    monkeypatch.setattr(module, "ResidentDilutionAdapter", adapter("Dilution"))
    monkeypatch.setattr(module, "ResidentWallLossAdapter", adapter("WallLoss"))
    monkeypatch.setattr(
        module, "ResidentNucleationAdapter", adapter("Nucleation")
    )
    monkeypatch.setattr(
        module, "ResidentDiagnosticsExecutor", adapter("Diagnostics")
    )


@pytest.mark.warp
@pytest.mark.gpu_parity
@pytest.mark.parametrize("communication_family", ["GAS", "PARTICLES"])
def test_complete_loop_repeats_canonical_order_without_payload_transfers(
    monkeypatch: pytest.MonkeyPatch, communication_family: str
) -> None:
    """Two complete map-family rows retain identities and dispatch in order."""
    module = _scheduler_module()
    request, guard = _request(module)
    events: list[str] = []
    identities = tuple(
        getattr(request, name) for name in ("session", "registry")
    )
    _install_recorders(monkeypatch, module, request, events)
    scheduler = module.ResidentSimulationScheduler(request)
    monkeypatch.setattr(scheduler, "_validate", lambda _duration: None)

    scheduler.execute(0.0)
    scheduler.execute(0.0)

    assert communication_family in {"GAS", "PARTICLES"}
    expected_trace = list(_NODE_IDS[:-1]) + [
        "vapor_pressure_refresh",
        "saturation_refresh",
        "diagnostics",
    ]
    assert events == expected_trace * 2
    assert guard.completed_steps == 2
    assert identities == (request.session, request.registry)


@pytest.mark.warp
def test_wall_loss_failure_closes_the_open_token_before_later_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A writer failure is propagated after scheduler cleanup, not retried."""
    module = _scheduler_module()
    request, guard = _request(module)
    events: list[str] = []
    _install_recorders(monkeypatch, module, request, events)
    scheduler = module.ResidentSimulationScheduler(request)
    monkeypatch.setattr(scheduler, "_validate", lambda _duration: None)

    class FailingWallLoss:
        """Model a late resident writer failure after token entry."""

        def execute(self, _item: object) -> None:
            raise RuntimeError("wall-loss writer failed")

    cleaned: list[object] = []
    monkeypatch.setattr(module, "ResidentWallLossAdapter", FailingWallLoss)
    monkeypatch.setattr(
        module,
        "_handle_failed_resident_operation",
        lambda *_args: cleaned.append("faulted"),
    )

    with pytest.raises(RuntimeError, match="wall-loss writer failed"):
        scheduler.execute(0.0)

    assert cleaned == ["faulted"]
    assert guard.completed_steps == 0
    assert events[-1] == "dilution"
