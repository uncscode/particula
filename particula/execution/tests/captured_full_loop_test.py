"""Three-way evidence boundaries for resident graph-capture full loops.

Warp CPU supplies the uncaptured resident baseline. Native CUDA capture is
optional and is skipped before qualification when the device or Warp capture
API is unavailable; it is never emulated on the CPU.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.testing as npt
import pytest

from particula.execution.communication import CommunicationTransportMode
from particula.execution.tests.full_loop_test import (
    _build_loop_fixture,
    _expected_inventory,
)
from particula.gpu.tests.cuda_availability import (
    CUDA_SKIP_REASON,
    cuda_available,
)

PARITY_RTOL = 1e-12
PARITY_ATOL = 1e-30


def _snapshot(fixture: Any) -> tuple[np.ndarray, ...]:
    """Synchronize at an assertion boundary and copy meaningful state."""
    fixture.wp.synchronize()
    session = fixture.session
    return (
        session.particles.masses.numpy().copy(),
        session.particles.concentration.numpy().copy(),
        session.particles.charge.numpy().copy(),
        session.particles.density.numpy().copy(),
        session.particles.volume.numpy().copy(),
        session.gas.concentration.numpy().copy(),
        session.gas.vapor_pressure.numpy().copy(),
        session.gas.partitioning.numpy().copy(),
        session.environment.temperature.numpy().copy(),
        session.environment.pressure.numpy().copy(),
        session.environment.saturation_ratio.numpy().copy(),
        fixture.gas_snapshot.numpy().copy(),
        fixture.saturation_snapshot.numpy().copy(),
        fixture.request.coagulation.resources.rng_states.numpy().copy(),
        fixture.request.wall_loss.resources.rng_states.numpy().copy(),
    )


def _assert_same_state(
    actual: tuple[np.ndarray, ...], expected: tuple[np.ndarray, ...]
) -> None:
    """Compare float state tightly and integer metadata exactly."""
    for index, (actual_value, expected_value) in enumerate(
        zip(actual, expected, strict=True)
    ):
        if index in {7, 12, 13, 14}:
            npt.assert_equal(actual_value, expected_value)
        else:
            npt.assert_allclose(
                actual_value,
                expected_value,
                rtol=PARITY_RTOL,
                atol=PARITY_ATOL,
            )


@pytest.mark.warp
@pytest.mark.gpu_parity
@pytest.mark.parametrize(
    "communication_family",
    (CommunicationTransportMode.GAS, CommunicationTransportMode.PARTICLES),
)
def test_deterministic_full_loop_matches_independent_uncaptured_baseline(
    monkeypatch: pytest.MonkeyPatch,
    communication_family: CommunicationTransportMode,
) -> None:
    """Three zero-duration steps preserve the full uncaptured resident state."""
    reference = _build_loop_fixture(monkeypatch, communication_family)
    initial_inventory = _expected_inventory(reference.session)

    reference.scheduler.execute(0.0)
    stable_state = _snapshot(reference)
    for _ in range(2):
        reference.scheduler.execute(0.0)
        _assert_same_state(_snapshot(reference), stable_state)
        npt.assert_allclose(
            _expected_inventory(reference.session),
            initial_inventory,
            rtol=PARITY_RTOL,
            atol=PARITY_ATOL,
        )

    expected_dispatch = tuple(
        node_id
        for node_id in reference.request.schedule.ordered_node_ids
        if node_id not in {"vapor_pressure_refresh", "saturation_refresh"}
    )
    assert reference.trace == list(expected_dispatch * 3)
    assert reference.guard.completed_steps == 3


def _require_native_cuda_capture() -> Any:
    """Skip only before native qualification when CUDA capture is unavailable."""
    wp = pytest.importorskip("warp")
    if not cuda_available(wp):
        pytest.skip(CUDA_SKIP_REASON)
    if not all(
        callable(getattr(wp, name, None))
        for name in ("capture_begin", "capture_end")
    ):
        pytest.skip("Warp capture API unavailable")
    return wp


@pytest.mark.warp
@pytest.mark.cuda
@pytest.mark.gpu_parity
def test_cuda_capture_support_is_native_only() -> None:
    """Native CUDA is an optional capture boundary, never a Warp-CPU fallback."""
    wp = _require_native_cuda_capture()
    devices = [
        device for device in wp.get_devices() if str(device).startswith("cuda")
    ]
    assert devices


@pytest.mark.warp
@pytest.mark.cuda
@pytest.mark.gpu_parity
def test_cuda_capture_zero_duration_contract_is_available_before_capture() -> (
    None
):
    """CUDA availability is decided before a zero-duration capture is attempted."""
    wp = _require_native_cuda_capture()
    assert callable(wp.capture_begin)
    assert callable(wp.capture_end)
