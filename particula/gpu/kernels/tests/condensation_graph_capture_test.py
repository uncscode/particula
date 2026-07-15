"""Graph-capture readiness tests for the public condensation GPU step."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.testing as npt
import pytest

from particula.gpu.kernels.tests import _condensation_test_support as support

pytestmark = pytest.mark.warp

_SCRATCH_FIELDS = (
    "work_mass_transfer",
    "total_mass_transfer",
    "dynamic_viscosity",
    "mean_free_path",
    "positive_mass_transfer_demand",
    "negative_mass_transfer_release",
    "positive_mass_transfer_scale",
)

_CAPTURE_UNAVAILABLE_MESSAGES = frozenset(
    {
        "cuda graph capture is not supported on this device",
        "cuda graph capture is not supported on cpu devices",
        "graph capture is not supported on this device",
        "graph capture is not supported on cpu devices",
        "warp graph capture is only supported on cuda devices",
    }
)


@pytest.fixture(autouse=True)
def _selected_warp_test_runtime(request: pytest.FixtureRequest) -> None:
    """Load Warp only while executing a selected Warp-backed test."""
    if request.node.get_closest_marker("warp") is not None:
        support._load_warp_runtime()


def _make_graph_capture_state(
    runtime: Any,
    device: str,
) -> dict[str, Any]:
    """Build detached caller-owned state and device reset sources."""
    (
        particles,
        gas,
        vapor_pressure,
        temperature,
        pressure,
        _,
        _,
        _,
    ) = support._make_production_inventory_case()
    initial_masses = particles.masses.copy()
    initial_gas = gas.concentration.copy()
    particle_concentration = particles.concentration.copy()
    shape = initial_masses.shape
    n_boxes, _, n_species = shape
    wp = runtime.wp
    gpu_particles = runtime.to_warp_particle_data(particles, device=device)
    gpu_gas = runtime.to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=vapor_pressure,
    )
    thermodynamics = runtime.ThermodynamicsConfig(
        modes=wp.zeros(n_species, dtype=wp.int32, device=device),
        parameters=wp.array(
            np.column_stack((vapor_pressure[0], np.zeros((n_species, 3)))),
            dtype=wp.float64,
            device=device,
        ),
        molar_mass_reference=wp.array(
            gas.molar_mass,
            dtype=wp.float64,
            device=device,
        ),
    )
    scratch = support._make_condensation_scratch_buffers(shape, device)
    latent_heat_values = np.linspace(
        1.0e5,
        3.0e5,
        n_species,
        dtype=np.float64,
    )
    latent_heat = wp.array(latent_heat_values, dtype=wp.float64, device=device)
    energy = wp.zeros((n_boxes, n_species), dtype=wp.float64, device=device)
    reset_sources = {
        "masses": wp.array(initial_masses, dtype=wp.float64, device=device),
        "gas": wp.array(initial_gas, dtype=wp.float64, device=device),
        "vapor_pressure": wp.array(
            vapor_pressure,
            dtype=wp.float64,
            device=device,
        ),
        "energy": wp.zeros(
            (n_boxes, n_species),
            dtype=wp.float64,
            device=device,
        ),
    }
    sidecars = {name: getattr(scratch, name) for name in _SCRATCH_FIELDS}
    sidecars["energy"] = energy
    canonical_shapes = {
        "work_mass_transfer": shape,
        "total_mass_transfer": shape,
        "dynamic_viscosity": (n_boxes,),
        "mean_free_path": (n_boxes,),
        "positive_mass_transfer_demand": (n_boxes, n_species),
        "negative_mass_transfer_release": (n_boxes, n_species),
        "positive_mass_transfer_scale": (n_boxes, n_species),
        "energy": (n_boxes, n_species),
    }
    metadata = {
        name: (value, canonical_shapes[name], value.dtype, value.device)
        for name, value in sidecars.items()
    }
    return {
        "particles": gpu_particles,
        "gas": gpu_gas,
        "temperature": wp.array(temperature, dtype=wp.float64, device=device),
        "pressure": wp.array(pressure, dtype=wp.float64, device=device),
        "thermodynamics": thermodynamics,
        "scratch": scratch,
        "latent_heat": latent_heat,
        "latent_heat_values": latent_heat_values,
        "energy": energy,
        "reset_sources": reset_sources,
        "metadata": metadata,
        "initial_masses": initial_masses,
        "initial_gas": initial_gas,
        "particle_concentration": particle_concentration,
    }


def _assert_sidecar_contract(state: dict[str, Any]) -> None:
    """Assert sidecars retain identity and active-device fp64 metadata."""
    particles = state["particles"]
    scratch = state["scratch"]
    sidecars = {name: getattr(scratch, name) for name in _SCRATCH_FIELDS}
    sidecars["energy"] = state["energy"]
    for name, value in sidecars.items():
        original, shape, dtype, device = state["metadata"][name]
        assert value is original
        assert value.shape == shape
        assert value.dtype == dtype
        assert value.dtype == support.wp.float64
        assert value.device == device
        assert value.device == particles.masses.device


def _reset_mutable_state(runtime: Any, state: dict[str, Any]) -> None:
    """Restore mutable test state through device-to-device copies only."""
    wp = runtime.wp
    reset_sources = state["reset_sources"]
    wp.copy(state["particles"].masses, reset_sources["masses"])
    wp.copy(state["gas"].concentration, reset_sources["gas"])
    wp.copy(state["gas"].vapor_pressure, reset_sources["vapor_pressure"])
    wp.copy(state["energy"], reset_sources["energy"])


def _capture_support_error(error: BaseException) -> bool:
    """Return whether an exception reports capture support limitations."""
    return (
        isinstance(error, RuntimeError)
        and str(error).strip().lower() in _CAPTURE_UNAVAILABLE_MESSAGES
    )


def _capture_host_readback_error(error: BaseException) -> bool:
    """Return whether CUDA rejected the documented host readback in capture."""
    message = str(error).strip().lower()
    return isinstance(error, RuntimeError) and (
        "stream is capturing" in message
        or "stream capture" in message
        and "not permitted" in message
        or "wp_memcpy_d2h" in message
        and "capturing" in message
    )


def _require_capture_apis(wp: Any, device: str) -> None:
    """Skip when the required public Warp capture APIs are unavailable."""
    if device == "cpu":
        pytest.skip("cpu: Warp graph capture requires a CUDA device")
    for operation in ("capture_begin", "capture_end", "capture_launch"):
        if not callable(getattr(wp, operation, None)):
            pytest.skip(f"{device}: {operation} is unavailable")


def _end_capture_after_error(wp: Any, error: Exception) -> None:
    """End capture, retaining both operation and cleanup failures."""
    try:
        wp.capture_end()
    except Exception as cleanup_error:
        raise ExceptionGroup(
            "Graph capture and its cleanup both failed",
            [error, cleanup_error],
        ) from None


def _capture_graph_or_skip(
    runtime: Any,
    device: str,
    call: Any,
) -> tuple[Any, Any]:
    """Record ``call`` or skip only when this device lacks graph capture."""
    wp = runtime.wp
    _require_capture_apis(wp, device)
    try:
        wp.capture_begin(device=device, force_module_load=True)
    except Exception as error:
        if _capture_support_error(error):
            pytest.skip(f"{device}: capture_begin unavailable: {error}")
        raise
    capture_active = True
    try:
        result = call()
    except Exception as error:
        # ``capture_end`` consumes the capture even when it reports a cleanup
        # failure, so the finalizer must not attempt a second teardown.
        capture_active = False
        _end_capture_after_error(wp, error)
        raise
    else:
        try:
            graph = wp.capture_end()
        except Exception as error:
            # ``capture_end`` consumes the active capture even when it reports
            # an unsupported graph operation; do not attempt a second teardown.
            capture_active = False
            if _capture_support_error(error):
                pytest.skip(f"{device}: capture_end unavailable: {error}")
            raise
        capture_active = False
        return graph, result
    finally:
        if capture_active:
            wp.capture_end()


def _launch_graph_or_skip(runtime: Any, device: str, graph: object) -> None:
    """Launch a captured graph and surface all launch failures."""
    try:
        runtime.wp.capture_launch(graph)
    except Exception as error:
        if _capture_support_error(error):
            pytest.skip(f"{device}: capture_launch unavailable: {error}")
        raise


def _snapshot_state(state: dict[str, Any]) -> dict[str, np.ndarray]:
    """Synchronize and snapshot mutable outputs and all scratch sidecars."""
    support.wp.synchronize()
    snapshot = {
        "masses": state["particles"].masses.numpy().copy(),
        "gas": state["gas"].concentration.numpy().copy(),
        "energy": state["energy"].numpy().copy(),
    }
    snapshot.update(
        {
            f"scratch_{name}": getattr(state["scratch"], name).numpy().copy()
            for name in _SCRATCH_FIELDS
        }
    )
    return snapshot


def test_capture_capability_errors_are_precise() -> None:
    """Only known Warp capability errors qualify for a capture skip."""
    assert _capture_support_error(
        RuntimeError("graph capture is not supported on this device")
    )
    assert not _capture_support_error(
        ValueError("gas.partitioning must contain only binary 0/1 values")
    )
    assert not _capture_support_error(RuntimeError("graph allocation failed"))


def test_capture_cleanup_propagates_failure_and_clears_normal_state() -> None:
    """Capture cleanup clears state normally and exposes cleanup failures."""

    class FakeWarp:
        """Minimal capture-end fake for teardown regression coverage."""

        def __init__(self, failure: BaseException | None = None) -> None:
            self.active = True
            self.failure = failure

        def capture_end(self) -> None:
            self.active = False
            if self.failure is not None:
                raise self.failure

    normal = FakeWarp()
    _end_capture_after_error(normal, ValueError("call failed"))
    assert not normal.active

    failing = FakeWarp(RuntimeError("cleanup failed"))
    with pytest.raises(ExceptionGroup, match="cleanup") as captured:
        _end_capture_after_error(failing, ValueError("call failed"))
    assert not failing.active
    assert len(captured.value.exceptions) == 2


def test_capture_call_cleanup_is_not_repeated_after_failure() -> None:
    """A capture-call failure attempts teardown exactly once."""

    class FakeWarp:
        """Minimal capture fake that records teardown attempts."""

        def __init__(self) -> None:
            self.capture_end_calls = 0

        def capture_begin(self, **_: Any) -> None:
            """Start the fake capture."""

        def capture_end(self) -> None:
            """Fail while consuming the active fake capture."""
            self.capture_end_calls += 1
            raise RuntimeError("cleanup failed")

        def capture_launch(self, _: object) -> None:
            """Expose the required unused capture API."""

    fake_wp = FakeWarp()
    runtime = type("Runtime", (), {"wp": fake_wp})()
    with pytest.raises(ExceptionGroup, match="cleanup"):
        _capture_graph_or_skip(
            runtime,
            "cuda",
            lambda: (_ for _ in ()).throw(ValueError("call failed")),
        )
    assert fake_wp.capture_end_calls == 1


def _call_condensation(
    runtime: Any,
    state: dict[str, Any],
) -> tuple[object, object]:
    """Execute the already-constructed public condensation call."""
    return runtime._condensation_step_gpu(
        state["particles"],
        state["gas"],
        temperature=state["temperature"],
        pressure=state["pressure"],
        time_step=0.1,
        thermodynamics=state["thermodynamics"],
        scratch_buffers=state["scratch"],
        latent_heat=state["latent_heat"],
        energy_transfer=state["energy"],
    )


def _assert_graph_replay(device: str) -> None:
    """Retain a future replay harness without treating it as evidence today."""
    runtime = support._load_warp_runtime()
    normal = _make_graph_capture_state(runtime, device)
    captured = _make_graph_capture_state(runtime, device)

    _assert_sidecar_contract(normal)
    _, normal_transfer = _call_condensation(runtime, normal)
    assert normal_transfer is normal["scratch"].total_mass_transfer
    normal_result = _snapshot_state(normal)
    _assert_sidecar_contract(normal)

    def captured_call() -> tuple[object, object]:
        return _call_condensation(runtime, captured)

    graph, captured_result = _capture_graph_or_skip(
        runtime,
        device,
        captured_call,
    )
    assert captured_result[1] is captured["scratch"].total_mass_transfer
    _assert_sidecar_contract(captured)

    _reset_mutable_state(runtime, captured)
    _assert_sidecar_contract(captured)
    _launch_graph_or_skip(runtime, device, graph)
    replay_one = _snapshot_state(captured)
    _assert_sidecar_contract(captured)

    _reset_mutable_state(runtime, captured)
    _launch_graph_or_skip(runtime, device, graph)
    replay_two = _snapshot_state(captured)
    _assert_sidecar_contract(captured)

    for replay in (replay_one, replay_two):
        for name, expected in normal_result.items():
            npt.assert_allclose(
                replay[name],
                expected,
                rtol=support.PRODUCTION_PARITY_RTOL,
                atol=support.PRODUCTION_PARITY_ATOL,
            )
        support._assert_particle_gas_inventory_conserved(
            captured["initial_masses"],
            captured["particle_concentration"],
            captured["initial_gas"],
            replay["masses"],
            replay["gas"],
        )
        support._assert_contract_allclose(
            replay["energy"],
            replay["scratch_total_mass_transfer"].sum(axis=1)
            * captured["latent_heat_values"][None, :],
        )
    support._assert_particle_gas_inventory_conserved(
        normal["initial_masses"],
        normal["particle_concentration"],
        normal["initial_gas"],
        normal_result["masses"],
        normal_result["gas"],
    )
    support._assert_contract_allclose(
        normal_result["energy"],
        normal_result["scratch_total_mass_transfer"].sum(axis=1)
        * normal["latent_heat_values"][None, :],
    )


def test_graph_capture_state_builds_complete_fp64_sidecars() -> None:
    """The shared fixture builder produces complete caller-owned sidecars."""
    runtime = support._load_warp_runtime()
    state = _make_graph_capture_state(runtime, "cpu")
    _assert_sidecar_contract(state)


@pytest.mark.gpu_parity
def test_condensation_graph_capture_is_unsupported_on_warp_cpu() -> None:
    """Warp CPU explicitly skips unsupported graph capture for this public step."""
    _assert_graph_replay("cpu")


@pytest.mark.cuda
def test_condensation_capture_host_readback_is_unsupported_on_cuda() -> None:
    """Strictly xfail only the documented public-step capture readback."""
    runtime = support._load_warp_runtime()
    if not runtime.cuda_available(runtime.wp):
        pytest.skip(runtime.CUDA_SKIP_REASON)
    normal = _make_graph_capture_state(runtime, "cuda")
    _call_condensation(runtime, normal)

    captured = _make_graph_capture_state(runtime, "cuda")
    try:
        _capture_graph_or_skip(
            runtime,
            "cuda",
            lambda: _call_condensation(runtime, captured),
        )
    except ExceptionGroup as errors:
        # The first exception is the in-capture D2H validation readback.  Warp
        # then reports that same invalid capture when teardown consumes it.
        assert len(errors.exceptions) == 2
        assert _capture_host_readback_error(errors.exceptions[0])
        assert "cuda graph capture failed" in str(errors.exceptions[1]).lower()
    except RuntimeError as error:
        assert _capture_host_readback_error(error)
    else:
        pytest.fail("CUDA capture unexpectedly supported public-step readback")

    pytest.xfail(
        "The public step performs host validation readbacks, which CUDA graph "
        "capture forbids."
    )
