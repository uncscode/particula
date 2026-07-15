"""Graph-capture readiness tests for the public condensation GPU step."""

from __future__ import annotations

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


@pytest.fixture(autouse=True)
def _selected_warp_test_runtime(request: pytest.FixtureRequest) -> None:
    """Load Warp only while executing a selected Warp-backed test."""
    if request.node.get_closest_marker("warp") is not None:
        support._load_warp_runtime()


def _make_graph_capture_state(
    runtime: object,
    device: str,
) -> dict[str, object]:
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


def _assert_sidecar_contract(state: dict[str, object]) -> None:
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


def _reset_mutable_state(runtime: object, state: dict[str, object]) -> None:
    """Restore mutable test state through device-to-device copies only."""
    wp = runtime.wp
    reset_sources = state["reset_sources"]
    wp.copy(state["particles"].masses, reset_sources["masses"])
    wp.copy(state["gas"].concentration, reset_sources["gas"])
    wp.copy(state["gas"].vapor_pressure, reset_sources["vapor_pressure"])
    wp.copy(state["energy"], reset_sources["energy"])


def _capture_support_error(error: Exception) -> bool:
    """Return whether an exception describes unavailable capture support."""
    if not isinstance(error, (RuntimeError, NotImplementedError)):
        return False
    message = str(error).lower()
    return any(term in message for term in ("captur", "graph", "not supported"))


def _end_capture_best_effort(wp: object) -> bool:
    """End an active capture, returning whether the cleanup succeeded."""
    try:
        wp.capture_end()
    except Exception:
        return False
    return True


def _require_capture_apis(wp: object, device: str) -> None:
    """Skip when a device cannot provide the public graph-capture APIs."""
    for operation in ("capture_begin", "capture_end", "capture_launch"):
        if not callable(getattr(wp, operation, None)):
            pytest.skip(f"{device}: {operation} is unavailable")
    if not wp.get_device(device).is_cuda:
        pytest.skip(f"{device}: capture_begin unavailable: requires CUDA")


def _capture_graph_or_skip(
    runtime: object,
    device: str,
    call: object,
) -> tuple[object, object]:
    """Record ``call`` or skip only when this device lacks graph capture."""
    wp = runtime.wp
    _require_capture_apis(wp, device)
    began = False
    result = None
    try:
        wp.capture_begin(device=device, force_module_load=True)
        began = True
    except Exception as error:
        if _capture_support_error(error):
            pytest.skip(f"{device}: capture_begin unavailable: {error}")
        raise
    try:
        result = call()
    except Exception as error:
        if _capture_support_error(error):
            began = not _end_capture_best_effort(wp)
            pytest.skip(f"{device}: capture_recording unavailable: {error}")
        raise
    try:
        graph = wp.capture_end()
        began = False
    except Exception as error:
        if _capture_support_error(error):
            pytest.skip(f"{device}: capture_end unavailable: {error}")
        raise
    finally:
        if began:
            _end_capture_best_effort(wp)
    return graph, result


def _launch_graph_or_skip(runtime: object, device: str, graph: object) -> None:
    """Launch a captured graph or skip only for launch capability failures."""
    try:
        runtime.wp.capture_launch(graph)
    except Exception as error:
        if _capture_support_error(error):
            pytest.skip(f"{device}: capture_launch unavailable: {error}")
        raise


def _snapshot_state(state: dict[str, object]) -> dict[str, np.ndarray]:
    """Synchronize and snapshot mutable outputs outside a capture region."""
    support.wp.synchronize()
    return {
        "masses": state["particles"].masses.numpy().copy(),
        "gas": state["gas"].concentration.numpy().copy(),
        "transfer": state["scratch"].total_mass_transfer.numpy().copy(),
        "energy": state["energy"].numpy().copy(),
    }


def _call_condensation(
    runtime: object,
    state: dict[str, object],
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
    """Compare independent normal and twice-replayed captured public calls."""
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
        for name in ("masses", "gas", "transfer", "energy"):
            npt.assert_allclose(
                replay[name],
                normal_result[name],
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
            replay["transfer"].sum(axis=1)
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
        normal_result["transfer"].sum(axis=1)
        * normal["latent_heat_values"][None, :],
    )


@pytest.mark.gpu_parity
def test_condensation_graph_replay_warp_cpu() -> None:
    """Warp CPU graph replay retains public condensation state contracts."""
    _assert_graph_replay("cpu")


@pytest.mark.cuda
@pytest.mark.gpu_parity
def test_condensation_graph_replay_cuda() -> None:
    """CUDA graph replay retains public condensation state contracts."""
    runtime = support._load_warp_runtime()
    if not runtime.cuda_available(runtime.wp):
        pytest.skip(runtime.CUDA_SKIP_REASON)
    _assert_graph_replay("cuda")
