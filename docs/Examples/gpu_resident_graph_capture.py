"""Demonstrate qualified native-CUDA resident graph capture.

Lazy qualification runs before CPU-state construction, resident imports, or
device allocation. This fixed-identity example explicitly captures once,
replays twice, invalidates, retires, and renews a graph. CPU and Warp-CPU are
not fallback or emulation paths. It provides no automatic recapture, migration,
resize/compaction, hidden transfer or synchronization, retry/rollback,
checkpointed or serialized opaque handles, or performance claims.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
from particula.gas import EnvironmentData, GasData
from particula.particles import ParticleData

_FORCE_NO_NATIVE_CAPTURE_ENV = "PARTICULA_EXAMPLE_FORCE_NO_NATIVE_CAPTURE"
_UNAVAILABLE_OUTPUT = (
    "Canonical path: docs/Examples/gpu_resident_graph_capture.py",
    "Native CUDA graph capture is unavailable or disabled.",
    "No CPU or Warp-CPU fallback ran; no fixture, upload, or capture ran.",
)


@dataclass
class ExampleRun:
    """Retain bounded, address-free observations from the walkthrough.

    Attributes:
        output: Deterministic status lines produced by the example.
        session: Resident session created by the enabled execution path.
        registry: Resource registry bound to ``session``.
        guard: Step guard bound to ``session`` and ``registry``.
        binding: Graph-capture binding used for the execution.
        captured: Retired graph-capture record from the first capture.
        renewed_capture: Fresh graph-capture record from renewal.
        gas_snapshot: Synchronized gas diagnostic observation, if enabled.
        saturation_snapshot: Synchronized saturation observation, if enabled.
        replay_count: Number of replays performed before retirement.
        invalidated: Whether structural drift invalidated the first capture.
        retired: Whether the first capture was explicitly retired.
        renewed: Whether a fresh lifecycle was created after retirement.
        synchronized: Whether the enabled path synchronized before host reads.
    """

    output: list[str]
    session: Any | None = None
    registry: Any | None = None
    guard: Any | None = None
    binding: Any | None = None
    captured: Any | None = None
    renewed_capture: Any | None = None
    gas_snapshot: np.ndarray | None = None
    saturation_snapshot: np.ndarray | None = None
    replay_count: int = 0
    invalidated: bool = False
    retired: bool = False
    renewed: bool = False
    synchronized: bool = False


def _disabled_output() -> list[str]:
    """Return deterministic output for intentionally unavailable execution.

    Returns:
        The fixed status lines used when native CUDA capture is unavailable.
    """
    return list(_UNAVAILABLE_OUTPUT)


def _qualified_native_cuda() -> str | None:
    """Find the first capture-capable native CUDA device without setup work.

    Only a missing Warp module, no native CUDA device, or missing capture API is
    treated as unavailable. Other import, loader, or probe failures propagate.

    Returns:
        The selected opaque native device name, or ``None`` when the explicit
        unavailable conditions are met.

    Raises:
        Exception: Propagates unexpected Warp import, device enumeration, or
            capture-API probe failures.
    """
    if os.getenv(_FORCE_NO_NATIVE_CAPTURE_ENV) == "1":
        return None
    try:
        warp = importlib.import_module("warp")
    except ModuleNotFoundError as error:
        if error.name == "warp":
            return None
        raise
    native_devices = warp.get_devices()
    native = next(
        (
            str(device)
            for device in native_devices
            if str(device).startswith("cuda")
        ),
        None,
    )
    if native is None:
        return None
    if not all(
        callable(getattr(warp, name, None))
        for name in ("capture_begin", "capture_end", "capture_launch")
    ):
        return None
    return native


def _load_enabled_runtime() -> SimpleNamespace:
    """Load concrete resident and graph-capture seams after preflight.

    Returns:
        A namespace containing the lazily loaded runtime modules.

    Raises:
        ImportError: If an enabled-path concrete module cannot be imported.
    """
    names = (
        "warp",
        "particula.execution",
        "particula.execution.graph_capture",
        "particula.execution.resident_scheduler",
    )
    loaded = {
        name.rsplit(".", 1)[-1]: importlib.import_module(name) for name in names
    }
    # The maintained resident example owns the identical bounded composition.
    loaded["resident_example"] = importlib.import_module(
        "docs.Examples.gpu_resident_multi_timestep"
    )
    return SimpleNamespace(**loaded)


class _WarpNativeCaptureAdapter:
    """Expose only the selected Warp native capture vocabulary.

    Attributes:
        _warp: Lazily imported Warp module.
        _native: Opaque native device selected during preflight.
        _graph_capture: Concrete graph-capture module providing callables.
    """

    def __init__(self, warp: Any, native: str, graph_capture: Any) -> None:
        """Bind Warp and graph-capture objects for one native device.

        Args:
            warp: Lazily imported Warp module.
            native: Opaque native CUDA device selected during preflight.
            graph_capture: Concrete graph-capture module.
        """
        self._warp = warp
        self._native = native
        self._graph_capture = graph_capture

    def runtime_available(self) -> bool:
        """Return the completed preflight result without another probe.

        Returns:
            ``True`` because construction occurs only after successful
            preflight.
        """
        return True

    def device_available(self, device: Any) -> bool:
        """Accept only the preflight-selected native CUDA device.

        Args:
            device: Candidate graph-capture device.

        Returns:
            Whether ``device`` has the exact preflight-selected native name.
        """
        return device.native == self._native

    def capture_api_available(self, device: Any) -> bool:
        """Accept APIs only for the selected native CUDA device.

        Args:
            device: Candidate graph-capture device.

        Returns:
            Whether ``device`` has the exact preflight-selected native name.
        """
        return device.native == self._native

    def capture_callables(self, device: Any) -> Any:
        """Return direct Warp capture callables and opaque-handle cleanup.

        Args:
            device: Device whose native name is passed to Warp capture.

        Returns:
            Concrete graph-capture callables for the preflight-selected device.
        """

        def begin() -> None:
            """Begin capture on the selected opaque native device."""
            self._warp.capture_begin(
                device=device.native,
                force_module_load=True,
            )

        def release(handle: object) -> None:
            """Release an opaque native handle when it exposes cleanup."""
            destroy = getattr(handle, "destroy", None)
            if callable(destroy):
                destroy()

        def abort() -> object:
            """End incomplete capture and return its cleanup handle."""
            return self._warp.capture_end()

        return self._graph_capture.GraphCaptureNativeCallables(
            begin,
            self._warp.capture_end,
            lambda: None,
            self._warp.capture_launch,
            release,
            abort,
        )


def _build_cpu_state() -> tuple[ParticleData, GasData, EnvironmentData]:
    """Build the fixed float64 resident state for the bounded walkthrough.

    Returns:
        Particle, gas, and environment containers for the resident setup.
    """
    return _load_enabled_runtime().resident_example._build_cpu_state()


def _compose_request(
    runtime: SimpleNamespace,
    session: Any,
    registry: Any,
    guard: Any,
    gas: GasData,
    environment: EnvironmentData,
    initial_total_mass: np.ndarray,
) -> tuple[Any, Any, Any]:
    """Compose the canonical twelve-node request and publish its resources.

    Args:
        runtime: Lazily loaded runtime modules.
        session: Active resident session.
        registry: Resource registry bound to ``session``.
        guard: Closed step guard bound to ``session`` and ``registry``.
        gas: CPU gas container used to construct the request.
        environment: CPU environment container used to construct the request.
        initial_total_mass: CPU-derived particle-plus-gas inventory.

    Returns:
        The resident request and its gas and saturation diagnostic outputs.
    """
    resident_runtime = runtime.resident_example._load_enabled_runtime()
    communication = SimpleNamespace(**vars(resident_runtime.communication))

    def graph_capture_map(
        _form: Any,
        transport_mode: Any,
        edge_capacity: int,
        source_indices: Any,
        destination_indices: Any,
        species_indices: Any,
        edge_rates: Any,
    ) -> Any:
        """Construct the required closed one-dimensional map before
        publication.
        """
        return resident_runtime.communication.CommunicationMap(
            resident_runtime.communication.CommunicationMapForm.ONE_DIMENSIONAL,
            transport_mode,
            edge_capacity,
            source_indices,
            destination_indices,
            species_indices,
            edge_rates,
        )

    communication.CommunicationMap = graph_capture_map
    capture_runtime = SimpleNamespace(
        **vars(resident_runtime), communication=communication
    )
    request, gas_output, saturation_output = runtime.resident_example._request(
        capture_runtime,
        session,
        registry,
        guard,
        1.0,
        gas,
        environment,
        initial_total_mass,
    )
    return request, gas_output, saturation_output


def _close_enabled_binding(
    graph_capture: Any | None,
    binding: Any | None,
    session: Any | None,
    registry: Any | None,
    guard: Any | None,
) -> None:
    """Close capture and session independently, retaining teardown failures."""
    failures: list[BaseException] = []
    if graph_capture is not None and binding is not None:
        try:
            graph_capture.close_resident_graph_capture(binding)
        except BaseException as error:
            failures.append(error)
    if session is not None and registry is not None and guard is not None:
        try:
            session.close(registry, guard)
        except BaseException as error:
            failures.append(error)
    if failures:
        if len(failures) > 1:
            raise failures[0] from failures[1]
        raise failures[0]


def run_example() -> ExampleRun:  # noqa: C901
    """Capture and replay a qualified native CUDA graph without fallback.

    Unavailable native capture returns deterministic status lines without
    fixture construction, resident setup, upload, or capture. The enabled path
    has one host-read boundary: explicit synchronization after two replays.
    It then deliberately invalidates, retires, and renews the fixed binding.

    Returns:
        Bounded status and synchronized observations from the walkthrough.

    Raises:
        Exception: Propagates unexpected qualification, enabled-path setup,
            capture, replay, renewal, or teardown failures after cleanup.
    """
    native = _qualified_native_cuda()
    if native is None:
        return ExampleRun(output=_disabled_output())
    runtime = _load_enabled_runtime()
    execution = runtime.execution
    graph_capture: Any | None = runtime.graph_capture
    device = execution.Device(execution.Backend.WARP, native)
    particles, gas, environment = _build_cpu_state()
    initial_total_mass = particles.volume[:, None] * (
        np.sum(particles.masses * particles.concentration[:, :, None], axis=1)
        + gas.concentration
    )
    session = None
    binding = None
    registry = None
    guard = None
    try:
        resident_runtime = runtime.resident_example._load_enabled_runtime()
        session = resident_runtime.gpu_session.setup_resident_session(
            particles, gas, environment, device
        )
        registry = resident_runtime.gpu_resources.GPUResourceRegistry(session)
        guard = resident_runtime.gpu_session.ResidentStepGuard(
            session, registry
        )
        request, gas_output, saturation_output = _compose_request(
            runtime,
            session,
            registry,
            guard,
            gas,
            environment,
            initial_total_mass,
        )
        signature = graph_capture.create_resident_graph_capture_signature(
            request
        )
        lifecycle = graph_capture.create_graph_capture_lifecycle(
            graph_capture.GraphCaptureCapability(
                device, graph_capture.GraphCaptureAvailability.AVAILABLE
            ),
            signature,
        )
        binding = graph_capture.ResidentGraphCaptureBinding(
            request, session, registry, guard, lifecycle
        )
        graph_capture._attach_resident_graph_capture_binding(request, binding)
        session.initialize_streams(registry, guard)
        prepared = runtime.resident_scheduler.prepare_resident_simulation(
            request, 1.0
        )
        capture_set = registry.validate_capture_resource_set(
            request.capture_resource_requirements
        )
        adapter = _WarpNativeCaptureAdapter(runtime.warp, native, graph_capture)
        qualification = graph_capture.qualify_prepared_resident_graph_capture(
            binding, prepared, capture_set, adapter
        )
        captured = graph_capture.capture_prepared_resident_graph(qualification)
        for _ in range(2):
            graph_capture.replay_captured_resident_graph(captured, 1.0)
        runtime.warp.synchronize_device(session.particles.masses.device)
        gas_snapshot = gas_output.numpy().copy()
        saturation_snapshot = saturation_output.numpy().copy()
        original_update = request.environment_update
        object.__setattr__(request, "environment_update", object())
        try:
            try:
                graph_capture.replay_captured_resident_graph(captured, 1.0)
            except ValueError:
                invalidated = (
                    binding.lifecycle.state
                    is graph_capture.GraphCaptureLifecycleState.INVALIDATED
                )
            else:
                raise RuntimeError("structural signature drift was accepted.")
        finally:
            object.__setattr__(request, "environment_update", original_update)
        if not invalidated:
            raise RuntimeError("structural signature drift did not invalidate.")
        graph_capture.retire_resident_graph_capture(binding)
        renewed_lifecycle = graph_capture.renew_resident_graph_capture(
            binding,
            graph_capture.create_resident_graph_capture_signature(request),
        )
        if renewed_lifecycle is lifecycle:
            raise RuntimeError("renewal reused the retired lifecycle.")
        renewed_prepared = (
            runtime.resident_scheduler.prepare_resident_simulation(request, 1.0)
        )
        renewed_set = registry.validate_capture_resource_set(
            request.capture_resource_requirements
        )
        renewed_qualification = (
            graph_capture.qualify_prepared_resident_graph_capture(
                binding, renewed_prepared, renewed_set, adapter
            )
        )
        renewed_capture = graph_capture.capture_prepared_resident_graph(
            renewed_qualification
        )
        return ExampleRun(
            output=[
                "Qualified native CUDA capture completed without fallback.",
                "Captured once, replayed twice, then explicitly retired and "
                "renewed.",
                "Opaque handles are not serialized; no performance claim is "
                "made.",
            ],
            session=session,
            registry=registry,
            guard=guard,
            binding=binding,
            captured=captured,
            renewed_capture=renewed_capture,
            gas_snapshot=gas_snapshot,
            saturation_snapshot=saturation_snapshot,
            replay_count=2,
            invalidated=True,
            retired=True,
            renewed=True,
            synchronized=True,
        )
    finally:
        _close_enabled_binding(graph_capture, binding, session, registry, guard)


def main() -> None:
    """Run the example and print its deterministic status lines.

    The unavailable path prints the fixed capability status. The enabled path
    prints the bounded native-CUDA capture walkthrough status.
    """
    for line in run_example().output:
        print(line)


if __name__ == "__main__":
    main()
