"""Run one qualified native-CUDA resident graph-capture walkthrough.

This fixed-identity example is native-CUDA-only: CPU and Warp-CPU are not
fallback or emulation paths.  It demonstrates explicit capture, two replays,
structural invalidation, retirement, and renewal.  It does not provide automatic
recapture, migration, resize/compaction, hidden transfer or synchronization,
retry/rollback, checkpointed or serialized opaque handles, or performance
claims.
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
    """Retain bounded address-free observations from the capture walkthrough."""

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
    """Return deterministic output for intentional unavailable execution."""
    return list(_UNAVAILABLE_OUTPUT)


def _qualified_native_cuda() -> str | None:
    """Return the first capture-capable native CUDA device without setup work."""
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
    """Load only concrete resident and graph-capture seams after preflight."""
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
    """Expose only the selected Warp native capture vocabulary."""

    def __init__(self, warp: Any, native: str, graph_capture: Any) -> None:
        self._warp = warp
        self._native = native
        self._graph_capture = graph_capture

    def runtime_available(self) -> bool:
        """Return the completed preflight result without another probe."""
        return True

    def device_available(self, device: Any) -> bool:
        """Accept only the preflight-selected native CUDA device."""
        return device.native == self._native

    def capture_api_available(self, device: Any) -> bool:
        """Accept APIs only for the preflight-selected native CUDA device."""
        return device.native == self._native

    def capture_callables(self, device: Any) -> Any:
        """Return direct Warp capture callables and exact opaque-handle cleanup."""

        def begin() -> None:
            self._warp.capture_begin(
                device=device.native,
                force_module_load=True,
            )

        def release(handle: object) -> None:
            destroy = getattr(handle, "destroy", None)
            if callable(destroy):
                destroy()

        return self._graph_capture.GraphCaptureNativeCallables(
            begin,
            self._warp.capture_end,
            lambda: None,
            self._warp.capture_launch,
            release,
        )


def _build_cpu_state() -> tuple[ParticleData, GasData, EnvironmentData]:
    """Build the fixed float64 resident state used for the bounded walkthrough."""
    return _load_enabled_runtime().resident_example._build_cpu_state()


def _compose_request(
    runtime: SimpleNamespace,
    session: Any,
    registry: Any,
    guard: Any,
    gas: GasData,
    environment: EnvironmentData,
) -> tuple[Any, Any, Any]:
    """Compose the canonical twelve-node request and publish its resources."""
    particles = session.particles
    initial_total_mass = particles.volume.numpy()[:, None] * (
        np.sum(
            particles.masses.numpy()
            * particles.concentration.numpy()[:, :, None],
            axis=1,
        )
        + session.gas.concentration.numpy()
    )
    resident_runtime = runtime.resident_example._load_enabled_runtime()
    request, gas_output, saturation_output = runtime.resident_example._request(
        resident_runtime,
        session,
        registry,
        guard,
        1.0,
        gas,
        environment,
        initial_total_mass,
    )
    # Resident capture accepts the fixed closed one-dimensional map form.
    object.__setattr__(
        request.communication.resources.configuration.communication_map,
        "form",
        resident_runtime.communication.CommunicationMapForm.ONE_DIMENSIONAL,
    )
    return request, gas_output, saturation_output


def run_example() -> ExampleRun:  # noqa: C901
    """Capture and replay a qualified native CUDA graph without fallback.

    The only host read boundary is one explicit synchronization after the two
    replay calls.  Enabled-path failures propagate after exact teardown.
    """
    native = _qualified_native_cuda()
    if native is None:
        return ExampleRun(output=_disabled_output())
    runtime = _load_enabled_runtime()
    execution = runtime.execution
    graph_capture = runtime.graph_capture
    device = execution.Device(execution.Backend.WARP, native)
    particles, gas, environment = _build_cpu_state()
    session = None
    binding = None
    registry = None
    guard = None
    try:
        session = runtime.resident_example._load_enabled_runtime().gpu_session.setup_resident_session(
            particles, gas, environment, device
        )
        registry = runtime.resident_example._load_enabled_runtime().gpu_resources.GPUResourceRegistry(
            session
        )
        guard = runtime.resident_example._load_enabled_runtime().gpu_session.ResidentStepGuard(
            session, registry
        )
        request, gas_output, saturation_output = _compose_request(
            runtime, session, registry, guard, gas, environment
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
                runtime.resident_scheduler.prepare_resident_simulation(
                    request, 1.0
                )
            except ValueError:
                invalidated = True
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
                "Captured once, replayed twice, then explicitly retired and renewed.",
                "Opaque handles are not serialized; no performance claim is made.",
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
        if binding is not None:
            graph_capture.close_resident_graph_capture(binding)
        if session is not None and registry is not None and guard is not None:
            session.close(registry, guard)


def main() -> None:
    """Run the example and print deterministic status lines."""
    for line in run_example().output:
        print(line)


if __name__ == "__main__":
    main()
