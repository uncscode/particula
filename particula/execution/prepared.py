"""Owned preparation, execution and observation of native Warp simulations.

Construction is routed through runnable preparation. Primary arrays and process
sidecars remain private and resident until an explicit observation. Warp CPU is
an explicit regression target; requesting CUDA never falls back to the CPU.
"""

from __future__ import annotations

from contextlib import contextmanager
from numbers import Integral
from threading import RLock
from typing import Any, Callable

from particula.aerosol import Aerosol
from particula.execution import Backend, Device


class PreparedSimulation:
    """Own a fixed-layout simulation and its explicit CPU observation boundary.

    Use a runnable's ``prepare`` method to construct this object. ``step`` and
    ``run`` enqueue work without downloading state. ``synchronize`` waits for
    pending work, and ``snapshot`` returns detached CPU data for analysis.
    Closing waits for outstanding work and releases owned storage without an
    automatic download. Instances serialize their own operations.
    """

    _wp: Any
    _device: Device
    _time_step: float
    _lock: Any
    _closed: bool
    _last_steps: int
    _last_time: float
    _operations: tuple[Callable[[], None], ...] | None
    _session: Any
    _registry: Any
    _guard: Any
    _stream: Any
    _observers: list[Callable[[], Any]]
    _plans: tuple[Any, ...]

    def __init__(self) -> None:
        """Reject direct construction without a validated process plan."""
        raise TypeError("Create a PreparedSimulation with runnable.prepare()")

    @classmethod
    def _upload(
        cls,
        aerosol: Aerosol,
        *,
        device: Device,
        time_step: float,
        root_seed: int = 0,
        logical_box_ids: tuple[str, ...] | None = None,
    ) -> PreparedSimulation:
        """Upload preflighted state and own its exact resident binding."""
        from particula._cpu_batch import duration
        from particula.execution.gpu_resources import GPUResourceRegistry
        from particula.execution.gpu_session import (
            ResidentStepGuard,
            _preflight_cpu_session,
            setup_resident_session,
        )

        if type(aerosol) is not Aerosol:
            raise TypeError("prepared execution requires a native Aerosol")
        if type(device) is not Device or device.backend is not Backend.WARP:
            raise ValueError("prepared execution requires Backend.WARP")
        dt = duration(time_step, 1)
        _preflight_cpu_session(
            aerosol.particles, aerosol.gas, aerosol.environment, device
        )
        import warp as wp

        target = wp.get_device(device.native)
        result = object.__new__(cls)
        result._wp = wp
        result._device = Device(Backend.WARP, str(target))
        result._time_step = dt
        result._lock = RLock()
        result._closed = False
        result._last_steps = 0
        result._last_time = 0.0
        result._operations = None
        result._session = None
        result._registry = None
        result._guard = None
        result._observers = []
        result._plans = ()
        result._stream = wp.Stream(target) if target.is_cuda else None
        try:
            with result._scope():
                result._session = setup_resident_session(
                    aerosol.particles,
                    aerosol.gas,
                    aerosol.environment,
                    result._device,
                    root_seed=root_seed,
                    logical_box_ids=logical_box_ids,
                )
                result._registry = GPUResourceRegistry(result._session)
                result._guard = ResidentStepGuard(
                    result._session, result._registry
                )
        except BaseException:
            # Uploaded host buffers and native arrays must outlive queued work
            # even when construction fails before a guard can be published.
            result._wait()
            raise
        return result

    @contextmanager
    def _scope(self):
        """Select this owner's device/stream without inter-stream events."""
        with self._wp.ScopedDevice(self._device.native):
            with self._wp.ScopedStream(self._stream, sync_enter=False):
                yield

    def _require_active(self) -> None:
        """Reject terminal state before observing or enqueueing anything."""
        from particula.execution.gpu_session import ResidentLifecycle

        if self._closed:
            raise RuntimeError("prepared simulation is closed")
        if self._session.lifecycle is not ResidentLifecycle.ACTIVE:
            raise RuntimeError("prepared simulation is faulted")
        self._guard.assert_step_closed()

    def _wait(self) -> None:
        """Wait only for this simulation's CUDA stream (CPU work is eager)."""
        if self._stream is not None:
            self._wp.synchronize_stream(self._stream)

    def _fault(self) -> None:
        """Preserve resident fault and graph invalidation semantics."""
        from particula.execution.gpu_session import (
            _fault_resident_session_with_context,
        )

        _fault_resident_session_with_context(
            self._session, self._registry, self._guard
        )

    @property
    def device(self) -> Device:
        """Return the explicit backend and canonical native device."""
        return self._device

    @property
    def time_step(self) -> float:
        """Return the frozen duration of one process sequence, in seconds."""
        return self._time_step

    @property
    def completed_steps(self) -> int:
        """Return enqueued steps; synchronization establishes device success."""
        if self._closed:
            return self._last_steps
        return self._guard.completed_steps

    @property
    def simulated_time(self) -> float:
        """Return enqueued simulated time in seconds."""
        if self._closed:
            return self._last_time
        return float(self._guard.simulated_time)

    @property
    def closed(self) -> bool:
        """Report whether owned resources have been released."""
        return self._closed

    def step(self) -> None:
        """Enqueue one frozen process sequence without a host observation.

        A launch failure faults the binding. Earlier writes are not rolled back.
        Asynchronous errors surface at a later synchronization or snapshot.
        """
        with self._lock:
            self._require_active()
            if self._operations is None:
                raise RuntimeError("prepared process plan has not been bound")
            token = self._guard.begin_step(self._time_step)
            try:
                with self._scope():
                    for operation in self._operations:
                        operation()
                self._guard.complete_step(token)
            except BaseException:
                try:
                    self._guard._abort_step(token)
                finally:
                    self._fault()
                raise

    def run(self, steps: int) -> None:
        """Enqueue a nonnegative integer number of fixed-duration steps."""
        if isinstance(steps, bool) or not isinstance(steps, Integral):
            raise TypeError("steps must be a non-boolean integer")
        if steps < 0:
            raise ValueError("steps must be nonnegative")
        with self._lock:
            self._require_active()
            for _ in range(steps):
                self.step()

    def synchronize(self) -> None:
        """Wait for pending work and surface asynchronous execution failures."""
        with self._lock:
            self._require_active()
            try:
                self._wait()
                with self._scope():
                    for observe in self._observers:
                        observe()
            except BaseException:
                self._fault()
                raise

    def snapshot(self) -> Aerosol:
        """Download detached native CPU state, preserving names and measures.

        Snapshot arrays never alias initial state, resident state, or earlier
        snapshots, including when explicitly targeting Warp CPU. Derived vapor
        pressure and random streams belong to execution/checkpoint state rather
        than the CPU aerosol schema.
        """
        from particula.gpu.conversion import (
            from_warp_environment_data,
            from_warp_gas_data,
            from_warp_particle_data,
        )

        with self._lock:
            self.synchronize()
            with self._scope():
                return Aerosol(
                    particles=from_warp_particle_data(
                        self._session.particles, sync=False
                    ).copy(),
                    gas=from_warp_gas_data(
                        self._session.gas,
                        name=list(self._session.metadata.gas_names),
                        sync=False,
                    ).copy(),
                    environment=from_warp_environment_data(
                        self._session.environment, sync=False
                    ).copy(),
                )

    def close(self) -> None:
        """Wait and release owned storage without downloading a final state."""
        with self._lock:
            if self._closed:
                return
            self._guard.assert_step_closed()
            try:
                self._wait()
            finally:
                try:
                    self._session.close(self._registry, self._guard)
                finally:
                    self._last_steps = self._guard.completed_steps
                    self._last_time = float(self._guard.simulated_time)
                    self._closed = True
                    self._operations = None
                    self._plans = ()
                    self._observers = []
                    self._guard = self._registry = self._session = None
                    self._stream = None

    def __enter__(self) -> PreparedSimulation:
        """Enter an active simulation's explicit ownership scope."""
        self._require_active()
        return self

    def __exit__(self, *_exc: object) -> None:
        """Release resources on normal exit and exceptions."""
        self.close()


def prepare_simulation(
    processes: tuple[Any, ...],
    aerosol: Aerosol,
    *,
    backend: Backend = Backend.WARP,
    device: str = "cuda:0",
    time_step: float,
    root_seed: int = 0,
    logical_box_ids: tuple[str, ...] | None = None,
) -> PreparedSimulation:
    """Validate the ordered process plan before uploading native state."""
    from particula._cpu_batch import duration
    from particula.execution.gpu_session import _preflight_cpu_session
    from particula.execution.prepared_processes import (
        _compile_process,
        _plan_process,
    )

    if backend is not Backend.WARP:
        raise ValueError("prepared execution requires Backend.WARP")
    target = Device(backend, device)
    dt = duration(time_step, 1)
    if type(aerosol) is not Aerosol:
        raise TypeError("prepared execution requires a native Aerosol")
    _preflight_cpu_session(
        aerosol.particles, aerosol.gas, aerosol.environment, target
    )
    if not processes:
        raise ValueError("prepared execution requires at least one process")
    plans = tuple(_plan_process(process, aerosol) for process in processes)
    if len({plan.family for plan in plans}) != len(plans):
        raise ValueError(
            "prepared execution supports one occurrence per family"
        )
    owner = PreparedSimulation._upload(
        aerosol,
        device=target,
        time_step=dt,
        root_seed=root_seed,
        logical_box_ids=logical_box_ids,
    )
    try:
        with owner._scope():
            owner._operations = tuple(
                _compile_process(plan, owner) for plan in plans
            )
            owner._plans = plans
    except BaseException:
        owner.close()
        raise
    return owner
