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
from particula.execution.prepared_types import (
    SimulationCheckpoint,
    SimulationDiagnostics,
)


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
    _diagnostics: Callable[[], Any] | None
    _refresh: Callable[[], None] | None
    _device_failure: Any
    _bindings: dict[str, Any]
    _observers: list[Callable[[], Any]]
    _plans: tuple[Any, ...]
    _auxiliary: dict[str, Any]
    _capture: Any

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
        result = cls._empty(device, dt)
        wp = result._wp
        target = result._device.native
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
                result._device_failure = wp.zeros(
                    1, dtype=wp.int32, device=target
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

    @classmethod
    def _empty(cls, device: Device, time_step: float) -> PreparedSimulation:
        """Create an owner and stream before either upload or exact restart."""
        import warp as wp

        target = wp.get_device(device.native)
        result = object.__new__(cls)
        result._wp = wp
        result._device = Device(Backend.WARP, str(target))
        result._time_step = time_step
        result._lock = RLock()
        result._closed = False
        result._last_steps = 0
        result._last_time = 0.0
        result._operations = None
        result._session = None
        result._registry = None
        result._guard = None
        result._diagnostics = None
        result._refresh = None
        result._device_failure = None
        result._bindings = {}
        result._observers = []
        result._plans = ()
        result._auxiliary = {}
        result._capture = None
        result._stream = wp.Stream(target) if target.is_cuda else None
        return result

    def _bind(self, plans: tuple[Any, ...]) -> None:
        """Compile a validated process plan against the owned resident state."""
        from particula.execution.prepared_cohort import wrap_cohort_operations
        from particula.execution.prepared_diagnostics import setup_diagnostics
        from particula.execution.prepared_processes import _compile_process
        from particula.execution.prepared_updates import compose_operations

        with self._scope():
            self._plans = plans
            operations = tuple(_compile_process(plan, self) for plan in plans)
            operations = wrap_cohort_operations(self, plans, operations)
            self._operations = compose_operations(self, plans, operations)
            self._diagnostics = setup_diagnostics(self)
            self._plans = plans

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

        # Revoke first, but retain the graph until outstanding work completes.
        if self._capture is not None:
            self._capture.invalidate()
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
        with self._lock:
            if self._closed:
                return self._last_steps
            return self._guard.completed_steps

    @property
    def simulated_time(self) -> float:
        """Return enqueued simulated time in seconds."""
        with self._lock:
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
                    if self._capture is None:
                        for operation in self._operations:
                            operation()
                    else:
                        from particula.execution.prepared_capture import replay

                        replay(self)
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
                    failure = int(self._device_failure.numpy()[0])
                    messages = {
                        2: (
                            "Brownian GPU coagulation requires "
                            "equal active counts"
                        ),
                        3: "ion source capacity exhausted",
                        4: "ion source Poisson mean exceeds supported range",
                        5: "ion cleanup requires unit active counts",
                        6: "seed cohort lineage invariant violated",
                        7: (
                            "atmospheric coagulation requires valid unit ions "
                            "and a supported model environment"
                        ),
                        8: "ion attachment rate is unsupported",
                    }
                    if failure in messages:
                        raise ValueError(messages[failure])
                    for observe in self._observers:
                        observe()
                    if failure:
                        raise ValueError(
                            "prepared nucleation failed; dependent work stopped"
                        )
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

    def diagnostics(self) -> SimulationDiagnostics:
        """Observe detached box/species reductions without downloading slots.

        Returns:
            SimulationDiagnostics with number, charge and mass concentrations,
            active-slot counts, forcing, and optional last-step latent energy.
        """
        with self._lock:
            self.synchronize()
            if self._diagnostics is None:
                raise RuntimeError("diagnostics are not prepared")
            try:
                with self._scope():
                    return self._diagnostics()
            except BaseException:
                self._fault()
                raise

    @property
    def gas_names(self) -> tuple[str, ...]:
        """Return the fixed full-gas order for updates and analysis."""
        with self._lock:
            self._require_active()
            return self._session.metadata.gas_names

    def update_environment(
        self, *, temperature=None, pressure=None, saturation_ratio=None
    ) -> None:
        """Replace specified environment fields between run segments.

        Temperature [K] and pressure [Pa] accept scalars or per-box vectors.
        Saturation accepts a scalar, full-gas vector or box-by-gas matrix; it
        is externally prescribed when no condensation model is present. With
        condensation, saturation is derived and cannot be prescribed here.
        This explicit forcing boundary waits for prior work and its uploads.
        Invalid values reject before any resident field is changed.
        """
        from particula.execution.prepared_updates import (
            apply_forcing,
            forcing_array,
        )

        with self._lock:
            self._require_active()
            if saturation_ratio is not None and self._refresh is not None:
                raise ValueError("condensation owns derived saturation_ratio")
            if all(
                value is None
                for value in (temperature, pressure, saturation_ratio)
            ):
                raise ValueError("supply at least one environment field")
            environment = self._session.environment
            updates = []
            for name, value, positive in (
                ("temperature", temperature, True),
                ("pressure", pressure, True),
                ("saturation_ratio", saturation_ratio, False),
            ):
                if value is not None:
                    target = getattr(environment, name)
                    values = forcing_array(
                        value, target.shape, name, positive=positive
                    )
                    updates.append((target, values))
            apply_forcing(self, tuple(updates))

    def update_gas(self, concentration) -> None:
        """Replace full-gas concentration [kg/m³] between run segments.

        Accept a full-gas vector or box-by-gas matrix in ``gas_names`` order.
        This explicit boundary waits for pending work, uploads owned values,
        and refreshes derived saturation when condensation supplies its law.
        It leaves particle state and process configuration unchanged.
        """
        from particula.execution.prepared_updates import (
            apply_forcing,
            forcing_array,
        )

        with self._lock:
            self._require_active()
            target = self._session.gas.concentration
            values = forcing_array(
                concentration, target.shape, "concentration", positive=False
            )
            apply_forcing(self, ((target, values),))

    def ion_source_diagnostics(self):
        """Return cumulative injected pairs and mass [kg] for each box.

        This explicit observation synchronizes and returns detached arrays.
        Mass is extensive, in the native particle species order.
        """
        import numpy as np

        with self._lock:
            self.synchronize()
            source = next(
                (p for p in self._plans if p.family == "ion_source"), None
            )
            if source is None:
                raise ValueError("simulation has no ion source")
            pairs = self._auxiliary["ion_source_pairs"].numpy().copy()
            mass = np.zeros((pairs.size, self._session.dimensions.n_species))
            for polarity in ("positive", "negative"):
                mass[:, source.values[polarity + "_species"]] = (
                    pairs * source.values[polarity + "_mass"]
                )
            return {"injected_pairs": pairs, "injected_mass": mass}

    def ion_cleanup_diagnostics(self):
        """Observe cumulative exported particles and mass [kg, box by species].

        Exported charge is exactly zero by construction. Returned arrays are
        detached; this explicit observation synchronizes the prepared owner.
        """
        with self._lock:
            self.synchronize()
            if "ion_export_mass" not in self._auxiliary:
                raise ValueError("simulation has no ion cleanup")
            return {
                "exported_particles": self._auxiliary["ion_export_particles"]
                .numpy()
                .copy(),
                "exported_mass": self._auxiliary["ion_export_mass"]
                .numpy()
                .copy(),
            }

    def cohort_diagnostics(self):
        """Observe detached per-box original-lineage outcome counts.

        Counts partition initial seeds into reached, captured and unresolved.
        Unresolved lineages are censored at the chosen observation end time.
        These are lineage counts, not the number of surviving particles.
        """
        with self._lock:
            self.synchronize()
            if "cohort_statistics" not in self._auxiliary:
                raise ValueError("simulation has no seed cohort")
            values = self._auxiliary["cohort_statistics"].numpy()
            return {
                name: values[:, column].copy()
                for column, name in enumerate(
                    ("initial", "reached", "captured", "unresolved")
                )
            }

    def checkpoint(self) -> SimulationCheckpoint:
        """Observe and save exact continuation bytes plus frozen configuration.

        Unlike a CPU snapshot, this in-memory checkpoint preserves random
        streams, GPU-only fields and step counters. Resume on the same device
        through the original runnable or an equivalently configured sequence.
        """
        from particula.execution.prepared_checkpoint import checkpoint

        with self._lock:
            return checkpoint(self)

    @property
    def captured(self) -> bool:
        """Report whether subsequent steps use this owner's live CUDA graph."""
        with self._lock:
            return self._capture is not None and self._capture.valid

    def capture(self) -> None:
        """Record one CUDA timestep without executing it or advancing time.

        This explicit setup boundary synchronizes prior work. Later step/run
        calls replay the owned graph. Shape, process order, duration and buffer
        identities stay fixed; forcing updates may change values in place.
        Warp CPU rejects. Warp may instantiate its executable on first replay;
        performance measurements should warm up that replay separately.
        """
        from particula.execution.prepared_capture import capture

        with self._lock:
            capture(self)

    def uncapture(self) -> None:
        """Wait, release the graph and return to ordinary prepared execution."""
        with self._lock:
            self.synchronize()
            self._capture = None

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
                    self._capture = None
                    self._operations = None
                    self._plans = ()
                    self._auxiliary = {}
                    self._observers = []
                    self._diagnostics = None
                    self._refresh = None
                    self._device_failure = None
                    self._bindings = {}
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
    from particula.execution.prepared_cohort import validate_cohort_plans
    from particula.execution.prepared_ions import validate_ion_plans
    from particula.execution.prepared_processes import _plan_process

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
    validate_cohort_plans(plans)
    validate_ion_plans(plans)
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
        owner._bind(plans)
    except BaseException:
        owner.close()
        raise
    return owner
