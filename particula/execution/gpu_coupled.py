"""Adaptive GPU condensation with a fixed prepared event grid.

The default explicitly transfers state between stages. With ``resident=True``,
particle/gas state stays on CUDA and native conditional graphs own every
adaptive decision. Both modes retain prepared event buffers and RNG streams;
the host orchestrates fixed physical boundaries and requested observations.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from numbers import Integral

import numpy as np
import warp as wp

from particula import _cpu_batch
from particula.dynamics import (
    Coagulation,
    Dilution,
    Nucleation,
    WallLoss,
)
from particula.dynamics.condensation.condensation_strategies import (
    CondensationIsothermal,
)
from particula.execution._adaptive_condensation_plan import (
    plan_adaptive_condensation,
)
from particula.execution.gpu_condensation_ode import GPUCondensationODE
from particula.execution.ode import AdaptiveSolver, StepStatistics
from particula.execution.prepared import prepare_simulation
from particula.execution.prepared_processes import _plan_process
from particula.execution.prepared_updates import apply_forcing
from particula.gas.vapor_pressure_strategies import WaterBuckStrategy
from particula.util.constants import GAS_CONSTANT


@dataclass(frozen=True)
class GPUCondensationControls:
    """Scalar adaptive controls, reset at every complete event interval.

    Particle absolute tolerance is kg per representative; gas absolute
    tolerance is kg/m³. Step durations are seconds. Error control covers
    condensation only, not error from the enclosing sequential composition.
    """

    method: str = "ros3"
    rtol: float = 1e-5
    particle_atol: float = 1e-25
    gas_atol: float = 1e-12
    first_step: float = 1e-6
    min_step: float = 1e-14
    max_step: float = 1e3
    max_trials: int = 100

    def __post_init__(self):
        """Reject invalid controls before physical state is allocated."""
        if self.method not in ("rk23", "ros3"):
            raise ValueError("coupled condensation supports rk23 or ros3")
        for name in (
            "rtol",
            "particle_atol",
            "gas_atol",
            "first_step",
            "min_step",
            "max_step",
        ):
            value = getattr(self, name)
            if type(value) not in (int, float):
                raise TypeError(f"{name} must be a scalar")
            AdaptiveSolver._scalar(value, name, positive=name != "rtol")
        if not self.min_step <= self.first_step <= self.max_step:
            raise ValueError("require min_step <= first_step <= max_step")
        if type(self.max_trials) is not int or self.max_trials < 1:
            raise ValueError("max_trials must be a positive integer")


def _validate_events(processes, aerosol):
    """Exclude derived-field consumers and unsupported custom ownership."""
    if type(processes) is not tuple or not processes:
        raise TypeError("events must be a nonempty tuple of native runnables")
    if any(
        type(process) not in (Coagulation, Dilution, Nucleation, WallLoss)
        for process in processes
    ):
        raise ValueError("unsupported coupled event family")
    plans = tuple(_plan_process(process, aerosol) for process in processes)
    for plan in plans:
        if plan.family not in {
            "nucleation",
            "coagulation",
            "wall_loss",
            "dilution",
        }:
            raise ValueError("unsupported coupled event family")
        if (
            plan.family == "nucleation"
            and dict(plan.values["config"])["saturation_lower"] is not None
        ):
            raise ValueError("coupled events cannot consume derived saturation")
    if aerosol.fields.specs:
        raise ValueError(
            "coupled condensation does not support extension fields"
        )


def _condensation_profile(aerosol, strategy, method):
    """Validate the shared GPU profile, including empty boundary populations."""
    _cpu_batch.validate(aerosol)
    if type(strategy) is not CondensationIsothermal:
        raise TypeError("GPU coupled rates require isothermal condensation")
    if aerosol.particles.distribution_type not in (
        "discrete",
        "particle_resolved",
    ):
        raise ValueError("GPU coupled rates require discrete or resolved state")
    plan = plan_adaptive_condensation(strategy, aerosol)
    return plan


class GPUCoupledSimulation:
    """Advance one fixed event interval and one adaptive condensation flow.

    ``order`` is ``events-first`` or ``condensation-first``. Each event runs
    once per cycle, regardless of rejected condensation trials. Temperature and
    gas updates are explicit between-cycle boundaries. An error after any
    physical work faults the owner; whole-cycle rollback is not promised.
    Observations/checkpoints are available only between complete cycles.

    Args:
        aerosol: Native state, detached during setup.
        events: Nonempty tuple of supported native event runnables.
        strategy: Supported isothermal condensation strategy.
        device: Explicit Warp device, with no automatic fallback.
        time_step: Frozen event interval in seconds.
        order: Sequential ordering of events and condensation.
        controls: Scalar condensation error and timestep controls.
        root_seed: Seed for persistent prepared event random streams.
        logical_box_ids: Optional stable identifiers for those streams.
        time: Initial physical time in seconds.
        resident: Keep stage handoff and adaptive decisions on CUDA.
    """

    def __init__(
        self,
        aerosol,
        events,
        strategy,
        *,
        device,
        time_step,
        order="events-first",
        controls=None,
        root_seed=0,
        logical_box_ids=None,
        time=0.0,
        resident=False,
    ):
        """Validate both components and allocate persistent event resources."""
        if type(resident) is not bool:
            raise TypeError("resident must be a bool")
        if resident and (
            not wp.get_device(device).is_cuda
            or not wp.is_conditional_graph_supported()
        ):
            raise ValueError(
                "resident control requires native CUDA conditional graphs"
            )
        _validate_events(events, aerosol)
        self._configure(
            strategy,
            order,
            GPUCondensationControls() if controls is None else controls,
        )
        plan = _condensation_profile(
            aerosol, self._strategy, self.controls.method
        )
        self._vapor = plan.values["vapor"]
        self._molar_mass = aerosol.gas.molar_mass.copy()
        if wp.get_device(device).is_capturing:
            raise RuntimeError("coupled setup cannot run during CUDA capture")
        self._events = prepare_simulation(
            events,
            aerosol,
            device=device,
            time_step=time_step,
            root_seed=root_seed,
            logical_box_ids=logical_box_ids,
            time=time,
        )
        self._initialize_counters()
        try:
            self._initialize_condensation(aerosol, resident)
        except BaseException:
            self._events.close()
            raise

    def _initialize_condensation(self, aerosol, resident):
        """Allocate one resident adaptive owner when explicitly requested."""
        if type(resident) is not bool:
            raise TypeError("resident must be a bool")
        self.resident = resident
        self._resident_flow = None
        if resident:
            from particula.execution.gpu_resident_condensation import (
                GPUResidentCondensation,
            )

            self._resident_flow = GPUResidentCondensation(
                aerosol,
                self._strategy,
                device=self.device.native,
                time=self.time,
                **asdict(self.controls),
            )

    def _configure(self, strategy, order, controls):
        if order not in ("events-first", "condensation-first"):
            raise ValueError("order must be events-first or condensation-first")
        if type(controls) is not GPUCondensationControls:
            raise TypeError("controls must be GPUCondensationControls")
        controls.__post_init__()
        self._strategy = deepcopy(strategy)
        self._order = order
        self._controls = controls

    @property
    def order(self):
        """Return the frozen sequential composition order."""
        return self._order

    @property
    def controls(self):
        """Return immutable condensation controls."""
        return self._controls

    def _initialize_counters(self):
        self._time = self._events.current_time
        self._completed_steps = self._events.completed_steps
        self.statistics = StepStatistics()
        self.control_readbacks = 0
        self.coupling_cycles = 0
        self.coupling_downloads = 0
        self.coupling_uploads = 0

    def _ready(self):
        self._events._require_active()
        if wp.get_device(self.device.native).is_capturing:
            raise RuntimeError(
                "coupled host control cannot run during CUDA capture"
            )

    @property
    def device(self):
        """Return the prepared event device."""
        return self._events.device

    @property
    def time_step(self):
        """Return the frozen event interval [s]."""
        return self._events.time_step

    @property
    def time(self):
        """Return the last complete cycle time, never an intermediate time."""
        with self._events._lock:
            return self._time

    @property
    def completed_steps(self):
        """Return successfully completed coupled cycles."""
        with self._events._lock:
            return self._completed_steps

    def _condense(self, start, end):
        if self.resident:
            flow = self._resident_flow
            flow._reset_from_session(
                self._events._session, start, self.controls.first_step
            )
            flow.advance(end)
            flow._export_to_session(self._events._session)
            self._accumulate_condensation_work(flow)
            return
        initial = self._events.snapshot()
        self.coupling_downloads += 1
        if not np.any(initial.particles.concentration > 0):
            return
        flow = GPUCondensationODE(
            initial,
            self._strategy,
            device=self.device.native,
            time=start,
            **asdict(self.controls),
        )
        self.coupling_uploads += 1
        while flow.time < end:
            flow.step(end)
        final = flow.snapshot()
        self.coupling_downloads += 1
        _cpu_batch.validate(final)
        # Only the accepted mass and gas fields may cross this boundary.
        for name in ("concentration", "charge", "density", "volume"):
            if not np.array_equal(
                getattr(initial.particles, name), getattr(final.particles, name)
            ):
                raise ValueError(
                    "condensation changed event-owned particle fields"
                )
        if final.particles.masses.shape != initial.particles.masses.shape or (
            final.gas.concentration.shape != initial.gas.concentration.shape
        ):
            raise ValueError("condensation changed fixed field shapes")
        apply_forcing(
            self._events,
            (
                (
                    self._events._session.particles.masses,
                    final.particles.masses,
                ),
                (
                    self._events._session.gas.concentration,
                    final.gas.concentration,
                ),
            ),
        )
        self.coupling_uploads += 1
        self._accumulate_condensation_work(flow)

    def _accumulate_condensation_work(self, flow):
        """Record work without downloading physical state."""
        self.control_readbacks += flow.control_readbacks
        for name, value in vars(flow.statistics).items():
            setattr(
                self.statistics, name, getattr(self.statistics, name) + value
            )

    def step(self, bound=None):
        """Complete one fixed cycle, rejecting a shorter bound before work."""
        with self._events._lock:
            self._ready()
            start = self._time
            end = self._events._initial_time + (
                self._events.simulated_time + self.time_step
            )
            if not np.isfinite(end) or end <= start:
                raise ValueError("event grid cannot advance representable time")
            if bound is not None:
                bound = AdaptiveSolver._scalar(bound, "bound", positive=False)
                if bound < end:
                    raise ValueError(
                        "bound must include a complete fixed event interval"
                    )
            try:
                # Reuse one stream and retain all event bindings/RNG throughout.
                with self._events._scope():
                    if self.order == "events-first":
                        self._events.step()
                    self._condense(start, end)
                    if self.order == "condensation-first":
                        self._events.step()
                    self._events.synchronize()
                self._time = self._events.current_time
                self._completed_steps = self._events.completed_steps
                self.coupling_cycles += 1
            except BaseException:
                self._events._fault()
                raise
            return self._time

    def run(self, steps):
        """Complete a nonnegative integer number of fixed coupled cycles."""
        if (
            isinstance(steps, bool)
            or not isinstance(steps, Integral)
            or steps < 0
        ):
            raise ValueError("steps must be a nonnegative integer")
        for _ in range(steps):
            self.step()

    def _saturation(self, gas, temperature):
        pressure = np.column_stack(
            [
                WaterBuckStrategy().pure_vapor_pressure(temperature)
                if mode
                else np.full_like(temperature, value)
                for mode, value in self._vapor
            ]
        )
        partial = gas * GAS_CONSTANT * temperature[:, None] / self._molar_mass
        # Involatile lanes follow the zero convention of GPU diagnostics.
        return np.divide(
            partial, pressure, out=np.zeros_like(partial), where=pressure > 0
        )

    def snapshot(self):
        """Explicitly download detached native state after a complete cycle."""
        with self._events._lock:
            self._ready()
            result = self._events.snapshot()
            result.environment.saturation_ratio[:] = self._saturation(
                result.gas.concentration, result.environment.temperature
            )
            return result

    def diagnostics(self):
        """Download box reductions and derive observed saturation."""
        with self._events._lock:
            self._ready()
            result = self._events.diagnostics()
            return replace(
                result,
                saturation_ratio=self._saturation(
                    result.gas_concentration, result.temperature
                ),
            )

    def update_environment(self, *, temperature=None, pressure=None):
        """Explicitly upload between-cycle temperature [K] and pressure [Pa]."""
        with self._events._lock:
            self._ready()
            self._events.update_environment(
                temperature=temperature, pressure=pressure
            )

    def update_gas(self, concentration):
        """Explicitly upload full-gas concentration [kg/m³] between cycles."""
        with self._events._lock:
            self._ready()
            self._events.update_gas(concentration)

    def capture_events(self):
        """Capture only the fixed event sequence."""
        with self._events._lock:
            self._ready()
            self._events.capture()

    def checkpoint(self):
        """Explicitly download exact event/RNG continuation between cycles."""
        from particula.execution.gpu_coupled_restart import checkpoint_coupled

        with self._events._lock:
            self._ready()
            return checkpoint_coupled(self)

    def close(self):
        """Release persistent event resources, including captured graphs."""
        self._events.close()

    def __enter__(self):
        """Enter the active owner's explicit resource scope."""
        self._ready()
        return self

    def __exit__(self, *_exc):
        """Release event resources on normal exit and exceptions."""
        self.close()
