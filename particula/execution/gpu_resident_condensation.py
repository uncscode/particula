"""Adaptive condensation with native CUDA graph control and resident state."""

import numpy as np
import warp as wp

from particula import _cpu_batch
from particula.execution import _condensation_device_control as control
from particula.execution import _condensation_ros3_kernels as ros
from particula.execution._gpu_condensation_block_ros3 import BlockROS3Stages
from particula.execution.gpu_condensation_ode import GPUCondensationODE
from particula.execution.ode import AdaptiveSolver, StepStatistics


class GPUResidentCondensation(GPUCondensationODE):
    """Run RK23 or material-coupled ROS3 with all adaptive decisions on CUDA.

    Construction uploads native state once and captures a conditional trial
    loop without advancing it. ``advance(bound)`` launches that loop; accepted
    state, clock, retry count and step size remain on device. Only completed
    observation-boundary control metadata is downloaded. ``snapshot()`` is an
    explicit full-state download. No Warp CPU or host-control fallback exists.

    Environment is constant between explicit bounds. Arbitrary host forcing
    callbacks are rejected; callers may update environment at segment bounds.
    Tolerances and physical units match GPUCondensationODE. CUDA conditional
    graph support (Toolkit and driver >=12.4) is required.
    """

    def __init__(self, aerosol, strategy, *, device, method="ros3", **kwargs):
        """Upload fixed-shape buffers and capture the adaptive trial loop."""
        if kwargs.get("forcing") is not None:
            raise ValueError(
                "resident control requires explicit environment segments"
            )
        native = wp.get_device(device)
        if not native.is_cuda or not wp.is_conditional_graph_supported():
            raise ValueError(
                "resident control requires native CUDA conditional graphs"
            )
        self._device_control = False
        super().__init__(
            aerosol, strategy, device=device, method=method, **kwargs
        )
        self._control = self._array(
            np.array(
                [
                    self.next_step,
                    self.time,
                    self.next_step,
                    self.time,
                    self.min_step,
                    self.max_step,
                    0.0,
                ]
            )
        )
        self._physical_atol = self._array(
            np.concatenate(
                (
                    self._tolerance(
                        kwargs.get("particle_atol", 1e-24),
                        self._shape,
                        "particle_atol",
                    ).ravel(),
                    self._tolerance(
                        kwargs.get("gas_atol", 1e-12),
                        self._gas_shape,
                        "gas_atol",
                    ).ravel(),
                )
            )
        )
        self._condition = self._array(np.zeros(1, dtype=np.int32), wp.int32)
        self._counters = self._array(np.zeros(5, dtype=np.int32), wp.int32)
        self._sticky_invalid = self._array(
            np.zeros(1, dtype=np.int32), wp.int32
        )
        self._accepted = self._array(np.zeros(1, dtype=np.int32), wp.int32)
        if method == "ros3":
            pairs = tuple(
                (s, t)
                for s, t in self._plan.values["pairs"]
                if aerosol.gas.partitioning[s]
            )
            self._ros3 = BlockROS3Stages(self, pairs)
            self._ros3.h = self._control
        self._device_control = True
        # Compile the exact trial body out of capture; this never commits state.
        self._device_trial()
        wp.load_module(module=control, device=self.device)
        wp.synchronize_device(self.device)
        with wp.ScopedCapture(device=self.device) as capture:
            wp.capture_while(self._condition, self._iteration)
        self._graph = capture.graph
        self.control_readbacks = 0

    def _status(self):
        if not self._device_control:
            return super()._status()
        self._launch(
            control.accumulate_invalid, 1, [self._invalid, self._sticky_invalid]
        )

    def _device_trial(self):
        """Build a trial without acceptance, host scalar reads or uploads."""
        if self.method == "rk23":
            self._rate(self.time, self._state, self._rates[0])
            for coefficients, output, rate in (
                ((0.5, 0, 0), self._stage, self._rates[1]),
                ((0, 0.75, 0), self._stage, self._rates[2]),
                ((2 / 9, 1 / 3, 4 / 9), self._candidate, self._rates[3]),
            ):
                self._launch(
                    control.rk_stage,
                    self._size,
                    [
                        self._state,
                        *self._rates[:3],
                        self._control,
                        *(wp.float64(v) for v in coefficients),
                        output,
                    ],
                )
                self._rate(self.time, output, rate)
            self._launch(
                control.rk_error,
                self._size,
                [
                    self._state,
                    self._candidate,
                    *self._rates,
                    self._atol,
                    self._rtol,
                    self._control,
                    self._norm,
                ],
            )
            return
        stages = self._ros3
        stages.linearize(self)
        first, second, third = stages.increments
        for stage, rate, result in (
            (1, self._rates[0], first),
            (2, self._rates[1], second),
            (3, self._rates[1], third),
        ):
            self._launch(
                control.ros_rhs,
                self._size,
                [
                    rate,
                    first,
                    second,
                    self._control,
                    stage,
                    stages.rhs,
                ],
            )
            stages.solve(self, None, result)
            if stage == 1:
                self._launch(
                    ros.stage_state,
                    self._size,
                    [self._state, first, self._stage],
                )
                self._rate(self.time, self._stage, self._rates[1])
        self._launch(
            ros.finish,
            self._size,
            [
                self._state,
                first,
                second,
                third,
                self._atol,
                self._rtol,
                self._candidate,
                self._norm,
            ],
        )
        if self._provider is None:
            self._unpack_trial(self._candidate)
        else:
            self._rate(self.time, self._candidate, self._rates[1])

    def _iteration(self):
        self._begin_activity_trial()
        self._launch(
            control.begin_trial,
            1,
            [self._control, self._sticky_invalid, self._norm, self._accepted],
        )
        self._device_trial()
        self._launch(
            control.decide,
            1,
            [
                self._control,
                self._norm,
                self._sticky_invalid,
                self._accepted,
                self._condition,
                self._counters,
                self.max_trials,
            ],
        )
        self._launch(
            control.commit_candidate,
            self._size,
            [self._state, self._candidate, self._accepted],
        )

    def advance(self, bound):
        """Reach an explicit observation/forcing bound [s] entirely on CUDA."""
        self._check_ready()
        bound = AdaptiveSolver._scalar(bound, "bound", positive=False)
        if bound < self.time:
            raise ValueError("bound precedes accepted time")
        if bound == self.time:
            return self.time
        self._launch(
            control.set_bound,
            1,
            [self._control, self._condition, self._counters, wp.float64(bound)],
        )
        try:
            wp.capture_launch(self._graph)
            metadata = self._control.numpy()
            counters = self._counters.numpy()
        except Exception:
            self._faulted = True
            raise
        self.control_readbacks += 2
        self._time = float(metadata[1])
        self.next_step = float(metadata[2])
        self.last_error = float(metadata[6])
        self.statistics.accepted = int(counters[0])
        self.statistics.rejected = int(counters[1])
        self.statistics.domain_rejected = int(counters[2])
        trials = int(counters[0] + counters[1])
        self.statistics.rate_calls = trials * (
            4
            if self.method == "rk23"
            else 2 + int(self._provider is not None) + 4 * self._ros3.width
        )
        self.statistics.jacobian_calls = 0 if self.method == "rk23" else trials
        self.statistics.factorizations = self.statistics.jacobian_calls
        self.statistics.linear_solves = 3 * self.statistics.jacobian_calls
        if counters[3]:
            self._faulted = True
            failure = self._activity_failure()
            if failure:
                raise RuntimeError(failure)
            raise RuntimeError(
                "resident adaptive step below min_step"
                if counters[3] == 1
                else "resident maximum trial count exceeded"
            )
        return self.time

    def step(self, bound):
        """Advance to the bound, including all internal accepted steps."""
        return self.advance(bound)

    def _reset_from_session(self, session, time, first_step):
        """Import event buffers on device at a complete split boundary."""
        self._check_ready()
        for target, source in (
            (self._masses, session.particles.masses),
            (self._counts, session.particles.concentration),
            (self._volume, session.particles.volume),
            (self._gas, session.gas.concentration),
            (self._temperature, session.environment.temperature),
            (self._pressure, session.environment.pressure),
        ):
            wp.copy(target, source)
        self._reset_coordinates(time, first_step)

    def restart_controller(self, first_step):
        """Reset scaling and step history on CUDA at an accepted boundary.

        ``first_step`` is in seconds, within this owner's timestep limits.
        Physical state and accepted time are retained; per-interval statistics
        reset. This matches a split flow's fresh-controller policy without
        downloading or re-uploading physical state.
        """
        self._check_ready()
        first_step = AdaptiveSolver._scalar(
            first_step, "first_step", positive=True
        )
        if not self.min_step <= first_step <= self.max_step:
            raise ValueError("require min_step <= first_step <= max_step")
        self._unpack_trial(self._state)
        self._reset_coordinates(self.time, first_step)

    def _reset_coordinates(self, time, first_step):
        """Reset scaled coordinates and control using device state."""
        self._launch(
            control.import_physical,
            self._size,
            [
                self._masses,
                self._gas,
                self._physical_atol,
                self._state,
                self._scale,
                self._initial_scaled,
                self._initial_physical,
                self._atol,
            ],
        )
        self._launch(
            control.reset_clock,
            1,
            [
                self._control,
                self._counters,
                wp.float64(time),
                wp.float64(first_step),
            ],
        )
        self._time = time
        self.statistics = StepStatistics()
        self.next_step = first_step
        self.control_readbacks = 0

    def _export_to_session(self, session):
        """Commit accepted physical masses/gas to the same event buffers."""
        self._unpack_trial(self._state)
        wp.copy(session.particles.masses, self._masses)
        wp.copy(session.gas.concentration, self._gas)

    def snapshot(self):
        """Explicitly download accepted native state and current environment."""
        self._check_ready()
        self._unpack_trial(self._state)
        result = _cpu_batch.detached(self._base)
        result.particles.masses[:] = self._masses.numpy()
        result.particles.concentration[:] = self._counts.numpy()
        result.particles.volume[:] = self._volume.numpy()
        result.gas.concentration[:] = self._gas.numpy()
        result.environment.temperature[:] = self._temperature.numpy()
        result.environment.pressure[:] = self._pressure.numpy()
        return result
