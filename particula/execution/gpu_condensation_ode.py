"""Native Warp condensation with resident RK3(2) or ROS3 trial state.

Construction explicitly uploads detached native state/configuration. Each
step uses a host-controlled shared maximum-error clock and downloads only
scalar validation/error reductions; snapshots explicitly download accepted
state. This is not CUDA-capturable or a CPU fallback for requested CUDA.
"""

from copy import deepcopy
from typing import Any

import numpy as np
import warp as wp

from particula import _cpu_batch
from particula.dynamics import CondensationIsothermal
from particula.execution import _condensation_ode_kernels as algebra
from particula.execution import _condensation_reservoirs as reservoirs
from particula.execution._adaptive_condensation_physics import material_rates
from particula.execution._adaptive_condensation_plan import (
    plan_adaptive_condensation,
)
from particula.execution.forcing import forcing_interval
from particula.execution.ode import (
    AdaptiveSolver,
    StepStatistics,
    TrialDomainError,
)
from particula.gpu.kernels import condensation as physics
from particula.gpu.kernels.thermodynamics import _refresh_vapor_pressure_kernel


class GPUCondensationODE:
    """Own native condensation on an explicitly requested Warp device.

    particle_atol is kg per particle; gas_atol is kg/m3. Scalar or full native
    shapes are accepted. rtol is a dimensionless scalar. All boxes share one
    accepted time and a maximum error, including small species. The physical
    profile supports mapped constant/Buck vapor pressures, ideal/kappa activity
    for all materials and mass/molar/volume surface phases. Zero constant vapor
    pressure also represents involatile uptake.
    Native discrete and resolved measures are supported; PDF and latent heat
    are rejected. Full active-slot evaporation remains a domain boundary.
    ROS3 retains cross-material Jacobian blocks and conservative gas coupling
    in storage linear in particle count and quadratic in exchanging materials.

    Optional forcing is a pure time -> (temperature[K], pressure[Pa]) callback.
    Every stage explicitly uploads those box values. EnvironmentForcing tables
    impose knot boundaries; opaque discontinuous callbacks require explicit
    step boundaries. No arbitrary/stochastic runnable is adapted.
    Capture is unsupported because each trial reads control reductions on host.
    """

    def __init__(
        self,
        aerosol,
        strategy,
        *,
        device,
        method="rk23",
        time=0.0,
        particle_atol=1e-24,
        gas_atol=1e-12,
        rtol=1e-5,
        first_step=1e-3,
        min_step=1e-14,
        max_step=1.0,
        max_trials=100,
        forcing=None,
    ):
        """Validate physics and explicitly upload fixed buffers."""
        if method not in ("rk23", "ros3"):
            raise ValueError("native GPU condensation supports rk23 or ros3")
        if type(strategy) is not CondensationIsothermal:
            raise TypeError(
                "GPU adaptive rates require isothermal condensation"
            )
        _cpu_batch.validate(aerosol)
        if aerosol.particles.distribution_type not in (
            "discrete",
            "particle_resolved",
        ):
            raise ValueError(
                "GPU adaptive condensation requires discrete or resolved state"
            )
        self._base = _cpu_batch.detached(aerosol)
        self._strategy = deepcopy(strategy)

        self._forcing = forcing
        self._plan = plan_adaptive_condensation(self._strategy, self._base)
        pairs = tuple(
            (source, target)
            for source, target in self._plan.values["pairs"]
            if self._base.gas.partitioning[source]
        )
        self._time = AdaptiveSolver._scalar(time, "time", positive=False)
        self.min_step = AdaptiveSolver._scalar(min_step, "min_step")
        self.max_step = AdaptiveSolver._scalar(max_step, "max_step")
        self.next_step = AdaptiveSolver._scalar(first_step, "first_step")
        if not self.min_step <= self.next_step <= self.max_step:
            raise ValueError("require min_step <= first_step <= max_step")
        if type(max_trials) is not int or max_trials < 1:
            raise ValueError("max_trials must be a positive integer")
        self.max_trials = max_trials
        self.method = method
        self.statistics = StepStatistics()
        self.last_error = None
        self.control_readbacks = 0
        self._faulted = False
        self.device = wp.get_device(device)
        self._check_ready()
        self._shape = aerosol.particles.masses.shape
        self._gas_shape = aerosol.gas.concentration.shape
        self._mass_size = aerosol.particles.masses.size
        self._size = self._mass_size + aerosol.gas.concentration.size
        self._host_scale = np.concatenate(
            (
                np.broadcast_to(
                    np.maximum(
                        aerosol.particles.masses.sum(axis=2, keepdims=True),
                        1e-30,
                    ),
                    self._shape,
                ).ravel(),
                np.maximum(aerosol.gas.concentration, 1e-12).ravel(),
            )
        )
        physical = np.concatenate(
            (
                aerosol.particles.masses.ravel(),
                aerosol.gas.concentration.ravel(),
            )
        )
        absolute = (
            np.concatenate(
                (
                    self._tolerance(
                        particle_atol, self._shape, "particle_atol"
                    ).ravel(),
                    self._tolerance(
                        gas_atol, self._gas_shape, "gas_atol"
                    ).ravel(),
                )
            )
            / self._host_scale
        )
        relative = AdaptiveSolver._scalar(rtol, "rtol", positive=False)
        self.device_storage_bytes = 0
        self._host_initial_physical = physical.copy()
        self._host_initial_scaled = physical / self._host_scale
        self._initial_physical = self._array(self._host_initial_physical)
        self._initial_scaled = self._array(self._host_initial_scaled)
        self._state = self._array(self._host_initial_scaled)
        self._scale = self._array(self._host_scale)
        self._atol = self._array(absolute)
        self._rtol = self._array(np.full(self._size, relative))
        self._candidate = self._array(np.zeros(self._size))
        self._stage = self._array(np.zeros(self._size))
        self._rates = tuple(self._array(np.zeros(self._size)) for _ in range(4))
        self._invalid = self._array(np.zeros(1, dtype=np.int32), wp.int32)
        self._norm = self._array(np.zeros(1))
        self._allocate_physics()
        # Probe the actual device rate before exposing a usable owner.
        self._rate(self.time, self._state, self._rates[0])
        self._ros3: Any = None
        if method == "ros3":
            from particula.execution._gpu_condensation_ros3 import ROS3Stages

            if len(pairs) == 1:
                self._ros3 = ROS3Stages(self, pairs[0])
            else:
                from particula.execution._gpu_condensation_block_ros3 import (
                    BlockROS3Stages,
                )

                self._ros3 = BlockROS3Stages(self, pairs)
        self.statistics = StepStatistics()
        self.control_readbacks = 0

    @staticmethod
    def _tolerance(value, shape, name):
        values = np.asarray(value, dtype=float)
        if (
            values.shape not in ((), shape)
            or not np.all(np.isfinite(values))
            or np.any(values <= 0)
        ):
            raise ValueError(
                f"{name} must be positive finite scalar or native shape"
            )
        return np.broadcast_to(values, shape)

    def _array(self, values, dtype=wp.float64):
        array: Any = wp.array(values, dtype=dtype, device=self.device)
        self.device_storage_bytes += array.size * (
            8 if dtype == wp.float64 else 4
        )
        return array

    def _allocate_physics(self):
        config = self._plan.values
        base = self._base
        self._masses = self._array(base.particles.masses)
        self._gas = self._array(base.gas.concentration)
        self._counts = self._array(base.particles.concentration)
        self._volume = self._array(base.particles.volume)
        self._density = self._array(base.particles.density)
        self._molar = self._array(base.gas.molar_mass)
        self._temperature = self._array(base.environment.temperature)
        self._pressure = self._array(base.environment.pressure)
        self._viscosity = self._array(np.zeros(self._shape[0]))
        self._mean_free_path = self._array(np.zeros(self._shape[0]))
        self._vapor = self._array(np.zeros(self._gas_shape))
        self._effective_tension = self._array(np.zeros(self._shape[:2]))
        self._mass_rate = self._array(np.zeros(self._shape))
        self._gas_partials = self._array(
            np.zeros((*self._gas_shape, max(1, min(64, self._shape[1]))))
        )
        self._tension = self._array(config["tensions"])
        self._surface_density = self._array(config["surface_density"])
        self._surface_reference = self._array(config["surface_reference"])
        self._phases = self._array(config["phases"], wp.int32)
        self._activity_density = self._array(config["activity_density"])
        self._bat_table = self._array(config["bat_table"])
        self._kappas = self._array(config["kappas"])
        self._references = self._array(config["references"])
        self._accommodation = self._array(config["accommodation"])
        self._diffusion = self._array(config["diffusion"])
        self._latent = self._array(np.zeros(self._gas_shape[1]))
        self._vapor_modes = self._array(
            [mode for mode, _ in config["vapor"]], wp.int32
        )
        parameters = np.zeros((self._gas_shape[1], 4))
        parameters[:, 0] = [value for _, value in config["vapor"]]
        self._vapor_parameters = self._array(parameters)
        forward = np.full(self._shape[2], -1, dtype=np.int32)
        reverse = np.full(self._gas_shape[1], -1, dtype=np.int32)
        for source, target in config["pairs"]:
            if base.gas.partitioning[source]:
                forward[target], reverse[source] = source, target
        self._forward = self._array(forward, wp.int32)
        self._reverse = self._array(reverse, wp.int32)
        self._provider = None
        self._flux_guard = None
        if config["activity_mode"] == 3:
            self._provider = self._strategy.activity_strategy.bind_device(
                self._shape, np.flatnonzero(forward >= 0), self.device
            )
            self._bat_table = self._provider.values.flatten()
            if getattr(self._provider, "hybrid", False):
                from particula.gpu.kernels.activity_guard import BoxFluxGuard

                if self.method == "ros3" and not callable(
                    getattr(self._provider, "evaluate_direct", None)
                ):
                    raise ValueError(
                        "hybrid ROS3 requires direct device linearization"
                    )
                self._flux_guard = BoxFluxGuard(self._provider)
            self._provider_stage = 0

    def _check_ready(self):
        if self._faulted:
            raise RuntimeError(
                "GPU adaptive owner faulted after a device failure"
            )
        if self.device.is_capturing:
            raise RuntimeError(
                "adaptive host control cannot execute during CUDA capture"
            )

    def _launch(self, kernel, dim, inputs):
        wp.launch(kernel, dim=dim, inputs=inputs, device=self.device)

    def _status(self):
        self.control_readbacks += 1
        if self._invalid.numpy()[0]:
            raise TrialDomainError(
                self._activity_failure()
                or "GPU trial leaves the physical inventory domain"
            )

    def _activity_failure(self):
        if self._provider is not None:
            return self._provider.failure_message()
        return ""

    def _environment(self, time):
        if self._forcing is None:
            return (
                self._base.environment.temperature,
                self._base.environment.pressure,
            )
        values = self._forcing(time)
        if not isinstance(values, (tuple, list)) or len(values) != 2:
            raise ValueError("forcing must return temperature and pressure")
        return tuple(
            self._tolerance(value, (self._shape[0],), "forcing")
            for value in values
        )

    def _unpack_trial(self, state):
        """Validate and unpack device coordinates without a physical rate."""
        self._invalid.zero_()
        self._launch(
            algebra.unpack,
            self._size,
            [
                state,
                self._scale,
                self._initial_scaled,
                self._initial_physical,
                self._masses,
                self._gas,
                self._invalid,
            ],
        )
        self._launch(
            algebra.check_slots,
            self._shape[:2],
            [
                self._masses,
                self._counts,
                self._invalid,
                wp.int32(
                    self._base.particles.distribution_type
                    == "particle_resolved"
                ),
            ],
        )
        self._status()

    def _material_rates(self, activities, dest):
        self._launch(
            material_rates,
            self._shape,
            [
                self._masses,
                self._counts,
                self._density,
                self._gas,
                self._vapor,
                self._molar,
                self._tension,
                self._surface_density,
                self._surface_reference,
                self._phases,
                self._references,
                self._activity_density,
                self._kappas,
                activities,
                wp.int32(self._plan.values["activity_mode"]),
                wp.int32(self._plan.values["water"]),
                self._accommodation,
                self._diffusion,
                self._mean_free_path,
                self._temperature,
                self._forward,
                dest,
            ],
        )

    def _rate(self, time, state, result, *, linearize=False):
        self._unpack_trial(state)
        if self._forcing is not None:
            temperature, pressure = self._environment(time)
            self._temperature.assign(np.array(temperature, copy=True))
            self._pressure.assign(np.array(pressure, copy=True))
        self._launch(
            _refresh_vapor_pressure_kernel,
            self._gas_shape,
            [
                self._vapor_modes,
                self._vapor_parameters,
                self._temperature,
                self._vapor,
            ],
        )
        self._launch(
            physics._prepare_environment_properties_kernel,
            self._shape[0],
            [
                self._temperature,
                self._pressure,
                self._viscosity,
                self._mean_free_path,
            ],
        )
        if self._provider is not None:
            evaluate = (
                self._provider.evaluate_direct
                if linearize and self._flux_guard is not None
                else self._provider.evaluate
            )
            evaluate(
                self._masses,
                self._counts,
                self._temperature,
                self._invalid,
                stage=self._provider_stage,
            )
            self._provider_stage += 1
            self._status()
        self.statistics.rate_calls += 1
        if self._flux_guard is not None and not linearize:
            self._flux_guard.refine(
                self._material_rates,
                self._masses,
                self._counts,
                self._temperature,
                self._invalid,
                self._provider_stage - 1,
            )
            self._status()
        self._material_rates(self._bat_table, self._mass_rate)
        self._launch(
            algebra.pack_rate,
            self._mass_size,
            [
                self._mass_rate,
                self._counts,
                self._volume,
                self._reverse,
                self._scale,
                result,
                self._invalid,
                wp.int32(
                    self._base.particles.distribution_type
                    == "particle_resolved"
                ),
            ],
        )
        self._launch(
            reservoirs.gas_parts,
            self._gas_partials.shape,
            [
                self._mass_rate,
                self._counts,
                self._volume,
                self._reverse,
                int(
                    self._base.particles.distribution_type
                    == "particle_resolved"
                ),
                self._gas_partials,
            ],
        )
        self._launch(
            reservoirs.finish_gas,
            self._gas_shape,
            [
                self._gas_partials,
                self._mass_size,
                self._scale,
                result,
                self._invalid,
            ],
        )
        self._status()

    def _combine(self, h, coefficients, output):
        self._launch(
            algebra.stage,
            self._size,
            [
                self._state,
                *self._rates[:3],
                wp.float64(h),
                *(wp.float64(value) for value in coefficients),
                output,
            ],
        )

    def _begin_activity_trial(self):
        if self._provider is not None:
            self._provider.detail.zero_()
            self._provider_stage = 0

    def _trial(self, h):
        self._begin_activity_trial()
        if self._ros3 is not None:
            return self._ros3.trial(self, h)
        k1, k2, k3, k4 = self._rates
        self._rate(self.time, self._state, k1)
        self._combine(h, (0.5, 0, 0), self._stage)
        self._rate(self.time + h / 2, self._stage, k2)
        self._combine(h, (0, 0.75, 0), self._stage)
        self._rate(self.time + 3 * h / 4, self._stage, k3)
        self._combine(h, (2 / 9, 1 / 3, 4 / 9), self._candidate)
        self._rate(self.time + h, self._candidate, k4)
        self._norm.zero_()
        self._launch(
            algebra.error_norm,
            self._size,
            [
                self._state,
                self._candidate,
                k1,
                k2,
                k3,
                k4,
                self._atol,
                self._rtol,
                wp.float64(h),
                self._norm,
            ],
        )
        self.control_readbacks += 1
        return float(self._norm.numpy()[0])

    @property
    def time(self):
        """Return the accepted shared physical time [s]."""
        return self._time

    def step(self, bound):
        """Accept one bounded device trial with scalar control readbacks."""
        self._check_ready()
        bound = AdaptiveSolver._scalar(bound, "bound", positive=False)
        with forcing_interval(self, bound) as endpoint:
            return self._step(endpoint)

    def _step(self, bound):
        """Advance within one smooth forcing interval."""
        if bound < self.time:
            raise ValueError("bound precedes accepted time")
        if bound == self.time:
            return self.time
        h = min(self.next_step, self.max_step, bound - self.time)
        floor = min(self.min_step, bound - self.time)
        for _ in range(self.max_trials):
            if h < floor or self.time + h == self.time:
                raise RuntimeError(
                    self._activity_failure()
                    or "accuracy requires a step below min_step"
                )
            domain_failure = False
            try:
                norm = self._trial(h)
            except TrialDomainError:
                domain_failure, norm = True, np.inf
            except Exception:
                # Device/configuration failures are not physical-domain retries.
                self._faulted = True
                raise
            if norm <= 1:
                self._state, self._candidate = self._candidate, self._state
                self._time = bound if h == bound - self.time else self.time + h
                self.statistics.accepted += 1
                self.last_error = norm
                factor = (
                    5.0
                    if norm == 0
                    else min(5.0, max(0.2, 0.9 * norm ** (-1 / 3)))
                )
                self.next_step = min(
                    self.max_step, max(self.min_step, h * factor)
                )
                return self.time
            self.statistics.rejected += 1
            self.statistics.domain_rejected += int(domain_failure)
            h *= (
                0.5
                if not np.isfinite(norm)
                else max(0.1, min(0.5, 0.9 * norm ** (-1 / 3)))
            )
        raise RuntimeError(
            self._activity_failure()
            or "maximum trial count exceeded without accepting a step"
        )

    def snapshot(self):
        """Download detached accepted native state, never trial buffers."""
        self._check_ready()
        scaled = self._state.numpy()
        values = scaled * self._host_scale
        unchanged = scaled == self._host_initial_scaled
        values[unchanged] = self._host_initial_physical[unchanged]
        result = _cpu_batch.detached(self._base)
        result.particles.masses[:] = values[: self._mass_size].reshape(
            self._shape
        )
        result.gas.concentration[:] = values[self._mass_size :].reshape(
            self._gas_shape
        )
        temperature, pressure = self._environment(self.time)
        result.environment.temperature[:] = temperature
        result.environment.pressure[:] = pressure
        return result
