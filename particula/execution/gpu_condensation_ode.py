"""Native Warp condensation with device-resident RK3(2) trial state.

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
from particula.dynamics import CondensationIsothermal, MassCondensation
from particula.execution import _condensation_ode_kernels as algebra
from particula.execution.forcing import forcing_interval
from particula.execution.ode import (
    AdaptiveSolver,
    StepStatistics,
    TrialDomainError,
)
from particula.execution.prepared_condensation import plan_condensation
from particula.gpu.kernels import condensation as physics
from particula.gpu.kernels.thermodynamics import _refresh_vapor_pressure_kernel
from particula.util import constants


class GPUCondensationODE:
    """Own native condensation on an explicitly requested Warp device.

    particle_atol is kg per particle; gas_atol is kg/m3. Scalar or full native
    shapes are accepted. rtol is a dimensionless scalar. All boxes share one
    accepted time and a maximum error, including small species. The physical
    profile is the existing mapped constant/Buck GPU rate with one volatile
    material; zero constant vapor pressure also represents involatile uptake.
    Native discrete and resolved measures are supported; PDF and latent heat
    are rejected. Full active-slot evaporation remains a domain boundary.

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
        if method != "rk23":
            raise ValueError("native GPU condensation supports rk23 only")
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
        if not np.any(aerosol.particles.concentration > 0):
            raise ValueError(
                "GPU adaptive condensation requires an active population"
            )
        self._base = _cpu_batch.detached(aerosol)
        self._strategy = deepcopy(strategy)
        self._forcing = forcing
        self._plan = plan_condensation(
            MassCondensation(self._strategy), self._base, allow_zero_vapor=True
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
        self._state = self._array(physical / self._host_scale)
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
        self._tension = self._array(config["tensions"])
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
                "GPU trial leaves the physical inventory domain"
            )

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

    def _rate(self, time, state, result):
        self._invalid.zero_()
        self._launch(
            algebra.unpack,
            self._size,
            [state, self._scale, self._masses, self._gas, self._invalid],
        )
        self._launch(
            algebra.check_slots,
            self._shape[:2],
            [self._masses, self._counts, self._invalid],
        )
        self._status()
        if self._forcing is not None:
            temperature, pressure = self._environment(time)
            self._temperature.assign(np.array(temperature, copy=True))
            self._pressure.assign(np.array(pressure, copy=True))
        config = self._plan.values
        if config["surface_mode"]:
            self._launch(
                physics._effective_surface_tension_kernel,
                self._shape[:2],
                [
                    self._masses,
                    self._density,
                    self._tension,
                    self._effective_tension,
                ],
            )
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
        self.statistics.rate_calls += 1
        self._launch(
            physics._mapped_condensation_mass_transfer_kernel,
            self._shape[:2],
            [
                self._masses,
                self._counts,
                self._density,
                self._gas,
                self._vapor,
                self._molar,
                self._tension,
                self._kappas,
                self._references,
                self._effective_tension,
                wp.int32(1),
                wp.int32(config["activity_mode"]),
                wp.int32(config["surface_mode"]),
                wp.int32(config["water"]),
                self._accommodation,
                self._diffusion,
                self._latent,
                wp.int32(0),
                self._viscosity,
                self._mean_free_path,
                wp.float64(constants.GAS_CONSTANT),
                wp.float64(constants.BOLTZMANN_CONSTANT),
                self._temperature,
                wp.float64(1.0),
                self._mass_rate,
                self._forward,
                wp.float64(config.get("kelvin_density", 0.0)),
            ],
        )
        self._launch(
            algebra.pack_rate,
            self._size,
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

    def _trial(self, h):
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
                raise RuntimeError("accuracy requires a step below min_step")
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
            "maximum trial count exceeded without accepting a step"
        )

    def snapshot(self):
        """Download detached accepted native state, never trial buffers."""
        self._check_ready()
        values = self._state.numpy() * self._host_scale
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
