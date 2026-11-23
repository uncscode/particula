"""Native CPU condensation as explicitly declared continuous mass rates.

The owner detaches initial state and process configuration. Trial coordinates
are scaled per-particle kg and gas kg/m3, with independent absolute tolerances.
Full species maps, resolved counts/volume and PDF quadrature remain native.
This path integrates isothermal rates, not inventory-limited finite-step maps.
"""

from copy import copy, deepcopy

import numpy as np

from particula import _cpu_batch
from particula.dynamics.condensation.condensation_strategies import (
    CondensationIsothermal,
    CondensationIsothermalStaggered,
)
from particula.dynamics.condensation.species_mapping import (
    condensation_population_weights,
    validate_condensation_state,
)
from particula.execution.forcing import _EnvironmentSegment, forcing_interval
from particula.execution.ode import AdaptiveSolver, ODEProblem, TrialDomainError


class CondensationODE:
    """Own detached native state and one shared deterministic CPU controller.

    particle_atol is per-particle kg; gas_atol is kg/m3. Both accept scalars or
    their full native array shape. rtol is dimensionless. All boxes share one
    accepted clock and a maximum component error, so no small box/species can
    be hidden in an average. Forcing, if supplied, is a pure function of time
    returning (temperature[K], pressure[Pa]), each scalar or one value per box.
    EnvironmentForcing tables impose knot boundaries; opaque discontinuous
    callbacks require explicit run boundaries. Full evaporation of an
    active slot is a domain boundary; unsupported crossings fail at min_step.
    """

    def __init__(
        self,
        aerosol,
        strategy,
        *,
        particle_atol=1e-24,
        gas_atol=1e-12,
        rtol=1e-5,
        time=0.0,
        method="rk23",
        forcing=None,
        dynamic_viscosity=None,
        **controls,
    ):
        """Validate physics and construct scaled continuous coordinates."""
        if type(strategy) not in (
            CondensationIsothermal,
            CondensationIsothermalStaggered,
        ):
            raise TypeError(
                "continuous condensation supports isothermal strategies only"
            )
        _cpu_batch.validate(aerosol)
        self._base = _cpu_batch.detached(aerosol)
        self._strategy = deepcopy(strategy)

        self._linear_strategy = self._strategy
        activity = self._strategy.activity_strategy
        if (
            method == "ros3"
            and getattr(activity, "evaluation", None) == "hybrid"
        ):
            direct = getattr(activity, "direct_for_linearization", None)
            if not callable(direct):
                raise ValueError(
                    "hybrid ROS3 requires a direct linearization provider"
                )
            self._linear_strategy = copy(self._strategy)
            self._linear_strategy.activity_strategy = direct()
        self._forcing = forcing
        self._viscosity = (
            None
            if dynamic_viscosity is None
            else self._tolerances(
                dynamic_viscosity,
                (aerosol.particles.n_boxes,),
                "dynamic_viscosity",
            ).copy()
        )
        self._shape = aerosol.particles.masses.shape
        mass_indices: list[tuple[int, int, int]] = []
        gas_indices: list[tuple[int, int]] = []
        self._maps = []
        self._weights = []
        for box in range(aerosol.particles.n_boxes):
            state = _cpu_batch.box_view(self._base, box)
            sources, targets = validate_condensation_state(
                state.particles,
                state.gas,
                self._strategy.species_map,
                float(state.environment.temperature[0]),
                float(state.environment.pressure[0]),
                self._strategy.skip_partitioning_indices,
            )
            self._maps.append(tuple(zip(sources, targets, strict=True)))
            self._weights.append(
                condensation_population_weights(state.particles)
            )
            active = np.flatnonzero(state.particles.concentration[0] > 0)
            for slot in active:
                for target in targets:
                    mass_indices.append((box, slot, int(target)))
            if self._strategy.update_gases:
                gas_indices.extend((box, int(source)) for source in sources)
        if not mass_indices:
            raise ValueError(
                "continuous condensation requires an active mapped population"
            )
        self._mass_indices = tuple(mass_indices)
        self._gas_indices = tuple(gas_indices)
        self._mass_count = len(self._mass_indices)
        values = self._pack(self._base)
        # Initial slot mass scales empty material lanes too.
        mass_scale = [
            self._base.particles.masses[b, n].sum()
            for b, n, _ in self._mass_indices
        ]
        gas_scale = [
            max(self._base.gas.concentration[b, g], 1e-12)
            for b, g in self._gas_indices
        ]
        self._scale = np.array(mass_scale + gas_scale)
        self._initial_physical = values.copy()
        self._initial_scaled = values / self._scale
        mass_tol = self._tolerances(particle_atol, self._shape, "particle_atol")
        gas_tol = self._tolerances(
            gas_atol, self._base.gas.concentration.shape, "gas_atol"
        )
        tolerance = np.array(
            [mass_tol[index] for index in self._mass_indices]
            + [gas_tol[index] for index in self._gas_indices]
        )
        self.problem = ODEProblem(
            self._rate,
            self._admissible,
            self._jacobian,
            self._partial_time,
            validate_candidate=(
                self._rate
                if getattr(
                    self._strategy.activity_strategy,
                    "prepared_provider_version",
                    None,
                )
                == 1
                else None
            ),
        )
        self.solver = AdaptiveSolver(
            self.problem,
            self._initial_scaled,
            time=time,
            method=method,
            atol=tolerance / self._scale,
            rtol=rtol,
            **controls,
        )
        # Configuration errors surface at construction, not as rejected trials.
        self._rate(time, self.solver.state)

    @staticmethod
    def _tolerances(value, shape, name):
        result = np.asarray(value, dtype=np.float64)
        if (
            result.shape not in ((), shape)
            or not np.all(np.isfinite(result))
            or np.any(result <= 0)
        ):
            raise ValueError(
                f"{name} must be positive finite scalar or full native shape"
            )
        return np.broadcast_to(result, shape)

    def _pack(self, state):
        return np.array(
            [state.particles.masses[index] for index in self._mass_indices]
            + [state.gas.concentration[index] for index in self._gas_indices]
        )

    def _unpack(self, time, values):
        candidate = _cpu_batch.detached(self._base)
        physical = values * self._scale
        # An unchanged coordinate must survive a scale/unscale round trip
        # exactly. Changed coordinates retain their ordinary physical value.
        unchanged = values == self._initial_scaled
        physical[unchanged] = self._initial_physical[unchanged]
        for index, value in zip(
            self._mass_indices, physical[: self._mass_count], strict=True
        ):
            candidate.particles.masses[index] = value
        for gas_index, value in zip(
            self._gas_indices, physical[self._mass_count :], strict=True
        ):
            candidate.gas.concentration[gas_index] = value
        if self._forcing is not None:
            temperature, pressure = self._forcing(time)
            for name, value in (
                ("temperature", temperature),
                ("pressure", pressure),
            ):
                target = getattr(candidate.environment, name)
                value = np.asarray(value, dtype=np.float64)
                if (
                    value.shape not in ((), target.shape)
                    or not np.all(np.isfinite(value))
                    or np.any(value <= 0)
                ):
                    raise ValueError(
                        "forcing requires positive finite scalar or box arrays"
                    )
                target[:] = value
        return candidate

    def _admissible(self, values):
        if np.any(values < 0) or not np.all(np.isfinite(values)):
            return False
        masses = self._base.particles.masses.copy()
        for index, value in zip(
            self._mass_indices,
            values[: self._mass_count] * self._scale[: self._mass_count],
            strict=True,
        ):
            masses[index] = value
        active = self._base.particles.concentration > 0
        return bool(np.all(masses.sum(axis=2)[active] > 0))

    def _rate(self, time, values):
        return self._strategy_rate(time, values, self._strategy)

    def _strategy_rate(self, time, values, strategy):
        if not self._admissible(values):
            raise TrialDomainError("condensation trial has invalid inventory")
        state = self._unpack(time, values)
        mass_rate = np.zeros_like(state.particles.masses)
        gas_rate = np.zeros_like(state.gas.concentration)
        for box, pairs in enumerate(self._maps):
            view = _cpu_batch.box_view(state, box)
            try:
                rate = strategy.mass_transfer_rate(
                    view.particles,
                    view.gas,
                    float(view.environment.temperature[0]),
                    float(view.environment.pressure[0]),
                    None
                    if self._viscosity is None
                    else float(self._viscosity[box]),
                )
            except getattr(
                strategy.activity_strategy, "trial_failure_types", ()
            ) as error:
                raise TrialDomainError(
                    f"condensation activity at box {box}, "
                    f"rate at t={time:g}: {error}"
                ) from error
            mass_rate[box] = rate
            if strategy.update_gases:
                for source, target in pairs:
                    gas_rate[box, source] = -np.dot(
                        self._weights[box], rate[:, target]
                    )
        return (
            np.array(
                [mass_rate[index] for index in self._mass_indices]
                + [gas_rate[index] for index in self._gas_indices]
            )
            / self._scale
        )

    def _jacobian(self, time, values):
        from particula.execution._condensation_jacobian import (
            coordinate_steps,
            sparse_jacobian,
            supports_local_rates,
        )

        if values.size > 64 and supports_local_rates(self._strategy):
            return sparse_jacobian(self, time, values)

        # Keep physical perturbations local after large wet-to-dry changes.
        differences = coordinate_steps(self, values)

        def rate(state):
            return self._strategy_rate(time, state, self._linear_strategy)

        base = rate(values)
        jacobian = np.empty((values.size, values.size))
        for index in range(values.size):
            delta = differences[index]
            plus, minus = values.copy(), values.copy()
            plus[index] += delta
            minus[index] -= delta
            if self._admissible(minus):
                jacobian[:, index] = (rate(plus) - rate(minus)) / (2 * delta)
            else:
                jacobian[:, index] = (rate(plus) - base) / delta
        return jacobian

    def _partial_time(self, time, values):
        if self._forcing is None:
            return np.zeros_like(values)
        delta = np.sqrt(np.finfo(float).eps) * max(abs(time), 1.0)
        if type(self._forcing) is _EnvironmentSegment:
            delta = min(delta, self._forcing.end - time)
        if delta <= 0:
            return np.zeros_like(values)
        return (
            self._strategy_rate(time + delta, values, self._linear_strategy)
            - self._strategy_rate(time, values, self._linear_strategy)
        ) / delta

    @property
    def time(self):
        """Return accepted absolute time [s]."""
        return self.solver.time

    def step(self, bound):
        """Accept one physical step no later than bound [s]."""
        bound = AdaptiveSolver._scalar(bound, "bound", positive=False)
        with forcing_interval(self, bound) as endpoint:
            return self.solver.step(endpoint)

    def snapshot(self):
        """Return detached accepted native state, including forcing."""
        return self._unpack(self.time, self.solver.state)
