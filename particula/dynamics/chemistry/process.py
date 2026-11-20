"""Adaptive native gas chemistry over a prescribed-temperature interval."""

import json
from dataclasses import astuple, dataclass
from math import isfinite

import numpy as np

from particula.box_process import BoxProcess
from particula.dynamics.chemistry.binding import BoundMechanism
from particula.execution.ode import AdaptiveSolver, ODEProblem


@dataclass(frozen=True)
class ChemistrySolver:
    """Per-interval adaptive controls shared by NumPy and resident execution.

    Absolute tolerance is in mol/m³ of air. A tuple follows normalized
    mechanism species order. The trial budget counts all accepted and
    rejected trials in one box during one process interval.
    """

    method: str = "rk23"
    rtol: float = 1e-7
    atol: float | tuple[float, ...] = 1e-14
    first_step: float = 1e-3
    min_step: float = 1e-14
    max_step: float = 1.0
    max_trials: int = 100_000

    def __post_init__(self):
        """Validate tolerances and positive finite timestep limits."""
        if self.method not in ("rk23", "ros3"):
            raise ValueError("gas chemistry method must be rk23 or ros3")
        if not isfinite(self.rtol) or self.rtol < 0:
            raise ValueError("rtol must be finite and nonnegative")
        atol = np.asarray(self.atol, dtype=np.float64)
        if (
            atol.ndim > 1
            or not atol.size
            or not np.all(np.isfinite(atol) & (atol > 0))
        ):
            raise ValueError("atol must contain finite positive tolerances")
        object.__setattr__(
            self, "atol", tuple(atol.tolist()) if atol.ndim else float(atol)
        )
        if (
            not all(
                isfinite(h)
                for h in (self.min_step, self.first_step, self.max_step)
            )
            or not 0 < self.min_step <= self.first_step <= self.max_step
        ):
            raise ValueError("require 0 < min_step <= first_step <= max_step")
        if type(self.max_trials) is not int or self.max_trials < 1:
            raise ValueError("max_trials must be a positive integer")


_DEFAULT_SOLVER = ChemistrySolver()


class GasChemistry(BoxProcess):
    """Advance bound gas concentrations in kg/m³ with staged box updates.

    Temperature is prescribed and constant within each process interval.
    Each interval starts a fresh controller, so only accepted concentrations
    and the simulation clock carry forward to the next process boundary.
    """

    _namespace = "gas_chemistry"

    def __init__(
        self,
        bound: BoundMechanism,
        solver: ChemistrySolver = _DEFAULT_SOLVER,
        *,
        name="chemistry",
    ):
        """Retain the immutable mechanism, gas layout and solver controls."""
        if not isinstance(bound, BoundMechanism) or not isinstance(
            solver, ChemistrySolver
        ):
            raise TypeError("expected a bound mechanism and ChemistrySolver")
        if isinstance(solver.atol, tuple) and len(solver.atol) != len(
            bound.indices
        ):
            raise ValueError("per-species atol must match the mechanism")
        self.bound = bound
        self.solver = solver
        super().__init__(
            name=name,
            extension_id="particula.gas_chemistry",
            reads=("gas.concentration", "environment.temperature"),
            writes=("gas.concentration",),
            parameters=self.configuration(),
        )

    def configuration(self):
        """Return native mechanism, binding and solver identity for restart."""
        return (
            json.dumps(
                self.bound.mechanism.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
            ),
            self.bound.gas_names,
            self.bound.indices,
            self.bound.molar_masses,
            astuple(self.solver),
        )

    def validate(self, aerosol):
        """Check bound layout, temperature and participating gas inventory."""
        self.bound.validate(aerosol.gas)
        state = aerosol.gas.concentration[:, self.bound.indices]
        temp = aerosol.environment.temperature
        if not np.all(np.isfinite(state) & (state >= 0)):
            raise ValueError(
                "chemistry needs finite nonnegative concentrations"
            )
        if not np.all(np.isfinite(temp) & (temp > 0)):
            raise ValueError("chemistry needs finite positive temperatures")

    def integrate(self, state, temperature, duration):
        """Advance mol/m³ for one interval and return solver statistics."""
        kinetics = self.bound.kinetics(temperature)
        settings = self.solver
        problem = ODEProblem(
            kinetics.rate,
            lambda y: bool(np.all(y >= 0)),
            kinetics.jacobian,
            lambda t, y: np.zeros_like(y),
        )
        ode = AdaptiveSolver(
            problem,
            state,
            method=settings.method,
            rtol=settings.rtol,
            atol=settings.atol,
            first_step=settings.first_step,
            min_step=settings.min_step,
            max_step=settings.max_step,
            max_trials=settings.max_trials,
        )
        while ode.time < duration:
            remaining = (
                settings.max_trials
                - ode.statistics.accepted
                - ode.statistics.rejected
            )
            if remaining <= 0:
                raise RuntimeError("gas chemistry trial budget exceeded")
            ode.max_trials = remaining
            ode.step(duration)
        return ode.state, ode.statistics

    def advance(self, aerosol, *, time, time_step, context):
        """Convert bound columns to molar units and publish accepted mass."""
        gas = aerosol.gas.concentration[0]
        idx = np.asarray(self.bound.indices)
        initial = gas[idx] / self.bound.molar_masses
        state, _ = self.integrate(
            initial,
            aerosol.environment.temperature[0],
            time_step,
        )
        changed = (state != initial) & np.any(self.bound.stoichiometry, axis=0)
        gas[idx[changed]] = (
            state[changed] * np.asarray(self.bound.molar_masses)[changed]
        )
