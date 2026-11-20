"""Closed gas, bulk particle and collision-limited uptake in resolved boxes."""

import json
from dataclasses import astuple

import numpy as np

from particula.box_process import BoxProcess
from particula.dynamics.chemistry.process import (
    _DEFAULT_SOLVER,
    ChemistrySolver,
    _integrate_problem,
)
from particula.dynamics.chemistry.resolved_binding import (
    BULK,
    GAS,
    UPTAKE,
    ResolvedBinding,
)
from particula.execution.ode import ODEProblem
from particula.util.constants import GAS_CONSTANT


class _BoxKinetics:
    """Scale each particle's integrated moles by its initial volume."""

    def __init__(self, bound, aerosol):
        particles = aerosol.particles
        masses = particles.masses[0]
        self.active = particles.concentration[0] > 0
        self.weights = particles.concentration[0] / particles.volume[0]
        self.reference = (masses / particles.density).sum(axis=1)
        self.pm = np.asarray(bound.particle_masses)
        self.gm = np.asarray(bound.gas_masses)
        self.ng = self.gm.size
        self.partial_volume = self.pm / particles.density
        reference = np.where(self.active, self.reference, 1.0)
        self.initial = np.concatenate(
            (
                aerosol.gas.concentration[0] / self.gm,
                (masses / self.pm / reference[:, None]).ravel(),
            )
        )
        temp = aerosol.environment.temperature[0]
        a, exponent, energy, t_ref = np.asarray(bound.parameters).T
        self.k = (
            a
            * (temp / t_ref) ** exponent
            * np.exp(-energy / (GAS_CONSTANT * temp))
        )
        self.orders = np.asarray(bound.orders)
        self.net = np.asarray(bound.net)
        self.kinds = bound.kinds
        self.sources = bound.sources
        self.speed = np.sqrt(8 * GAS_CONSTANT * temp / (np.pi * self.gm))
        if not np.all(np.isfinite(self.k)):
            raise ValueError("nonfinite chemistry rate coefficient")

    def admissible(self, state):
        """Reject a negative concentration or an emptied active slot."""
        particle = state[self.ng :].reshape(-1, self.pm.size)
        return bool(
            np.all(np.isfinite(state))
            and np.all(state >= 0)
            and np.all(particle[self.active].sum(axis=1) > 0)
        )

    def rate(self, time, state):
        """Return gas mol/m³-air/s and material mol/m³-initial-particle/s."""
        gas = state[: self.ng]
        particle = state[self.ng :].reshape(-1, self.pm.size)
        dg = np.zeros_like(gas)
        dp = np.zeros_like(particle)
        active = self.active
        ratio = particle[active] @ self.partial_volume
        concentration = particle[active] / ratio[:, None]
        volume = self.reference[active] * ratio
        # Surface area of spheres with the instantaneous additive volume.
        radius = np.cbrt(volume * (3 / (4 * np.pi)))
        area = 4 * np.pi * radius**2
        for r, kind in enumerate(self.kinds):
            if kind == GAS:
                flux = self.k[r] * np.prod(gas ** self.orders[r, : self.ng])
                dg += self.net[r, : self.ng] * flux
            elif kind == BULK:
                flux = (
                    self.k[r]
                    * ratio
                    * np.prod(
                        concentration ** self.orders[r, self.ng :], axis=1
                    )
                )
                dp[active] += flux[:, None] * self.net[r, self.ng :]
            else:
                source = self.sources[r]
                extent = (
                    (self.k[r] * self.speed[source] * 0.25) * gas[source] * area
                )
                dg[source] -= self.weights[active] @ extent
                dp[active] += (extent / self.reference[active])[
                    :, None
                ] * self.net[r, self.ng :]
        return np.concatenate((dg, dp.ravel()))

    def publish(self, aerosol, final):
        """Publish changed lanes and preserve inactive and inert values."""
        g = final[: self.ng]
        changed = g != self.initial[: self.ng]
        if np.any(changed):
            aerosol.gas.concentration[0, changed] = (
                g[changed] * self.gm[changed]
            )
        p = final[self.ng :].reshape(-1, self.pm.size)
        change = p != self.initial[self.ng :].reshape(p.shape)
        out = p * self.pm * self.reference[:, None]
        aerosol.particles.masses[0, change] = out[change]


class ResolvedChemistry(BoxProcess):
    """Couple a version-2 mechanism across gas and resolved particle masses.

    Bulk rates use mol/m³ of instantaneous particle material. Uptake uses
    a spherical collision area, prescribed net uptake probability and mean
    molecular speed. RK23 absolute tolerances use mol/m³ of air for gas
    and mol/m³ of the initial particle volume for each process interval.
    """

    _namespace = "resolved_chemistry"

    def __init__(
        self,
        bound: ResolvedBinding,
        solver=_DEFAULT_SOLVER,
        *,
        name="resolved_chemistry",
    ):
        """Select the immutable layout and per-box adaptive controls."""
        if not isinstance(bound, ResolvedBinding) or not isinstance(
            solver, ChemistrySolver
        ):
            raise TypeError("expected a resolved binding and ChemistrySolver")
        if solver.method != "rk23" or isinstance(solver.atol, tuple):
            raise ValueError(
                "resolved chemistry requires RK23 and scalar molar atol"
            )
        self.bound = bound
        self.solver = solver
        writes = ["particles.masses"]
        if GAS in bound.kinds or UPTAKE in bound.kinds:
            writes.append("gas.concentration")
        super().__init__(
            name=name,
            extension_id="particula.resolved_chemistry",
            reads=(
                "particles.masses",
                "particles.concentration",
                "particles.volume",
                "gas.concentration",
                "environment.temperature",
            ),
            writes=tuple(writes),
            representations=("particle_resolved",),
            parameters=self.configuration(),
        )

    def configuration(self):
        """Return restart identity including densities, mappings and solver."""
        b = self.bound
        return (
            json.dumps(
                b.mechanism.to_dict(), sort_keys=True, separators=(",", ":")
            ),
            b.indices,
            b.gas_names,
            b.gas_masses,
            b.density,
            astuple(self.solver),
        )

    def validate(self, aerosol):
        """Validate current representation, layout and inventories."""
        self.bound.validate(aerosol)

    def advance(self, aerosol, *, time, time_step, context):
        """Integrate one simultaneous closed material balance within a box."""
        kinetics = _BoxKinetics(self.bound, aerosol)
        final, _ = _integrate_problem(
            ODEProblem(kinetics.rate, kinetics.admissible),
            kinetics.initial,
            time_step,
            self.solver,
            "resolved chemistry",
        )
        kinetics.publish(aerosol, final)
