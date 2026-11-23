"""Closed parcel with analytical particle masses and finite vapor species.

Mixing ratios refer to one kg of dry air. The dilute first law neglects
condensate volume and sensible heat, dry aerosol loading and entrainment.
The continuum diffusion law uses unit accommodation, unit ventilation and
a single linearized conductive heat balance at the surface of each droplet.
"""

from dataclasses import dataclass
from operator import index

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import Radau

from particula.gas.properties.thermal_conductivity import (
    get_thermal_conductivity,
)
from particula.particles.activity_strategies import ActivityStrategy
from particula.util.constants import (
    GAS_CONSTANT,
    MOLECULAR_WEIGHT_AIR,
    STANDARD_GRAVITY,
)

FloatArray = NDArray[np.float64]
RD = GAS_CONSTANT / MOLECULAR_WEIGHT_AIR
CP = 1005.0


@dataclass(frozen=True)
class ParcelVapor:
    """Pure species properties for dilute continuum transfer to one liquid.

    name and material_index bind the gas to an analytical particle material.
    molar_mass is kg/mol, reference_pressure is Pa at reference_temperature
    (K), latent_heat is J/kg and diffusivity is m²/s at 273.15 K/101325 Pa.
    Unit accommodation applies to this continuum model. With enabled=False
    the vapor remains part of the gas law and its condensed material still
    contributes to AIOMFAC activities and wet radius.
    """

    name: str
    material_index: int
    molar_mass: float
    reference_pressure: float
    reference_temperature: float
    latent_heat: float
    diffusivity: float
    accommodation: float = 1.0
    enabled: bool = True

    def __post_init__(self):
        """Reject unsupported transport properties before any integration."""
        props = [
            self.molar_mass,
            self.reference_pressure,
            self.reference_temperature,
            self.latent_heat,
            self.diffusivity,
        ]
        if not self.name or not np.all(np.isfinite(props)) or min(props) <= 0:
            raise ValueError(
                "vapor properties must be named, positive and finite"
            )
        if self.accommodation != 1:
            raise ValueError("continuum parcel requires unit accommodation")
        if (
            isinstance(self.material_index, (bool, np.bool_))
            or index(self.material_index) < 0
        ):
            raise ValueError("material_index must be a nonnegative integer")


@dataclass(frozen=True)
class MultiVaporParcelInitialState:
    """An explicitly wet class population and finite vapor reservoirs.

    masses has shape (class, material) in kg/particle. number gives represented
    counts per class. vapor holds kg/kg dry air per gas in vapors order.
    density is kg/m³ per material. pressure, temperature, dry_air_mass,
    height and surface_tension use Pa, K, kg, m and N/m respectively.
    A vapor named 'water' identifies the supersaturation diagnostic.
    """

    masses: FloatArray
    number: FloatArray
    vapor: FloatArray
    material_names: tuple[str, ...]
    density: FloatArray
    vapors: tuple[ParcelVapor, ...]
    activity_strategy: ActivityStrategy
    pressure: float
    temperature: float
    dry_air_mass: float = 1.0
    height: float = 0.0
    surface_tension: float = 0.072

    def __post_init__(self):
        """Freeze caller-owned input arrays and verify extensive inventories."""
        for key in ("masses", "number", "vapor", "density"):
            value = np.array(getattr(self, key), dtype=np.float64, copy=True)
            if not np.all(np.isfinite(value)) or np.any(value < 0):
                raise ValueError(f"invalid parcel {key}")
            value.setflags(write=False)
            object.__setattr__(self, key, value)
        object.__setattr__(self, "material_names", tuple(self.material_names))
        object.__setattr__(self, "vapors", tuple(self.vapors))
        width = len(self.material_names)
        if (
            not width
            or not all(isinstance(k, str) and k for k in self.material_names)
            or len(set(self.material_names)) != width
            or self.density.shape != (width,)
            or np.any(self.density <= 0)
            or self.number.ndim != 1
            or not self.number.size
            or self.masses.shape != (self.number.size, width)
            or np.any((self.number > 0) != (self.masses.sum(axis=1) > 0))
        ):
            raise ValueError("invalid material layout, count or class mass")
        names = [v.name for v in self.vapors]
        targets = [v.material_index for v in self.vapors]
        if (
            "water" not in names
            or len(set(names)) != len(names)
            or self.vapor.shape != (len(names),)
            or len(set(targets)) != len(targets)
            or max(targets) >= width
        ):
            raise ValueError(
                "invalid parcel vapor layout or missing water reservoir"
            )
        props = [self.temperature, self.pressure, self.dry_air_mass]
        if (
            not np.all(np.isfinite(props))
            or min(props) <= 0
            or not np.isfinite(self.height)
            or not np.isfinite(self.surface_tension)
            or self.surface_tension < 0
        ):
            raise ValueError("invalid parcel environment")
        validate = getattr(
            self.activity_strategy, "validate_material_layout", None
        )
        if validate is not None:
            validate(width, targets)


@dataclass(frozen=True)
class MultiVaporParcelTrajectory:
    """Histories at common output times in SI and kg per kg of dry air.

    masses is (time, class, material) in kg/particle, wet_radius is (time,
    class) in m. vapor and liquid are (time, vapor) mixing ratios in declared
    gas order. volume is physical parcel m³, supersaturation is RH minus one.
    """

    time: FloatArray
    height: FloatArray
    pressure: FloatArray
    temperature: FloatArray
    volume: FloatArray
    vapor: FloatArray
    liquid: FloatArray
    masses: FloatArray
    wet_radius: FloatArray
    supersaturation: FloatArray
    initial: MultiVaporParcelInitialState


def multivapor_trajectory(
    initial, time, pressure, temperature, vapor, masses, updraft
):
    """Construct the shared observable record from CPU or device outputs."""
    time, pressure, temperature, vapor, masses = map(
        np.asarray, (time, pressure, temperature, vapor, masses)
    )
    gas_r = GAS_CONSTANT / np.array([v.molar_mass for v in initial.vapors])
    volume = (RD + vapor @ gas_r) * temperature / pressure
    targets = [v.material_index for v in initial.vapors]
    water = [v.name for v in initial.vapors].index("water")
    spec = initial.vapors[water]
    p_sat = spec.reference_pressure * np.exp(
        spec.latent_heat
        / gas_r[water]
        * (1 / spec.reference_temperature - 1 / temperature)
    )
    radius = np.cbrt(3 * (masses @ (1 / initial.density)) / (4 * np.pi))
    liquid = (
        np.einsum("c,tcs->ts", initial.number, masses[:, :, targets])
        / initial.dry_air_mass
    )
    return MultiVaporParcelTrajectory(
        time,
        initial.height + updraft * time,
        pressure,
        temperature,
        volume * initial.dry_air_mass,
        vapor,
        liquid,
        masses,
        radius,
        vapor[:, water] * gas_r[water] * temperature / (volume * p_sat) - 1,
        initial,
    )


class _TrialDomainError(ValueError):
    """A numerical trial left the physical domain before acceptance."""


def _uncertain_fluxes(
    rates, driving, slope, sink, offset, errors, latent, enabled
):
    """Bound the shared surface temperature before each species flux."""
    eps = np.expm1(errors)
    dheat = (abs(driving) * eps) @ latent
    dsink = (abs(slope) * eps) @ latent
    safe = sink > dsink
    doffset = (dheat + abs(offset) * dsink) / np.where(safe, sink - dsink, 1.0)
    bound = (
        eps * (abs(driving) + abs(slope * offset[:, None]))
        + abs(slope) * (1 + eps) * doffset[:, None]
    )
    return np.any(errors > 0, axis=1) & (
        ~safe | np.any((abs(rates) <= bound) & enabled, axis=1)
    )


def _activity_intervals(provider, masses, temperature, targets):
    """Return activities and optional error estimates for occupied classes."""
    args = (masses, temperature, targets, np.ones(len(masses), dtype=bool))
    if getattr(provider, "evaluation", None) == "hybrid" and hasattr(
        provider, "neutral_with_error"
    ):
        return provider.neutral_with_error(*args)
    return provider.neutral_activities(*args), None


class ParcelTendencies:
    """Evaluate the coupled instantaneous mass, pressure and heat budgets."""

    def __init__(self, initial, updraft, phase_exchange=True):
        """Bind property vectors and the conserved hydrostatic loading."""
        self.initial = initial
        self.updraft = updraft
        self.targets = np.array([v.material_index for v in initial.vapors])
        self.gas_r = GAS_CONSTANT / np.array(
            [v.molar_mass for v in initial.vapors]
        )
        self.latent = np.array([v.latent_heat for v in initial.vapors])
        self.diffusivity = np.array([v.diffusivity for v in initial.vapors])
        self.p0 = np.array([v.reference_pressure for v in initial.vapors])
        self.t0 = np.array([v.reference_temperature for v in initial.vapors])
        self.enabled = np.array(
            [phase_exchange and v.enabled for v in initial.vapors]
        )
        self.loading = (
            1
            + initial.vapor.sum()
            + initial.number
            @ initial.masses[:, self.targets].sum(axis=1)
            / initial.dry_air_mass
        )

    def rates(self, pressure, temperature, vapor, masses):
        """Return dp (Pa/s), dT (K/s), dq (kg/kg/s), and J (kg/particle/s)."""
        if (
            temperature <= 0
            or pressure <= 0
            or np.any(vapor < 0)
            or np.any(masses < 0)
        ):
            raise _TrialDomainError(
                "parcel trial crossed a nonnegative inventory or state boundary"
            )
        state = self.initial
        volume = (RD + vapor @ self.gas_r) * temperature / pressure
        dp = -self.loading * STANDARD_GRAVITY * self.updraft / volume
        flux = np.zeros((len(state.number), len(self.targets)))
        active = state.number > 0
        if active.any() and self.enabled.any():
            radius = np.cbrt(
                3 * (masses[active] @ (1 / state.density)) / (4 * np.pi)
            )[:, None]
            if np.any(radius <= 0):
                raise _TrialDomainError("occupied parcel class has zero volume")
            try:
                activity, errors = _activity_intervals(
                    state.activity_strategy,
                    masses[active],
                    temperature,
                    self.targets,
                )
            except getattr(
                state.activity_strategy, "trial_failure_types", ()
            ) as error:
                slot = int(np.flatnonzero(active)[error.batch_index[0]])
                raise _TrialDomainError(
                    f"parcel activity at box 0, slot {slot}, "
                    f"Radau rate evaluation: {error.reason}"
                ) from error
            curvature = (
                2
                * state.surface_tension
                / (
                    self.gas_r
                    * temperature
                    * state.density[self.targets]
                    * radius
                )
            )
            psat = self.p0 * np.exp(
                self.latent / self.gas_r * (1 / self.t0 - 1 / temperature)
            )
            mobility = (
                4
                * np.pi
                * radius
                * self.diffusivity
                * (temperature / 273.15) ** 1.94
                * 101325
                / pressure
                / (self.gas_r * temperature)
            ) * self.enabled
            conduction = (
                4 * np.pi * radius[:, 0] * get_thermal_conductivity(temperature)
            )

            def surface_rates(activity):
                equilibrium = activity * psat * np.exp(curvature)
                isothermal = mobility * (
                    vapor * self.gas_r * temperature / volume - equilibrium
                )
                # Linearize surface equilibrium vapor density about the trial
                # temperature. All vapors share one conductive heat sink.
                slope = (
                    mobility
                    * equilibrium
                    * (self.latent / (self.gas_r * temperature) - 1 - curvature)
                    / temperature
                )
                sink = conduction + slope @ self.latent
                surface_offset = (isothermal @ self.latent) / sink
                return (
                    isothermal - slope * surface_offset[:, None],
                    slope,
                    sink,
                    surface_offset,
                    mobility * equilibrium,
                )

            rates, slope, sink, offset, driving = surface_rates(activity)
            if errors is not None:
                uncertain = _uncertain_fluxes(
                    rates,
                    driving,
                    slope,
                    sink,
                    offset,
                    errors,
                    self.latent,
                    self.enabled,
                )
                if uncertain.any():
                    try:
                        exact = state.activity_strategy.refine_neutrals(
                            masses[active], temperature, self.targets, uncertain
                        )
                    except getattr(
                        state.activity_strategy, "trial_failure_types", ()
                    ) as error:
                        slot = int(np.flatnonzero(active)[error.batch_index[0]])
                        raise _TrialDomainError(
                            f"parcel activity at box 0, slot {slot}: "
                            f"{error.reason}"
                        ) from error
                    activity[uncertain] = exact[uncertain]
                    rates = surface_rates(activity)[0]
            # A dry material lane can take up vapor at its deliquescence
            # boundary. It has no mass available for evaporation.
            dry = masses[active][:, self.targets] == 0
            flux[active] = np.where(dry, np.maximum(rates, 0.0), rates)
        dq = -(state.number @ flux) / state.dry_air_mass
        dtemp = (volume * dp - dq @ self.latent) / CP
        if not np.isfinite(dtemp) or not np.all(np.isfinite(flux)):
            raise _TrialDomainError(
                "unrepresentable parcel heat or mass transfer"
            )
        return dp, dtemp, dq, flux


def cpu_multivapor_reference(
    initial, time, updraft, phase_exchange=True, rtol=2e-10
):
    """Integrate coupled inventories independently using scaled SciPy Radau.

    An initial class mass scale per lane and independent vapor variables keep
    the embedded error estimate sensitive to both trace and bulk species.
    Failed physical Radau trials retry from the accepted state at a smaller
    step. Provider errors on the initial state are reported immediately.
    """
    law = ParcelTendencies(initial, updraft, phase_exchange)
    mass0 = initial.masses[:, law.targets]
    scale = np.maximum(
        mass0, np.maximum(initial.masses.sum(axis=1)[:, None] * 1e-6, 1e-30)
    )
    qscale = np.maximum(
        initial.vapor + initial.number @ mass0 / initial.dry_air_mass, 1e-18
    )
    nv = len(law.targets)

    def decode(y):
        masses = initial.masses.copy()
        masses[:, law.targets] = y[2 + nv :].reshape(scale.shape) * scale
        return (
            y[0] * initial.pressure,
            y[1] * initial.temperature,
            y[2 : 2 + nv] * qscale,
            masses,
        )

    def rhs(_, y):
        dp, dt, dq, flux = law.rates(*decode(y))
        return np.r_[
            dp / initial.pressure,
            dt / initial.temperature,
            dq / qscale,
            (flux / scale).ravel(),
        ]

    y0 = np.r_[1.0, 1.0, initial.vapor / qscale, (mass0 / scale).ravel()]
    rhs(0.0, y0)
    end = float(time[-1])

    def new_solver(t, y, max_step):
        return Radau(
            rhs,
            t,
            y,
            end,
            rtol=rtol,
            atol=rtol * 1e-3,
            max_step=max_step,
            first_step=min(max_step, 1e-3, end - t),
        )

    solver = new_solver(0.0, y0, end)
    results = [decode(y0)]
    pos = 1
    while solver.status == "running":
        old_t, old_y = solver.t, solver.y.copy()
        try:
            message = solver.step()
            if solver.status == "failed":
                raise _TrialDomainError(f"multi-vapor Radau failed: {message}")
            rhs(solver.t, solver.y)
        except _TrialDomainError as exc:
            step = min(solver.h_abs, end - old_t, solver.max_step) * 0.25
            if step <= max(1e-12, abs(old_t) * np.finfo(float).eps * 16):
                raise ValueError(
                    f"multi-vapor Radau exhausted trial at {old_t:g} s: {exc}"
                ) from exc
            solver = new_solver(old_t, old_y, step)
            continue
        dense = solver.dense_output()
        while pos < len(time) and time[pos] <= solver.t:
            results.append(decode(dense(time[pos])))
            pos += 1
    pressure, temperature, vapor, masses = map(
        np.stack, zip(*results, strict=True)
    )
    return multivapor_trajectory(
        initial, time, pressure, temperature, vapor, masses, updraft
    )
