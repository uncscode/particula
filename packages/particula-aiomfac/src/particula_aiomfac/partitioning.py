"""Bulk vapor, liquid and optional crystal equilibrium at fixed temperature."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.special import expit

from ._equilibrium import ConvergenceError
from .phases import (
    LiquidEquilibrium,
    _LiquidSystem,
    _stability,
    component_molar_masses,
)
from .solids import SolidLiquidEquilibrium, solve_solid_liquid


@dataclass(frozen=True)
class GasLiquidEquilibrium:
    """Gas concentrations in µg/m³ in declaration order. RH is a ratio.

    Water exchanged is positive for net uptake from the external reservoir.
    With crystals enabled, solid_liquid owns the condensed physical inventory,
    including dry states where liquids is None.
    """

    liquids: LiquidEquilibrium | None
    gas_concentrations: np.ndarray
    relative_humidity: float
    water_exchanged: float
    vapor_residual: float
    solid_liquid: SolidLiquidEquilibrium | None = None


def _saturation_vector(provider, totals, temperature, saturations, rh):
    names = tuple(provider.mixture.components)
    if (
        totals.shape != (len(names),)
        or not np.isfinite(totals).all()
        or np.any(totals < 0)
    ):
        raise ValueError(
            "expected one nonnegative finite analytical inventory per component"
        )
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite in K")
    if names[provider.water_index] not in saturations:
        raise ValueError("supply the pure-water saturation concentration")
    if rh is not None and not 0 <= rh <= 1:
        raise ValueError("relative_humidity must be in [0, 1]")
    provider.material_lanes(saturations)
    cstar = np.zeros_like(totals)
    for name, concentration in saturations.items():
        if not np.isfinite(concentration) or concentration <= 0:
            raise ValueError(
                "saturation concentrations must be positive and finite"
            )
        cstar[names.index(name)] = concentration
    return cstar


def _crystal_assemblage(state, dry_feed):
    dry = state(dry_feed)[1]
    if (
        dry.dry_ion_masses.sum() > 2e-12 * dry_feed.sum()
        or not dry.solid_masses.any()
    ):
        raise ValueError(
            "dry-to-wet stability requires a fully assigned mineral inventory"
        )
    return dry


def _incipient_brine(state, dry_feed, dry, composition):
    """Resolve the saturated liquid at one trial neutral mass composition.

    Every crystal initially present must remain in excess. The first mineral
    exhausted under proportional uptake sets the extent of this assemblage.
    """
    present = dry.solid_masses > 0
    sample = np.min(dry.solid_masses[present])
    for _ in range(15):
        _, brine = state(dry_feed + sample * composition)
        if np.all(
            brine.solid_masses[present] >= 0.05 * dry.solid_masses[present]
        ):
            consumed = dry.solid_masses - brine.solid_masses
            dissolving = consumed > 0
            if dissolving.any():
                endpoint = sample * np.min(
                    dry.solid_masses[dissolving] / consumed[dissolving]
                )
                return brine, endpoint
        sample *= 0.1
    raise ConvergenceError(
        "could not resolve a saturated incipient brine with the dry mineral assemblage"
    )


def _crystal_water_partition(state, totals, cstar, water, water_name, rh):
    """Partition water among vapor, brine and a dry mineral assemblage."""
    dry_feed = totals.copy()
    dry_feed[water] = 0.0
    dry = _crystal_assemblage(state, dry_feed)
    water_column = dry.neutral_ids.index(water_name)
    unit = np.zeros_like(dry_feed)
    unit[water] = 1.0

    def wet(water_mass):
        phases, solids = state(dry_feed + water_mass * unit)
        return phases.alpha.neutral_activities[water_column], solids

    brine, saturated_water = _incipient_brine(state, dry_feed, dry, unit)
    drh = brine.liquids.alpha.neutral_activities[water_column]
    if not 0 < drh < 1:
        raise ConvergenceError("invalid water activity of saturated brine")
    saturated_gas = cstar[water] * drh

    def result(condensed_water, solids, gas_water, residual):
        gas = np.zeros_like(totals)
        gas[water] = gas_water
        exchange = (
            0.0 if rh is None else gas_water + condensed_water - totals[water]
        )
        return GasLiquidEquilibrium(
            solids.liquids,
            gas,
            gas_water / cstar[water],
            exchange,
            residual,
            solids,
        )

    if rh is None:
        vapor = totals[water]
        if vapor <= saturated_gas:
            return result(0.0, dry, vapor, 0.0)
        brine_water = vapor - saturated_gas
        if brine_water <= saturated_water:
            aw, solids = wet(brine_water)
            return result(
                brine_water, solids, saturated_gas, abs(np.log(aw / drh))
            )

        # Above complete dissolution both terms in the budget vary with the
        # water mass. The dimensionless coordinate keeps the tolerance relative.
        def budget(fraction):
            aw = wet(vapor * fraction)[0]
            return fraction + aw * cstar[water] / vapor - 1

        lo = saturated_water / vapor
        fraction = brentq(budget, lo, 1.0, xtol=1e-14, rtol=1e-14)
        water_mass = vapor * fraction
        aw, solids = wet(water_mass)
        # Evaluating the vapor from activity preserves a trace vapor at large
        # water budgets. The root independently enforces the total budget.
        gas_water = aw * cstar[water]
        error = abs((water_mass + gas_water) / vapor - 1)
        return result(water_mass, solids, gas_water, error)

    if abs(rh - drh) <= 1e-10:
        raise ValueError(
            "prescribed deliquescence RH has no unique brine amount"
        )
    if rh < drh:
        return result(0.0, dry, rh * cstar[water], 0.0)
    if rh == 1:
        raise ValueError(
            "saturated bulk reservoir requires unbounded water uptake"
        )

    def mismatch(log_ratio):
        return wet(saturated_water * np.exp(log_ratio))[0] - rh

    for high in range(1, 38, 2):
        if mismatch(high) >= 0:
            break
    else:
        raise ConvergenceError(
            "could not bracket the deliquesced water inventory"
        )
    ratio = brentq(mismatch, 0, high, xtol=1e-13)
    water_mass = saturated_water * np.exp(ratio)
    aw, solids = wet(water_mass)
    return result(water_mass, solids, rh * cstar[water], abs(np.log(aw / rh)))


class _CrystalVaporStability:
    """Tangent grand potential of an incipient brine against a vapor reservoir."""

    def __init__(self, state, dry_feed, dry, volatile, gas, model):
        self.state = state
        self.dry_feed = dry_feed
        self.dry = dry
        self.lanes = np.flatnonzero(volatile)
        self.mw = component_molar_masses(model)[self.lanes]
        moles = gas[self.lanes] / self.mw
        self.feed = moles / moles.sum()
        self.cols = [
            int(np.where(model.neutral_indices == lane)[0][0])
            for lane in self.lanes
        ]

    def mass_fractions(self, x):
        composition = np.zeros_like(self.dry_feed)
        weights = x * self.mw
        composition[self.lanes] = weights / weights.sum()
        return composition

    def potentials(self, x):
        brine, _ = _incipient_brine(
            self.state, self.dry_feed, self.dry, self.mass_fractions(x)
        )
        return np.log(brine.liquids.alpha.neutral_activities[self.cols])


def _crystal_vapor_stability(
    state, trial_state, totals, cstar, water, rh, model
):
    """Check for the appearance of liquid and bracket its initial inventory."""
    gas = np.where(cstar > 0, totals, 0.0)
    if rh is not None:
        gas[water] = rh * cstar[water]
    dry_feed = np.where(cstar == 0, totals, 0.0)
    dry = _crystal_assemblage(state, dry_feed)
    active = gas > 0
    # TPD searches for the very first *individual* liquid. A two-liquid trial
    # has a mole-weighted mean grand potential bounded below by its best liquid.
    trial = _CrystalVaporStability(
        trial_state, dry_feed, dry, active, gas, model
    )
    reference = np.log(gas[active] / cstar[active])
    if len(reference) == 1:
        composition = np.ones(1)
        minimum = float(trial.potentials(composition)[0] - reference[0])
    else:
        minimum, composition = _stability(trial, reference)[0]
    if minimum >= -1e-8:
        # Any closed organic inventory fixes the dry-limit uptake to zero.
        # At incipient saturation it stays entirely in the vapor.
        return (gas, dry_feed, None, dry), None
    direction = trial.mass_fractions(composition)
    closed = active.copy()
    if rh is not None:
        closed[water] = False
    limiting = np.min(totals[closed] / direction[closed])

    # The incipient-liquid grand potential is negative. Along this path at
    # least one closed vapor is eventually exhausted; find a finite wet start
    # where its directional derivative vanishes before freeing all transfers.
    def slope(fraction):
        if fraction == 0:
            return minimum
        if fraction == 1:
            return np.inf
        condensed = fraction * limiting * direction
        phases, _ = state(dry_feed + condensed)
        vapor = gas.copy()
        vapor[closed] -= condensed[closed]
        ratio = phases.alpha.neutral_activities[trial.cols] / (
            vapor[active] / cstar[active]
        )
        return composition @ np.log(ratio)

    fraction = brentq(slope, 0, 1, xtol=1e-14)
    if not 0 < fraction < 1:
        raise ConvergenceError(
            "could not resolve incipient liquid uptake from the finite vapor"
        )
    return None, fraction * limiting * direction


def _vapor_only_state(
    provider, backend, state, totals, cstar, gas, temperature, rh
):
    """Check whether a vapor with no permanent condensed inventory can condense."""
    occupied = gas > 0
    zeros = np.zeros_like(gas)
    water = provider.water_index
    if not np.any(occupied):
        return gas, zeros, *state(zeros)
    reference = np.log(gas[occupied] / cstar[occupied])
    system = _LiquidSystem(
        provider.prepared, gas, temperature, evaluator=backend
    )
    if len(reference) > 1:
        minimum = _stability(system, reference)[0][0]
    else:
        lane = int(np.flatnonzero(occupied)[0])
        mu = system.potentials(np.ones(1))[0]
        minimum = float(mu - reference[0])
        if minimum < -1e-9 and (rh is None or lane != water):
            gas = gas.copy()
            gas[lane] = cstar[lane] * np.exp(mu)
            condensed = zeros.copy()
            condensed[lane] = totals[lane] - gas[lane]
            return gas, condensed, *state(condensed)
        if rh is not None and lane == water and abs(minimum) <= 1e-9:
            raise ValueError(
                "pure water at prescribed saturation has no unique liquid amount"
            )
    if minimum >= -1e-9:
        return gas, zeros, *state(zeros)
    return None


def _wet_partition(
    state, totals, cstar, water, names, rh, neutral_ids, start=None
):
    if rh == 1:
        raise ValueError(
            "saturated bulk reservoir may require unbounded water uptake"
        )
    closed = (cstar > 0) & (totals > 0)
    if rh is not None:
        closed[water] = False
    variable = np.flatnonzero(closed)
    wet = rh is not None and rh > 0
    scale = max(
        totals[np.arange(len(names)) != water].sum(), np.finfo(float).tiny
    )
    cols = [neutral_ids.index(names[lane]) for lane in variable]
    if wet:
        cols.append(neutral_ids.index(names[water]))

    def evaluate(v):
        condensed = totals.copy()
        gas = np.zeros_like(totals)
        # Each closed neutral preserves its inventory even near depletion.
        condensed[variable] = totals[variable] * expit(v[: len(variable)])
        gas[variable] = totals[variable] * expit(-v[: len(variable)])
        if rh is not None:
            gas[water] = rh * cstar[water]
            condensed[water] = scale * np.exp(v[-1]) if wet else 0.0
        liquids, solids = state(condensed)
        target = gas[variable] / cstar[variable]
        if wet:
            target = np.r_[target, rh]
        residual = np.log(liquids.alpha.neutral_activities[cols]) - np.log(
            target
        )
        return residual, gas, condensed, liquids, solids

    guess = np.log(scale / cstar[variable])
    if wet:
        guess = np.r_[guess, np.log(rh / (1 - rh))]
    if start is not None:
        guess = np.log(start[variable] / (totals[variable] - start[variable]))
        if wet:
            guess = np.r_[guess, np.log(start[water] / scale)]
    if guess.size:
        solution = least_squares(
            lambda v: evaluate(v)[0],
            np.clip(guess, -40, 40),
            bounds=(-80.0, 80.0),
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
        )
        guess = solution.x
    residual, gas, condensed, liquids, solids = evaluate(guess)
    error = float(np.max(np.abs(residual), initial=0.0))
    if not np.isfinite(error) or error > 1e-7:
        raise ConvergenceError(f"gas-liquid equilibrium log residual {error:g}")
    return gas, condensed, liquids, solids, error


def gas_liquid_equilibrium(
    provider,
    *,
    inventories,
    temperature,
    saturation_concentrations,
    relative_humidity=None,
    execution="cpu",
    device=None,
    phase_mode="stable",
    solids=None,
):
    """Partition analytical inventories in µg/m³ over ideal vapor and condensed phases.

    Saturation concentrations (µg/m³) designate exchanging neutral compounds.
    Include water. An omitted RH closes all inventories. At fixed RH,
    initial water is a reference for net reservoir exchange. The configured
    minerals apply by default. An empty solids mapping disables crystals.
    Dry seeds require a complete set of minerals for their constituent ions.
    A multistart stability search tests an incipient brine against the gas.
    """
    if phase_mode not in ("stable", "single_liquid"):
        raise ValueError("phase_mode must be stable or single_liquid")
    rh = relative_humidity
    names = tuple(provider.mixture.components)
    water = provider.water_index
    totals = np.asarray(inventories, dtype=float)
    cstar = _saturation_vector(
        provider, totals, temperature, saturation_concentrations, rh
    )
    minerals = provider.solids if solids is None else solids
    backend = provider.equilibrium_backend(execution, device)

    def state(masses, mode=phase_mode):
        if minerals:
            crystal = solve_solid_liquid(
                provider,
                masses,
                temperature,
                minerals,
                execution=execution,
                device=device,
                phase_mode=mode,
            )
            return crystal.liquids, crystal
        if not masses.any():
            return None, None
        if mode == "stable":
            return backend.phase_equilibrium(masses, temperature, water), None
        direct = backend.evaluate(masses, temperature, diagnostics=True)
        return LiquidEquilibrium(
            masses.copy(), None, direct, None, 0.0, np.nan
        ), None

    neutral = np.isin(names, provider.prepared.neutrals)
    organic = neutral.copy()
    organic[water] = False
    answer, start = None, None
    if (
        minerals
        and np.any(totals[~neutral])
        and not np.any(totals[organic & (cstar == 0)])
    ):
        if not np.any(totals[organic]):
            return _crystal_water_partition(
                state, totals, cstar, water, names[water], rh
            )
        answer, start = _crystal_vapor_stability(
            state,
            lambda m: state(m, "single_liquid"),
            totals,
            cstar,
            water,
            rh,
            provider.prepared,
        )
    elif not np.any(totals[cstar == 0]):
        gas = np.where(cstar > 0, totals, 0.0)
        if rh is not None:
            gas[water] = rh * cstar[water]
        answer = _vapor_only_state(
            provider, backend, state, totals, cstar, gas, temperature, rh
        )
    if answer is None:
        gas, condensed, phases, crystal, error = _wet_partition(
            state,
            totals,
            cstar,
            water,
            names,
            rh,
            provider.prepared.neutrals,
            start,
        )
    else:
        gas, condensed, phases, crystal = answer
        error = 0.0
    exchange = (
        gas[water] + condensed[water] - totals[water] if rh is not None else 0.0
    )
    return GasLiquidEquilibrium(
        phases, gas.copy(), gas[water] / cstar[water], exchange, error, crystal
    )
