# %% [markdown]
# Replay Manchester chamber chemistry and condensation with Particula.
# Each box represents one chamber condition in SI units.
# %%
"""Construct and advance the Manchester chamber in Particula."""

from dataclasses import dataclass

import numpy as np
import particula as par
from particula.dynamics.chemistry import ReactionMechanism
from particula.dynamics.chemistry.adapter import GasChemistryAdapter
from particula.dynamics.chemistry.process import ChemistrySolver, GasChemistry
from particula.dynamics.condensation.species_mapping import (
    CondensationSpeciesMap,
)

R, PRESSURE, TREF = 8.314462618, 101325.0, 298.15
AP, O2 = 0.136238, 0.031998
OM = AP + O2
NAMES = (
    "A",
    "B",
    "oxygen",
    "intermediate",
    "vapor",
    "other",
    "wallA",
    "wallV",
    "oxid",
    "preloss",
    "aging",
    "vaporloss",
)
MW = np.array([AP, AP, O2, OM, OM, AP, AP, OM, 1.0, 1.0, 1.0, 1.0])
# Campaign Table 1 seed mass and injection. Temperature and number nearest
# lights-on from the archived RHT, WCPC or DMPS series. Run 1 uses WCPC.
# Run 6 has no RHT file and uses the campaign nominal temperature.
RUNS = {
    1: (309.0, 72.6, 10500.0, 26.29),
    4: (155.0, 45.7, 2460.0, 26.75),
    5: (155.0, 47.8, 16522.7, 26.59),
    6: (103.0, 51.0, 29100.0, 24.0),
}
SEED_DENSITY = 1770.0
OA_DENSITY = 1400.0


@dataclass(frozen=True)
class Case:
    """Define the chemical and physical assumptions for one box."""

    run: int
    fraction: float  # Mole fraction of precursor entering condensable branch
    vapor_wall_h: float
    particle_wall_h: float
    lag_min: float
    injection: float = 1.0
    number_factor: float = 1.0
    precursor_wall_s: float = 0.0
    wall_share: float = 1.0


def chamber(cases, slots=1, sigma=1.5):
    """Initialize resolved seed and a branched precursor inventory."""
    from scipy.special import ndtri

    nbox = len(cases)
    gas = np.zeros((nbox, len(NAMES)))
    masses = np.zeros((nbox, slots, 2))
    counts = np.zeros((nbox, slots))
    temps = np.empty(nbox)
    # Equal-number quadrature on a lognormal; scale diameters to preserve
    # measured total seed mass and number exactly, also for a single slot.
    shape = np.exp(np.log(sigma) * ndtri((np.arange(slots) + 0.5) / slots))
    mass_shape = shape**3 / np.mean(shape**3)
    for i, c in enumerate(cases):
        ppb, seed, count, celsius = RUNS[c.run]
        temps[i] = celsius + 273.15
        molecules = ppb * c.injection * 1e-9 * PRESSURE / (R * temps[i])
        gas[i, :3] = [
            molecules * AP * c.fraction,
            molecules * AP * (1 - c.fraction),
            molecules * O2 * 2,
        ]
        gas[i, 10:] = [1 / (60 * c.lag_min), c.vapor_wall_h / 3600]
        number = count * 1e6 * c.number_factor
        counts[i] = number / slots
        masses[i, :, 0] = seed * 1e-9 * mass_shape / number
    return par.Aerosol(
        particles=par.particles.ParticleData(
            masses=masses,
            concentration=counts,
            density=np.array([SEED_DENSITY, OA_DENSITY]),
            charge=np.zeros_like(counts),
            volume=np.ones(nbox),
            distribution_type="particle_resolved",
        ),
        gas=par.gas.GasData(
            list(NAMES),
            MW.copy(),
            gas,
            np.array([name == "vapor" for name in NAMES]),
        ),
        environment=par.gas.EnvironmentData(
            temperature=temps,
            pressure=np.full(nbox, PRESSURE),
            saturation_ratio=np.zeros_like(gas),
        ),
    )


def mechanism():
    """Build the mass-balanced reduced α-pinene oxidation mechanism."""

    def reaction(name, start, end, drive, oxygen=False):
        reactants = {"gas:" + start: 1}
        orders = {"gas:" + start: 1, "gas:" + drive: 1}
        if oxygen:
            reactants["gas:oxygen"] = 1
            orders["gas:oxygen"] = 0
        return dict(
            id=name,
            reactants=reactants,
            products={"gas:" + end: 1},
            orders=orders,
            rate=dict(type="constant", k=1.0, units="m^3/mol/s"),
        )

    return ReactionMechanism.from_dict(
        dict(
            version=1,
            basis="mol/m^3-air",
            species={"gas:" + s: {} for s in NAMES},
            reactions=[
                reaction("oxidation", "A", "intermediate", "oxid", True),
                reaction("other_products", "B", "other", "oxid"),
                reaction("product_formation", "intermediate", "vapor", "aging"),
                reaction("vapor_deposition", "vapor", "wallV", "vaporloss"),
                reaction("precursor_loss_A", "A", "wallA", "preloss"),
                reaction("precursor_loss_B", "B", "wallA", "preloss"),
            ],
        )
    )


def processes(state, cstar, accommodation, seed_access):
    """Bind gas chemistry and reversible mass exchange to one state."""
    chemistry = GasChemistry(
        mechanism().bind(state.gas),
        ChemistrySolver(method="rk23", rtol=2e-6, atol=1e-15),
    )
    # The available absorbing mass is organic plus accessible seed.
    activity = par.particles.ActivityIdealMolar(
        np.array([OM / seed_access, OM])
    )
    strategy = par.dynamics.CondensationIsothermal(
        molar_mass=MW.copy(),
        diffusion_coefficient=6e-6,
        accommodation_coefficient=float(accommodation),
        activity_strategy=activity,
        vapor_pressure_strategy=par.gas.ConstantVaporPressureStrategy(
            cstar * 1e-9 * R * TREF / OM
        ),
        surface_strategy=par.particles.SurfaceStrategyVolume(
            surface_tension=0.03, density=1400.0
        ),
        species_map=CondensationSpeciesMap(NAMES, ((4, 1),)),
    )
    return chemistry, par.dynamics.MassCondensation(strategy)


def inventory(state):
    """Return suspended OA, C10 equivalents and total reactive mass per m³."""
    gas = state.gas.concentration
    oa = (state.particles.masses[:, :, 1] * state.particles.concentration).sum(
        axis=1
    )
    carbon = (gas[:, [0, 1, 3, 4, 5, 6, 7]] / MW[[0, 1, 3, 4, 5, 6, 7]]).sum(
        axis=1
    ) + oa / OM
    total_mass = gas[:, :8].sum(axis=1) + oa
    return oa, carbon, total_mass


def check_inventory(state):
    """Reject negative or nonfinite gas or particle inventories."""
    gas = state.gas.concentration
    if (
        not np.isfinite(gas).all()
        or not np.isfinite(state.particles.masses).all()
        or gas.min() < 0
        or state.particles.masses.min() < 0
    ):
        raise ArithmeticError("nonphysical chamber state")


def replay(
    cases,
    forcing,
    *,
    cstar=10.0,
    accommodation=0.1,
    seed_access=1.0,
    slots=1,
    dt=12,
    interval=360,
    end=18000,
    device="cuda:0",
    number_forcing=None,
):
    """Advance native processes with piecewise forcing and midpoint removal."""
    if interval % (2 * dt) or end % interval:
        raise ValueError("interval must contain an even number of steps")
    state = chamber(cases, slots)
    chemistry, transfer = processes(state, cstar, accommodation, seed_access)
    registry = par.ProcessRegistry()
    registry.register(GasChemistry, GasChemistryAdapter())
    oa, carbon0, mass0 = inventory(state)
    loss = np.zeros(len(cases))
    max_carbon = max_mass = max_forcing = 0.0
    history = []
    pref_s = np.array([c.precursor_wall_s for c in cases])
    rates = np.array([c.particle_wall_h for c in cases])
    sequence = chemistry | transfer
    prec0 = state.gas.concentration[:, :2].sum(axis=1)

    def advance(duration):
        nonlocal state
        if device == "numpy":
            for _ in range(duration // dt):
                chemistry.execute(state, float(dt))
                transfer.execute(state, float(dt))
        else:
            with sequence.prepare(
                state,
                device=device,
                time_step=float(dt),
                process_registry=registry,
            ) as sim:
                sim.run(duration // dt)
                state = sim.snapshot()

    times = np.arange(0, end + 1, interval)
    residual = np.array([np.interp(times, *forcing[c.run]) for c in cases])
    number_curves = None
    wall_share = np.array([c.wall_share for c in cases])
    if number_forcing is not None:
        number_curves = np.array(
            [np.interp(times, *number_forcing[c.run]) for c in cases]
        )
    for j, t in enumerate(times):
        oa, carbon, mass = inventory(state)
        gas = state.gas.concentration
        check_inventory(state)
        max_carbon = max(
            max_carbon, np.max(abs((carbon + loss / OM) / carbon0 - 1))
        )
        max_mass = max(max_mass, np.max(abs((mass + loss) / mass0 - 1)))
        max_forcing = max(
            max_forcing,
            np.max(abs(gas[:, :2].sum(axis=1) / prec0 - residual[:, j])),
        )
        history.append(
            np.stack(
                [
                    oa * 1e9,
                    gas[:, 4] * 1e9,
                    gas[:, 7] * 1e9,
                    loss * 1e9,
                    gas[:, 6] * 1e9,
                    gas[:, :2].sum(axis=1) * 1e9,
                ],
                axis=1,
            )
        )
        if t == end:
            break
        apparent = np.log(residual[:, j] / residual[:, j + 1]) / interval
        nonreactive = np.minimum(pref_s, apparent)
        gas[:, 8] = apparent - nonreactive
        gas[:, 9] = nonreactive
        advance(interval // 2)
        keep = np.exp(-rates * interval / 3600)
        if number_curves is not None:
            keep = number_curves[:, j + 1] / number_curves[:, j]
        # Mean-particle closure for count-conditioned tests. At wall_share=1
        # particles leave with their material; at zero, consolidation keeps
        # material suspended as the mean particle grows. This is a number and
        # mass accounting end member, not a size-resolved coagulation kernel.
        retained_mass = keep**wall_share
        loss += inventory(state)[0] * (1 - retained_mass)
        state.particles.concentration *= keep[:, None]
        state.particles.masses *= (retained_mass / keep)[:, None, None]
        advance(interval // 2)
    checks = dict(
        carbon_rel=float(max_carbon),
        mass_rel=float(max_mass),
        forcing_fraction_abs=float(max_forcing),
    )
    if max(max_carbon, max_mass) > 2e-7 or max_forcing > 2e-5:
        raise ArithmeticError(checks)
    return times, np.stack(history), checks
