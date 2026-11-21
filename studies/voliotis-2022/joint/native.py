# %%
"""Replay joint mechanisms using native gas chemistry and condensation."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import particula as par
from engine import (
    AP,
    CONDITIONS,
    FORCING,
    NOX,
    OM,
    ORDER,
    PRESSURE,
    RESULT,
    TIMES,
    TREF,
    Engine,
    R,
)
from optimize import definition
from particula.dynamics.chemistry import ReactionMechanism
from particula.dynamics.chemistry.adapter import GasChemistryAdapter
from particula.dynamics.chemistry.process import ChemistrySolver, GasChemistry
from particula.dynamics.condensation.species_mapping import (
    CondensationSpeciesMap,
)

NAMES = (
    "A",
    "unproductive_A",
    "O2",
    "I",
    "SV",
    "LV",
    "other",
    "wall",
    "effective_rate",
)
MW = np.array([AP, AP, OM - AP, OM, OM, OM, AP, OM, 1.0])


def make_state(p):
    """Initialize five matched bulk seed and gas boxes."""
    gas = np.zeros((5, len(NAMES)))
    mass = np.zeros((5, 1, 3))
    counts = np.zeros((5, 1))
    temps = []
    for j, r in enumerate(ORDER):
        ppb, seed, n, tc = CONDITIONS[r]
        temp = 273.15 + tc
        temps.append(temp)
        moles = ppb * 1e-9 * PRESSURE / (R * temp) * p[9 + j]
        odds = p[0] / (1 - p[0]) * (NOX[r] / 30.0) ** p[16]
        frac = odds / (1 + odds)
        gas[j, :3] = moles * np.array(
            [frac * AP, (1 - frac) * AP, 2 * (OM - AP)]
        )
        counts[j, 0] = n * 1e6
        mass[j, 0, 0] = seed * 1e-9 / counts[j, 0]
    return par.Aerosol(
        particles=par.particles.ParticleData(
            masses=mass,
            concentration=counts,
            density=np.array([1770.0, 1400.0, 1400.0]),
            charge=np.zeros_like(counts),
            volume=np.ones(5),
            distribution_type="particle_resolved",
        ),
        gas=par.gas.GasData(
            list(NAMES),
            MW.copy(),
            gas,
            np.array([s in ("SV", "LV") for s in NAMES]),
        ),
        environment=par.gas.EnvironmentData(
            temperature=np.array(temps),
            pressure=np.full(5, PRESSURE),
            saturation_ratio=np.zeros_like(gas),
        ),
    )


def mechanism(p):
    """Generate a native, mass-balanced mechanism from the global fit."""
    reactions = []

    def add(label, react, prod, k, units, orders=None):
        if k == 0:
            return
        entry = dict(
            id=label,
            reactants={"gas:" + s: v for s, v in react.items()},
            products={"gas:" + s: v for s, v in prod.items()},
            rate=dict(type="constant", k=float(k), units=units),
        )
        if orders is not None:
            entry["orders"] = {"gas:" + s: v for s, v in orders.items()}
        reactions.append(entry)

    add(
        "precursor_oxidation",
        {"A": 1, "O2": 1},
        {"I": 1},
        1,
        "m^3/mol/s",
        {"A": 1, "O2": 0, "effective_rate": 1},
    )
    add(
        "unresolved_products",
        {"unproductive_A": 1},
        {"other": 1},
        1,
        "m^3/mol/s",
        {"unproductive_A": 1, "effective_rate": 1},
    )
    add(
        "semivolatile_products",
        {"I": 1},
        {"SV": 1},
        (1 - p[6]) / (60 * p[3]),
        "1/s",
    )
    add(
        "lower_volatility_products",
        {"I": 1},
        {"LV": 1},
        p[6] / (60 * p[3]),
        "1/s",
    )
    add("gas_aging", {"SV": 1}, {"LV": 1}, p[8] / 3600, "1/s")
    add(
        "competing_pair_reaction",
        {"I": 2},
        {"LV": 2},
        p[15] * OM / (2 * 1e-8 * 3600),
        "m^3/mol/s",
    )
    for s in ("SV", "LV"):
        add("wall_" + s, {s: 1}, {"wall": 1}, p[5] / 3600, "1/s")
    return ReactionMechanism.from_dict(
        dict(
            version=1,
            basis="mol/m^3-air",
            species={"gas:" + s: {} for s in NAMES},
            reactions=reactions,
        )
    )


def transfer(p, source, target, cstar):
    """Bind one volatile lane and full particle composition for its activity."""
    strategy = par.dynamics.CondensationIsothermal(
        molar_mass=MW.copy(),
        diffusion_coefficient=6e-6,
        accommodation_coefficient=float(p[2]),
        activity_strategy=par.particles.ActivityIdealMolar(np.full(3, OM)),
        surface_strategy=par.particles.SurfaceStrategyVolume(
            surface_tension=0.03, density=1400.0
        ),
        vapor_pressure_strategy=par.gas.ConstantVaporPressureStrategy(
            float(cstar) * 1e-9 * R * TREF / OM
        ),
        species_map=CondensationSpeciesMap(NAMES, ((source, target),)),
    )
    return par.dynamics.MassCondensation(strategy)


def advance_half(state, processes, sequence, reg, dt, split, device):
    """Advance a half interval using the supported native execution path."""
    if device == "numpy":
        for _ in range(round(180 / dt)):
            for op in processes:
                op.execute(state, dt)
    elif len(processes) == 3:
        for _ in range(round(180 / split)):
            for section in (sequence, processes[2]):
                with section.prepare(
                    state,
                    device=device,
                    time_step=float(dt),
                    process_registry=reg,
                ) as sim:
                    sim.run(round(split / dt))
                    state = sim.snapshot()
    else:
        with sequence.prepare(
            state,
            device=device,
            time_step=float(dt),
            process_registry=reg,
        ) as sim:
            sim.run(round(180 / dt))
            state = sim.snapshot()
    return state


def native_replay(
    family, dt=6.0, end=18000, device="cuda:0", proxy_hazard=1.0, split=36.0
):
    """Use native chamber inventories to verify one parameter vector."""
    row = json.loads((RESULT / (family + ".json")).read_text())
    p = np.array(row["physical"])
    state = make_state(p)
    mech = mechanism(p)
    mech.save(RESULT / (family + "_mechanism.json"))
    chem = GasChemistry(
        mech.bind(state.gas),
        ChemistrySolver(method="rk23", rtol=1e-7, atol=1e-16),
    )
    processes = [chem, transfer(p, 4, 1, p[1])]
    if p[6] > 0 or p[8] > 0 or p[15] > 0:
        processes.append(transfer(p, 5, 2, p[1] / p[7]))
    sequence = processes[0] | processes[1]
    # The prepared interface admits one condensation family per owner. Use
    # two native owners and exchange complete native states at split boundaries.
    # Each gas transfers against the full evolving mixed-particle composition.
    if not (180 % split == 0 and split % dt == 0):
        raise ValueError(
            "split and native step must divide the forcing half interval"
        )
    reg = par.ProcessRegistry()
    reg.register(GasChemistry, GasChemistryAdapter())
    # Controller knots have the same 360-s clock as the original study.
    f = np.array(
        [
            np.interp(TIMES, *FORCING[1 if r == 2 else r])
            ** (proxy_hazard if r == 2 else 1.0)
            for r in ORDER
        ]
    )
    count_mode = definition(family)[1]
    counts = Engine(dt=18).inputs[2].numpy() if count_mode else None
    deposited = np.zeros(5)
    ys = []
    budgets = []
    max_force = 0.0
    initial_ap = state.gas.concentration[:, :2].sum(axis=1)
    initial_reactive_mass = state.gas.concentration[:, :8].sum(axis=1)
    for j, t in enumerate(TIMES):
        material = (
            state.particles.masses * state.particles.concentration[:, :, None]
        ).sum(axis=1)
        organics = material[:, 1:].sum(axis=1)
        g = state.gas.concentration
        if (
            not np.isfinite(g).all()
            or np.any(g < 0)
            or not np.isfinite(material).all()
            or np.any(material < 0)
        ):
            raise ArithmeticError("Invalid native state")
        total = (g[:, [0, 1, 6]] / AP).sum(axis=1) + (
            g[:, [3, 4, 5, 7]].sum(axis=1) + organics + deposited
        ) / OM
        carbon = np.max(abs(total / (initial_ap / AP) - 1))
        mass = np.max(
            abs(
                (g[:, :8].sum(axis=1) + organics + deposited)
                / initial_reactive_mass
                - 1
            )
        )
        budgets.append([carbon, mass])
        max_force = max(
            max_force, np.max(abs(g[:, :2].sum(axis=1) / initial_ap - f[:, j]))
        )
        ys.append(
            np.column_stack(
                [organics, material[:, 1], material[:, 2], g[:, 7], deposited]
            )
            * 1e9
        )
        if t >= end:
            break
        g[:, 8] = np.log(f[:, j] / f[:, j + 1]) / 360.0
        for half in range(2):
            state = advance_half(
                state, processes, sequence, reg, dt, split, device
            )
            if half == 0:
                keep = (
                    counts[:, j + 1] / counts[:, j]
                    if count_mode
                    else np.full(5, np.exp(-p[4] * 0.1))
                )
                mass_keep = keep ** (p[14] if count_mode else 1.0)
                oa = (
                    state.particles.masses[:, :, 1:].sum(axis=2)
                    * state.particles.concentration
                ).sum(axis=1)
                deposited += oa * (1 - mass_keep)
                state.particles.concentration *= keep[:, None]
                state.particles.masses *= (mass_keep / keep)[:, None, None]
    err = np.max(budgets, axis=0)
    if max(err) > 2e-7 or max_force > 2e-5:
        raise ArithmeticError((err, max_force))
    return np.stack(ys), dict(
        carbon_rel=err[0],
        mass_rel=err[1],
        max_forcing_abs=max_force,
        fit_hash=hashlib.sha256(np.array(p, dtype="<f8").tobytes()).hexdigest(),
        physical=p.tolist(),
        step_s=dt,
        split_s=split,
        device=device,
        proxy_hazard=proxy_hazard,
    )


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("families", nargs="+")
    a.add_argument("--dt", type=float, default=6.0)
    a.add_argument("--split", type=float, default=36.0)
    a.add_argument("--end", type=int, default=18000)
    a.add_argument("--device", default="cuda:0")
    opt = a.parse_args()
    for family in opt.families:
        history, checks = native_replay(
            family, opt.dt, opt.end, opt.device, split=opt.split
        )
        rows = []
        for i, t in enumerate(TIMES[: len(history)]):
            for j, r in enumerate(ORDER):
                rows.append(
                    dict(
                        run=r,
                        time_s=t,
                        **dict(
                            zip(
                                (
                                    "soa",
                                    "SV_particle",
                                    "LV_particle",
                                    "wall_vapor",
                                    "wall_particle",
                                ),
                                history[i, j],
                                strict=True,
                            )
                        ),
                    )
                )
        path = RESULT / (
            f"{family}_{opt.device.replace(':', '')}_{opt.dt:g}s"
            f"_{opt.end}_split{opt.split:g}"
        )
        pd.DataFrame(rows).to_csv(str(path) + ".csv", index=False)
        Path(str(path) + ".json").write_text(
            json.dumps(checks, indent=2) + "\n"
        )
        print(family, checks, flush=True)
