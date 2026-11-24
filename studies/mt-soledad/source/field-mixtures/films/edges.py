"""Size-dependent gas exchange and mass-conserving surface redistribution."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from particula.particles.properties.vapor_correction_module import (
    get_vapor_transition_correction,
)
from response import hydration, kelvin_nm

HERE = Path(__file__).resolve().parent
STUDY = HERE.parents[1]
R = 8.314462618
MW = np.array([0.092094, 0.116116])
DENSITY = 1500.0  # same effective organic dry volume as chemistry ledger
MW_H = 0.01801528
PURITY_T = [
    328.4,
    333.4,
    338.3,
    343.3,
    343.4,
    348.3,
    353.3,
    358.3,
    358.4,
    363.4,
    368.4,
    373.4,
]
PURITY_PA = [
    7.05,
    10.45,
    14.14,
    20.96,
    20.19,
    27.02,
    38.10,
    52.78,
    51.48,
    75.87,
    100.96,
    134.93,
]
LEV = np.polyfit(1 / np.array(PURITY_T), np.log(PURITY_PA), 1)


def saturation(T):
    return np.array(
        [
            0.0060 * np.exp(-78400 / R * (1 / T - 1 / 295.15)),
            np.exp(np.polyval(LEV, 1 / T)),
        ]
    )


def critical(r, ss, k, T, lost_volume, lost_osm, film, pressure, nq):
    q = np.geomspace(0.01, 1.0e5, nq)
    ratio = np.cbrt(1 - lost_volume + q)
    A = kelvin_nm(T)
    # Number of C16 molecules per nm³ of original dry volume.
    c16 = 602.214076 * r.hexadecanol / (242.446 * r.dry_volume)
    if not film or c16 == 0:
        ds = A / (ratio * (np.log1p(ss / 100) + np.log1p((k - lost_osm) / q)))
    else:
        lo = np.zeros_like(q)
        hi = np.full_like(q, 400.0)
        for _ in range(35):
            d = (lo + hi) * 0.5
            capacity = 6 * ratio**2 / (d * 0.21)
            ns = np.minimum(c16, capacity)
            coverage = ns / capacity
            # The competition branch treats C16 molecules as dissolved neutral
            # osmoles initially. k already includes that increment in this branch.
            if film == "solubility_capped":
                cap = 4.1e-8 * 0.997 * q / 242.446 * 602.214076
                expelled = np.minimum(c16, cap) - np.minimum(c16 - ns, cap)
            else:
                expelled = ns
            solute = k - lost_osm - expelled / (997 / MW_H * 6.02214076e23 * 1e-27)
            sigma = 0.072 - pressure * coverage
            log_s = -np.log1p(solute / q) + A * (sigma / 0.072) / (d * ratio)
            lo = np.where(log_s > np.log1p(ss / 100), d, lo)
            hi = np.where(log_s <= np.log1p(ss / 100), d, hi)
        ds = (lo + hi) * 0.5
    j = ds.argmax()
    if j in (0, len(q) - 1):
        raise ValueError("Critical peak")
    return ds[j]


def flux(r, T, P, RH, k, Ddry, gas, retained, alpha, activity, seconds):
    # Electroneutral organic proxies only. Retain all unassigned organic mass.
    moles = np.array([r.glycerol, r.levulinic]) / r.dry_volume / MW * 1e3
    volume = np.pi / 6 * (Ddry * 1e-9) ** 3
    mass = moles * MW * volume
    wet = float(hydration(np.array([Ddry]), np.array([k]), RH, kelvin_nm(T))[0]) * 1e-9
    vbar = np.sqrt(8 * R * T / (np.pi * MW))
    diffus = 8e-6 * (T / 295.0) ** 1.75 * (99400 / P)
    kn = 6 * diffus / vbar / wet
    beta = get_vapor_transition_correction(kn, alpha)
    rate = 2 * np.pi * wet * diffus * beta * MW / (R * T)
    pgas = np.maximum(0, np.asarray(gas)) * 1e-9 * P
    # Free-molecular upper bound on the change in incoming mass from removing
    # measured ambient gas for 20 s, constant initial surface area, no retention loss.
    collision = np.pi * wet**2 * vbar / 4 * MW / (R * T) * pgas * seconds
    aqu = (wet / (Ddry * 1e-9)) ** 3 - 1
    x = moles / ((k + aqu) * 997 / MW_H)
    if activity == "equilibrated":
        peq = pgas
    else:
        curvature = np.exp(4 * 0.04 * MW / (DENSITY * R * T * wet))
        peq = saturation(T) * np.minimum(1, x * float(activity)) * curvature
    lam = np.divide(rate * peq, mass, out=np.zeros_like(mass), where=mass > 0)
    # An absent fitted species is outside the modeled particle pool; gas that
    # could newly condense is independently reported by the collision limit.
    exchanged = (
        np.divide(pgas * retained - peq, peq, out=np.zeros_like(mass), where=peq > 0)
        * mass
        * (-np.expm1(-lam * seconds))
    )
    if ((mass + exchanged) < -1e-30).any():
        raise ValueError("negative molecular inventory")
    dv = -exchanged.sum() / DENSITY / volume
    osm = (-exchanged / MW).sum() * MW_H / 997 / volume
    return dv, osm, mass, exchanged, collision, x, peq, pgas


def solve(r, setting, g, meas, nq):
    T = meas["T"]
    RH = meas.RH
    pressure = meas.pressure_kPa * 1000
    ss = meas.calibrated_S
    k = r.kappa08
    before = critical(r, ss, k, T, 0, 0, False, 0, nq)
    c16osm = (r.hexadecanol / r.dry_volume / 0.242446 * 1.0e3) * MW_H / 997
    records = []

    def add(name, mode, alpha, a, ret, t, d, bypass, extras):
        db = hydration(np.array([bypass]), np.array([k]), RH, kelvin_nm(T))[0]
        dd = hydration(np.array([d]), np.array([k]), RH, kelvin_nm(T))[0]
        records.append(
            dict(
                filter_id=r.filter_id,
                utc=r.utc,
                recipe=r.recipe,
                S=setting,
                mechanism=name,
                activity=mode,
                alpha=alpha,
                surface_pressure=a,
                gas_remaining=ret,
                transit_s=t,
                bypass_dry_nm=bypass,
                denuded_original_dry_nm=d,
                dry_shift=d - bypass,
                bypass_mobility_nm=db,
                mobility_shift=dd - db,
                obs_bypass=meas.D_b,
                obs_shift=meas["shift"],
                **extras,
            )
        )

    add("no_change", "", 1, 0, 1, 0, before, before, {})
    # Same mass and neutral molecules, all initially dissolved; segregate up
    # to a finite monolayer at the Köhler peak and optionally depress tension.
    dissolved = critical(r, ss, k + c16osm, T, 0, 0, False, 0, nq)
    for delta in [0, 0.015, 0.035]:
        ds = critical(r, ss, k + c16osm, T, 0, 0, True, delta, nq)
        add(
            "redistribute_C16",
            "dissolved_to_interface",
            1,
            delta,
            1,
            0,
            ds,
            dissolved,
            {},
        )
        actual = critical(r, ss, k, T, 0, 0, "solubility_capped", delta, nq)
        add(
            "redistribute_C16",
            "solubility_capped",
            1,
            delta,
            1,
            0,
            actual,
            before,
            {},
        )
    for mode in ["equilibrated", "0.1", "1.0"]:
        for alpha in [1.0, 0.01]:
            for retained, seconds in [
                (0.0, 20.0),
                (0.5, 20.0),
                (1.0, 20.0),
                (0.0, 10.0),
                (0.0, 30.0),
            ]:
                d = before
                for _ in range(8):
                    v, krem, m, exc, col, x, peq, pg = flux(
                        r,
                        T,
                        pressure,
                        RH,
                        k,
                        d,
                        g,
                        retained,
                        alpha,
                        mode,
                        seconds,
                    )
                    next_d = critical(r, ss, k, T, v, krem, False, 0, nq)
                    iteration_error = abs(next_d - d)
                    if iteration_error < 0.00001:
                        break
                    d = next_d
                if iteration_error > 1e-3:
                    raise ValueError("Evaporation-size iteration")
                extra = {}
                for i, name in enumerate(["glycerol", "levulinic"]):
                    extra.update(
                        {
                            name + "_loss_fraction": -exc[i] / m[i]
                            if m[i] > 0
                            else np.nan,
                            name + "_inlet_ppbv": g[i],
                            name + "_peq_Pa": peq[i],
                            name + "_inlet_Pa": pg[i],
                            name + "_collision_kg": col[i],
                            name + "_initial_kg": m[i],
                        }
                    )
                add(
                    "gas_exchange",
                    mode,
                    alpha,
                    0,
                    retained,
                    seconds,
                    next_d,
                    before,
                    extra,
                )
    return records


def run(src, dest, grid):
    r = pd.read_csv(src)
    g = pd.read_csv(STUDY / "extension/chemistry/cims.csv.gz")
    g["utc"] = g.time_unix // 21600 * 21600
    channels = ["propanetriol_ppbv", "levulinic_ppbv"]
    stats = g.groupby("utc")[channels].agg(["count", "mean", "min", "max"])
    stats.columns = ["_".join(c) for c in stats.columns]
    stats = stats.loc[r.utc.unique()]
    stats.to_csv(dest / "gas-observations.csv")
    z = (
        pd.read_csv(STUDY / "extension/edges.csv")
        .query("kind=='inversion' and phase==-1 and window==120 and S in [0.4,0.8]")
        .set_index(["utc", "S"])
    )
    rows = []
    for recipe in r.itertuples(index=False):
        if "max3" in recipe.recipe:
            continue
        for S in [0.4, 0.8]:
            if (recipe.utc, S) in z.index:
                gs = stats.loc[recipe.utc]
                if min(gs[s + "_count"] for s in channels) < 36:
                    continue
                ppb = [gs[s + "_mean"] for s in channels]
                rows += solve(recipe, S, ppb, z.loc[(recipe.utc, S)], grid)
    pd.DataFrame(rows).to_csv(
        dest / "activation.csv.gz",
        index=False,
        float_format="%.10g",
        compression={"method": "gzip", "mtime": 0},
    )


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("recipes", type=Path)
    a.add_argument("output", type=Path)
    a.add_argument("--grid", type=int, default=1025)
    cfg = a.parse_args()
    cfg.output.mkdir(parents=True, exist_ok=True)
    run(cfg.recipes, cfg.output, cfg.grid)
