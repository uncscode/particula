"""Conditional outlet growth of finite-inventory patches on measured seed populations."""

# %%
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from models import UPPER, average, get_cases, intervals, scores
from particula.dynamics.condensation.mass_transfer import (
    get_first_order_mass_transport_k,
)
from particula.gas.properties.thermal_conductivity import (
    get_thermal_conductivity,
)
from particula.particles import get_kelvin_radius
from scipy.optimize import minimize_scalar

HERE = Path(__file__).resolve().parent
STUDY = HERE.parents[1]
DROPS = STUDY / "mechanisms/droplets"
EDGES = np.r_[0.5, UPPER]
AVOG = 6.02214076e23
WATER_R = 461.52
SIGMA = 0.072
RHO_W = 997.0
BASE = dict(
    alpha=2.85e-5,
    alpha_bare=1.0,
    area=0.21,
    molar=0.24244,
    rho_d=1450.0,
    kappa=0.30,
    budget="finite",
    reservoir=True,
)
GRID = np.array(
    [
        0,
        0.025,
        0.05,
        0.075,
        0.10,
        0.125,
        0.15,
        0.175,
        0.20,
        0.225,
        0.25,
        0.275,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.70,
        0.80,
        0.90,
        1.0,
    ]
)
GRID = np.unique(np.r_[GRID, np.arange(0.220, 0.246, 0.001)])
SCENARIOS = {
    "central": {},
    "pinned": {"reservoir": False},
    "permeable": {"alpha": 0.05},
    "short_chain": {"alpha": 0.0024, "molar": 0.18633},
    "long_chain": {"alpha": 1.7e-5, "molar": 0.25647},
    "low_budget": {
        "budget": "low",
        "rho_d": 1200.0,
        "area": 0.20,
        "kappa": 0.2,
        "alpha_bare": 0.5,
    },
    "high_budget": {
        "budget": "high",
        "rho_d": 1600.0,
        "area": 0.22,
        "kappa": 0.5,
    },
    "coverage_only": {},
    "correlated": {"rank": 1},
    "anticorrelated": {"rank": -1},
    "access10": {"access": 0.1},
    "access25": {"access": 0.25},
    "access50": {"access": 0.5},
    "access75": {"access": 0.75},
    "sulfate_mixed": {"core_k": 0.61, "core_rho": 1770.0, "organic_k": 0.1},
    "nitrate_mixed": {"core_k": 0.67, "core_rho": 1720.0, "organic_k": 0.1},
    "sulfate_weak_organic": {
        "core_k": 0.61,
        "core_rho": 1770.0,
        "organic_k": 0.03,
    },
    "sulfate_strong_organic": {
        "core_k": 0.61,
        "core_rho": 1770.0,
        "organic_k": 0.20,
    },
}

for salt, kappa_s, rho_s in [
    ("sulfate", 0.61, 1770.0),
    ("nitrate", 0.67, 1720.0),
]:
    for pct in (5, 10, 25, 50, 75, 100):
        SCENARIOS[f"{salt}_C16_{pct}"] = {
            "core_k": kappa_s,
            "core_rho": rho_s,
            "organic_k": 0.1,
            "c16_share": pct / 100,
        }


def make_data():
    data = get_cases("selected", DROPS)
    chem = pd.read_csv(STUDY / "extension/chemistry/ams-band-6h.csv.gz")
    chem = chem[chem.coverage_pass].set_index("time_unix")
    z = np.load(STUDY / "extension/blocks-final.npz")
    d = z["diam"]
    keep = (d >= 120) & (d <= 250)
    for split, c in data.items():
        weights = []
        state = []
        for utc in c["utc"]:
            mask = (z["keys"][:, 0] == utc // 21600) & (
                z["keys"][:, 1] == round(c["S"] * 10)
            )
            v = z["sums"][mask, 0, 3].sum(axis=0)[keep] * np.gradient(np.log(d))[keep]
            weights.append(v / v.sum())
            state.append(np.average(z["env"][mask], axis=0, weights=z["n"][mask]))
        c["state"] = np.array(state)
        c["weights"] = np.array(weights)
        c["dm"] = d[keep] / 1000
        cf = chem.reindex(c["utc"])
        c["covered"] = np.isfinite(cf.organics_finite_fraction.to_numpy())
        c["budget"] = cf.organics_finite_fraction.fillna(
            chem.organics_finite_fraction.median()
        ).values
        c["low"] = np.where(
            c["covered"],
            np.minimum(cf.organics_finite_fraction, cf.organics_zero_fraction),
            chem.organics_finite_fraction.quantile(0.1),
        )
        c["high"] = np.where(
            c["covered"],
            np.maximum(cf.organics_finite_fraction, cf.organics_zero_fraction),
            chem.organics_finite_fraction.quantile(0.9),
        )
    return data


def physics(c, i, p):
    rh, T, P, ss = c["state"][i]
    Dm = c["dm"]
    worg = c["budget" if p["budget"] == "finite" else p["budget"]][i]
    if "budget_pos" in p:
        worg = c["low"][i] + p["budget_pos"] * (c["high"][i] - c["low"][i])
    if "core_k" in p:
        corvol = (1 - worg) / p["core_rho"]
        if "c16_share" in p:
            filmvol = worg * p["c16_share"] / 820.0
            orgvol = worg * (1 - p["c16_share"]) / 1300.0
        else:
            filmvol = 0.0
            orgvol = worg / 1200.0
        volume = corvol + orgvol + filmvol
        p = {
            **p,
            "rho_d": 1 / volume,
            "kappa": (orgvol * p["organic_k"] + corvol * p["core_k"]) / volume,
        }
    A = 2e6 * get_kelvin_radius(SIGMA, RHO_W, 0.01801528, T)
    a = rh / 100 * np.exp(-A / Dm)
    dry = Dm / (1 + p["kappa"] * a / (1 - a)) ** (1 / 3)
    m = p["rho_d"] * np.pi / 6 * (dry * 1e-6) ** 3
    organic = worg * m * p.get("c16_share", 1.0)
    # Width of an initial monolayer across all potentially film-forming molecules.
    area_max = organic / p["molar"] * AVOG * p["area"] * 1e-6  # nm2 to um2
    Dv = 2.50e-5 * (T / 295) ** 1.75 * 99.4 / P
    k = get_first_order_mass_transport_k(1e-6, 1.0, Dv)
    Dv = k / (4 * np.pi * 1e-6)
    es = 611.2 * np.exp(17.67 * (T - 273.15) / (T - 29.65))
    vapor = RHO_W * WATER_R * T / (Dv * es)
    heat = (
        RHO_W
        * 2.45e6
        / (get_thermal_conductivity(T) * T)
        * (2.45e6 / (WATER_R * T) - 1)
    )
    mfp = 3 * Dv / np.sqrt(8 * WATER_R * T / np.pi) * 1e6
    return Dm, dry, A, area_max, ss, vapor, heat, mfp, p


def travel(x, dry, A, ss, vapor, heat, mfp, cov, p):
    # dt/dlogD for D in micrometres. D^2 growth includes Fuchs-Sutugin
    # gas transfer and quasi-steady latent heat resistance.
    water = (x / dry) ** 3 - 1
    seq = (water / (water + p["kappa"])) * np.exp(A / x)
    drive = (1 + ss / 100) - seq
    if np.any(drive <= 0):
        raise ValueError("Growth path crosses a subcritical equilibrium")
    alpha = (1 - cov) * p["alpha_bare"] + cov * p["alpha"]
    kn = 2 * mfp / x
    beta = (0.75 * alpha * (1 + kn)) / (
        kn * kn + kn + 0.283 * alpha * kn + 0.75 * alpha
    )
    return x * x * (vapor / beta + heat) / (4e12 * drive)


def window(c, i, config, qgrid, nstep, store):
    dm, dry, A, area, ss, vapor, heat, mfp, config = physics(c, i, config)
    mids = c["weights"][i]
    # The physical integration is uniform in log D within segments whose
    # endpoints explicitly include the film-break diameter and optical edges.
    # The inverse maps the time to each coated edge into a bypass diameter.
    out = np.zeros((len(qgrid), 20))
    lookup = []
    maximum = 0.0
    cov_metric = []
    b = c["b"][i]
    joint = np.broadcast_to(b, (len(mids), len(b)))
    if config.get("rank", 0):
        ps = np.r_[0, np.cumsum(mids)]
        order = np.arange(len(b))[:: config["rank"]]
        po = np.r_[0, np.cumsum(b[order])]
        masses = np.maximum(
            0,
            np.minimum(ps[1:, None], po[None, 1:])
            - np.maximum(ps[:-1, None], po[None, :-1]),
        )
        joint = np.zeros_like(masses)
        joint[:, order] = np.divide(
            masses,
            mids[:, None],
            out=np.zeros_like(masses),
            where=mids[:, None] > 0,
        )
    for d0, ds, a, w, b in zip(dm, dry, area, mids, joint, strict=True):
        grid = np.geomspace(d0, 10.001, nstep)
        active = qgrid * a
        initial = np.minimum(active, np.pi * d0 * d0)
        access = config.get("access", float(config["reservoir"]))
        interface = initial + access * (active - initial)
        breaks = np.sqrt(interface / np.pi)
        grid = np.unique(np.r_[grid, EDGES, breaks[(breaks > d0) & (breaks < 10.001)]])
        grid = grid[grid >= d0]
        # Integrate using midpoint quadrature rather than an endpoint at the jump in
        # uncovered area. Add an interior logarithmic mesh just past full cover.
        # Effective alpha initially rises steeply after a packed film starts breaking.
        offsets = np.geomspace(1e-7, 0.06, 18)
        fine = breaks[:, None] * (1 + offsets)
        grid = np.unique(np.r_[grid, fine[(fine > d0) & (fine < 10.001)]])
        log_grid = np.log(grid)
        h = np.diff(log_grid)
        x = np.exp(log_grid[:-1] + h / 2)
        tbare = np.r_[
            0, np.cumsum(travel(x, ds, A, ss, vapor, heat, mfp, 0, config) * h)
        ]
        cap = np.minimum(interface[:, None], np.pi * x * x)
        inactive = (1 - qgrid) * a
        reservoir = active[:, None] - cap
        total = cap + reservoir + inactive[:, None]
        maximum = max(maximum, float(np.max(abs(total - a) / a)))
        cov = cap / (np.pi * x * x)
        ts = np.column_stack(
            [
                np.zeros(len(qgrid)),
                np.cumsum(
                    travel(x, ds, A, ss, vapor, heat, mfp, cov, config) * h,
                    axis=1,
                ),
            ]
        )
        indices = np.searchsorted(grid, EDGES)
        # If coated edge needs more exposure time than the largest bypass
        # droplet, its preimage is beyond the measured input support.
        preimages = np.array([np.interp(row[indices], tbare, grid) for row in ts])
        width = np.maximum(
            0,
            np.minimum(EDGES[1:, None], preimages[:, None, 1:])
            - np.maximum(EDGES[:-1, None], preimages[:, None, :-1]),
        )
        pre = width / np.diff(EDGES)[None, :, None]
        out += w * np.einsum("i,qij->qj", b, pre)
        if store:
            lookup.append(preimages)
        cov_metric.append(
            (
                w,
                d0,
                ds,
                a,
                float(np.interp(np.dot(b, (EDGES[1:] + EDGES[:-1]) / 2), grid, tbare)),
            )
        )
    return out, np.array(lookup), maximum, cov_metric


def calc(data, setting, grid, nstep, cache=False):
    cfg = {**BASE, **SCENARIOS[setting]}
    pred = {}
    details = []
    lookups = {}
    residual = 0
    for split, c in data.items():
        out = []
        lu = []
        for i in range(len(c["day"])):
            y, table, err, ctx = window(c, i, cfg, grid, nstep, cache)
            out.append(y)
            lu.append(table)
            residual = max(residual, err)
            if setting == "central":
                for w, dm, dry, area, t in ctx:
                    details.append(
                        (
                            split,
                            c["utc"][i],
                            c["covered"][i],
                            dm,
                            dry,
                            w,
                            c["budget"][i],
                            area,
                            t,
                        )
                    )
        pred[split] = np.array(out)
        if cache:
            lookups[split] = np.array(lu)
    return pred, lookups, residual, details


def metrics_summary(y, b, p, day):
    s = scores(y, p, b, EDGES) if (p.sum(axis=1) > 0).all() else None
    if s is None:
        tot = p.sum(axis=1)
        obs = y.sum(axis=1)
        s = dict(
            observed_ratio=obs,
            model_ratio=tot,
            ratio_error=tot - obs,
            amplitude_abs_error=abs(tot - obs),
        )
    return list(intervals(s, day))


def fit(data, pred, grid, setting):
    controls = [
        "bare",
        "uniform_count",
        "uniform_full",
        "subset_full",
        "all_organics",
        "both_ordered",
        "reverse_order",
    ]
    train = data["train"]
    b = train["b"]
    y = train["y"]
    p = pred["train"]
    day = train["day"]
    valid = train["covered"] if setting == "coverage_only" else np.ones(len(day), bool)
    b = b[valid]
    y = y[valid]
    p = p[valid]
    day = day[valid]
    target = average(y.sum(axis=1), day)

    def cost(pv):
        return average(
            ((np.sqrt(np.maximum(pv, 0)) - np.sqrt(y)) ** 2).sum(axis=1), day
        )

    hom = np.array([cost(x) for x in p.transpose(1, 0, 2)])
    amplitude = np.array([average(x.sum(axis=1), day) for x in p.transpose(1, 0, 2)])
    cand = []
    for j in range(len(grid)):

        def loss(f):
            return cost((1 - f) * b + f * p[:, j])

        fit = minimize_scalar(loss, bounds=(0, 1), method="bounded")
        v, f = min((loss(0), 0.0), (fit.fun, fit.x), (loss(1), 1.0))
        cand.append((v, f, j))
    _, frac, j = min(cand)
    select = [
        (0, 0.0),
        (np.argmin(abs(amplitude - target)), 1.0),
        (np.argmin(hom), 1.0),
        (j, frac),
        (len(grid) - 1, 1.0),
        (j, frac),
        (j, frac),
    ]
    records = []
    pars = []
    profiles = []
    for label, (j, f) in zip(controls, select, strict=True):
        pars.append([setting, label, grid[j], f])
        for split, c in data.items():
            valid = (
                c["covered"]
                if setting == "coverage_only"
                else np.ones(len(c["day"]), bool)
            )
            B = c["b"][valid]
            Y = c["y"][valid]
            D = c["day"][valid]
            model = (1 - f) * B + f * pred[split][valid, j]
            if label == "both_ordered":
                # Ratio response cancels for identical branches even when both
                # have different absolute distributions from the clean reference.
                model = B
            if label == "reverse_order":
                # Reverse the experiment at the same uncoated-equivalent
                # exposure. The two scenarios refer to different bypass states.
                # Re-express on the coated rather than bare reference basis.
                Z = model.sum(axis=1)
                if (Z <= 0).any():
                    continue
                metrics = scores(Y, B / Z[:, None], B, EDGES)
                metrics["model_mean_shift"] = (B - model / Z[:, None]) @ (
                    (EDGES[1:] + EDGES[:-1]) / 2
                )
                metrics.pop("mean_diameter_error")
                metrics.pop("shape_TV")
                res = list(intervals(metrics, D))
            else:
                res = metrics_summary(Y, B, model, D)
            for k, m, lo, hi in res:
                records.append(
                    [
                        setting,
                        split,
                        label,
                        k,
                        m,
                        lo,
                        hi,
                        len(np.unique(D)),
                        len(D),
                    ]
                )
            if setting == "central" and split == "test" and label != "reverse_order":
                for ib in range(20):
                    profiles.append(
                        [
                            label,
                            EDGES[ib],
                            EDGES[ib + 1],
                            average(B[:, ib], D),
                            average(Y[:, ib], D),
                            average(model[:, ib], D),
                        ]
                    )
    return pars, records, profiles


def main(out, nstep, settings):
    tic = time.monotonic()
    data = make_data()
    pars = []
    results = []
    profiles = []
    details = []
    errs = {}
    fields = []
    out.mkdir(parents=True, exist_ok=True)
    for setting in settings:
        pred, lut, e, d = calc(data, setting, GRID, nstep, cache=setting == "central")
        errs[setting] = e
        a, b, p = fit(data, pred, GRID, setting)
        pars.extend(a)
        results.extend(b)
        profiles.extend(p)
        details.extend(d)
        chosen = next(x for x in a if x[1] == "subset_full")
        same = chosen[2] * chosen[3]
        uniform, _, _, _ = calc(data, setting, np.array([same]), nstep)
        pars.append([setting, "equal_inventory_uniform", same, 1.0])
        for split, c in data.items():
            use = (
                c["covered"]
                if setting == "coverage_only"
                else np.ones(len(c["day"]), bool)
            )
            D = c["day"][use]
            for k, m, lo, hi in metrics_summary(
                c["y"][use], c["b"][use], uniform[split][use, 0], D
            ):
                results.append(
                    [
                        setting,
                        split,
                        "equal_inventory_uniform",
                        k,
                        m,
                        lo,
                        hi,
                        len(np.unique(D)),
                        len(D),
                    ]
                )
        for split, c in data.items():
            for i, q in enumerate(GRID):
                Y = pred[split][:, i]
                fields.append(
                    [
                        setting,
                        split,
                        q,
                        average(Y.sum(axis=1), c["day"]),
                        average(
                            (Y - Y.sum(axis=1)[:, None] * c["b"])
                            @ ((EDGES[1:] + EDGES[:-1]) / 2),
                            c["day"],
                        ),
                    ]
                )
        if setting == "central":
            chosen = next(x for x in a if x[1] == "subset_full")
            j = np.flatnonzero(GRID == chosen[2])[0]
            for split, c in data.items():
                np.savez_compressed(
                    out / (split + "-map.npz"),
                    preimage=lut[split][:, :, j, :],
                    seed_weights=c["weights"],
                    b=c["b"],
                    y=c["y"],
                    day=c["day"],
                    utc=c["utc"],
                    predicted=(1 - chosen[3]) * c["b"] + chosen[3] * pred[split][:, j],
                    fitted_fraction=chosen[3],
                )
    for filename, frames, cols in [
        ("parameters", pars, "scenario model active_share coated_fraction"),
        (
            "scores",
            results,
            "scenario split model metric mean low high days windows",
        ),
        ("profiles", profiles, "model low_um high_um bypass denuded pred"),
        (
            "grid",
            fields,
            "scenario split active_share count conditional_moment_change",
        ),
        (
            "inventories",
            details,
            "split utc covered Dmob_um Ddry_um weight org_frac max_area_um2 clean_time_s",
        ),
    ]:
        pd.DataFrame(frames, columns=cols.split()).to_csv(
            out / (filename + ".csv"), index=False
        )
    (out / "metadata.json").write_text(
        json.dumps(
            dict(steps=nstep, elapsed=time.monotonic() - tic, mass_residual=errs),
            indent=2,
        )
        + "\n"
    )
    print(nstep, errs, time.monotonic() - tic)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--steps", type=int, default=257)
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    args = ap.parse_args()
    main(args.out, args.steps, args.scenarios)
