"""Fixed-inventory aerosol mixing and observed CCN activation curves at Mt Soledad."""

# %%
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import warp as wp
from particula.particles import get_kelvin_radius
from scipy.optimize import least_squares, isotonic_regression
from scipy.special import expit, ndtr
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent
STUDY = HERE.parents[1]
sys.path.insert(0, str(STUDY))
from activation import oracle, tables  # noqa: E402

SPECIES = ("organics", "nitrate", "sulfate", "ammonium", "chloride")
LAYOUTS = (
    "uniform",
    "small_nitrate",
    "large_nitrate",
    "nitrate_quartile",
    "organic_half",
    "organic_external",
)
DAY0 = int(np.datetime64("2023-04-01").astype(int))


def measured(root, kind):
    """Use all eligible CCN paired scans within a block and common-time ambient bulk AMS."""
    chem = root / "chemistry"
    raw = pd.read_csv(chem / "ams.csv.gz").dropna(
        subset=[s + "_fraction" for s in SPECIES]
    )
    group = raw.groupby(raw.time_unix // 21600)
    mass = group[[s + "_ug_m3" for s in SPECIES]].mean().loc[group.size() >= 24]
    kk = pd.read_csv(chem / "ams-kappa.csv.gz")
    variants = ["kappa_sol_01", "kappa_sol_OC", "kappa_sol_f44"]
    kg = kk.dropna(subset=variants).copy()
    kg = kg.groupby(kg.time_unix // 21600)
    kappa = kg[variants].mean().loc[kg.size() >= 24]
    band = (
        pd.read_csv(chem / "ams-band-6h.csv.gz")
        .query("coverage_pass")
        .set_index("time_unix")
    )
    band.index = band.index // 21600
    product = mass.join(kappa, how="inner").join(
        band[[s + "_finite_ug_m3" for s in SPECIES]]
    )
    with np.load(root / "blocks-final.npz") as archive:
        raw = {k: archive[k] for k in ("keys", "diam", "n", "sums", "env", "valid")}
    keys, d = raw["keys"], raw["diam"]
    obs, forcing, composition, bandmass, kv, record = [], [], [], [], [], []
    plateau = (d >= 120) & (d <= 250)
    for key, ss in sorted(set(map(tuple, keys[:, :2]))):
        if key not in product.index:
            continue
        ix = (keys[:, 0] == key) & (keys[:, 1] == ss)
        if (n := raw["n"][ix].sum()) < 4:
            continue
        spec = raw["sums"][ix].sum(axis=0)[0]
        off = 2 if kind == "inversion" else 0
        x, y = spec[off : off + 2]
        if not np.all(x > 0) or y[plateau].sum() <= 0:
            continue
        valid = raw["valid"][ix].sum(axis=0)
        if np.any(valid == 0):
            continue
        spec_shape = spec[2] / valid  # Per-channel mean dN/dlnD from the inverted CPC.
        if np.any(spec_shape <= 0):
            continue
        env = np.average(raw["env"][ix], axis=0, weights=raw["n"][ix])
        row = product.loc[key]
        denom = y[plateau].sum() / x[plateau].sum()
        obs.append([x, y / x / denom, spec_shape])
        composition.append(row[[s + "_ug_m3" for s in SPECIES]])
        bandmass.append(row[[s + "_finite_ug_m3" for s in SPECIES]])
        kv.append(row[variants])
        forcing.append(env)
        day = key // 4
        record.append([key * 21600, day, (day - DAY0) % 3 == 0, ss / 10, denom, n])
    labels = pd.DataFrame(record, columns="utc day held S plateau pairs".split())
    x, observed, shape = np.asarray(obs).transpose(1, 0, 2)
    size = (d >= 30) & (d <= 250)
    width = np.full(d.size, np.diff(np.log(d)).mean())
    width[[0, -1]] *= 0.5  # Integrate exactly the 30–310 nm covered interval.
    volume = shape * d**3 * width
    return dict(
        labels=labels,
        d=d,
        weight=x[:, size] / x[:, size].sum(axis=1, keepdims=True),
        obs=observed,
        dndln=shape,
        bin_count=shape * width,
        volume=volume / volume.sum(axis=1, keepdims=True),
        size=size,
        level=plateau,
        x=x,
        mass=np.array(composition),
        band=np.array(bandmass),
        kv=np.array(kv),
        env=np.array(forcing),
    )


def inventory(mass):
    """Separate neutral NH4NO3, organics and remaining measured ions without adding material."""
    groups = np.zeros((len(mass), 3, len(SPECIES)))
    groups[:, 0, 1] = np.minimum(mass[:, 1], mass[:, 3] * 62.005 / 18.039)
    groups[:, 0, 3] = groups[:, 0, 1] * 18.039 / 62.005
    groups[:, 1, 0] = mass[:, 0]
    groups[:, 2] = mass - groups[:, 0] - groups[:, 1]
    if np.min(groups) < -1e-12:
        raise ValueError("Negative species inventory")
    groups = np.maximum(groups, 0)
    density = np.array([1.72, 1.50, 1.78])
    mass_by_group = groups.sum(axis=2)
    vol = mass_by_group / density
    return groups, vol / vol.sum(axis=1, keepdims=True)


def distribution(
    data, mass, kb, layout, count=128, steep=2, subset=0.25, ko=0.1, kn=0.67
):
    groups, vf = inventory(mass)
    w = data["volume"]
    size = data["d"][None, :]
    a, b, c = vf.T[:, :, None]
    other_k = (kb[:, None] - a * kn - b * ko) / c
    if np.any(other_k <= 0):
        raise ValueError("No nonnegative residual kappa for composition and mean kappa")
    no_nitrate_org = b / (1 - a)
    p = np.ones_like(w)
    high_n = low_n = np.broadcast_to(a, w.shape)
    high_o = low_o = np.broadcast_to(b, w.shape)
    if layout in ("small_nitrate", "large_nitrate"):
        exponent = (-steep if layout == "small_nitrate" else steep) * np.log(size / 100)
        lo, hi = np.full_like(a, -35), np.full_like(a, 35)
        for _ in range(52):
            mid = (lo + hi) / 2
            too_little = (w * expit(mid + exponent)).sum(axis=1, keepdims=True) < a
            lo, hi = (
                np.where(too_little, mid, lo),
                np.where(too_little, hi, mid),
            )
        high_n = low_n = expit((lo + hi) / 2 + exponent)
        high_o = low_o = (1 - high_n) * no_nitrate_org
    if layout == "nitrate_quartile":
        p = np.broadcast_to(np.maximum(subset, a), w.shape)
        high_n, low_n = a / p, np.zeros_like(w)
        high_o, low_o = (
            (1 - high_n) * no_nitrate_org,
            (1 - low_n) * no_nitrate_org,
        )
    if layout in ("organic_external", "organic_half"):
        p = np.broadcast_to(b, w.shape)
        high_o, low_o = np.ones_like(w), np.zeros_like(w)
        high_n, low_n = np.zeros_like(w), np.broadcast_to(a / (1 - b), w.shape)
        if (
            layout == "organic_half"
        ):  # Move halfway from internal to fully segregated compositions.
            high_n, low_n = (high_n + a) / 2, (low_n + a) / 2
            high_o, low_o = (high_o + b) / 2, (low_o + b) / 2
    # Exactly represent count particles per cell, including one with the boundary composition.
    occupied = np.floor(p * count).astype(int)
    remainder = p * count - occupied
    boundary = (remainder > 0).astype(int)
    fractions = np.stack(
        [occupied / count, boundary / count, 1 - (occupied + boundary) / count],
        axis=-1,
    )
    nval = np.stack(
        [high_n, remainder * high_n + (1 - remainder) * low_n, low_n], axis=-1
    )
    oval = np.stack(
        [high_o, remainder * high_o + (1 - remainder) * low_o, low_o], axis=-1
    )
    rval = 1 - nval - oval
    if np.min(rval) < -1e-10 or np.min(fractions) < 0:
        raise ValueError("Particle class overflow")
    popvol = w[:, :, None] * fractions
    recovered = np.stack(
        [(popvol * z).sum(axis=(1, 2)) for z in [nval, oval, rval]], axis=1
    )
    by_species = np.einsum("bg,bgi->bi", recovered / vf, groups)
    residual = np.max(np.abs(by_species - mass) / np.maximum(mass, 1e-12))
    closure = np.max(np.abs(fractions.sum(axis=-1) - 1))
    if max(residual, closure) > 1e-8:
        raise ValueError(f"Inventory mismatch {residual} {closure}")
    k = nval * kn + oval * ko + rval * other_k[:, :, None]
    return k, fractions, residual


def diameters(tables_, ks, k):
    if k.min() < ks[0] or k.max() > ks[-1]:
        raise ValueError(f"Critical lookup range {k.min()}..{k.max()}")
    u = (np.log(k) - np.log(ks[0])) / (np.log(ks[-1] / ks[0]) / (len(ks) - 1))
    ix = np.minimum(u.astype(int), len(ks) - 2)
    block = np.arange(len(k))[:, None, None]
    return tables_[block, ix] * (1 - u + ix) + tables_[block, ix + 1] * (u - ix)


def ratio(data, crit, frac, log_scale, spread):
    curve = (
        frac
        * ndtr((np.log(data["d"])[None, :, None] - np.log(crit) - log_scale) / spread)
    ).sum(axis=-1)
    level = data["level"]
    x = data["x"][:, level]
    norm = (x * curve[:, level]).sum(axis=1) / x.sum(axis=1)
    if np.any(norm < 0.1):
        raise ValueError("Plateau not reached")
    return curve / norm[:, None]


def fit(data, crit, frac, train):
    labels = data["labels"].iloc[train]
    nd = labels.groupby("day").utc.transform("size").to_numpy()
    size = data["size"]
    # Optimize on the training subset only, each day equally weighted.
    sub = {**data, "x": data["x"][train]}
    obs, weight = data["obs"][train][:, size], data["weight"][train]

    def residual(theta):
        pred = ratio(sub, crit[train], frac[train], *theta)
        return ((pred[:, size] - obs) * np.sqrt(weight / nd[:, None])).ravel()

    f = least_squares(residual, [0.15, 0.18], bounds=([-0.7, 0.01], [0.7, 0.70]))
    if not f.success:
        raise ValueError(f.message)
    return f.x


def edges(data, curves):
    d = data["d"][data["size"]]
    out = []
    for v, w in zip(curves[:, data["size"]], data["weight"], strict=True):
        u = isotonic_regression(v, weights=w).x
        e = [
            np.exp(np.interp(x, u, np.log(d))) if u[0] <= x <= u[-1] else np.nan
            for x in [0.25, 0.5, 0.75]
        ]
        out.append([e[1], e[2] - e[0]])
    return np.array(out)


def score(data, curve, reference_edges):
    diff = curve[:, data["size"]] - data["obs"][:, data["size"]]
    ep = edges(data, curve)
    return np.c_[
        np.sqrt((data["weight"] * diff**2).sum(axis=1)),
        ep,
        ep - reference_edges,
    ]


def execute(ext, dest, kind, grid, count, draws, seed):
    started = time.monotonic()
    dest.mkdir(exist_ok=True, parents=True)
    data = measured(ext, kind)
    labels = data["labels"]
    train = np.flatnonzero((~labels.held) & (labels.S == 0.4))
    env = data["env"]
    kelvin = 2e9 * get_kelvin_radius(0.072, 1000, 0.01801528, env[:, 1])
    cond = np.c_[env[:, 3], env[:, 0], kelvin]
    ks = np.geomspace(0.008, 5.0, grid)
    table = [tables(cond, ks, h) for h in (False, True)]
    checks = [
        abs(table[h][i, j] - oracle(cond[i], ks[j], h))
        for i in [0, len(cond) // 2, -1]
        for j in [grid // 3, 2 * grid // 3]
        for h in (0, 1)
    ]
    if max(checks) > 1e-3:
        raise ValueError("Critical threshold independent check failed")
    ref = edges(data, data["obs"])
    rows, summaries, pars, uncertain, curves = [], [], [], [], []
    max_residual = 0
    U = qmc.Sobol(6, scramble=True, seed=seed).random_base2(draws)
    for row in range(-1, len(U)):
        if row == -1:
            variant, hydrate, steep, subset, ko, kn = 0, 0, 2.0, 0.25, 0.1, 0.67
        else:
            u = U[row]
            variant = int(u[0] * 3)
            hydrate = int(u[1] * 2)
            steep, subset, ko, kn = (
                1 + 2 * u[2],
                0.15 + 0.25 * u[3],
                0.05 + 0.2 * u[4],
                0.58 + 0.17 * u[5],
            )
        locfit = {}
        for source in ["bulk", "band"] if row == -1 else ["bulk"]:
            mass = data["mass"] if source == "bulk" else data["band"]
            if source == "band":
                allowed = np.isfinite(mass).all(axis=1)
                mass = np.where(allowed[:, None], mass, data["mass"])
            else:
                allowed = np.ones(len(mass), dtype=bool)
            for layout in LAYOUTS:
                k, frac, err = distribution(
                    data,
                    mass,
                    data["kv"][:, variant],
                    layout,
                    count,
                    steep,
                    subset,
                    ko,
                    kn,
                )
                max_residual = max(max_residual, err)
                crit = diameters(table[hydrate], ks, k)
                if row == -1 or layout == "uniform":
                    theta = fit(data, crit, frac, train[allowed[train]])
                    if layout == "uniform":
                        locfit[source] = theta
                if row == -1:
                    pars.append([source, layout, *theta])
                    maps = {
                        "shared": locfit[source],
                        "refit": theta,
                        "chemical": [0.0, 0.06],
                    }
                else:
                    maps = {"shared": locfit[source]}
                for mapping, theta0 in maps.items():
                    curve = ratio(data, crit, frac, *theta0)
                    if row >= 0:
                        sel = labels.held & (labels.S == 0.8)
                        z = (
                            curve[sel][:, data["size"]]
                            - data["obs"][sel][:, data["size"]]
                        ) ** 2
                        rm = np.sqrt((z * data["weight"][sel]).sum(axis=1))
                        uncertain.append(
                            (
                                row,
                                layout,
                                np.mean(
                                    pd.Series(rm)
                                    .groupby(labels.day[sel].to_numpy())
                                    .mean()
                                ),
                            )
                        )
                        continue
                    scores = score(data, curve, ref)
                    for i in np.flatnonzero(allowed):
                        rows.append(
                            [
                                source,
                                layout,
                                mapping,
                                *labels.iloc[i].tolist(),
                                *scores[i],
                                *ref[i],
                            ]
                        )
                    for S in [0.4, 0.6, 0.8, 1.0]:
                        for held in [False, True] if S == 0.4 else [True]:
                            sel = (labels.S == S) & (labels.held == held) & allowed
                            if not np.any(sel):
                                continue
                            daily = (
                                pd.DataFrame(np.c_[scores[sel], ref[sel]])
                                .groupby(labels.day[sel].to_numpy())
                                .mean()
                            )
                            rng = np.random.default_rng(seed + int(10 * S))
                            ids = rng.integers(len(daily), size=(2000, len(daily)))
                            for j, metric in enumerate(
                                [
                                    "rmse",
                                    "D50",
                                    "width",
                                    "d50_error",
                                    "width_error",
                                    "observed_D50",
                                    "observed_width",
                                ]
                            ):
                                arr = daily.iloc[:, j].to_numpy()
                                if np.isfinite(arr).any():
                                    valid_vals = arr[np.isfinite(arr)]
                                    good_ids = (
                                        ids
                                        if len(valid_vals) == len(arr)
                                        else rng.integers(
                                            len(valid_vals),
                                            size=(2000, len(valid_vals)),
                                        )
                                    )
                                    sims = valid_vals[good_ids].mean(axis=1)
                                    summaries.append(
                                        [
                                            source,
                                            layout,
                                            mapping,
                                            S,
                                            held,
                                            int(sel.sum()),
                                            int(np.isfinite(arr).sum()),
                                            metric,
                                            np.nanmean(arr),
                                            *np.nanpercentile(sims, [2.5, 97.5]),
                                        ]
                                    )
                            if source == "bulk" and mapping == "shared":
                                for tag, m in [
                                    ("pred", curve),
                                    ("obs", data["obs"]),
                                ]:
                                    avg = (
                                        pd.DataFrame(m[sel])
                                        .groupby(labels.day[sel].to_numpy())
                                        .mean()
                                        .mean()
                                        .to_numpy()
                                    )
                                    curves += [
                                        [layout, S, held, tag, d, a]
                                        for d, a in zip(data["d"], avg, strict=True)
                                    ]
    pd.DataFrame(
        rows,
        columns=[
            "chem",
            "layout",
            "map",
            *labels.columns,
            "rmse",
            "D50",
            "width",
            "d50_err",
            "width_err",
            "obs_d50",
            "obs_width",
        ],
    ).to_csv(
        dest / "blocks.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    pd.DataFrame(
        summaries,
        columns="chem layout map S held blocks days metric mean low high".split(),
    ).to_csv(dest / "summary.csv", index=False)
    pd.DataFrame(pars, columns="chem layout log_scale spread".split()).to_csv(
        dest / "fit.csv", index=False
    )
    pd.DataFrame(uncertain, columns=["draw", "layout", "held08_rmse"]).to_csv(
        dest / "physical_draws.csv", index=False
    )
    pd.DataFrame(
        curves,
        columns="layout S held kind diameter_nm normalized_fraction".split(),
    ).to_csv(
        dest / "curves.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    source_files = [
        ext / "blocks-final.npz",
        *(
            ext / "chemistry" / f
            for f in ["ams.csv.gz", "ams-kappa.csv.gz", "ams-band-6h.csv.gz"]
        ),
    ]
    meta = dict(
        grid=grid,
        physical_draws=len(U),
        particle_representatives_per_channel=count,
        seed=seed,
        kind=kind,
        cuda_device=wp.get_device("cuda:0").name,
        version=wp.__version__,
        windows=len(labels),
        training_blocks=len(train),
        dry_hydrated_critical_cases=2 * len(labels) * grid,
        max_oracle_error_nm=max(checks),
        max_relative_species_residual=max_residual,
        seconds=time.monotonic() - started,
        hashes={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files
        },
    )
    (dest / "run.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ext", type=Path, default=STUDY / "extension")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--kind", choices=["inversion", "signal"], default="inversion")
    p.add_argument("--grid", type=int, default=513)
    p.add_argument("--count", type=int, default=128)
    p.add_argument("--draws", type=int, default=4)
    p.add_argument("--seed", type=int, default=7106)
    a = p.parse_args()
    execute(a.ext, a.out, a.kind, a.grid, a.count, a.draws, a.seed)
