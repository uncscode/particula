"""Propagate finite, chemistry-bounded surface inventories at three exposures.

A clean equivalent exposure is inferred for every original measured bypass
diameter. Scaling this absolute clean exposure on BOTH paths produces the new
bypass and coated outlet populations at each hypothesized growth duration.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from film import BASE, EDGES, make_data, physics, travel

HERE = Path(__file__).resolve().parent
STUDY = HERE.parents[1]
DURATION = np.array([0.5, 1.0, 2.0])
COVERED = 0.210347173986785  # fitted previously, never fitted here
FINE = np.arange(0.5, 18.0001, 0.025)
CLOSE = np.geomspace(1e-7, 0.08, 22)
UTC_DAY = 86400


def transport(case, i, worg, c16, kappa, density, resolution, atlas=False):
    state = {**case, "budget": case["budget"].copy()}
    state["budget"][i] = worg
    cfg = {**BASE, "kappa": kappa, "rho_d": density}
    dm, dry, A, maxarea, ss, vapor, heat, mfp, cfg = physics(state, i, cfg)
    prob = max(COVERED, c16)
    if not 0 < prob <= 1 or not 0 <= c16 <= 1:
        raise ValueError("Molecular budget")
    # At each seed size, maintain identical total C16 as equivalent patch
    # inventory. The subset carries at most the full organic inventory. This
    # is an upper bound on concentration of an equivalent film-forming mass;
    # dilute bulk κ, dry volume and clean exposure are kept fixed in this test.
    states = [
        ("clean", 0.0, 1.0, 0.0),
        ("initial_all", c16, 1.0, 0.0),
        ("reserve_all", c16, 1.0, 1.0),
        ("initial_subset", c16 / prob, prob, 0.0),
        ("reserve_subset", c16 / prob, prob, 1.0),
    ]
    optical = np.unique(np.r_[EDGES, FINE])
    mids = (optical[1:] + optical[:-1]) / 2
    # Raw optical edge centroids support direct comparison to the measured mean.
    coarse = (EDGES[1:] + EDGES[:-1]) / 2
    points = np.searchsorted(optical, EDGES)
    accum = {key: np.zeros((3, len(optical) + 1)) for key, *_ in states}
    exposure = 0.0
    balance = 0.0
    lookup = np.empty((len(dm), len(states), 3, len(EDGES))) if atlas else None
    cdf = np.r_[0.0, np.cumsum(case["b"][i])]
    assert abs(cdf[-1] - 1) < 1e-12
    for j, (start, solid, a, w) in enumerate(
        zip(dm, dry, maxarea, case["weights"][i], strict=True)
    ):
        available = np.array([qa * a for _, qa, _, _ in states])
        initial = np.minimum(available, np.pi * start**2)
        access = np.array([r for *_, r in states])
        surface_limit = initial + access * (available - initial)
        breaks = np.sqrt(surface_limit / np.pi)
        mesh = np.unique(
            np.r_[
                np.geomspace(start, 18.01, resolution),
                optical,
                breaks,
                (breaks[:, None] * (1 + CLOSE)).ravel(),
            ]
        )
        mesh = mesh[(mesh >= start) & (mesh <= 18.01)]
        h = np.diff(np.log(mesh))
        mid = np.sqrt(mesh[1:] * mesh[:-1])
        cover = np.minimum(1, surface_limit[:, None] / (np.pi * mid**2))
        time_map = np.column_stack(
            [
                np.zeros(len(states)),
                np.cumsum(
                    travel(mid, solid, A, ss, vapor, heat, mfp, cover, cfg) * h,
                    axis=1,
                ),
            ]
        )
        exposure += w * np.dot(case["b"][i], np.interp(coarse, mesh, time_map[0]))
        for idx, (key, _, frac, _) in enumerate(states):
            # m_interfacial + m_reserve = m_available independent of access.
            mass_surface = np.minimum(surface_limit[idx], np.pi * mid**2)
            if np.any(mass_surface > available[idx] * (1 + 1e-12)):
                raise ValueError(key)
            if idx:
                balance = max(
                    balance,
                    abs(available[idx] * frac - c16 * a) / max(a, 1e-300),
                )
            tedges = np.interp(optical, mesh, time_map[idx])
            pre = np.interp(tedges[None, :] / DURATION[:, None], time_map[0], mesh)
            if atlas:
                lookup[j, idx] = pre[:, points]
            f = np.interp(pre, EDGES, cdf)
            discrete = np.column_stack([f[:, 0], np.diff(f, axis=1), 1 - f[:, -1]])
            if idx == 0:
                base = discrete
            accum[key] += w * ((1 - frac) * base + frac * discrete)
    rows = []
    spectra = []
    for key, arr in accum.items():
        if np.max(abs(arr.sum(axis=1) - 1)) > 1e-11 or arr.min() < -1e-12:
            raise ValueError(key)
        if arr[:, -1].max() > 1e-10:
            raise ValueError("18 µm diameter ceiling")
        for j, factor in enumerate(DURATION):
            mass = arr[j, 1:-1]
            total = mass.sum()
            b = accum["clean"][j, 1:-1]
            bcount = b.sum()
            spec = np.add.reduceat(mass, points[:-1])
            # The last coarse bin ends at 10.0 µm. Remove the over-range portion.
            over = mass[points[-1] :].sum()
            spec[-1] -= over
            bins = spec.sum()
            rows.append(
                [
                    key,
                    factor,
                    prob,
                    exposure * factor,
                    total / bcount,
                    total,
                    arr[j, 0],
                    over,
                    np.dot(mass, mids) / total if total > 0 else np.nan,
                    np.dot(b, mids) / bcount,
                    spec @ coarse / bins if bins > 0 else np.nan,
                    balance,
                ]
            )
            for c, amount in enumerate(spec):
                spectra.append([key, factor, c, amount])
    return (rows, spectra, lookup) if atlas else (rows, spectra)


def run(recipes, steps, folder, detail=False):
    tic = time.monotonic()
    data = make_data()
    results = []
    profiles = []
    for r in recipes.itertuples(index=False):
        for split, case in data.items():
            # Transfer each strict recipe to all withheld reference 0.8% boxes,
            # plus all optical windows whose UTC exactly matches that recipe.
            use = np.flatnonzero(case["utc"] == r.utc)
            if split == "test":
                use = np.arange(len(case["utc"]))
            for i in use:
                utc = case["utc"][i]
                label = "same UTC" if utc == r.utc else "campaign reference"
                prefix = [
                    r.filter_id,
                    r.utc,
                    r.recipe,
                    split,
                    utc,
                    label,
                    case["S"],
                    r.organic_mass_fraction,
                    r.c16_organic_fraction,
                    r.kappa08,
                    r.density,
                ]
                rows, sp = transport(
                    case,
                    i,
                    r.organic_mass_fraction,
                    r.c16_organic_fraction,
                    r.kappa08,
                    r.density,
                    steps,
                )
                obs = case["y"][i]
                b = case["b"][i]
                m = (EDGES[:-1] + EDGES[1:]) / 2
                observed = [obs.sum(), np.dot(obs, m) / obs.sum(), np.dot(b, m)]
                results.extend([prefix + x + observed for x in rows])
                if detail and "max3" not in r.recipe:
                    profiles.extend([prefix[:7] + x for x in sp if x[1] == 1])
    ids = "filter_id recipe_utc recipe split optical_utc coupling S_pct organic_fraction c16_of_org kappa08 density".split()
    cols = "model exposure subset_fraction mean_exposure_s count_ratio count_fraction censored fraction_above10 wet_um bypass_wet_um binned_wet_um balance observed_ratio observed_wet_um observed_bypass_wet_um".split()
    folder.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results, columns=ids + cols)
    df.to_csv(
        folder / "outlet.csv.gz",
        index=False,
        float_format="%.11g",
        compression={"method": "gzip", "mtime": 0},
    )
    if detail:
        pd.DataFrame(
            profiles, columns=ids[:7] + "model exposure channel count".split()
        ).to_csv(
            folder / "spectra.csv.gz",
            index=False,
            float_format="%.10g",
            compression={"method": "gzip", "mtime": 0},
        )
    reduced = df[df.split == "test"].copy()
    reduced["optical_day"] = reduced.optical_utc // 86400
    keys = ["filter_id", "recipe_utc", "recipe", "model", "exposure"]
    vals = [
        "count_ratio",
        "count_fraction",
        "censored",
        "fraction_above10",
        "wet_um",
        "bypass_wet_um",
        "binned_wet_um",
        "observed_ratio",
        "observed_wet_um",
        "observed_bypass_wet_um",
        "mean_exposure_s",
    ]
    summary = (
        reduced.groupby(keys + ["optical_day"])[vals]
        .mean()
        .groupby(keys)
        .mean()
        .reset_index()
    )
    summary.to_csv(folder / "held-summary.csv", index=False)
    (folder / "outlet-run.json").write_text(
        json.dumps(
            dict(
                steps=steps,
                seconds=time.monotonic() - tic,
                rows=len(df),
                mass_error=df.balance.max(),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("recipes", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--steps", type=int, default=513)
    ap.add_argument("--detail", action="store_true")
    arg = ap.parse_args()
    run(pd.read_csv(arg.recipes), arg.steps, arg.output, arg.detail)
