"""Compare the same hour's aged and dark optical columns and incident sunlight."""

import argparse
import json
import os
import time
from functools import lru_cache

import numpy as np
import pandas as pd
from optical import HERE, TMP, bulk_moments, optical_column, radiance
from scipy.interpolate import interp1d

COLUMN = TMP.parent / "columns"
FLUX = [
    "up_black_wm2",
    "down_black_wm2",
    "up_surface02_wm2",
    "up_surface08_wm2",
    "up_direct_black_wm2",
    "up_diffuse_black_wm2",
    "positive_budget_residual_wm2",
    "diffuse_reciprocity_residual_wm2",
    "isotropic_sky_albedo",
    "up_site_wm2",
    "up_swapped_spectra_black_wm2",
    "up_swapped_spectra_site_wm2",
    "up_circumsolar_black_wm2",
    "up_circumsolar_site_wm2",
]
PROP = [
    "wet_path_gm2",
    "column_reff_um",
    "tau_geometric",
    "tau_550nm",
    "scatter_550nm",
    "g_550nm",
]


@lru_cache(maxsize=1)
def solar():
    """Join the observed irradiance, effective surface albedo and optional colors."""
    f = pd.read_csv(HERE.parent / "forcing" / "forcing.csv")
    if f.duplicated(["month", "hour_in_week"]).any() or len(f) != 2016:
        raise ValueError("Incomplete solar forcing")
    if np.max(abs(f.ghi_wm2 - f.dhi_wm2 - f.direct_horizontal_wm2)) > 2e-5:
        raise ValueError("Downwelling closure")
    # Low-sun weeks occasionally have no radiation above ten W m-2. Retain
    # all daylight at/above one W m-2 then tabulate threshold sensitivity.
    albedos = []
    for month, g in f.groupby("month"):
        estimates = {}
        for cutoff in (1, 5, 10, 20):
            lit = g[(g.ghi_wm2 >= cutoff) & g.fmi_refl_wm2.notna()]
            v = (
                lit.fmi_refl_wm2.sum() / lit.ghi_wm2.sum()
                if len(lit)
                else np.nan
            )
            estimates[f"albedo_min{cutoff}"] = v
            estimates[f"hours_min{cutoff}"] = len(lit)
        chosen = 10 if estimates["hours_min10"] >= 5 else 1
        value = estimates[f"albedo_min{chosen}"]
        if not 0 <= value <= 1:
            raise ValueError(("surface boundary outside range", month, value))
        albedos.append(
            dict(
                month=month,
                site_albedo=value,
                chosen_min_ghi_wm2=chosen,
                **estimates,
            )
        )
    site = pd.DataFrame(albedos)
    boundary_file = HERE / f"surface_boundary.{os.getpid()}.part"
    site.to_csv(boundary_file, index=False, float_format="%.8g")
    boundary_file.replace(HERE / "surface_boundary.csv")
    f = f.merge(
        site[["month", "site_albedo"]], on="month", validate="many_to_one"
    )
    data = HERE / "spectral-forcing.csv.gz"
    if data.is_file():
        c = pd.read_csv(data)[
            [
                "month",
                "hour_in_week",
                "solar_uv_fraction",
                "solar_par_fraction",
                "ceres_color_valid",
            ]
        ]
        f = f.merge(c, on=["month", "hour_in_week"], validate="one_to_one")
    return f


def read_cases(paths, bands, order, layers, step, drop_cut):
    """Read physical snapshots and integrate their spectral and bulk optical state."""
    records = []
    for path in paths:
        with np.load(path) as col:
            rr, nn, z = col["wet_radius_m"], col["number_m3"], col["height_m"]
            if drop_cut:
                nn = nn * (rr > drop_cut * 1e-6)
            opts = optical_column(rr, nn, z, bands, order, layers, step)
            visible = optical_column(rr, nn, z, 0, 2, 1, step)[:, 1, 0, :]
            bulk = bulk_moments(rr, nn, z)
            props = np.column_stack(
                (
                    bulk,
                    visible[:, 0],
                    visible[:, 1],
                    visible[:, 2] / visible[:, 1],
                )
            )
            records.append((col["hour_in_week"], opts, props))
    h, c, p = (np.concatenate([r[i] for r in records]) for i in range(3))
    j = np.argsort(h)
    return h[j], c[j], p[j]


@lru_cache(maxsize=16)
def dark_moments(prefix, profile, w, bands, order, layers, step, drop_cut):
    """Cache the matched no-chemistry cloud for a given physical grid and regime."""
    paths = list(COLUMN.glob(f"{prefix}_m00_{profile}_w{w:g}_dark_*.npz"))
    if len(paths) != 1:
        raise FileNotFoundError(paths)
    _, c, p = read_cases(paths, bands, order, layers, step, drop_cut)
    return c, p


def hourly(a, h, t):
    """Interpolate positive layer-integrated moments within one sampled week."""
    if len(h) == 1:
        return np.broadcast_to(a, (len(t),) + a.shape[1:])
    if np.array_equal(t, h):
        return a
    if np.min(t) < h[0] or np.max(t) > h[-1] or len(set(h)) != len(h):
        raise ValueError(("Incomplete endpoints", h))
    return interp1d(h, a, axis=0, assume_sorted=True)(t)


def group(
    prefix,
    m,
    profile,
    w,
    model,
    bands,
    streams,
    order,
    layers,
    step,
    drop_cut,
    thin,
    sample_step=1,
    boundary="observed",
):
    """Pair a complete sampled week with its dark control under one boundary."""
    files = list(
        COLUMN.glob(f"{prefix}_m{m:02d}_{profile}_w{w:g}_{model}_*.npz")
    )
    if not files:
        raise FileNotFoundError((m, profile, w, model))
    stamps = []
    for path in files:
        with np.load(path) as metadata:
            stamps.extend(metadata["hour_in_week"])
    h = np.sort(stamps)
    if (
        len(h) < 2
        or h[0] != 0
        or h[-1] != 167
        or max(np.diff(h))
        > (6 if prefix == "optref" or prefix.startswith("resolved") else 3)
        or (prefix == "hourly" and len(h) != 168)
    ):
        raise ValueError(
            ("Missing hourly cloud coverage", prefix, m, profile, w, model, h)
        )
    h, raw, p = read_cases(files, bands, order, layers, step, drop_cut)
    if sample_step > 1:
        keep = (h % sample_step == 0) | (h == 167)
        h, raw, p = h[keep], raw[keep], p[keep]
    f = (
        solar()
        .query("month==@m")
        .sort_values("hour_in_week")
        .reset_index(drop=True)
    )
    if boundary == "fixed":
        # Counterfactual incident light isolates intrinsic optical state even
        # at timestamps that had no actual sunlight. Keep outside flux totals.
        f = f.assign(
            ghi_wm2=100.0,
            direct_horizontal_wm2=50.0,
            dhi_wm2=50.0,
            mu0=0.5,
            dni_wm2=100.0,
            ceres_color_valid=False,
        )
    elif boundary != "observed":
        clear = (
            pd.read_csv(HERE / "clear-forcing.csv.gz")
            .query("month==@m")
            .set_index("hour_in_week")
            .loc[f.hour_in_week]
        )
        g = clear.clear_ghi.to_numpy()
        native = np.divide(
            clear.clear_dirh, clear.clear_ghi, out=np.zeros(len(f)), where=g > 0
        )
        if boundary == "clear_native":
            frac = native
            mu = clear.clear_mu.to_numpy()
        elif boundary == "clear_amount":
            obs = f.ghi_wm2.to_numpy()
            frac = np.divide(
                f.direct_horizontal_wm2,
                obs,
                where=obs > 0,
                out=np.asarray(native).copy(),
            )
            mu = np.where(obs > 0, f.mu0, clear.clear_mu)
        else:
            raise ValueError("Unknown radiation boundary")
        f = f.copy()
        f["ghi_wm2"] = g
        f["direct_horizontal_wm2"] = g * frac
        f["dhi_wm2"] = g * (1 - frac)
        f["mu0"] = mu
        f["dni_wm2"] = np.divide(
            g * frac, mu, where=mu > 0, out=np.zeros(len(f))
        )
        if (f.direct_horizontal_wm2 - f.mu0 * f.dni_wm2).abs().max() > 1e-6:
            raise ValueError("Supplied clear beam closure")
        f["ceres_color_valid"] = False
    t = f.hour_in_week.to_numpy()
    dark_prefix = (
        "hourly"
        if prefix.startswith("snow_") or prefix == "assignments"
        else "resolved"
        if prefix.startswith("resolved_")
        else prefix
    )
    d, dp = dark_moments(
        dark_prefix, profile, w, bands, order, layers, step, drop_cut
    )
    c = hourly(raw, h, t)
    p = hourly(p, h, t)
    dark = np.broadcast_to(d, c.shape)
    radi = radiance(c, f, bands, streams, order, thin)
    darkrad = radiance(dark, f, bands, streams, order, thin)
    out = f[
        [
            "month",
            "hour_in_week",
            "local_solar_hour",
            "mu0",
            "ghi_wm2",
            "direct_horizontal_wm2",
            "dhi_wm2",
            "site_albedo",
        ]
    ].copy()
    out["profile"], out["w"], out["model"], out["prefix"] = (
        profile,
        w,
        model,
        prefix,
    )
    for j, key in enumerate(FLUX):
        out[key] = radi[:, j]
        out["delta_" + key] = radi[:, j] - darkrad[:, j]
    extra = (
        "up_two_band_surface_wm2",
        "up_ceres_color_black_wm2",
        "up_ceres_color_site_wm2",
    )
    for j, key in enumerate(extra, start=14):
        out[key] = radi[:, j]
        out["delta_" + key] = radi[:, j] - darkrad[:, j]
    for j, key in enumerate(PROP):
        out[key] = p[:, j]
        out["delta_" + key] = p[:, j] - dp[0, j]
    return out


def run(args):
    """Write complete paired groups atomically for the selected cloud resolutions."""
    dest = TMP / args.tag
    dest.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    rows = 0
    missing = []
    for m in args.months:
        for profile in args.profiles:
            for w in args.updrafts:
                for model in args.models:
                    prefix = args.prefix or (
                        "hourly"
                        if model in ("ranked", "secondary")
                        else "assignments"
                    )
                    file = (
                        dest / f"{prefix}_{m:02}_{profile}_{w:g}_{model}.csv.gz"
                    )
                    if file.is_file() and not args.redo:
                        continue
                    try:
                        out = group(
                            prefix,
                            m,
                            profile,
                            w,
                            model,
                            args.bands,
                            args.streams,
                            args.order,
                            args.layers,
                            args.step,
                            args.drop_cut,
                            args.thin,
                            args.sample_step,
                            args.boundary,
                        )
                    except FileNotFoundError:
                        missing.append(file.name)
                        continue
                    except ValueError as e:
                        if "Missing hourly cloud coverage" in str(e):
                            missing.append(file.name)
                            continue
                        raise
                    temp = file.with_name(f"{file.name}.{os.getpid()}.partial")
                    out.to_csv(
                        temp,
                        index=False,
                        compression="gzip",
                        float_format="%.9g",
                    )
                    temp.replace(file)
                    rows += len(out)
                    if rows % 1344 == 0:
                        print(
                            f"{rows} hours processed in {time.monotonic() - started:.1f}s",
                            flush=True,
                        )
    result = dict(
        tag=args.tag,
        new_hours=rows,
        elapsed_s=time.monotonic() - started,
        missing_group_count=len(missing),
        missing=missing[:8],
    )
    print(json.dumps(result), flush=True)
    if missing:
        raise ValueError(f"Incomplete input coverage: {missing}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--months", type=int, nargs="+", default=list(range(1, 13)))
    p.add_argument("--profiles", nargs="+", default=["clean", "rich"])
    p.add_argument("--updrafts", nargs="+", type=float, default=[0.25, 2.0])
    p.add_argument(
        "--models",
        nargs="+",
        default=["ranked", "secondary", "mixed", "inverse"],
    )
    p.add_argument("--prefix")
    p.add_argument("--tag", default="primary")
    p.add_argument("--bands", type=int, default=96)
    p.add_argument("--streams", type=int, default=8)
    p.add_argument("--order", type=int, default=16)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--drop-cut", type=float, default=0.0)
    p.add_argument("--thin", type=float, default=2e-5)
    p.add_argument("--redo", action="store_true")
    p.add_argument("--sample-step", type=int, default=1)
    p.add_argument(
        "--boundary",
        choices=["observed", "clear_native", "clear_amount", "fixed"],
        default="observed",
    )
    run(p.parse_args())
