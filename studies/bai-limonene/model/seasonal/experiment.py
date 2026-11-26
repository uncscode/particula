"""72-hour cohorts under measured forcing, followed by matched adiabatic liquid clouds."""

import argparse
import ctypes as ct
import os
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "sunlight"))
import aging
import cloud
from cloud import par

TMP = Path(os.environ["SEASONAL_TMP"])
PT = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
MODELS = {
    "ranked": ("ranked", 0.0),
    "mixed": ("independent", 0.0),
    "inverse": ("inverse", 0.0),
    "secondary": ("ranked", 1.0),
    "dark": ("ranked", 0.0),
    "moderate": ("ranked", 0.25),
    "weighted": ("ranked", 0.0),
    "weighted_dark": ("ranked", 0.0),
}


def contiguous(a):
    return np.ascontiguousarray(a, dtype=float)


def age_boxes(pop, mode, flux, sub):
    case = aging.Exposure(
        inventory="wilson",
        common_fresh_k=True,
        spread_aged=True,
        rate_assignment=MODELS[mode][0],
    )
    C, k, frac, _, _, na = aging.species(case)
    if mode.startswith("weighted"):
        # Cain et al. measurement applied as broad thermodenuder volatility groups.
        k[:na] = np.where(
            C[:na] < 0.001, 0.02, np.where(C[:na] <= 0.1, 0.12, 0.11)
        )
    mass = (
        np.pi / 6 * aging.RHO * (pop.original * 1e-9) ** 3 * pop.number * 1e15
    )
    tag = pop.tag == 0
    core = np.where(tag, 0.005 * mass, mass)
    M = np.zeros((len(tag), len(C)))
    M[tag, :na] = (mass - core)[tag, None] * frac
    lut = aging.optical_table("dose.csv")
    d = np.empty((len(flux), len(tag)))
    kap = np.empty_like(d)
    audit = np.empty((len(flux), 5))
    lib = ct.CDLL(str(TMP / "aging.so"))
    f = lib.seasonal_age
    f.argtypes = [ct.c_int] * 5 + [PT] * 13 + [ct.c_double] * 3 + [PT] * 3
    f.restype = None
    f(
        len(flux),
        len(tag),
        flux.shape[1],
        sub,
        len(lut),
        *[
            contiguous(x)
            for x in (
                flux,
                pop.original,
                pop.number,
                core,
                mass,
                np.where(tag, 0.0465, pop.kappa),
                core * np.where(tag, 1, 0.25),
                M,
                C,
                k,
                np.log(lut.diameter_nm),
                lut.ratio_mvos,
                lut.ratio_lvos,
            )
        ],
        aging.J.mvos,
        aging.J.lvos,
        MODELS[mode][1],
        d,
        kap,
        audit,
    )
    assert np.isfinite(d).all() and (kap > 0).all() and audit[:, 4].max() < 1e-9
    return d, kap, audit


def init_cloud(
    pop, d, k, points, grids=None, temperature=cloud.T0, pressure=cloud.P0
):
    # Volume weights belong only to the physical bins. Probes share their exact water history.
    xs = [np.linspace(np.log(e[0]), np.log(e[-1]), points) for e in pop.edges]
    if grids is not None:
        xs = grids
    ds = [d]
    ks = [k]
    for tag, x in enumerate(xs):
        ix = pop.tag == tag
        orig = np.log(pop.original[ix])
        xx = np.broadcast_to(x, (len(d), x.shape[-1]))
        ds.append(
            np.exp(
                xx
                + np.array(
                    [
                        np.interp(xi, orig, np.log(row[ix]) - orig)
                        for xi, row in zip(xx, d)
                    ]
                )
            )
        )
        ks.append(
            np.array([np.interp(xi, orig, row[ix]) for xi, row in zip(xx, k)])
        )
    dry = np.concatenate(ds, axis=1) * 5e-10
    kval = np.concatenate(ks, axis=1)
    n = np.pad(pop.number, (0, sum(x.shape[-1] for x in xs))) * 1e6 * cloud.VREF
    e = cloud.RH0 * par._saturation_pressure(temperature)
    target = par.RD / par.RV * e / (pressure - e)
    q = np.full((len(d), 1), target)
    for i in range(20):
        wet = cloud.wet_at(dry, kval, q, temperature, pressure)
        nxt = target - par.WATER_FACTOR * ((wet**3 - dry**3) @ n)[:, None]
        if np.max(abs(q - nxt)) < 1e-14:
            break
        q = nxt
    assert np.max(abs(q - nxt)) < 1e-13
    b = np.stack([dry, np.broadcast_to(n, dry.shape), wet, kval], axis=-1)
    air = np.column_stack(
        [np.full(len(d), pressure), np.full(len(d), temperature), q.ravel()]
    )
    return b, air, xs


def evolve(b, air, w, dt, levels, depth=100):
    nb, n = b.shape[:2]
    steps = levels * int(np.ceil(depth / w / dt / levels))
    dt = depth / w / steps
    v = np.full((nb, steps), float(w))
    out = np.zeros((nb, 8))
    wet = np.zeros((nb, n))
    tr = np.zeros((nb, levels, 5))
    r = np.zeros((nb, levels, n))
    f = ct.CDLL(os.environ["BAI_PARCEL_LIB"]).evolve_profiles
    f.argtypes = [PT] * 6 + [ct.c_int] * 3 + [ct.c_double, ct.c_int, PT]
    code = f(
        contiguous(b),
        contiguous(air),
        v,
        out,
        wet,
        tr,
        nb,
        n,
        steps,
        dt,
        levels,
        r,
    )
    assert (
        code == 0
        and np.array_equal(r[:, -1], wet)
        and np.isfinite(out).all()
        and out[:, 5].max() < 1e-10
    )
    return out, wet, tr, np.concatenate([b[:, None, :, 2], r], axis=1), dt


def distribution(pop, xs, b, wet, T, expansion, merge_physical=False):
    margin = cloud.margins(b, wet, T)
    n = len(pop.number)
    sizes = []
    weights = []
    for tag, x in enumerate(xs):
        m = margin[n : n + len(x)]
        log_d = np.log(2e6 * wet[n : n + len(x)])
        n += len(x)
        if merge_physical:
            ix = np.where(pop.tag == tag)[0]
            xx = np.r_[x, np.log(pop.original[ix])]
            order = np.argsort(xx)
            x = xx[order]
            m = np.r_[m, margin[ix]][order]
            log_d = np.r_[log_d, np.log(2e6 * wet[ix])][order]
        lo = x[:-1].copy()
        hi = x[1:].copy()
        sel = (m[:-1] >= 0) | (m[1:] >= 0)
        rise = (m[:-1] < 0) & (m[1:] >= 0)
        fall = (m[:-1] >= 0) & (m[1:] < 0)
        for edges, ix in [(lo, rise), (hi, fall)]:
            edges[ix] = (
                x[:-1][ix] - m[:-1][ix] * np.diff(x)[ix] / np.diff(m)[ix]
            )
        nodes = lo[sel, None] + (hi - lo)[sel, None] * np.linspace(0, 1, 17)
        sizes.extend(
            np.exp(
                np.interp(
                    0.5 * (nodes[:, 1:] + nodes[:, :-1]).ravel(), x, log_d
                )
            )
        )
        weights.extend(
            (np.diff(cloud.cdf(pop, tag, nodes), axis=1) * expansion).ravel()
        )
    d = np.asarray(sizes)
    w = np.asarray(weights)
    order = np.argsort(d)
    d = d[order]
    w = w[order]
    pos = w > 0
    d = d[pos]
    w = w[pos]
    total = w.sum()
    q = np.interp(
        np.array([0.10, 0.50, 0.90]) * total, np.cumsum(w) - w * 0.5, d
    )
    hist = np.histogram(d, bins=SPECTRUM, weights=w)[0]
    return total, *q, hist


SPECTRUM = np.r_[0, np.arange(1.0, 41.0), np.inf]


def process(pop, mode, hours, month, flux, opts):
    start = time.monotonic()
    if mode in {"dark", "weighted_dark"}:
        flux = np.zeros_like(flux)
    d, k, a = age_boxes(pop, mode, flux, opts.sub)
    dose = flux.sum(axis=1)
    if flux.shape[1] % 2:
        dose -= 0.5 * (flux[:, 0] + flux[:, -1])
    chem = time.monotonic() - start
    runrows = []
    spectra = []
    for w in opts.wind:
        for j in range(0, len(hours), opts.batch):
            sl = slice(j, j + opts.batch)
            b, air, xs = init_cloud(
                pop,
                d[sl],
                k[sl],
                opts.probes,
                temperature=opts.temperature,
                pressure=opts.pressure,
            )
            out, wet, tr, r, dt = evolve(
                b, air, w, opts.dt, opts.levels, opts.depth
            )
            if opts.zoom:
                grids = [[], []]
                for ii in range(len(out)):
                    margin = cloud.margins(b[ii], wet[ii], out[ii, 1])
                    n = len(pop.number)
                    for tag, x in enumerate(xs):
                        mv = margin[n : n + len(x)]
                        n += len(x)
                        cross = np.flatnonzero((mv[:-1] >= 0) != (mv[1:] >= 0))
                        if len(cross):
                            counts = np.full(
                                len(cross), opts.zoom // len(cross)
                            )
                            counts[: opts.zoom % len(cross)] += 1
                            extra = np.concatenate(
                                [
                                    np.linspace(x[i], x[i + 1], j + 2)[1:-1]
                                    for i, j in zip(cross, counts)
                                ]
                            )
                        else:
                            extra = np.linspace(x[0], x[-1], opts.zoom + 2)[
                                1:-1
                            ]
                        grids[tag].append(np.sort(np.r_[x, extra]))
                grid_arrays = list(map(np.array, grids))
                b, air, xs = init_cloud(
                    pop,
                    d[sl],
                    k[sl],
                    opts.probes,
                    grid_arrays,
                    opts.temperature,
                    opts.pressure,
                )
                out, wet, tr, r, dt = evolve(
                    b, air, w, opts.dt, opts.levels, opts.depth
                )
            z = np.linspace(0, opts.depth, opts.levels + 1)
            p = tr[:, :, 4]
            t = tr[:, :, 3]
            e = par._saturation_pressure(t) * (1 + tr[:, :, 1])
            q = par.RD / par.RV * e / (p - e)
            vol = np.column_stack(
                [
                    ((par.RD + air[:, 2] * par.RV) * air[:, 1] / air[:, 0]),
                    (par.RD + q * par.RV) * t / p,
                ]
            )
            n = len(pop.number)
            number = b[:, None, :n, 1] / vol[:, :, None]
            rw = r[:, :, :n]
            lwp = np.trapezoid(
                4
                * np.pi
                / 3
                * 1e6
                * ((rw**3 - b[:, None, :n, 0] ** 3) * number).sum(axis=2),
                z,
                axis=1,
            )
            ids = []
            for ii, hour in enumerate(hours[sl]):
                case_id = (
                    f"m{month:02d}_h{hour:03d}_{opts.profile}_w{w:g}_{mode}"
                    if month
                    else f"{mode}_{opts.profile}_w{w:g}"
                )
                ids.append(case_id)
                num, q10, q50, q90, hist = distribution(
                    pop,
                    [x[ii] if x.ndim == 2 else x for x in xs],
                    b[ii],
                    wet[ii],
                    out[ii, 1],
                    cloud.VREF / vol[ii, -1],
                    opts.merge_physical,
                )
                eff = (
                    (number[ii, -1] @ rw[ii, -1] ** 3)
                    / (number[ii, -1] @ rw[ii, -1] ** 2)
                    * 1e6
                )
                audit = a[j + ii]
                runrows.append(
                    dict(
                        case_id=case_id,
                        month=month,
                        hour_in_week=hour,
                        profile=opts.profile,
                        w=w,
                        model=mode,
                        activated_cm3=num,
                        p10_um=q10,
                        median_um=q50,
                        p90_um=q90,
                        re_all_um=eff,
                        lwp_gm2=lwp[ii],
                        peak_ss_pct=out[ii, 3] * 100,
                        water_error=out[ii, 5],
                        rh_iteration_error=out[ii, 6],
                        mass_error=audit[4],
                        organic_retained=audit[0],
                        secondary_reacted_fraction=audit[2],
                        duration_fast_reference_h=dose[j + ii, 0],
                        duration_slow_reference_h=dose[j + ii, 1],
                        parcel_dt_s=dt,
                    )
                )
                spectra.append([case_id, *hist])
            if opts.columns:
                dest = (
                    TMP
                    / "columns"
                    / f"{opts.label}_m{month:02d}_{opts.profile}_w{w:g}_{mode}_{j}.npz"
                )
                temp = dest.with_suffix(".pending")
                with temp.open("wb") as handle:
                    np.savez(
                        handle,
                        wet_radius_m=rw.astype("float32"),
                        number_m3=number.astype("float32"),
                        height_m=z,
                        case_id=ids,
                        month=np.full(len(ids), month),
                        hour_in_week=hours[sl],
                        profile=[opts.profile] * len(ids),
                        w=np.full(len(ids), w),
                        model=[mode] * len(ids),
                    )
                temp.replace(dest)
    dest = (
        Path(os.environ.get("BAI_SEASONAL_RESULTS", ROOT / "results"))
        / opts.label
    )
    dest.mkdir(parents=True, exist_ok=True)
    fname = f"m{month:02d}_{opts.profile}_{mode}"
    pd.DataFrame(runrows).to_csv(dest / (fname + ".csv"), index=False)
    pd.DataFrame(
        spectra,
        columns=["case_id", *["N_" + str(i) for i in range(len(SPECTRUM) - 1)]],
    ).to_csv(dest / (fname + ".spectrum.csv.gz"), index=False)
    print(
        fname,
        len(runrows),
        f"{time.monotonic() - start:.2f}s incl chemistry {chem:.2f}s",
        flush=True,
    )


def main():
    p = argparse.ArgumentParser()
    for name, default, typ in [
        ("label", "annual", str),
        ("bins", 48, int),
        ("subbins", 2, int),
        ("sub", 12, int),
        ("probes", 144, int),
        ("zoom", 0, int),
        ("dt", 0.15, float),
        ("levels", 25, int),
        ("batch", 60, int),
        ("stride", 3, int),
        ("depth", 100.0, float),
        ("temperature", 285.0, float),
        ("pressure", 90000.0, float),
        ("dose_scale", 1.0, float),
        ("age_hours", 72, int),
        ("loading", 300.0, float),
        ("seed_median", 100.0, float),
    ]:
        p.add_argument("--" + name, type=typ, default=default)
    p.add_argument("--midpoint", action="store_true")
    p.add_argument(
        "--forcing",
        choices=[
            "central",
            "iso",
            "bb",
            "uvi_gray",
            "uvi_shape",
            "snow_45",
            "snow_65",
        ],
        default="central",
    )
    p.add_argument("--columns", action="store_true")
    p.add_argument("--months", nargs="+", type=int, default=list(range(1, 13)))
    p.add_argument("--profiles", nargs="+", default=["clean", "rich"])
    p.add_argument(
        "--modes",
        nargs="+",
        default=["ranked", "secondary", "mixed", "inverse"],
    )
    p.add_argument("--wind", nargs="+", type=float, default=[2.0, 0.25])
    p.add_argument("--hours", nargs="+", type=int)
    p.add_argument("--merge-physical", action="store_true")
    p.add_argument("--flat-dose", action="store_true")
    o = p.parse_args()
    for m in o.months:
        if m:
            table = pd.concat(
                [
                    pd.read_csv(ROOT / "forcing/history.csv"),
                    pd.read_csv(ROOT / "forcing/forcing.csv"),
                ]
            )
            if o.forcing.startswith(("uvi_", "snow_")):
                source = o.forcing.split("_")[0]
                other = pd.concat(
                    [
                        pd.read_csv(
                            ROOT / f"forcing/history_{source}_sensitivity.csv"
                        ),
                        pd.read_csv(
                            ROOT / f"forcing/forcing_{source}_sensitivity.csv"
                        ),
                    ]
                )
                table = table.merge(
                    other[
                        [
                            "month",
                            "hour_in_week",
                            "j_fast_scale_" + o.forcing,
                            "j_slow_scale_" + o.forcing,
                        ]
                    ],
                    on=["month", "hour_in_week"],
                    validate="one_to_one",
                )
            table = (
                table[table.month == m].set_index("hour_in_week").sort_index()
            )
            hours = np.array(
                o.hours
                if o.hours
                else sorted(set(range(0, 168, o.stride)) | {167})
            )
            key = "" if o.forcing == "central" else "_" + o.forcing
            if o.age_hours <= 0 or o.age_hours % 2 or o.age_hours > 72:
                raise ValueError(
                    "Use a positive, even age up to the archived 72-hour antecedent."
                )
            length = o.age_hours + int(o.midpoint)
            if o.midpoint:
                assert o.sub % 2 == 0
            flux = np.array(
                [
                    table.loc[
                        h - length + 1 : h,
                        ["j_fast_scale" + key, "j_slow_scale" + key],
                    ].to_numpy()
                    for h in hours
                ]
            )
            assert (
                flux.shape == (len(hours), length, 2)
                and np.isfinite(flux).all()
                and flux.min() >= 0
            )
            flux *= o.dose_scale
            if o.flat_dose:
                dose = flux.sum(axis=1)
                if o.midpoint:
                    dose -= 0.5 * (flux[:, 0] + flux[:, -1])
                flux = np.broadcast_to(
                    dose[:, None, :] / o.age_hours, flux.shape
                ).copy()
        else:
            hours = np.array([0])
            flux = np.zeros((1, o.age_hours, 2))
        for profile in o.profiles:
            o.profile = profile
            pop = aging.inflow(
                profile, o.bins, o.subbins, loading=o.loading, med=o.seed_median
            )
            for mode in o.modes:
                process(pop, mode, hours, m, flux, o)


if __name__ == "__main__":
    main()
