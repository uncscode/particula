# %% Cloud-base response to chamber-constrained dry residue sizes
"""Coupled ascent with the same incoming spectrum, number and water inventory."""

import argparse
import csv
import json
import platform
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from particula.dynamics import parcel as par
from scipy.optimize import brentq
from scipy.special import ndtr

HERE = Path(__file__).resolve().parent
T0 = 285.0  # K
P0 = 90000.0  # Pa
RH0 = 0.999
E0 = RH0 * par._saturation_pressure(T0)
TOTAL_WATER = par.RD / par.RV * E0 / (P0 - E0)
V0 = (par.RD + par.RV * TOTAL_WATER) * T0 / P0  # m³/kg dry air
PAIRS = {
    "monoterpenes": ("pinene", "ocimene"),
    "sesquiterpenes": ("caryophyllene", "farnesene"),
}


@dataclass(frozen=True)
class Case:
    precursor: str = "fresh"
    rh: str = "dry"
    model: str = "primary"
    kappa: float = 0.1
    w: float = 0.2  # m/s
    n: int = 40  # weighted classes
    small: float = 65.0  # diameter support in nm
    large: float = 300.0
    median: float = 140.0  # geometric median before truncation, nm
    sigma: float = 1.4  # geometric standard deviation
    number: float = 300.0  # cm^-3 in common reference volume
    dz: float = 1.0  # output interval in m; adaptive internal steps
    rtol: float = 2e-5
    probes: int = 96  # additional unweighted threshold probes


def load_survival(path):
    if not path:
        return {}
    with Path(path).open() as handle:
        rows = list(csv.DictReader(handle))
    grouped = {}
    for row in rows:
        key = (row["precursor"], row["rh"], row["model"])
        grouped.setdefault(key, []).append(
            (
                float(row["initial_diameter_nm"]),
                float(row["volume_fraction_200min"]),
            )
        )
    tables = {}
    for key, data in grouped.items():
        d, v = np.array(sorted(data)).T
        assert np.all(np.diff(d) > 0) and np.all((v > 0) & (v <= 1.0000001)), (
            key
        )
        tables[key] = (d, v)
    return tables


def initial(case, tables, probe_nm=()):
    # Uniform spacing in log diameter and exact number CDF differences.
    edges = np.linspace(np.log(case.small), np.log(case.large), case.n + 1)
    x = np.r_[
        (edges[1:] + edges[:-1]) / 2,
        np.linspace(edges[0], edges[-1], case.probes),
        np.log(probe_nm),
    ]
    zedges = (edges - np.log(case.median)) / np.log(case.sigma)
    wt = np.r_[np.diff(ndtr(zedges)), np.zeros(case.probes + len(probe_nm))]
    wt /= wt.sum()
    ix = np.argsort(x)
    x, wt = x[ix], wt[ix]
    d = np.exp(x)
    if case.precursor == "fresh":
        v = np.ones_like(d)
    elif case.precursor == "pilot_half":
        v = np.full_like(d, 0.5)
    else:
        ds, values = tables[case.precursor, case.rh, case.model]
        if d[0] < ds[0] - 1.0e-9 or d[-1] > ds[-1] + 1.0e-9:
            raise ValueError(
                "Requested spectrum exceeds chamber model diameter grid"
            )
        v = np.interp(x, np.log(ds), values)
    dry = 0.5e-9 * d * np.cbrt(v)
    n = wt * case.number * 1.0e6 * V0

    def make(q):
        return par.ParcelInitialState(dry, n, case.kappa, P0, T0, q)

    def water_budget(q):
        wet = par._initial_radii(make(q))
        return q + par.WATER_FACTOR * n @ (wet**3 - dry**3) - TOTAL_WATER

    q = brentq(water_budget, 0.98 * TOTAL_WATER, TOTAL_WATER, xtol=1.0e-16)
    return make(q), x


def margin(wet, dry, temp, k):
    # The stationary point of the exact equilibrium saturation ratio:
    # d ln S_eq / d ln r changes sign from positive to negative at r_c.
    x = wet / dry
    u = x**3 - 1
    a = 2 * par.SIGMA / (par.RV * par.RHO_W * temp * dry)
    return np.log(a * u * (u + k) / (3 * k * x**4))


def observe(case, tr, initial, logr):
    ss = tr.supersaturation
    z = tr.height
    j0 = np.flatnonzero(ss >= 0)[0]
    assert j0 > 0
    base = np.interp(0.0, ss[j0 - 1 : j0 + 1], z[j0 - 1 : j0 + 1])
    height = base + 100.0
    j = np.searchsorted(z, height)
    assert j < len(z)
    mix = (height - z[j - 1]) / (z[j] - z[j - 1])
    wet = (1 - mix) * tr.wet_radius[j - 1] + mix * tr.wet_radius[j]
    temp = (1 - mix) * tr.temperature[j - 1] + mix * tr.temperature[j]
    volume = (1 - mix) * tr.volume[j - 1] + mix * tr.volume[j]
    dry = initial.dry_radius
    m = margin(wet, dry, temp, case.kappa)

    def cdf(x):
        coord = (x - np.log(case.median)) / np.log(case.sigma)
        lo, hi = (
            np.log([case.small, case.large]) - np.log(case.median)
        ) / np.log(case.sigma)
        return (ndtr(coord) - ndtr(lo)) / (ndtr(hi) - ndtr(lo))

    ix = np.flatnonzero(np.diff(m >= 0))
    roots = logr[ix] - m[ix] * (logr[ix + 1] - logr[ix]) / (m[ix + 1] - m[ix])
    bounds = np.r_[logr[0], roots, logr[-1]]
    yes = (np.arange(len(bounds) - 1) % 2) == (0 if m[0] >= 0 else 1)
    frac = np.sum(np.diff(cdf(bounds))[yes])
    wide = np.sum(cdf(logr[ix + 1]) - cdf(logr[ix]))
    histories = margin(
        tr.wet_radius[:j], dry, tr.temperature[:j, None], case.kappa
    )
    # Identify classes that crossed and fell back below critical. Only those
    # still supercritical at the sampling height contribute to the count.
    receded = (np.max(histories, axis=0) > 0) & (m < 0)
    water = tr.vapor + tr.liquid
    assert np.max(abs(water - TOTAL_WATER)) < 3.0e-9
    assert np.min(tr.wet_radius / dry) > 1 and np.all(
        np.isfinite(tr.wet_radius)
    )
    stop = np.argmax(ss[: j + 1])
    return dict(
        fraction_pct=float(100 * frac),
        activated_cm3=float(frac * initial.number.sum() / volume * 1.0e-6),
        max_threshold_interval_pp=float(100 * wide),
        count_distinct_crossings=len(roots),
        original_cutoff_nm=float(np.exp(roots[0])) if len(roots) == 1 else "",
        peak_ss_pct=float(100 * ss[stop]),
        peak_above_base_m=float(z[stop] - base),
        final_z_m=float(height),
        cloud_base_m=float(base),
        sampled_receded_pct=float(
            100 * np.average(receded, weights=initial.number)
        ),
        max_water_error_kgkg=float(np.max(abs(water - TOTAL_WATER))),
        final_lwc_gm3=float(
            ((1 - mix) * tr.liquid[j - 1] + mix * tr.liquid[j]) * 1000 / volume
        ),
    )


def simulate(case, tables, device, probe_nm=()):
    s, x = initial(case, tables, probe_nm)
    tr = par.run_parcel(
        s,
        updraft=case.w,
        timestep=case.dz / case.w,
        steps=round(107 / case.dz),
        device=device,
        adaptive=None if device == "cpu" else par.ParcelROS3(rtol=case.rtol),
    )
    return observe(case, tr, s, x)


def write_csv(path, rows):
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(stage, out, tables, device):
    from dataclasses import replace

    import particula
    import warp as wp

    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=True)
    if (out / f"{stage}.csv").exists():
        raise FileExistsError("Select a new directory for independent replay")
    rows = []

    @lru_cache(None)
    def one(c):
        res = dict(**asdict(c), **simulate(c, tables, device))
        rows.append(res)
        return res

    if stage == "pilot":
        for w in (0.2, 1.0):
            for p in ("fresh", "pilot_half"):
                one(Case(precursor=p, w=w, n=18, probes=34))
    else:
        models = sorted(set(k[2] for k in tables))
        comparisons = []
        for pair, (cyclic, acyclic) in PAIRS.items():
            for rh in ("dry", "wet"):
                for w in (0.2, 1.0):
                    c = Case(
                        precursor=cyclic,
                        rh=rh,
                        w=w,
                        kappa=0.10 if pair == "monoterpenes" else 0.05,
                    )
                    cases = [("primary", c)]
                    if stage == "matrix":
                        cases += [
                            (model, replace(c, model=model))
                            for model in models
                            if model != "primary"
                            and (cyclic, rh, model) in tables
                            and (acyclic, rh, model) in tables
                        ]
                        cases += [
                            (
                                "chamber-range",
                                replace(
                                    c,
                                    small=140.0,
                                    large=220.0,
                                    median=165.0,
                                    sigma=1.2,
                                ),
                            ),
                            ("lower-loading", replace(c, number=100.0)),
                            ("higher-loading", replace(c, number=800.0)),
                        ]
                    elif stage == "refine":
                        cases = [
                            (
                                "primary-refined",
                                replace(
                                    c, n=80, probes=240, rtol=1.0e-6, dz=0.5
                                ),
                            )
                        ]
                    for control, cc in cases:
                        ca = replace(cc, precursor=acyclic)
                        fresh, cyc, acy = (
                            one(
                                replace(
                                    cc,
                                    precursor="fresh",
                                    rh="dry",
                                    model="primary",
                                )
                            ),
                            one(cc),
                            one(ca),
                        )

                        def delta(k):
                            return (
                                one(replace(ca, kappa=k))["fraction_pct"]
                                - cyc["fraction_pct"]
                            )

                        if control.startswith("primary"):
                            # A fixed κ grid establishes the sign on both sides independently of root interpolation.
                            for k in (
                                cc.kappa * 0.4,
                                cc.kappa * 0.7,
                                cc.kappa * 1.4,
                            ):
                                delta(k)
                            lo, hi = cc.kappa * 0.2, cc.kappa * 3
                            if delta(lo) * delta(hi) < 0:
                                cross = brentq(delta, lo, hi, xtol=5.0e-5)
                                offset = delta(cross)
                            else:
                                cross, offset = "", ""
                        else:
                            cross, offset = "", ""
                        comparisons.append(
                            dict(
                                pair=pair,
                                rh=rh,
                                w=w,
                                control=control,
                                cyclic_kappa=cc.kappa,
                                unaged_pct=fresh["fraction_pct"],
                                cyclic_pct=cyc["fraction_pct"],
                                acyclic_pct=acy["fraction_pct"],
                                difference_pp=acy["fraction_pct"]
                                - cyc["fraction_pct"],
                                cyclic_peak_pct=cyc["peak_ss_pct"],
                                acyclic_peak_pct=acy["peak_ss_pct"],
                                cyclic_critical_d0_nm=cyc["original_cutoff_nm"],
                                acyclic_critical_d0_nm=acy[
                                    "original_cutoff_nm"
                                ],
                                cyclic_drops_cm3=cyc["activated_cm3"],
                                acyclic_drops_cm3=acy["activated_cm3"],
                                breakeven_acyclic_kappa=cross,
                                breakeven_residual_pp=offset,
                            )
                        )
        write_csv(out / f"{stage}-comparisons.csv", comparisons)
    write_csv(out / f"{stage}.csv", rows)
    report = dict(
        device=str(wp.get_device(device)),
        device_name=wp.get_device(device).name,
        stage=stage,
        python=platform.python_version(),
        source_version=particula.__version__,
        source_file=particula.__file__,
        warp=wp.__version__,
        runs=len(rows),
        wall_seconds=time.monotonic() - started,
        max_water_error=max(r["max_water_error_kgkg"] for r in rows),
    )
    (out / f"{stage}-runtime.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["pilot", "matrix", "refine"])
    parser.add_argument("out", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda:0"], default="cuda:0")
    parser.add_argument("--survival", type=Path)
    args = parser.parse_args()
    run(args.stage, args.out, load_survival(args.survival), args.device)
