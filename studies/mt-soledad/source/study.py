"""Fit on 0.4% training days and predict 0.6% on separate campaign dates."""

# %%
import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import warp as wp
from activation import conditions, oracle, tables
from scipy.optimize import minimize, minimize_scalar

MODELS = ("common", "constant", "rh", "throughput")
EPOCH = np.datetime64("2023-04-01").astype("datetime64[D]").astype(int)


def require(condition, message):
    """Stop a replay when a scientific or numerical check fails."""
    if not condition:
        raise RuntimeError(message)


def prepare(data, order, mask):
    """Collect the log bin edges and measured branch spectra."""
    bounds = data["bounds"][mask]
    density = data["cpc"][mask, order]
    lo, hi = np.log(bounds[:, :, 0]), np.log(bounds[:, :, 1])
    return lo, hi, density


def count(prepared, d, b, band=None):
    """Integrate the bounded spectrum above the activation diameter."""
    lo, hi, density = prepared
    if band is not None:
        lo = np.maximum(lo, np.log(band[0]))
        hi = np.minimum(hi, np.log(band[1]))
    width = np.maximum(0.0, hi - np.maximum(lo, np.log(d)[:, None]))
    return (density[:, b] * width).sum(axis=1)


def interp(grid, logk, table):
    """Interpolate each scan at its effective log hygroscopicity."""
    pos = (np.clip(logk, grid[0], grid[-1]) - grid[0]) / (grid[1] - grid[0])
    left = np.clip(np.floor(pos).astype(int), 0, len(grid) - 2)
    frac = pos - left
    return (
        table[np.arange(len(table)), left] * (1 - frac)
        + table[np.arange(len(table)), left + 1] * frac
    )


def upstream_multiplier(dry, kelvin, rh, k):
    # Common classifier hydration, even if downstream activation kappa differs.
    """Solve subsaturated growth on the common upstream classifier path."""
    r = rh / 100
    lower = np.ones_like(dry)
    upper = np.broadcast_to((1 + k * r / (1 - r)) ** (1 / 3), dry.shape).copy()
    for _ in range(32):
        mid = (lower + upper) / 2
        volume = mid**3 - 1
        below = volume / (volume + k) * np.exp(kelvin / (dry * mid)) < r
        lower = np.where(below, mid, lower)
        upper = np.where(below, upper, mid)
    return (lower + upper) / 2


def select_pairs(data, order, match, window, qc_screen):
    """Select matched spectra and assign independent UTC days."""
    day = data["time"][:, 0].astype(int) // 86400
    hold = (day - EPOCH) % 3 == 0
    eligible = data["mismatch"] <= match
    if window == "may":
        eligible &= day >= np.datetime64("2023-05-01").astype(int)
    number = (
        np.log(data["bounds"][:, :, 1] / data["bounds"][:, :, 0])[:, None, :]
        * data["cpc"][:, order]
    ).sum(axis=2)
    if qc_screen:
        eligible &= (data["obs"][:, order] <= 1.20 * number).all(axis=1)
    return day, hold, eligible, number


def fit_score(data, cutoff, grid, order, match, window, hydrated, qc_screen):
    """Fit the branch models and evaluate them on disjoint dates."""
    day, hold, eligible, number = select_pairs(data, order, match, window, qc_screen)
    kelvin = conditions(data)[:, 2].reshape(-1, 2)

    def diameter(k, upstream, b, selection, table):
        dry = interp(grid, k, table[:, b])
        if hydrated:
            return dry * upstream_multiplier(
                dry,
                kelvin[selection, b],
                data["rh"][selection, b],
                np.exp(upstream),
            )
        return dry

    train = eligible & ~hold & (abs(data["setpoint"][:, 0] - 0.4) < 0.025)
    test = eligible & hold & (abs(data["setpoint"][:, 0] - 0.6) < 0.025)
    prep = prepare(data, order, train)
    obs = data["obs"][train, order]

    def weights(days):
        _, inverse, n = np.unique(days, return_inverse=True, return_counts=True)
        return 1 / n[inverse]

    w = weights(day[train])

    def loss(res):
        return np.average(2 * 25 * (np.hypot(25, res) - 25), weights=w)

    tr_table = cutoff[train]

    def predicted(k, b, upstream=None):
        return count(
            prep,
            diameter(k, k if upstream is None else upstream, b, train, tr_table),
            b,
        )

    bounds = (grid[0], grid[-1])
    fit = minimize_scalar(
        lambda k: loss(predicted(k, 0) - obs[:, 0]),
        bounds=bounds,
        method="bounded",
    )
    require(fit.success, "Bypass fit check failed")
    base = fit.x
    bypass = predicted(base, 0)
    rh = data["rh"][train, 1]
    rh_limits = [float(rh.min()), float(rh.max())]
    x = (rh - 45) / 10
    delta = obs[:, 1] - obs[:, 0]

    def objective(params):
        return loss(
            predicted(base + params[0] + params[1] * x, 1, base) - bypass - delta
        )

    const = minimize_scalar(
        lambda a: objective((a, 0)), bounds=(-1.5, 1.5), method="bounded"
    )
    opt = minimize(
        objective,
        [const.x, 0.0],
        bounds=[(-1.5, 1.5), (-1, 1)],
        method="Powell",
        options={"xtol": 1e-7, "ftol": 1e-10},
    )
    require(const.success and opt.success, "Inlet fit check failed")
    unadjusted = predicted(base, 1)
    throughput = minimize_scalar(
        lambda factor: loss(factor * unadjusted - bypass - delta),
        bounds=(0.25, 1),
        method="bounded",
    )
    require(throughput.success, "Count fraction fit check failed")
    offsets = [(0.0, 0.0), (const.x, 0.0), opt.x]

    def evaluate(selection, band_index=None):
        prep = prepare(data, order, selection)
        obs = data["obs"][selection, order]
        band = None
        if band_index is not None:
            band = ((40, 90), (120, 250))[band_index]
            obs = data["band_obs"][selection, order, :, band_index]
        rh = data["rh"][selection, 1]
        table = cutoff[selection]
        x = (np.clip(rh, *rh_limits) - 45) / 10
        by = count(prep, diameter(base, base, 0, selection, table), 0, band)
        out = dict(
            day=day[selection],
            utc=data["time"][selection, 0].astype(int),
            inlet_rh=rh,
            bypass_obs=obs[:, 0],
            denuded_obs=obs[:, 1],
            obs=obs[:, 1] - obs[:, 0],
            bypass=by,
        )
        for name, (a, b) in zip(MODELS[:3], offsets, strict=True):
            den = count(
                prep,
                diameter(base + a + b * x, base, 1, selection, table),
                1,
                band,
            )
            total = ((prep[1] - prep[0]) * prep[2][:, 1]).sum(axis=1)
            require(
                np.all((den >= 0) & (den <= total + 1e-8)),
                "Count bounds check failed",
            )
            out[name] = den - by
        out["throughput"] = throughput.x * (by + out["common"]) - by
        out["max_observed_fraction"] = np.max(
            data["obs"][selection, order] / number[selection], axis=1
        )
        return pd.DataFrame(out)

    pars = dict(
        throughput=float(throughput.x),
        bypass_k=float(np.exp(base)),
        constant_log_multiplier=float(const.x),
        rh_intercept=float(opt.x[0]),
        rh_per_10pct=float(opt.x[1]),
        rh_training_range=rh_limits,
        train_pairs=int(train.sum()),
        train_days=len(set(day[train])),
    )
    return (
        pars,
        evaluate(test),
        evaluate(train),
        [evaluate(test, b) for b in (0, 1)],
    )


def summarize(frame):
    """Weight dates equally and resample dates in paired model comparisons."""
    results = []
    for band, mask in [
        ("all", frame.inlet_rh.notna()),
        ("<40", frame.inlet_rh < 40),
        ("40-50", (frame.inlet_rh >= 40) & (frame.inlet_rh < 50)),
        (">=50", frame.inlet_rh >= 50),
    ]:
        sub = frame.loc[mask].copy()
        if sub.empty:
            continue
        for m in MODELS:
            sub[m + "_bias"] = sub[m] - sub.obs
            sub[m + "_mae"] = abs(sub[m + "_bias"])
            sub[m + "_denuded"] = sub[m] + sub.bypass
        sub["rh_minus_common_mae"] = sub.rh_mae - sub.common_mae
        sub["rh_minus_throughput_mae"] = sub.rh_mae - sub.throughput_mae
        sub["rh_minus_constant_mae"] = sub.rh_mae - sub.constant_mae
        means = sub.groupby("day").mean()
        rng = np.random.default_rng(20261123)
        indices = rng.integers(len(means), size=(4000, len(means)))
        for key in [
            "obs",
            "bypass_obs",
            "denuded_obs",
            "bypass",
            *MODELS,
            *(m + s for m in MODELS for s in ["_bias", "_mae", "_denuded"]),
            "rh_minus_common_mae",
            "rh_minus_constant_mae",
            "rh_minus_throughput_mae",
        ]:
            values = means[key].values
            ci = np.percentile(values[indices].mean(axis=1), [2.5, 97.5])
            results.append(
                dict(
                    band=band,
                    metric=key,
                    estimate=values.mean(),
                    low=ci[0],
                    high=ci[1],
                    pairs=len(sub),
                    days=len(means),
                )
            )
    return pd.DataFrame(results)


# %%
def pilot_conditions(cc):
    """Select humidity extremes at both calibrated supersaturations."""
    # Check each supersaturation at the ends of its sampled humidity range.
    selected = []
    for nominal in (0.4, 0.6):
        group = cc[np.abs(cc[:, 0] - nominal) < 0.025]
        selected.extend(group[[np.argmin(group[:, 1]), np.argmax(group[:, 1])]])
    return np.asarray(selected)


def run(args):
    """Check the scalar reference before fitting the selected scenarios."""
    tic = time.monotonic()
    d = dict(np.load(args.input / "pairs.npz"))
    cc = conditions(d)
    wp.init()
    gpu = wp.get_device("cuda:0")

    def ram():
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()

    before = ram()
    test_k = np.array([0.028, 0.23, 1.1])
    selected = pilot_conditions(cc)
    pilot = []
    for hyd in (False, True):
        predicted = tables(selected, test_k, hyd)
        reference = np.array(
            [[oracle(row, k, hyd) for k in test_k] for row in selected]
        )
        for i, j in np.ndindex(reference.shape):
            pilot.append(
                dict(
                    hydrated=hyd,
                    S=selected[i, 0],
                    RH=selected[i, 1],
                    kelvin_nm=selected[i, 2],
                    k=test_k[j],
                    cpu_nm=reference[i, j],
                    cuda_nm=predicted[i, j],
                )
            )
        require(
            abs(reference - predicted).max() < 1e-5,
            "CPU peak parity check failed",
        )
    pd.DataFrame(pilot).to_csv(args.out / "pilot.csv", index=False)
    meta = dict(
        device=gpu.name,
        conditions=len(cc),
        before_MiB=before,
        pilot_MiB=ram(),
        pilot_seconds=time.monotonic() - tic,
        pilot_thresholds=len(pilot),
        warp=wp.__version__,
    )
    if args.pilot:
        print(json.dumps(meta))
        return
    summaries = []
    params = {}
    scenarios = [
        ("dry", 129, False, 0, 0.30, "all", False),
        ("may-coarse", 129, False, 0, 0.30, "may", False),
        ("fine", 257, False, 0, 0.30, "all", False),
        ("may", 257, False, 0, 0.30, "may", False),
        ("may-smoothed", 257, False, 1, 0.30, "may", False),
        ("may-stable", 257, False, 0, 0.20, "may", False),
        ("may-hydrated", 257, True, 0, 0.30, "may", False),
        ("may-qc", 257, False, 0, 0.30, "may", True),
    ]
    prev = None
    workloads = []
    for name, n, hyd, order, match, window, qc_screen in scenarios:
        grid = np.linspace(np.log(0.015), np.log(1.5), n)
        if n != prev:
            cutoff = tables(cc, np.exp(grid), False).reshape(-1, 2, n)
            workloads.append(dict(grid=n, hydrated=False, thresholds=cutoff.size))
            prev = n
        params[name], scored, train, bands = fit_score(
            d, cutoff, grid, order, match, window, hyd, qc_screen
        )
        stages = [("held", scored), ("train", train)]
        if name == "may":
            for label, frame in zip(("40-90nm", "120-250nm"), bands, strict=True):
                frame.to_csv(args.out / (label + "-pairs.csv"), index=False)
                stages.append((label, frame))
        for stage, f in stages:
            summary = summarize(f)
            summary["scenario"] = name
            summary["split"] = stage
            summaries.append(summary)
        scored.to_csv(args.out / (name + "-pairs.csv"), index=False)
    pd.concat(summaries).to_csv(args.out / "summary.csv", index=False)
    meta.update(
        total_seconds=time.monotonic() - tic, workloads=workloads, params=params
    )
    (args.out / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--pilot", action="store_true")
    args = p.parse_args()
    args.out.mkdir(exist_ok=True)
    run(args)
