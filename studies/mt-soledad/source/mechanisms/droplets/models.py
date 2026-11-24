"""Compare optical outcomes on identical measured bypass populations."""

# %%
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar
from particula.dynamics.condensation.mass_transfer import (
    get_first_order_mass_transport_k,
)
from particula.gas.properties.thermal_conductivity import (
    get_thermal_conductivity,
)

UPPER = np.r_[0.75, 1.0, np.arange(1.5, 10.01, 0.5)]
EPOCH = np.datetime64("2023-04-01").astype(int)
NAMES = [
    "unchanged",
    "scalar_loss",
    "scalar_count_efficiency",
    "uniform_amplitude",
    "uniform_joint",
    "thinning_plus_delay",
    "partially_delayed",
    "never_activated_subset",
]
S0 = 0.4
SEED = 32941


def growth_rate(S):
    """Continuum diffusion and latent heat resistance, wet D squared in µm²/s."""
    T = 295.0
    rho = 997.0
    rv = 461.5
    lv = 2.45e6
    es = 611.2 * np.exp(17.67 * (T - 273.15) / (T - 29.65))
    D = 2.50e-5
    # Reference drop radius 1 µm; continuum vapor transition factor is one.
    k = get_first_order_mass_transport_k(1e-6, 1.0, D)
    conductance = k / (4 * np.pi * 1e-6)
    resistance = rho * (
        rv * T / (conductance * es)
        + lv / (get_thermal_conductivity(T) * T) * (lv / (rv * T) - 1)
    )
    return 8e12 * (S / 100) / resistance


def get_cases(source, out):
    d = np.load(out / ("large.npz" if source == "selected" else "processed.npz"))
    if source == "selected":
        v = d["observations"]
        normalized = v[:, :, 3:] / v[:, :, :1]
        overflow = v[:, :, 2] / v[:, :, 0]
        measured = v[:, :, 1] / v[:, :, 0]
        min_n = 2
    else:
        normalized = d["spectra"]
        # The NetCDF histograms exclude overflow and have no branch-specific
        # raw CPC amplitude at identical time/voltage for the whole scan.
        overflow = np.zeros_like(normalized[:, :, 0])
        measured = normalized.sum(axis=2)
        min_n = 4
    keys = d["keys"]
    n = d["n"]
    ok = (n >= min_n) & (normalized.sum(axis=2) > 0).all(axis=1)
    day = keys[:, 0] // 4
    held = (day - EPOCH) % 3 == 0
    result = {}
    for label, keep in [
        ("train", ok & ~held & (keys[:, 1] == 4)),
        ("test", ok & held & (keys[:, 1] == 8)),
    ]:
        v = normalized[keep]
        base = v[:, 0].sum(axis=1)
        result[label] = dict(
            b=v[:, 0] / base[:, None],
            y=v[:, 1] / base[:, None],
            day=day[keep],
            utc=keys[keep, 0] * 21600,
            reported=measured[keep, 1] / measured[keep, 0],
            overflow=overflow[keep] / base[:, None],
            pairs=n[keep],
            S=keys[keep, 1][0] / 10,
        )
    return result


def average(values, day):
    return np.mean([values[day == key].mean(axis=0) for key in np.unique(day)], axis=0)


def transition(edges, loss_d2):
    """Uniform in each incoming optical diameter channel, explicit underflow."""
    a = np.sqrt(edges**2 + loss_d2)
    width = np.maximum(
        0,
        np.minimum(edges[1:, None], a[None, 1:])
        - np.maximum(edges[:-1, None], a[None, :-1]),
    )
    return width / np.diff(edges)[:, None]


def forward(model, b, S, params, edges):
    f, delay, mix = params
    if model in (
        "scalar_loss",
        "scalar_count_efficiency",
        "never_activated_subset",
    ):
        return f * b
    if model == "unchanged":
        return b.copy()
    delayed = b @ transition(edges, delay * growth_rate(S))
    return f * ((1 - mix) * b + mix * delayed)


def fit(c, edges):
    b, y, day = c["b"], c["y"], c["day"]
    frac = average(y.sum(axis=1), day)

    def cost(q):
        # Squared Hellinger distance for finite measures. It scores both
        # amplitude and the distribution of counts across optical bins.
        return average(((np.sqrt(np.maximum(0, q)) - np.sqrt(y)) ** 2).sum(axis=1), day)

    def amplitude(t):
        p = b @ transition(edges, t * growth_rate(S0))
        return average(p.sum(axis=1), day) - frac

    t_amp = brentq(amplitude, 0, (edges[-1] ** 2) / growth_rate(S0))

    def err(t):
        return cost(b @ transition(edges, t * growth_rate(S0)))

    grid = np.linspace(0, edges[-1] ** 2 / growth_rate(S0), 101)

    def grid_min(fn):
        vals = [fn(x) for x in grid]
        i = int(np.argmin(vals))
        r = minimize_scalar(
            fn,
            bounds=(grid[max(0, i - 1)], grid[min(len(grid) - 1, i + 1)]),
            method="bounded",
        )
        return min([(vals[i], grid[i]), (r.fun, r.x)])[1]

    t_joint = grid_min(err)

    def thin(t):
        p = b @ transition(edges, t * growth_rate(S0))
        denom = average(p.sum(axis=1), day)
        if denom < 1e-11:
            return 1.0, cost(p)
        f = np.clip((average((np.sqrt(p * y)).sum(axis=1), day) / denom) ** 2, 0, 1)
        return f, cost(f * p)

    t_thin = grid_min(lambda x: thin(x)[1])

    def partial(t):
        p = b @ transition(edges, t * growth_rate(S0))

        def fn(m):
            return cost((1 - m) * b + m * p)

        r = minimize_scalar(fn, bounds=(0, 1), method="bounded")
        score, m = min((fn(0), 0), (fn(1), 1), (r.fun, r.x))
        return m, score

    t_part = grid_min(lambda x: partial(x)[1])
    return {
        "unchanged": (1, 0, 0),
        "scalar_loss": (frac, 0, 0),
        "scalar_count_efficiency": (frac, 0, 0),
        "never_activated_subset": (frac, 0, 0),
        "uniform_amplitude": (1, t_amp, 1),
        "uniform_joint": (1, t_joint, 1),
        "thinning_plus_delay": (thin(t_thin)[0], t_thin, 1),
        "partially_delayed": (1, t_part, partial(t_part)[0]),
    }


def scores(y, p, b, edges):
    mids = (edges[:-1] + edges[1:]) / 2
    ptotal = p.sum(axis=1)
    if np.any(ptotal <= 0):
        raise ValueError("No modeled accepted droplets in a block")
    pshape = p / ptotal[:, None]
    ys = y / y.sum(axis=1)[:, None]
    return dict(
        observed_ratio=y.sum(axis=1),
        model_ratio=ptotal,
        ratio_error=ptotal - y.sum(axis=1),
        obs_mean_shift=(ys - b) @ mids,
        model_mean_shift=(pshape - b) @ mids,
        mean_diameter_error=(pshape - ys) @ mids,
        shape_TV=np.abs(pshape - ys).sum(axis=1) / 2,
        amplitude_abs_error=abs(ptotal - y.sum(axis=1)),
    )


def intervals(metrics, day):
    rng = np.random.default_rng(SEED)
    ds = np.unique(day)
    draws = rng.integers(len(ds), size=(10000, len(ds)))
    for name, v in metrics.items():
        by = np.array([v[day == k].mean() for k in ds])
        yield (
            name,
            by.mean(),
            *np.percentile(by[draws].mean(axis=1), [2.5, 97.5]),
        )


def run(path):
    tic = time.monotonic()
    summ = []
    profiles = []
    pars = []
    blocks = []
    for source in ("selected", "whole"):
        data = get_cases(source, path)
        for floor in (0.5, 0.65):
            edges = np.r_[floor, UPPER]
            params = fit(data["train"], edges)
            for model, param in params.items():
                pars.append((source, floor, model, *param))
                for split, c in data.items():
                    b, y, day = c["b"], c["y"], c["day"]
                    p = forward(model, b, c["S"], param, edges)
                    metrics = scores(y, p, b, edges)
                    metrics["reported_ratio"] = c["reported"]
                    metrics["observed_minus_reported"] = (
                        metrics["observed_ratio"] - c["reported"]
                    )
                    for metric, m, lo, hi in intervals(metrics, day):
                        summ.append(
                            (
                                source,
                                floor,
                                model,
                                split,
                                metric,
                                m,
                                lo,
                                hi,
                                len(np.unique(day)),
                                len(day),
                                c["pairs"].sum(),
                            )
                        )
                    if floor == 0.5:
                        for i, date in enumerate(day):
                            blocks.append(
                                (
                                    source,
                                    model,
                                    split,
                                    date,
                                    c["utc"][i],
                                    *[
                                        metrics[k][i]
                                        for k in (
                                            "observed_ratio",
                                            "model_ratio",
                                            "obs_mean_shift",
                                            "model_mean_shift",
                                            "shape_TV",
                                        )
                                    ],
                                )
                            )
                        for i in range(20):
                            profiles.append(
                                (
                                    source,
                                    model,
                                    split,
                                    i + 1,
                                    edges[i],
                                    edges[i + 1],
                                    average(b[:, i], day),
                                    average(y[:, i], day),
                                    average(p[:, i], day),
                                )
                            )
    pd.DataFrame(
        summ,
        columns="source floor model split metric mean low high days blocks pairs".split(),
    ).to_csv(path / "scores.csv", index=False)
    pd.DataFrame(
        pars,
        columns="source floor model counted_fraction delay_s delayed_fraction".split(),
    ).to_csv(path / "parameters.csv", index=False)
    pd.DataFrame(
        blocks,
        columns="source model split day utc observed_ratio model_ratio obs_shift model_shift shape_TV".split(),
    ).to_csv(path / "block-scores.csv", index=False)
    pd.DataFrame(
        profiles,
        columns="source model split channel lower_um upper_um bypass denuded predicted".split(),
    ).to_csv(path / "profiles.csv", index=False)
    print(
        "G in square micron/s",
        growth_rate(0.4),
        growth_rate(0.8),
        "elapsed",
        time.monotonic() - tic,
    )


if __name__ == "__main__":
    run(Path(__file__).resolve().parent)
