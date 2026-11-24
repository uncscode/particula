"""Compare field chemistry and observed growth to the identical bypass spectrum."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import isotonic_regression
from scipy.special import ndtr
from particula.particles import get_kelvin_radius

HERE = Path(__file__).resolve().parent
FIELD = HERE.parent
STUDY = FIELD.parent
APRIL = int(np.datetime64("2023-04-01").astype(int))
MAPPING = (
    pd.read_csv(STUDY / "mechanisms/mixing/final/fit.csv")
    .query("chem=='bulk' and layout=='uniform'")
    .iloc[0]
)


def held(utc):
    return (np.asarray(utc) // 86400 - APRIL) % 3 == 0


def kelvin_nm(temperature):
    return 2e9 * get_kelvin_radius(0.072, 997.0, 0.01801528, temperature)


def read_ccn():
    """Reconstruct the field curve and sample weights after archived QC."""
    obs = {}
    with np.load(STUDY / "extension/blocks-final.npz") as archive:
        keys, d, n, signals, valid, env = [
            archive[k] for k in ("keys", "diam", "n", "sums", "valid", "env")
        ]
    for block, setting in sorted(set(map(tuple, keys[:, :2]))):
        if setting not in (4, 8):
            continue
        sel = (keys[:, 0] == block) & (keys[:, 1] == setting)
        if n[sel].sum() < 4 or np.min(valid[sel].sum(axis=0)) == 0:
            continue
        cpc, ccn = signals[sel, 0, 2:4].sum(axis=0)
        big = (d >= 120) & (d <= 250)
        if np.any(cpc <= 0) or np.sum(ccn[big]) <= 0:
            continue
        data = dict(
            d=d,
            x=cpc,
            obs=(ccn / cpc) / (ccn[big].sum() / cpc[big].sum()),
            forcing=np.average(env[sel], axis=0, weights=n[sel]),
            n=n[sel].sum(),
        )
        obs[block * 21600, setting / 10] = data
    return obs


def crossings(curve, data):
    """Weighted isotonic half-height and interquartile width on the observed size range."""
    d = data["d"]
    keep = (d >= 30) & (d <= 250)
    fitted = isotonic_regression(curve[keep], weights=data["x"][keep]).x
    quant = [
        np.exp(np.interp(v, fitted, np.log(d[keep])))
        if fitted[0] <= v <= fitted[-1]
        else np.nan
        for v in (0.25, 0.50, 0.75)
    ]
    return quant[1], quant[2] - quant[0]


def score_response(dry, weight, data, scale, spread):
    """Apply the frozen response to particle threshold(s) and normalize as in the scans."""
    log_size = np.log(data["d"])[:, None]
    log_threshold = np.log(dry)[None, :] + scale
    prob = (
        ndtr((log_size - log_threshold) / spread)
        if spread > 0
        else (log_size >= log_threshold)
    )
    curve = prob @ weight
    d = data["d"]
    big = (d >= 120) & (d <= 250)
    plateau = np.average(curve[big], weights=data["x"][big])
    if plateau < 0.1:
        return np.nan, np.nan, np.nan
    curve = curve / plateau
    i = (d >= 30) & (d <= 250)
    rmse = np.sqrt(np.average((curve[i] - data["obs"][i]) ** 2, weights=data["x"][i]))
    mid, width = crossings(curve, data)
    return mid, width, rmse


def koehler_threshold(kappa, supersaturation, kelvin, nq=1537):
    """Maximize dry threshold at equal Köhler saturation over wet/dry water ratios."""
    kappa = np.atleast_1d(kappa)
    q = np.geomspace(1.0e-7, 1.0e5, nq)
    curve = kelvin / (
        np.cbrt(1 + q)
        * (np.log1p(supersaturation / 100) + np.log1p(kappa[:, None] / q))
    )
    result = curve.max(axis=1)
    result[kappa == 0] = kelvin / np.log1p(supersaturation / 100)
    return result


def hydration(dry, kappa, rh, kelvin):
    """Solve the upstream equilibrium mobility diameter for a fixed hygroscopicity."""
    lo = dry.copy()
    hi = dry * np.cbrt(1 + kappa * (rh / 100) / (1 - rh / 100))
    for _ in range(24):
        dm = (lo + hi) / 2
        aw = rh / 100 * np.exp(-kelvin / dm)
        grow = dry * np.cbrt(1 + kappa * aw / (1 - aw))
        lo = np.where(dm < grow, dm, lo)
        hi = np.where(dm >= grow, dm, hi)
    return (lo + hi) / 2


def gf_to_k(gf, kelvin):
    """60 nm subsaturated growth to effective κ under the common surface tension."""
    aw = 0.85 * np.exp(-kelvin / (60 * gf))
    return np.maximum(0, (gf**3 - 1) * (1 / aw - 1))


def obs_transfer():
    import json

    ccn = read_ccn()
    src = np.load(FIELD / "observations/pdf60.npz", allow_pickle=False)
    grid = src["gf"]
    if grid.ndim != 1 or np.any(np.diff(grid) <= 0):
        raise ValueError("Bad growth-factor grid.")
    quadrature = np.gradient(grid)
    quadrature[[0, -1]] *= 0.5
    rows = []
    errors = []
    for clock, utc, density in zip(
        src["clock"].astype(str), src["utc"], src["density"], strict=True
    ):
        w = density * quadrature
        if abs(w.sum() - 1) > 0.02 or not np.isfinite(w).all() or np.any(w < -1e-10):
            raise ValueError("Non-normalized density.")
        w = np.maximum(0, w)
        w /= w.sum()
        for s in [0.4, 0.8]:
            key = (utc, s)
            if key not in ccn:
                continue
            data = ccn[key]
            rh, t, _, ss = data["forcing"]
            a = kelvin_nm(t)
            if not 0.05 < ss < 1.1:
                raise ValueError(ss)
            k = gf_to_k(grid, a)
            physical = koehler_threshold(k, ss, a)
            if len(errors) < 8:
                errors.append(np.max(abs(koehler_threshold(k, ss, a, 3073) - physical)))
            mean_g = w @ grid
            for case, crit, weights, kap in [
                (
                    "mean_GF",
                    koehler_threshold(gf_to_k(np.array([mean_g]), a), ss, a),
                    np.ones(1),
                    gf_to_k(np.array([mean_g]), a),
                ),
                ("distribution", physical, w, k),
                (
                    "distribution_above_one",
                    physical[grid > 1],
                    w[grid > 1] / w[grid > 1].sum(),
                    k[grid > 1],
                ),
            ]:
                om, ow = crossings(data["obs"], data)
                for mapping, threshold, scale, spread in [
                    ("physical", crit, 0, 0),
                    ("inlet_hydrated", hydration(crit, kap, rh, a), 0, 0),
                    (
                        "previous_mapping",
                        crit,
                        MAPPING.log_scale,
                        MAPPING.spread,
                    ),
                ]:
                    pm, pw, rmse = score_response(
                        threshold, weights, data, scale, spread
                    )
                    rows.append(
                        (
                            clock,
                            utc,
                            s,
                            held(utc),
                            case,
                            mapping,
                            mean_g,
                            w[grid < 1].sum(),
                            om,
                            ow,
                            pm,
                            pw,
                            rmse,
                        )
                    )
    df = pd.DataFrame(
        rows,
        columns="clock utc S held case mapping gf_avg fraction_below_one obs_mid obs_width pred_mid pred_width rmse".split(),
    )
    df.to_csv(
        HERE / "kappa-transfer.csv.gz",
        index=False,
        float_format="%.9g",
        compression={"method": "gzip", "mtime": 0},
    )
    (HERE / "kappa-check.json").write_text(
        json.dumps(dict(max_refinement_nm=max(errors), rows=len(df)), indent=2) + "\n"
    )


if __name__ == "__main__":
    obs_transfer()
