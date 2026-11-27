# %% Compare coating geometries and transfer the fitted Particula chemistry.
"""Run from the pinned worktree with its Particula, AIOMFAC and JAX environment."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from fit import D, PARAMS, RH, VOL, config, make_model  # noqa: E402 (sibling study source)


def thickness(v, growth):
    """Shell thickness / total wet radius, with all water in the core."""
    brine = (1 - v) * growth**3
    return 1 - np.cbrt(brine / (brine + v))


def factor(v, kind, coefficient, growth):
    delta = thickness(v, growth)
    if kind == "constant":
        return np.ones_like(v)
    if kind == "patches":
        return np.maximum(0, 1 - coefficient * v)
    if kind == "composition":
        return np.exp(-coefficient * v / (1 - v))
    if kind == "resistance":
        return 1 / (1 + coefficient * delta)
    if kind == "attenuation":
        return np.exp(-coefficient * delta)
    raise ValueError(kind)


def constrained_fit(v, values, kind, growth):
    if kind == "constant":
        return 0.0
    opt = least_squares(
        lambda a: factor(v, kind, a[0], growth) - values,
        x0=[1.0],
        bounds=(0.0, 500.0),
        ftol=1e-13,
        gtol=1e-13,
        xtol=1e-13,
    )
    assert opt.success
    return opt.x[0]


def main():
    k = pd.read_csv(HERE / "figure6b.csv").sort_values("ovf_pct")
    sulfate = pd.read_csv(HERE / "figure6a.csv")
    s2 = pd.read_csv(HERE / "figure6-s2.csv")
    zero = k.loc[k.ovf_pct == 0, "k_per_h"].item()
    fresh = (
        pd.read_csv(HERE.parent / "thermo/observed-endmembers.csv")
        .query("salt=='nacl' and rh_pct==85")
        .iloc[0]
    )
    vv = k.ovf_pct.to_numpy() / 100
    obs = k.k_per_h.to_numpy() / zero
    calib = (vv > 0) & (vv <= 0.33)
    comp = np.arange(0, 71) / 100
    rate_rows = []
    fits = []
    coverage = []
    for pct, rel in zip(k.ovf_pct, obs):
        coverage.append(
            (
                pct,
                rel,
                max(0.0, 1 - rel),
                thickness(pct / 100, fresh.gf_liquid),
                thickness(pct / 100, 1.0),
            )
        )
    pd.DataFrame(
        coverage,
        columns="ovf_pct throughput effective_coverage_min wet_shell_thickness_fraction dry_shell_thickness_fraction".split(),
    ).to_csv(HERE / "geometry.csv", index=False)
    for geom, growth in [
        ("aiomfac", fresh.gf_liquid),
        ("observed", fresh.observed_gf),
        ("dry", 1.0),
    ]:
        for law in ["constant", "patches", "resistance", "attenuation", "composition"]:
            coef = constrained_fit(vv[calib], obs[calib], law, growth)
            curves = []
            for omit in np.where(calib)[0]:
                mask = calib.copy()
                mask[omit] = False
                b = constrained_fit(vv[mask], obs[mask], law, growth)
                curves.append(zero * factor(comp, law, b, growth))
            pred = zero * factor(comp, law, coef, growth)
            for pct, z, lo, hi in zip(
                np.arange(71), pred, np.min(curves, axis=0), np.max(curves, axis=0)
            ):
                rate_rows.append((geom, law, pct, z, lo, hi))
            zh = zero * factor(vv[~calib & (vv > 0)], law, coef, growth)
            fits.append(
                (
                    geom,
                    law,
                    coef,
                    np.sqrt(np.mean((zh - k.loc[vv > 0.33, "k_per_h"]) ** 2)),
                )
            )
    curves = pd.DataFrame(
        rate_rows,
        columns="geometry law ovf_pct predicted_k_per_h omit_one_min omit_one_max".split(),
    )
    curves.to_csv(HERE / "rate-curves.csv", index=False)
    pd.DataFrame(
        fits, columns="geometry law coefficient held_k_rmse_per_h".split()
    ).to_csv(HERE / "rate-fits.csv", index=False)
    _, predict, _ = make_model(24)
    fresh_core = (
        pd.read_csv(HERE.parent.parent / "xiong/endmembers.csv")
        .set_index("rh_pct")
        .nacl_gf
    )
    rows = []
    growth_rows = []
    for network, stage in [
        ("parallel_neutral", "primary"),
        ("retention_neutral", "retention"),
        ("direct", "primary"),
    ]:
        scores = pd.read_csv(HERE.parent / stage / "scores.csv").set_index("network")
        p = scores.loc[network, PARAMS].to_numpy(float)
        for pct in sulfate.ovf_pct.unique():
            kobs = k.loc[k.ovf_pct == pct, "k_per_h"].item()
            source = curves.query("geometry=='aiomfac' and ovf_pct==@pct")
            for law, ratio in [
                (r.law, r.predicted_k_per_h / zero) for r in source.itertuples()
            ] + [("measured_rate_ratio", kobs / zero)]:
                states, gf, _ = map(
                    np.asarray, predict(p, config(network)[0], float(ratio))
                )
                for diam in [50, 100, 150, 200]:
                    idx = np.where(D == diam)[0].item()
                    for time in [2, 3]:
                        j = 2 * time - 1
                        nacl, exposed_nacl, bisulfite, bisulfate, neutral_salt = states[
                            j, idx
                        ]
                        dry = (
                            (nacl + exposed_nacl + (1 - p[8]) * bisulfite) * VOL[0]
                            + (bisulfate + p[8] * bisulfite) * VOL[1]
                            + 2 * neutral_salt * VOL[2]
                        )
                        org = VOL[0] * pct / (100 - pct)
                        wet = dry * gf[j, idx, 4] ** 3
                        cs = (bisulfate + neutral_salt) * 1000 / wet
                        total = (bisulfate + neutral_salt) * 1000 / (wet + org)
                        measured = sulfate.query(
                            "ovf_pct==@pct and time_h==@time"
                        ).sulfate_mol_per_L.item()
                        rows.append(
                            (network, law, pct, diam, time, ratio, cs, total, measured)
                        )
                        if (
                            diam == 100
                            and network == "parallel_neutral"
                            and law == "attenuation"
                        ):
                            for h, rh in enumerate(RH):
                                rr = s2.query("ovf_pct==@pct and rh_pct==@rh")
                                gf0 = rr.loc[rr.time_h == 0, "gf"].item()
                                gft = rr.loc[rr.time_h == time, "gf"].item()
                                # Register the predicted fresh optical signal to each composition's measured baseline.
                                pure0 = fresh_core.loc[rh]
                                full = np.cbrt(
                                    (gf[j, idx, h] ** 3 * dry + org) / (dry + org)
                                )
                                fresh_full = np.cbrt(
                                    (pure0**3 * VOL[0] + org) / (VOL[0] + org)
                                )
                                growth_rows.append(
                                    (
                                        pct,
                                        time,
                                        rh,
                                        gft,
                                        gf0 * gf[j, idx, h] / pure0,
                                        gf0 * full / fresh_full,
                                    )
                                )
    pred = pd.DataFrame(
        rows,
        columns="network law ovf_pct reference_diameter_nm time_h rate_multiplier sulfate_inorganic_mol_L sulfate_whole_mol_L observed_mol_L".split(),
    )
    pred.to_csv(HERE / "sulfate-transfer.csv", index=False)
    pd.DataFrame(
        growth_rows,
        columns="ovf_pct time_h rh_pct observed_gf predicted_inorganic_gf predicted_whole_gf".split(),
    ).to_csv(HERE / "growth-transfer.csv", index=False)


if __name__ == "__main__":
    main()
