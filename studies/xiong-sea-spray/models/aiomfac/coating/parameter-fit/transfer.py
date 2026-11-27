# %% Cross-salt predictions after the independent training choice.
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from trial import HERE, prepare, binary_prediction, MW_SOS, MW_WATER
from stability import grid, onset
from boundary import cloud, state_at_rh, evaluate
from calibrate import TEMPERATURE


def make_model(s, r):
    return prepare(s, r["tail"], [r.B1, r.B2, r.C2], r.NH4_delta)


def prediction(s, v, r, **kwargs):
    return cloud(make_model(s, r), s, v, **kwargs)[0]


def stable_binary(r):
    data = pd.read_csv(HERE / "binary-water.csv")
    model = make_model("NaCl", r)
    ms = np.geomspace(0.015, 400.0, 1201)
    for temp, g in data.groupby("T_K"):
        c, mu = evaluate(model, "NaCl", ms, 1.0, 1.0, temp)
        _, feed = evaluate(model, "NaCl", g.m_SOS_mol_kg_water, 1.0, 1.0, temp)
        if (np.einsum("ij,ij->i", c, mu)[:, None] - c @ feed.T).min() < -1e-5:
            return False
    return True


def main():
    all_profiles = pd.concat(
        [
            pd.read_csv(HERE / f)
            for f in [
                "refined-calibration.csv",
                "extended-calibration.csv",
                "joint-calibration.csv",
            ]
        ],
        ignore_index=True,
    )
    keys = ["tail", "B1", "B2", "C2"]
    distinct = all_profiles.drop_duplicates(keys).copy()
    distinct["binary_stable"] = distinct.apply(stable_binary, axis=1)
    distinct[keys + ["binary_stable"]].to_csv(
        HERE / "binary-stability.csv", index=False
    )
    all_profiles = all_profiles.merge(
        distinct.loc[distinct.binary_stable, keys], on=keys
    )
    train = (
        all_profiles.drop_duplicates(keys + ["NH4_delta"])
        .dropna(subset="training_loss")
        .sort_values("training_loss")
    )
    phase = pd.read_csv(HERE / "phase-data.csv")
    sds = phase.query("salt=='AS'").srh_sd_pct.to_numpy()
    target = phase.query("salt=='AS'").srh_pct.to_numpy()
    # Recheck leading distinct fits after continuous optimization over trial phases.
    candidates = train.drop_duplicates(keys).head(12)
    top = pd.concat([train.head(15), candidates]).drop_duplicates()
    corrected = []
    for _, r in top.iterrows():
        vals = np.array([prediction("AS", v, r) for v in [50, 70, 90]])
        corrected.append(
            [
                *r.iloc[:6],
                r.training_loss,
                r.binary_loss + sum(((vals - target) / sds) ** 2),
                *vals,
            ]
        )
    profiles = pd.DataFrame(
        corrected,
        columns=[
            "tail",
            "B1",
            "B2",
            "C2",
            "NH4_delta",
            "binary_loss",
            "grid_loss",
            "polished_loss",
            "AS50",
            "AS70",
            "AS90",
        ],
    ).sort_values("polished_loss")
    profiles.to_csv(HERE / "final-calibration.csv", index=False)
    best = profiles.iloc[0]
    # No measured NaCl onset has entered any of the above rankings.
    profileset = all_profiles.drop_duplicates(keys)
    cross = []
    for _, r in profileset.iterrows():
        model = make_model("NaCl", r)
        rh = [
            onset(grid("NaCl", v, TEMPERATURE, model, refine=2, maxmol=160))[0]
            for v in [50, 70, 90]
        ]
        cross.append([r[k] for k in keys] + [r.binary_loss, *rh])
    curves = pd.DataFrame(
        cross, columns=keys + ["binary_loss", "NaCl50", "NaCl70", "NaCl90"]
    )
    curves.to_csv(HERE / "nacl-profiles.csv", index=False)
    profiles.merge(curves.drop(columns="binary_loss"), on=keys).to_csv(
        HERE / "transfer-profiles.csv", index=False
    )
    results = []
    for _, o in phase.iterrows():
        s = o.salt
        v = o.ovf_pct
        m = make_model(s, best)
        base, mf, mt, feed_os, incipient_os = cloud(m, s, v, details=True)
        state_m, tpd = state_at_rh(m, s, v)
        results.append(
            [
                s,
                v,
                o.srh_pct,
                o.srh_sd_pct,
                base,
                base - o.srh_pct,
                mf,
                mt,
                feed_os,
                incipient_os,
                state_m,
                tpd,
                "two liquids favored" if tpd < -1e-5 else "homogeneous within grid",
            ]
        )
    result = pd.DataFrame(
        results,
        columns=[
            "salt",
            "OVF_pct",
            "observed_RH",
            "observed_SD",
            "model_RH",
            "model_minus_observed",
            "cloud_feed_m",
            "cloud_incipient_m",
            "feed_anion_OS_fraction",
            "incipient_anion_OS_fraction",
            "m_at_85",
            "tpd_at_85",
            "phase_at_85",
        ],
    )
    result.to_csv(HERE / "predictions.csv", index=False)
    checks = []
    for temp, rho, cap, ref in [
        (273.15, 1.266, 160, 2),
        (277.15, 1.266, 160, 2),
        (298.15, 1.266, 160, 2),
        (275.15, 1.0, 160, 2),
        (275.15, 1.5, 160, 2),
        (275.15, 1.266, 400, 3),
    ]:
        for s, v in [
            ("NaCl", 50),
            ("NaCl", 70),
            ("NaCl", 90),
            ("AS", 50),
            ("AS", 70),
            ("AS", 90),
        ]:
            rh = prediction(s, v, best, temp=temp, rho=rho, cap=cap, refine=ref)
            checks.append([s, v, temp, rho, cap, ref, rh])
    pd.DataFrame(
        checks,
        columns=[
            "salt",
            "OVF_pct",
            "temperature_K",
            "SOS_density_g_cm3",
            "trial_max_m",
            "refine",
            "model_RH",
        ],
    ).to_csv(HERE / "sensitivity.csv", index=False)
    reciprocal = []
    spans = []
    for v in [50, 70]:
        hold = 120 - v
        o = phase[(phase.salt == "NaCl") & (phase.ovf_pct == v)].iloc[0]
        fit = curves.copy()
        fit["score"] = (
            fit.binary_loss + ((fit[f"NaCl{v}"] - o.srh_pct) / o.srh_sd_pct) ** 2
        )
        compatible = fit.loc[(fit[f"NaCl{v}"] - o.srh_pct).abs() <= o.srh_sd_pct]
        q = compatible.nsmallest(1, "score").iloc[0]
        near = compatible[compatible.score <= q.score + 4]
        other = near[f"NaCl{hold}"]
        spans.append([v, hold, len(near), other.isna().sum(), other.min(), other.max()])
        row = profileset[(profileset[keys] == q[keys]).all(axis=1)].iloc[0]
        reciprocal.append(
            [
                v,
                hold,
                *[q[k] for k in keys],
                prediction("NaCl", v, row),
                prediction("NaCl", hold, row),
            ]
        )
    pd.DataFrame(
        reciprocal,
        columns=[
            "calibration_OVF",
            "withheld_OVF",
            *keys,
            "fitted_RH",
            "withheld_model_RH",
        ],
    ).to_csv(HERE / "reciprocal.csv", index=False)
    pd.DataFrame(
        spans,
        columns=[
            "calibration_OVF",
            "withheld_OVF",
            "profiles",
            "no_onset_count",
            "finite_low",
            "finite_high",
        ],
    ).to_csv(HERE / "reciprocal-range.csv", index=False)
    binary = pd.read_csv(HERE / "binary-water.csv")
    binary["model_aw"] = np.exp(
        binary_prediction(best["tail"], [best.B1, best.B2, best.C2], binary)
    )
    binary.to_csv(HERE / "binary-fit.csv", index=False)
    # 100 nm check: convert kappa-free pure SOS activity to additive wet/dry volume.
    # RH curvature uses a bounded surface tension instead of assuming a measured value.
    growth = []
    model = make_model("NaCl", best)
    for rho in [1.0, 1.19, 1.266, 1.5]:
        for sigma in [0.0, 0.030, 0.072]:

            def gf(m):
                return (1 + rho / (m * MW_SOS)) ** (1 / 3)

            def resid(m):
                mu = evaluate(model, "NaCl", m, 1.0, 1.0, 297.15)[1][0, 0]
                return (
                    mu
                    + 4
                    * sigma
                    * MW_WATER
                    / (8.314462618 * 297.15 * 997.0 * 100e-9 * gf(m))
                    - np.log(0.85)
                )

            m = brentq(resid, 7.0, 50.0)
            growth.append([rho, sigma, m, gf(m)])
    pd.DataFrame(
        growth,
        columns=[
            "dry_density_g_cm3",
            "surface_tension_N_m",
            "m",
            "growth_factor_85",
        ],
    ).to_csv(HERE / "growth-check.csv", index=False)
    print(best.to_string())
    print(result.to_string(index=False))
    print(reciprocal, flush=True)


if __name__ == "__main__":
    main()
