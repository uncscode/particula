"""Check the independent NaCl transfer on the 610 Pa ESEM temperature/RH path."""

import numpy as np
import pandas as pd
import math
from scipy.optimize import brentq
from transfer import HERE, make_model
from trial import states, MW_WATER, prepare
from boundary import cloud, state_at_rh
from stability import grid, onset
from calibrate import fit_binary
from trial import binary_fit


def saturation(t):
    """Liquid water saturation pressure in Pa, T in K, Murphy & Koop (2005) eq. 10."""
    return math.exp(
        54.842763
        - 6763.22 / t
        - 4.210 * math.log(t)
        + 0.000367 * t
        + math.tanh(0.0415 * (t - 218.8))
        * (53.878 - 1331.22 / t - 9.44523 * math.log(t) + 0.014025 * t)
    )


def main():
    best = pd.read_csv(HERE / "final-calibration.csv").iloc[0]
    records = []
    for v in [50, 70, 90]:
        m = make_model("NaCl", best)

        def resid(t):
            return cloud(m, "NaCl", v, temp=t)[0] - 61000 / saturation(t)

        temp = brentq(resid, 273.0, 278.0, xtol=1e-6)
        rh, mf, mt, f, os = cloud(m, "NaCl", v, temp=temp, details=True)
        t85 = brentq(lambda t: 61000 / saturation(t) - 85, 273.0, 278.0)
        meq, d = state_at_rh(m, "NaCl", v, temp=t85)
        records.append([v, temp, rh, mf, mt, f, os, t85, meq, d])
    pd.DataFrame(
        records,
        columns=[
            "OVF_pct",
            "predicted_T_K",
            "model_RH_610Pa",
            "feed_equiv_m",
            "incipient_equiv_m",
            "OS_feed",
            "OS_incipient",
            "T_at85",
            "m_at85",
            "TPD_at85",
        ],
    ).to_csv(HERE / "chamber-path.csv", index=False)
    controls = []
    for kind, tail, par in [
        ("homolog_prior", 1.0, (0.0241, 0.224, 0.006081)),
        ("unrestricted_binary", 1.0, binary_fit(1.0)[0]),
    ]:
        m = prepare("NaCl", tail, par)
        for cap in (20, 80, 400) if kind == "unrestricted_binary" else (160,):
            for v in (50, 70, 90):
                g = grid("NaCl", v, 275.15, m, maxmol=cap, refine=2)
                value, trial, left = onset(g)
                controls.append([kind, v, cap, value, trial, left])
    pd.DataFrame(
        controls,
        columns=[
            "model",
            "OVF_pct",
            "upper_trial_m",
            "onset_RH",
            "incipient_m",
            "left_censored",
        ],
    ).to_csv(HERE / "controls.csv", index=False)
    # Assess compatibility of the homogeneous water-activity calibration assumption
    # and sensitivity to the weight assigned to the imposed 90% RH.
    binary = pd.read_csv(HERE / "binary-water.csv")
    stable = []
    for factor in [0.5, 1.0, 2.0, 1e8]:
        p, loss = fit_binary(best["tail"], best.C2, point_scale=factor)
        m = prepare("NaCl", best["tail"], p)
        trial = np.geomspace(0.02, 400.0, 1001)
        distances = []
        for T, group in binary.groupby("T_K"):
            c, mu = states(m, "NaCl", 1 / (MW_WATER * trial), 1, 1, T)
            _, feed = states(
                m, "NaCl", 1 / (MW_WATER * group.m_SOS_mol_kg_water), 1, 1, T
            )
            distances.extend(
                np.minimum(
                    0.0,
                    (np.einsum("ij,ij->i", c, mu)[:, None] - c @ feed.T).min(axis=0),
                )
            )
        preds = [cloud(m, "NaCl", v)[0] for v in (50, 70)]
        stable.append([factor, *p, loss, min(distances), *preds])
    pd.DataFrame(
        stable,
        columns=[
            "90RH_scale_factor",
            "B1",
            "B2",
            "C2",
            "binary_score",
            "worst_binary_tpd",
            "NaCl50",
            "NaCl70",
        ],
    ).to_csv(HERE / "binary-sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
