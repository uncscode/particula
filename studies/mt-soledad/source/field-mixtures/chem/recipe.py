"""Build group-constrained filter recipes without loading hygroscopic or CCN responses."""

import hashlib
import os
import re
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent
STUDY = HERE.parents[1]
SRC = Path(os.environ.get("SOLEDAD_INPUT", STUDY / "inputs"))
NAMES = ["glycerol", "butanediol", "succinic", "levulinic", "hexadecanol"]
# Screening surrogate formulae and the reported FTIR mass per mole of bonds.
MW = np.array([92.094, 90.122, 118.088, 116.116, 242.446])
GROUPS = ["OH", "CH", "COOH", "CO"]
BONDS = np.array([[3, 2, 0, 0, 1], [5, 8, 4, 7, 33], [0, 0, 2, 1, 0], [0, 0, 0, 1, 0]])
MATRIX = np.array([23.014, 7.0135, 45.017, 28.010])[:, None] * BONDS / MW
FIT = ["Alcohol_Sum_", "Alkane", "Acid_Sum_", "Carbonyl"]
ERROR = ["AlcoholSDV", "AlkaneSDV", "AcidSDV", "CarbonylSDV"]
MWA = np.array([62.004, 96.063, 35.453])
CHARGE = np.array([1, 2, 1])


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_xrf():
    p = SRC / "xrf.nc"

    def array(k):
        text = subprocess.check_output(
            ["h5dump", "-y", "-m", "%.17g", "-w", "0", "-d", k, p], text=True
        )
        s = re.search(r"\n   DATA \{(.*?)\n   \}", text, re.S).group(1)
        return np.fromstring(s, sep=",")

    return pd.DataFrame(
        {
            k: array(k)
            for k in [
                "start time",
                "stop time",
                "Na",
                "Mg",
                "K",
                "Ca",
                "NaUnc",
                "MgUnc",
                "KUnc",
                "CaUnc",
            ]
        }
    ).rename(columns={"start time": "begin", "stop time": "end"})


def fit(y, sd, budget, recipe, draw):
    # Match the detected group signals. Censored observations can occupy [0,3 sigma].
    detected = y > 0
    target = np.where(detected, np.maximum(0, y + draw * sd), 0)
    # Weighted L1 penalties; undetected groups penalized only above the proxy detection limit.
    upper = np.where(detected, target, 3 * sd)
    lo = np.where(detected, target, 0)
    mat = np.r_[
        np.c_[MATRIX, -np.diag(sd)],
        np.c_[-MATRIX, -np.diag(sd)],
        [np.r_[np.ones(5), np.zeros(4)]],
    ]
    # Tiny tie-break chooses the hydroxyl-rich carrier for high-solubility; low favors alkyl carrier.
    tie = (
        np.array([1, 0.2, 0.1, 0, 0])
        if recipe == "low"
        else np.array([0, 0.7, 0.1, 0, 1])
    )
    result = linprog(
        np.r_[0.004 * tie, np.ones(4)],
        A_ub=mat,
        b_ub=np.r_[upper, -lo, budget],
        bounds=(0, None),
        method="highs",
    )
    if not result.success:
        raise ValueError(result.message)
    mass = result.x[:5]
    modeled = MATRIX @ mass
    error = np.where(
        detected, abs(modeled - y) / sd, np.maximum(0, modeled - 3 * sd) / sd
    )
    return mass, modeled, error


def build():
    f = pd.read_csv(SRC / "ftir.csv")
    on = pd.read_csv(SRC / "organic-nitrate.csv").set_index("FilterID")
    flow = pd.read_csv(HERE.parent / "comparison/filter-flow.csv").set_index(
        "filter_id"
    )
    raw = pd.read_csv(STUDY / "extension/chemistry/ams.csv.gz").dropna(
        subset=["organics_fraction"]
    )
    f["start"] = (
        pd.to_datetime(f.RealStart, utc=True, errors="coerce").astype("int64") // 10**9
    )
    f["stop"] = (
        pd.to_datetime(f.RealStop, utc=True, errors="coerce").astype("int64") // 10**9
    )
    f = f.query(
        "1682899200<=stop and start<1688169600 and MaintenanceFlag==0 and x3OFG==1 and FilterMask in [1,2,3]"
    ).copy()
    nest = f.pivot_table(index="Start", columns="FilterMask", values="OMsum")
    medratio = nest.div(nest[1], axis=0).median()
    w = pd.read_csv(STUDY / "organic-films/bulk/composition.csv")
    xrf = load_xrf().query("1682899200<=end and begin<1688169600").copy()
    cols = ["Na", "Mg", "K", "Ca"]
    emw = np.array([22.990, 24.305 / 2, 39.098, 40.078 / 2])
    for i, x in xrf.iterrows():
        a = raw[raw.time_unix.between(x.begin, x.end, inclusive="left")]
        n = (
            a[[s + "_ug_m3" for s in ["nitrate", "sulfate", "chloride"]]]
            .mean()
            .to_numpy()
            / MWA
        )
        req = n @ CHARGE - a.ammonium_ug_m3.mean() / 18.038
        pm = f[(f.FilterMask == 1) & (abs(f.start - x.begin) < 120)]
        xrf.loc[i, "unweighted_missing_equiv"] = req
        xrf.loc[i, "basis"] = "ambient_mean"
        if len(pm) == 1 and pm.iloc[0].FilterID in flow.index:
            measured = flow.loc[pm.iloc[0].FilterID]
            if pd.notna(measured.ammonium_open):
                n = (
                    measured[
                        ["nitrate_open", "sulfate_open", "chloride_open"]
                    ].to_numpy(float)
                    / MWA
                )
                req = n @ CHARGE - measured.ammonium_open / 18.038
                xrf.loc[i, "basis"] = "valve_open_flow_weighted"
                xrf.loc[i, "filter_id"] = pm.iloc[0].FilterID
                xrf.loc[i, "n_open_flow"] = measured.n_open
        xrf.loc[i, "ams_scans"] = len(a)
        xrf.loc[i, "unmatched_eq_umol_m3"] = req
        xrf.loc[i, "Na_required_ug_m3"] = req * 22.99
        xrf.loc[i, "all_metal_eq_umol_m3"] = (x[cols] / emw).sum()
        xrf.loc[i, "metal_eq_unc_rss"] = np.linalg.norm(
            x[[j + "Unc" for j in cols]].to_numpy(float) / emw
        )
    xrf.to_csv(HERE / "xrf.csv", index=False)
    settings = np.load(STUDY / "extension/blocks-final.npz")
    keys = settings["keys"]
    env = settings["env"]
    n = settings["n"]
    r, ledger, unc, coverage = [], [], [], []
    for _, a in f.iterrows():
        mask = int(a.FilterMask)
        am = raw[raw.time_unix.between(a.start, a.stop, inclusive="left")]
        w0 = w[(w.utc >= a.start) & (w.utc + 21600 <= a.stop)]
        coverage.append(
            dict(
                filter_id=a.FilterID,
                mask=mask,
                start=a.start,
                stop=a.stop,
                enclosed_windows=len(w0),
                ams_samples=len(am),
            )
        )
        if w0.empty:
            continue
        ratio = (
            1
            if mask == 1
            else min(1.0, a.OMsum / nest.loc[a.Start, 1])
            if pd.notna(nest.loc[a.Start, 1])
            else medratio[mask]
        )
        val = flow.loc[a.FilterID, "organics_open"]
        if not np.isfinite(val):
            continue
        base = val * ratio
        y = a[FIT].to_numpy(float)
        sd = a[ERROR].to_numpy(float)
        budget = max(0.0, base - a.Amine)
        seed = int(a.FilterID[3:]) + 51093
        draws = qmc.Sobol(6, scramble=True, seed=seed).random_base2(6)
        for label in ["low", "high"]:
            central, modeled, error = fit(y, sd, budget, label, np.zeros(4))
            ledger.append(
                dict(
                    filter_id=a.FilterID,
                    mask=mask,
                    recipe=label,
                    cut_scale=ratio,
                    cut_scale_from_pair=bool(
                        mask == 1 or pd.notna(nest.loc[a.Start, 1])
                    ),
                    om_filter=a.OMsum,
                    ams_om_mean=base,
                    orgN_group_ug_m3=on.OrgN.get(a.FilterID, np.nan),
                    amine=a.Amine,
                    amine_sd=a.AmineSDV,
                    ams_scans=len(am),
                    **{
                        k + g: val
                        for k, vals in [
                            ("obs_", y),
                            ("sd_", sd),
                            ("pred_", modeled),
                            ("err_sigma_", error),
                        ]
                        for g, val in zip(GROUPS, vals)
                    },
                    **dict(zip(NAMES, central)),
                    fit_max_sigma=error.max(),
                )
            )
            for dnum, draw in enumerate(draws, 1):
                v, _, _ = fit(y, sd, budget, label, (draw[:4] - 0.5) * 3.46)
                unc.append(
                    [
                        a.FilterID,
                        mask,
                        label,
                        dnum,
                        *((v / base) if base else 0 * v),
                        draw[4],
                        draw[5],
                    ]
                )
            for _, b in w0.iterrows():
                row = dict(
                    filter_id=a.FilterID,
                    mask=mask,
                    utc=int(b.utc),
                    recipe=label,
                    start=int(a.start),
                    stop=int(a.stop),
                    cut_scale=ratio,
                    organic_total=b.organics_ug_m3 * ratio,
                    temperature_K=b.temperature_K,
                    fit_max_sigma=error.max(),
                    amine_ug_m3=a.Amine * b.organics_ug_m3 * ratio / base,
                    orgN_group_ug_m3=on.OrgN.get(a.FilterID, np.nan),
                )
                row.update(dict(zip(NAMES, central * row["organic_total"] / base)))
                row["residual"] = row["organic_total"] - sum(row[m] for m in NAMES)
                for k in ["nitrate", "sulfate", "chloride", "ammonium"]:
                    row[k] = b[k + "_ug_m3"] * ratio
                peq = (
                    row["nitrate"] / 62.004
                    + 2 * row["sulfate"] / 96.063
                    + row["chloride"] / 35.453
                    - row["ammonium"] / 18.038
                )
                row["missing_eq"] = peq
                row["dry_volume"] = (
                    row["organic_total"] / 1.5
                    + (
                        sum(
                            row[k]
                            for k in [
                                "nitrate",
                                "sulfate",
                                "chloride",
                                "ammonium",
                            ]
                        )
                        + 22.99 * peq
                    )
                    / 1.76
                )
                for ss in [4, 8]:
                    ix = (keys[:, 0] == b.utc // 21600) & (keys[:, 1] == ss)
                    row[f"ss0{ss}_pct"] = (
                        np.average(env[ix, 3], weights=n[ix]) if ix.any() else ss / 10
                    )
                    row[f"ss0{ss}_measured"] = ix.any()
                r.append(row)
    pd.DataFrame(r).to_csv(HERE / "recipes.csv", index=False, float_format="%.12g")
    pd.DataFrame(ledger).to_csv(
        HERE / "filter-fits.csv", index=False, float_format="%.12g"
    )
    pd.DataFrame(coverage).to_csv(HERE / "coverage.csv", index=False)
    pd.DataFrame(
        unc,
        columns=[
            "filter_id",
            "mask",
            "recipe",
            "draw",
            *NAMES,
            "residual_u",
            "solubility_u",
        ],
    ).to_csv(
        HERE / "draws.csv.gz",
        index=False,
        float_format="%.9g",
        compression={"method": "gzip", "mtime": 0},
    )
    print(
        pd.DataFrame(r).groupby("mask").size(),
        pd.DataFrame(ledger).fit_max_sigma.describe(),
    )


if __name__ == "__main__":
    build()
