"""Equilibrium inlet hydration and changes in organic/inorganic nitrate inventory."""

import json
import numpy as np
import pandas as pd
import warp as wp
from recipe import HERE, STUDY, MWA, CHARGE
from thermo import AIOMFACModel, spec, inventories, activity, inverse, evaluate


def run():
    r = pd.read_csv(HERE / "recipes.csv")
    r["draw"] = 0
    p = pd.read_csv(HERE / "predictions.csv.gz")
    data = np.load(STUDY / "extension/blocks-final.npz")
    k, env, n = data["keys"], data["env"], data["n"]
    lookup = {}
    for utc in r.utc.unique():
        for ss in (4, 8):
            ix = (k[:, 0] == utc // 21600) & (k[:, 1] == ss)
            include = ix if ix.any() else k[:, 0] == utc // 21600
            lookup[utc, ss] = np.average(env[include, 0], weights=n[include])
    rows = []
    bounds = []
    states = 0
    for label in p.closure.unique():
        model = AIOMFACModel.prepare(spec(label), execution="cuda", device="cuda:0")
        q = np.geomspace(0.15 if label != "sodium" else 0.005, 1e5, 257)
        qo = np.geomspace(0.005, 1e5, 257)
        u = np.r_[
            np.linspace(-np.log(0.995), -np.log(0.45), 700, endpoint=False),
            np.linspace(-np.log(0.45), -np.log(0.00007), 515),
        ]
        for cls in p[p.closure == label].class_id.unique():
            cases = p[(p.closure == label) & (p.class_id == cls)]
            incl = r.merge(
                cases[cases.model == "full"][["utc", "filter_id", "recipe"]],
                on=["utc", "filter_id", "recipe"],
            )
            for start in range(0, len(incl), 24):
                b = incl.iloc[start : start + 24]
                m, f, limit = inventories(b, label)
                if cls != "uniform":
                    frac = (1 + f) / 2 if cls == "org_rich" else f / 2
                    m[:, :6] *= (frac / f)[:, None]
                    m[:, 6:] *= ((1 - frac) / (1 - f))[:, None]
                temp = b.temperature_K.to_numpy()
                a, ideal, _ = activity(model, m, q, temp, limit, "full")
                states += len(q) * len(b)
                curves = {"full": inverse(q, a, u)}
                if cls == "uniform":
                    org, _, _ = activity(model, m, qo, temp, limit, "org")
                    salts, _, _ = activity(model, m, q, temp, limit, "salt")
                    curves.update(
                        ideal=inverse(q, ideal, u),
                        two_pool=inverse(qo, org, u) + inverse(q, salts, u),
                    )
                    states += 2 * len(q) * len(b)
                for name, vol in curves.items():
                    pc = b[["filter_id", "utc", "recipe", "mask"]].merge(
                        cases[cases.model == name],
                        on=["filter_id", "utc", "recipe", "mask"],
                    )
                    gf = np.cbrt(1 + vol)
                    for ss in (4, 8):
                        rh = np.array([lookup[int(utc), ss] for utc in pc.utc])
                        dry = pc[f"dcrit0{ss}_nm"].to_numpy()
                        sat = (
                            np.log(-np.expm1(-u))[None, :]
                            + (
                                4e9
                                * 0.072
                                * 0.01801528
                                / (8.314462618 * 997 * temp * dry)
                            )[:, None]
                            / gf
                        )
                        growth = []
                        blank = []
                        for i, z in enumerate(np.log(rh / 100)):
                            # Equilibrium on the dilute continuation; if two stable roots exist choose larger water.
                            ix = np.flatnonzero((sat[i, :-1] <= z) & (sat[i, 1:] >= z))
                            if not len(ix):
                                raise ValueError("Inlet root unbracketed")
                            j = ix[-1]
                            g = np.interp(z, sat[i, j : j + 2], gf[i, j : j + 2])
                            growth.append(g)
                            blank.append(g**3 - 1 < q[0])
                        pc[f"rh0{ss}_pct"] = rh
                        pc[f"dcrit0{ss}_wet_nm"] = dry * np.array(growth)
                        pc[f"rh0{ss}_setting_specific"] = b[
                            f"ss0{ss}_measured"
                        ].to_numpy()
                        pc[f"rh0{ss}_below_first_water_node"] = blank
                    rows.append(
                        pc[
                            [
                                c
                                for c in pc
                                if c
                                in [
                                    "filter_id",
                                    "mask",
                                    "utc",
                                    "recipe",
                                    "closure",
                                    "model",
                                    "class_id",
                                    "draw",
                                    "number_fraction",
                                ]
                                or c.startswith("rh0")
                                or c.endswith("_wet_nm")
                            ]
                        ]
                    )
    pd.concat(rows).to_csv(
        HERE / "inlet-hydration.csv.gz",
        index=False,
        float_format="%.8g",
        compression={"method": "gzip", "mtime": 0},
    )
    # Ammonium nitrate may evaporate after the selector. These two cases use the remaining dry diameter:
    # convert to original by dividing by residual volume fraction^(1/3), retaining volume prior to loss.
    model = AIOMFACModel.prepare(spec("sodium"), execution="cuda", device="cuda:0")
    for ret in (0, 0.5):
        sub = r.copy()
        lost = (1 - ret) * np.minimum(sub.nitrate, sub.ammonium * 62.004 / 18.038)
        sub["nitrate"] -= lost
        sub["ammonium"] -= lost * 18.038 / 62.004
        ratio = 1 - (lost * (1 + 18.038 / 62.004) / 1.76) / sub.dry_volume
        for i in range(0, len(sub), 24):
            b = sub.iloc[i : i + 24]
            df = evaluate(b, model, "sodium", "uniform", 257, False)[0]
            for ss in [4, 8]:
                df[f"original_dcrit0{ss}_nm"] = df[f"dcrit0{ss}_nm"] / np.cbrt(
                    ratio.iloc[i : i + 24].to_numpy()
                )
            df["test"] = f"nitrate_retained_{ret:g}"
            df["residual_dry_volume_ratio"] = ratio.iloc[i : i + 24].to_numpy()
            bounds.append(df)
            states += len(b) * 257
    band = (
        pd.read_csv(STUDY / "extension/chemistry/ams-band-6h.csv.gz")
        .query("coverage_pass")
        .set_index("time_unix")
    )
    b = r[r.utc.isin(band.index)].copy()
    anions = ["nitrate", "sulfate", "chloride"]
    size = band.loc[b.utc, [i + "_finite_ug_m3" for i in anions]].to_numpy() / MWA
    eq = b[anions].to_numpy() / MWA @ CHARGE
    b[anions] = size * (eq / (size @ CHARGE))[:, None] * MWA
    for i in range(0, len(b), 24):
        sub = b.iloc[i : i + 24]
        df = evaluate(sub, model, "sodium", "uniform", 257, False)[0]
        df["test"] = "band_anion_ratios"
        bounds.append(df)
        states += len(sub) * 257
    pd.concat(bounds).to_csv(
        HERE / "ion-sensitivity.csv.gz",
        index=False,
        float_format="%.8g",
        compression={"method": "gzip", "mtime": 0},
    )
    md = dict(
        activity_states=states,
        device=wp.get_device("cuda:0").name,
        hyd_rows=sum(map(len, rows)),
        band_windows=len(b) // 2,
    )
    (HERE / "hydration.json").write_text(json.dumps(md, indent=2) + "\n")


if __name__ == "__main__":
    run()
