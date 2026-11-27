# %% Evaluate chemistry-specific predictions in original measurement units
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.interpolate import RegularGridInterpolator
from fit import ROOT, PARAMS, D, VOL, config, make_model


def main():
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    scores = pd.concat(
        [pd.read_csv(ROOT / tag / "scores.csv") for tag in ("primary", "retention")]
    )
    spectra = pd.read_csv(ROOT.parent / "rosati/spectra.csv").set_index("diameter_nm")
    low = pd.read_csv(ROOT / "low_so2/s20.csv")
    fig5 = pd.read_csv(ROOT.parent / "xiong/fig5-sulfate.csv").merge(
        pd.read_csv(ROOT.parent / "xiong/fig5-log.csv"), on=["diameter_nm", "time_h"]
    )
    sizes = np.searchsorted(D, fig5.diameter_nm)
    ticks = np.rint(fig5.time_h * 2).astype(int) - 1
    _, predict, _ = make_model(24)
    c0 = 1000 / (
        VOL[0]
        * pd.read_csv(ROOT.parent / "xiong/endmembers.csv")
        .query("rh_pct==80")
        .nacl_gf.iloc[0]
        ** 3
    )
    budget = []
    comp = []
    dose = []
    dose_scores = []
    solid = []
    nested = []
    sat = pd.read_csv(ROOT / "thermo/sulfate-solids.csv").query("rh_pct==85")
    sat_lookup = {}
    for col in ["thenardite_saturation", "mirabilite_saturation"]:
        axes = sat.pivot(index="f", columns="neutral_na_share", values=col)
        sat_lookup[col] = RegularGridInterpolator(
            (axes.index, axes.columns), axes.to_numpy()
        )
    for r in scores.itertuples():
        name = r.network
        par = np.array([getattr(r, p) for p in PARAMS])
        flags, _ = config(name)

        def forward(multiplier):
            x, g, logratio = map(np.array, predict(par, flags, float(multiplier)))
            nacl, exposed_nacl, bisulfite, bisulfate, neutral_salt = x.transpose(
                2, 0, 1
            )
            # S(IV) dry partial volume is unknown. Use the same two-endpoint proxy as for GF.
            vo = (
                (nacl + exposed_nacl + (1 - par[8]) * bisulfite) * VOL[0]
                + (bisulfate + par[8] * bisulfite) * VOL[1]
                + 2 * neutral_salt * VOL[2]
            )
            wet_l = vo * g[:, :, 4] ** 3 / 1000
            return (
                x,
                (bisulfate + neutral_salt) / wet_l,
                np.log(c0 * wet_l / np.maximum(nacl + exposed_nacl, 1e-12)),
            )

        x, c, log_ratio = forward(1.0)
        nacl, exposed_nacl, bisulfite, bisulfate, neutral_salt = x.transpose(2, 0, 1)
        for target, pred, obs in [
            ("sulfate_mol_L", c, fig5.sulfate_mol_per_L),
            ("ln_c0_over_c", log_ratio, fig5.log_c0_c),
        ]:
            for d, t, z, v in zip(
                fig5.diameter_nm, fig5.time_h, obs, pred[ticks, sizes]
            ):
                comp.append((name, d, t, target, z, v))
        total = bisulfate + 2 * neutral_salt
        q = 2 * neutral_salt / np.maximum(total, 1e-12)
        for col, fun in sat_lookup.items():
            sat_values = fun(
                np.stack([np.clip(total, 0, 1), np.clip(q, 0, 1)], axis=-1)
            )
            solid.append(
                (
                    name,
                    col,
                    sat_values[:7].max(),
                    int((sat_values[:7] > 1).sum()),
                    total[:7].max(),
                    q[:7].max(),
                )
            )
        for j, t in enumerate(np.arange(0.5, 3.51, 0.5)):
            for shape in ["mean", "initial", "five_hour"]:
                a = []
                for k in [3, 5]:
                    d = D[k]
                    n = spectra.loc[d, shape + "_cm3_decade"]
                    mass = (
                        n
                        * np.pi
                        / 6
                        * (d * 1e-9) ** 3
                        * 2165
                        * 1e15
                        * 96.06
                        / 58.443
                        * (bisulfate[j, k] + neutral_salt[j, k])
                    )
                    a.append(mass)
                budget.append(
                    (
                        name,
                        shape,
                        t,
                        *a,
                        a[1] / a[0],
                        (bisulfate[j, 5] + neutral_salt[j, 5])
                        / (bisulfate[j, 3] + neutral_salt[j, 3]),
                    )
                )
        il = (low.time_h * 2).astype(int).to_numpy() - 1

        def diagnostics(speed):
            _, cs, ls = forward(speed)
            v = cs[il, 1]
            u = ls[il, 1]
            return v, u

        # Dose linearity was frozen before S20. The slope-ratio case is conditioned on all low-dose times.
        # An additional single clock is calibrated ONLY to sulfate in the first two lower-dose samples.
        def objective(s):
            z, _ = diagnostics(s)
            return np.sum((z[:2] - low.sulfate_mol_per_L.to_numpy()[:2]) ** 2)

        opt = minimize_scalar(objective, bounds=(0.025, 1.0), method="bounded")
        for scenario, speed in [
            ("linear_frozen", 0.10),
            ("published_slope_ratio", 0.2254 / 0.6523),
            ("early_two_sulfate", opt.x),
        ]:
            pred_c, pred_l = diagnostics(speed)
            for j, row in enumerate(low.itertuples()):
                dose.append(
                    (
                        name,
                        scenario,
                        speed,
                        row.time_h,
                        row.sulfate_mol_per_L,
                        pred_c[j],
                        row.log_c0_c,
                        pred_l[j],
                    )
                )
            dose_scores.append(
                (
                    name,
                    scenario,
                    speed,
                    np.sqrt(np.mean((pred_c - low.sulfate_mol_per_L) ** 2)),
                    np.sqrt(
                        np.mean(
                            (pred_c[2:] - low.sulfate_mol_per_L.to_numpy()[2:]) ** 2
                        )
                    ),
                    np.sqrt(np.mean((pred_l - low.log_c0_c) ** 2)),
                )
            )
        # A zero added rate must reproduce its parent under identical clock and optical settings.
        if name in [
            "neutral",
            "siv_parallel",
            "siv_sink",
            "parallel_neutral",
            "activity_neutral",
        ]:
            test = par.copy()
            parent = "direct"
            if name in ["neutral", "activity_neutral", "parallel_neutral"]:
                test[5] = 0.0
                parent = {
                    "neutral": "direct",
                    "parallel_neutral": "siv_parallel",
                    "activity_neutral": "chloride_activity",
                }[name]
            else:
                test[4] = 0.0
            xp = predict(test, flags, 1.0)
            yp = predict(test, config(parent)[0], 1.0)
            err = max(float(np.abs(a - b).max()) for a, b in zip(xp, yp))
            assert err < 1e-10
            (nested.append((name, parent, err)))

    def write(rows, cols, file):
        pd.DataFrame(rows, columns=cols.split()).to_csv(out / file, index=False)

    write(
        budget,
        "network spectrum time_h sulfate200_ug_m3_decade sulfate300_ug_m3_decade massratio_300_200 sulfur_per_Na_ratio",
        "budget.csv",
    )
    write(comp, "network diameter_nm time_h observable observed predicted", "fig5.csv")
    write(
        dose,
        "network scenario speed time_h observed_sulfate_mol_L predicted_sulfate_mol_L observed_ln_c0_over_c predicted_ln_c0_over_c",
        "dose.csv",
    )
    write(
        dose_scores,
        "network scenario speed sulfate_rmse_all sulfate_rmse_late ln_ratio_rmse_all",
        "dose-errors.csv",
    )
    write(
        solid,
        "network saturation max_ratio_at85 cells_above_one max_convertedNa max_neutral_Na_share",
        "phase-flags.csv",
    )
    write(nested, "child parent max_abs_difference", "zero-rate.csv")


if __name__ == "__main__":
    main()
