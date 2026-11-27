# %% Challenge the frozen fits and recover independent textual readouts.
import numpy as np
import pandas as pd
from fit import ROOT, PARAMS, VOL, make_model, config, observations


def main():
    models = [make_model(n, stride)[1] for n, stride in [(24, 1), (48, 1), (24, 2)]]
    text = pd.read_csv(ROOT.parent / "xiong/text-100nm.csv")
    ref = pd.read_csv(ROOT.parent / "xiong/endmembers.csv").set_index("rh_pct")
    # Independently read Figure 4b and the article's 80% GF changes at whole hours.
    published = pd.read_csv(ROOT.parent / "xiong/fig4b5c-100nm.csv")
    peaks = observations()[1]
    marker = (
        text.query("rh_pct == 80 and time_h <= 3")
        .merge(published, on="time_h")
        .merge(peaks, on="time_h")
    )
    marker["ratio_from_eq12"] = (
        (
            np.exp(marker.log_c0_c_ic)
            / (1 + marker.reported_percent_change_from_fresh / 100) ** 3
            - 1
        )
        * VOL[0]
        / VOL[1]
    )
    marker.to_csv(ROOT / "results/eq12-recovery.csv", index=False)
    numerical = []
    plus = []
    nested = []
    for stage in ("primary", "retention", "blank", "eq8"):
        for row in pd.read_csv(ROOT / stage / "scores.csv").itertuples():
            par = np.array([getattr(row, p) for p in PARAMS])
            flags, bd = config(row.network, row.variant)
            if len(bd) > len(par):
                par = np.r_[par, row.ic_blank]
            x, g, ratio = map(np.asarray, models[0](par, flags, 1.0))
            for typ, model in zip(("time_48", "compose_grid_stride2"), models[1:]):
                xx, gg, rr = map(np.asarray, model(par, flags, 1.0))
                numerical.append(
                    (
                        stage,
                        row.network,
                        row.variant,
                        typ,
                        np.max(abs(xx - x)),
                        np.max(abs(gg[:10] - g[:10])),
                        np.max(abs(rr[:6, 1] - ratio[:6, 1])),
                    )
                )
            pairs = {
                "retention": ("direct", 4),
                "retention_neutral": ("retention", 5),
                "retention_activity": ("chloride_activity", 4),
            }
            if stage == "retention":
                parent, k = pairs[row.network]
                p = par.copy()
                p[k] = 0.0
                err = max(
                    float(np.max(np.abs(a - b)))
                    for a, b in zip(
                        models[0](p, flags, 1.0), models[0](p, config(parent)[0], 1.0)
                    )
                )
                assert err < 1e-10
                nested.append((row.network, parent, err))
            if stage in ("primary", "retention"):
                for a in text.itertuples():
                    index = [40, 60, 70, 75, 80, 85].index(a.rh_pct)
                    gf0 = ref.loc[a.rh_pct, "nacl_gf"]
                    pp = 100 * (g[int(2 * a.time_h) - 1, 1, index] / gf0 - 1)
                    plus.append(
                        (
                            row.network,
                            a.time_h,
                            a.rh_pct,
                            a.reported_percent_change_from_fresh,
                            pp,
                        )
                    )
    out = ROOT / "results"
    pd.DataFrame(
        numerical,
        columns="stage network variant refinement max_inventory_error max_growth_error_thru5h max_ic_log_error_thru3h".split(),
    ).to_csv(out / "frozen-refinement.csv", index=False)
    pd.DataFrame(nested, columns="child parent max_abs_difference".split()).to_csv(
        out / "retention-zero.csv", index=False
    )
    pd.DataFrame(
        plus, columns="network time_h rh_pct observed_percent predicted_percent".split()
    ).to_csv(out / "whole-hour-growth.csv", index=False)


if __name__ == "__main__":
    main()
