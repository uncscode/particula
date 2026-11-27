# %% Bound the absolute sulfate burden and check whether the peak is identifiable
import numpy as np
import pandas as pd
from inverse import PACKET, SALT_DENSITY, M_SALT, M_SULFATE
from particula.particles.properties.convert_size_distribution import (
    get_distribution_in_dn,
)


def main():
    out = PACKET / "population"
    e = pd.read_csv(PACKET / "joint/extrema.csv").query(
        "scope=='all' and scatter_multiple==2"
    )
    authors = pd.read_csv(out / "author-retrieval.csv").query("diameter_nm==100")
    mid = authors.rename(columns={"graph_lo": "fmin", "graph_hi": "fmax"})
    df = pd.concat([a[["diameter_nm", "time_h", "fmin", "fmax"]] for a in (e, mid)])
    n = pd.read_csv(PACKET / "rosati/spectra.csv").set_index("diameter_nm")
    grid = 50 + (np.arange(1400) + 0.5) * 0.25
    w = get_distribution_in_dn(grid, np.ones_like(grid))

    def burden(d):
        return (
            np.pi
            / 6
            * (d * 1e-9) ** 3
            * SALT_DENSITY
            * 1000
            * 1e15
            * M_SULFATE
            / M_SALT
        )

    rows = []
    totals = []
    for shape in ["mean", "initial", "five_hour"]:
        nn = n[shape + "_cm3_decade"]
        err = n[shape + "_read_sensitivity"]
        for t, data in df.groupby("time_h"):
            g = data.set_index("diameter_nm").sort_index()
            d = g.index.to_numpy()
            vals = []
            integ = []
            for count, f in [(np.maximum(0, nn - err), g.fmin), (nn + err, g.fmax)]:
                vals.append(np.interp(d, n.index, count) * burden(d) * f)
                integ.append(
                    w
                    @ (
                        np.interp(grid, n.index, count)
                        * burden(grid)
                        * np.interp(np.log(grid), np.log(d), f)
                    )
                )
            for i, D in enumerate(d):
                rows.append(
                    (
                        shape,
                        t,
                        D,
                        vals[0].iloc[i],
                        vals[1].iloc[i],
                        vals[1].iloc[i] >= vals[0].max(),
                    )
                )
            totals.append((shape, t, *integ))
    pd.DataFrame(
        rows,
        columns=[
            "shape",
            "time_h",
            "diameter_nm",
            "sulfate_lo",
            "sulfate_hi",
            "peak_not_excluded",
        ],
    ).to_csv(out / "size-bounds.csv", index=False)
    pd.DataFrame(
        totals, columns=["shape", "time_h", "integral_lo", "integral_hi"]
    ).to_csv(out / "integral-bounds.csv", index=False)


if __name__ == "__main__":
    main()
