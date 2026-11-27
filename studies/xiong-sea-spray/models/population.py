# %% Recover the authors' two aqueous concentrations and compare population weights

import numpy as np
import pandas as pd
from particula.particles.properties.convert_size_distribution import (
    get_distribution_in_dn,
)
from inverse import PACKET, M_SALT, M_SULFATE, SALT_DENSITY

SHAPES = ("mean", "initial", "five_hour", "equal_mass")
OUT = PACKET / "population"


def recovered():
    cp = pd.read_csv(PACKET / "xiong/fig5-sulfate.csv")
    log = pd.read_csv(PACKET / "xiong/fig5-log.csv")
    df = cp.merge(log, on=["diameter_nm", "time_h"], validate="one_to_one")
    assert len(df) == 56
    ref = pd.read_csv(PACKET / "xiong/endmembers.csv").set_index("rh_pct").loc[80]

    def fraction(g, c, log_ratio):
        return c / (c + 1000 * SALT_DENSITY / M_SALT / g**3 * np.exp(-log_ratio))

    df["sodium_fraction"] = fraction(ref.nacl_gf, df.sulfate_mol_per_L, df.log_c0_c)
    for tag, s in [("lo", -1), ("hi", 1)]:
        df["graph_" + tag] = fraction(
            ref.nacl_gf + s * (ref.nacl_spread + 0.010),
            df.sulfate_mol_per_L + s * df.reading_halfwidth_mol_per_L,
            df.log_c0_c + s * df.reading_halfwidth_log,
        )
    df.to_csv(OUT / "author-retrieval.csv", index=False)
    fit = (
        df.groupby("diameter_nm")
        .apply(
            lambda g: np.dot(g.time_h, g.log_c0_c) / np.dot(g.time_h, g.time_h),
            include_groups=False,
        )
        .rename("extracted_slope_per_h")
        .reset_index()
    )
    fit = fit.merge(pd.read_csv(PACKET / "xiong/size-rates.csv"), on="diameter_nm")
    fit["signed_difference"] = fit.extracted_slope_per_h - fit.k_per_h
    fit.to_csv(OUT / "rates.csv", index=False)
    return df


def main():
    OUT.mkdir(exist_ok=True)
    src = recovered()
    specs = pd.read_csv(PACKET / "rosati/spectra.csv").set_index("diameter_nm")
    joint = pd.read_csv(PACKET / "joint/example.csv").query(
        "scope=='all' and scatter_multiple==1"
    )
    mid = src.query("diameter_nm==100")[
        ["diameter_nm", "time_h", "sodium_fraction"]
    ].rename(columns={"sodium_fraction": "f"})
    # 100 nm is the separately published Fig 5 retrieval, not a Table S1 observation.
    joint = pd.concat([joint[["diameter_nm", "time_h", "f"]], mid]).sort_values(
        ["diameter_nm", "time_h"]
    )
    src = src.rename(columns={"sodium_fraction": "f"})
    native = []
    points = []
    summaries = []
    cross = []
    d = np.sort(src.diameter_nm.unique())
    t = np.sort(src.time_h.unique())
    ix2, ix3 = np.searchsorted(d, [200, 300])

    def salt_mass(x):
        # kg per particle times 10^9 ug/kg and 10^6 cm^3/m^3.
        return np.pi / 6 * (x * 1e-9) ** 3 * SALT_DENSITY * 1000 * 1e15

    def spectrum(x, label):
        return np.interp(x, specs.index, specs[label + "_cm3_decade"])

    for label in SHAPES[:-1]:
        row = specs.loc[[200, 300]]
        n = row[label + "_cm3_decade"].to_numpy()
        e = row[label + "_read_sensitivity"].to_numpy()
        cross.append(
            (
                label,
                n[0] / n[1] / 3.375,
                (n[0] - e[0]) / (n[1] + e[1]) / 3.375,
                (n[0] + e[0]) / (n[1] - e[1]) / 3.375,
            )
        )
    pd.DataFrame(cross, columns=["shape", "threshold", "graph_lo", "graph_hi"]).to_csv(
        OUT / "crossovers.csv", index=False
    )
    for source, df in [("authors", src), ("joint_example", joint)]:
        f = (
            df.pivot(index="diameter_nm", columns="time_h", values="f")
            .loc[d, t]
            .to_numpy()
        )
        diffs = np.diff(f, axis=1, prepend=np.zeros((len(d), 1)))
        for bins in [350, 1400, 5600]:
            step = 350 / bins
            grid = 50 + (np.arange(bins) + 0.5) * step
            widths = get_distribution_in_dn(grid, np.ones(bins))
            assert np.isclose(widths.sum(), np.log10(400 / 50), rtol=0, atol=1e-11)
            mean_initial = np.sum(
                get_distribution_in_dn(grid, spectrum(grid, "mean")) * salt_mass(grid)
            )
            for shape in SHAPES:
                if shape == "equal_mass":
                    dry = np.full(len(grid), mean_initial / widths.sum())
                    dry_d = np.full(len(d), dry[0])
                    original = mean_initial
                else:
                    dry = spectrum(grid, shape) * salt_mass(grid)
                    dry_d = spectrum(d, shape) * salt_mass(d)
                    original = np.sum(dry * widths)
                norm = mean_initial / original
                for j, hour in enumerate(t):
                    for interp in ["log", "linear"] if bins == 1400 else ["log"]:
                        x = np.log(grid) if interp == "log" else grid
                        xd = np.log(d) if interp == "log" else d
                        yf = np.interp(x, xd, f[:, j])
                        dy = np.interp(x, xd, diffs[:, j])
                        factor = M_SULFATE / M_SALT
                        sulfate = dry * yf * factor
                        rate = dry * dy * factor / 0.5
                        mass = np.dot(widths, sulfate)
                        production = np.dot(widths, rate)
                        assert 0 <= mass <= original * factor * (1 + 1e-12)
                        summaries.append(
                            (
                                source,
                                shape,
                                bins,
                                interp,
                                hour,
                                original,
                                mean_initial,
                                mass,
                                production,
                                mass * norm,
                                production * norm,
                                grid[np.argmax(sulfate)],
                                grid[np.argmax(rate)],
                            )
                        )
                        if bins == 1400 and interp == "log":
                            for k, D in enumerate(d):
                                points.append(
                                    (
                                        source,
                                        shape,
                                        hour,
                                        D,
                                        f[k, j],
                                        dry_d[k],
                                        dry_d[k] * f[k, j] * factor,
                                        dry_d[k] * diffs[k, j] * factor / 0.5,
                                        dry_d[k] * f[k, j] * factor * norm,
                                    )
                                )
    pd.DataFrame(
        points,
        columns=[
            "source",
            "shape",
            "time_h",
            "diameter_nm",
            "sodium_fraction",
            "initial_salt_ug_m3_decade",
            "sulfate_ug_m3_decade",
            "rate_ug_m3_h_decade",
            "matched_sulfate_ug_m3_decade",
        ],
    ).to_csv(OUT / "at-sizes.csv", index=False)
    pd.DataFrame(
        summaries,
        columns=[
            "source",
            "shape",
            "bins",
            "interpolation",
            "time_h",
            "initial_salt_ug_m3",
            "matched_salt_ug_m3",
            "sulfate_ug_m3",
            "rate_ug_m3_h",
            "matched_sulfate_ug_m3",
            "matched_rate_ug_m3_h",
            "cumulative_peak_nm",
            "rate_peak_nm",
        ],
    ).to_csv(OUT / "integrals.csv", index=False)
    ratios = pd.read_csv(PACKET / "joint/ratios.csv").query("scope=='all'")
    for shape, T, lo, hi in cross:
        for r in ratios.itertuples():
            native.append(
                (
                    shape,
                    r.scatter_multiple,
                    r.time_h,
                    r.outer_lo / hi,
                    r.outer_hi / lo,
                    r.attainable_lo / T,
                    r.attainable_hi / T,
                    (hi / r.outer_lo) ** (1 / 3),
                )
            )
    pd.DataFrame(
        native,
        columns=[
            "shape",
            "scatter_multiple",
            "time_h",
            "massratio_conservative_lo",
            "massratio_conservative_hi",
            "attainable_lo_central_spectrum",
            "attainable_hi_central_spectrum",
            "relative_equivalent_diameter_for_guarantee",
        ],
    ).to_csv(OUT / "mass-ratios.csv", index=False)


if __name__ == "__main__":
    main()
