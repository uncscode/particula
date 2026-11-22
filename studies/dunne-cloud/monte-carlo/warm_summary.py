"""Summarize paired joint scenarios, retaining time and size resolution."""

import csv
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from warm import HERE, masks, write, rmse


def quantile(values):
    return dict(
        zip(("p05", "median", "p95"), np.quantile(values, [0.05, 0.5, 0.95]))
    )


def main():
    data = np.load(HERE / "warm-trajectories.npz")
    h, obs, Y = data["hours"], data["observed"], data["prediction"]
    families, bands = list(data["family"]), list(data["band"])
    cases = list(csv.DictReader((HERE / "warm-parameters.csv").open()))
    periods = masks()
    rows, pairs, replicates, band_rows, rank_rows = [], [], [], [], []
    measures = {}
    for j, family in enumerate(families):
        for k, mask in periods.items():
            measures[family, k] = rmse(Y[:, j][:, mask], obs[mask])
    # Select using the measured-acid comparator training data alone, with
    # the same member indices retained for every alternative and period.
    train = measures[families[0], "training"][:, 1]
    accepted = train <= np.quantile(train, 0.25)
    selectors = {
        "all": np.ones(len(cases), bool),
        "parent_training_quartile": accepted,
    }
    for j, family in enumerate(families):
        comparator = (
            "measured_clear" if family == "gas_matched_clear" else "measured"
        )
        for selection, take in selectors.items():
            lo, md, hi = np.quantile(Y[take, j], [0.05, 0.5, 0.95], axis=0)
            if selection == "all":
                for t in range(len(h)):
                    for b, band in enumerate(bands):
                        band_rows.append(
                            dict(
                                family=family,
                                band=band,
                                hour=h[t],
                                observed=obs[t, b],
                                p05=lo[t, b],
                                median=md[t, b],
                                p95=hi[t, b],
                            )
                        )
            for k, mask in periods.items():
                for b, band in enumerate(bands):
                    x = measures[family, k][:, b]
                    delta = x - measures[comparator, k][:, b]
                    peak = Y[take, j][:, mask, b].max(axis=1)
                    base = dict(
                        family=family,
                        selection=selection,
                        period=k,
                        band=band,
                        spectra=mask.sum(),
                        draws=take.sum(),
                    )
                    rows.append(
                        dict(
                            **base,
                            **quantile(x[take]),
                            observation_inside_fraction=np.mean(
                                (lo[mask, b] <= obs[mask, b])
                                & (obs[mask, b] <= hi[mask, b])
                            ),
                            observed_max=obs[mask, b].max(),
                            **{
                                f"max_{u}": v for u, v in quantile(peak).items()
                            },
                        )
                    )
                    pairs.append(
                        dict(
                            **base,
                            comparator=comparator,
                            **quantile(delta[take]),
                            fraction_lower=np.mean(delta[take] < 0),
                        )
                    )
                    if b == 1 and selection == "all":
                        for r in range(4):
                            ind = np.array(
                                [int(p["replicate"]) == r for p in cases]
                            )
                            replicates.append(
                                dict(
                                    family=family,
                                    period=k,
                                    replicate=r,
                                    **quantile(x[ind]),
                                    fraction_lower=np.mean(delta[ind] < 0),
                                )
                            )
                        for par in (
                            "acid",
                            "coag",
                            "wall_at_10",
                            "boundary",
                            "ozone",
                            "peroxide",
                        ):
                            rank_rows.append(
                                dict(
                                    family=family,
                                    period=k,
                                    parameter=par,
                                    rho=spearmanr(
                                        [float(p[par]) for p in cases], x
                                    ).statistic,
                                )
                            )
    write("warm-scores.csv", rows)
    write("warm-paired.csv", pairs)
    write("warm-bands.csv", band_rows)
    write("warm-sampling.csv", replicates)
    write("warm-rank-sensitivity.csv", rank_rows)
    write(
        "warm-joint-win.csv",
        [
            dict(
                family=f,
                comparator=(
                    base := "measured_clear"
                    if f == "gas_matched_clear"
                    else "measured"
                ),
                selection="all",
                fraction_lower_both=np.mean(
                    (measures[f, "peak"][:, 1] < measures[base, "peak"][:, 1])
                    & (
                        measures[f, "later"][:, 1]
                        < measures[base, "later"][:, 1]
                    )
                ),
            )
            for f in families
        ],
    )
    led = list(csv.DictReader((HERE / "warm-led.csv").open()))
    v = {
        f: np.array(
            [
                float(i["second_LED_rmse_molecules_cm3"])
                for i in led
                if i["family"] == f
            ]
        )
        for f in ("retained", "background", "delayed_return", "both")
    }
    assert len(v["retained"]) == len(cases)
    write(
        "warm-led-scores.csv",
        [
            dict(
                family=f,
                **quantile(u),
                fraction_below_background=np.mean(u < v["background"]),
            )
            for f, u in v.items()
        ],
    )
    plot(h, obs, Y, families, measures)


def plot(h, obs, Y, families, measures):
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(2, 1, sharey=True, figsize=(8.3, 6.2), layout="constrained")
    curves = [
        ("gas_matched", "#067b75", "-", "Model · reconstructed acid"),
        ("measured", "#361a56", (0, (5, 3)), "Model · measured acid"),
    ]
    for ax, (start, end), title, marker_stride in zip(
        axes, [(0, 6.0001), (6.36, h[-1])],
        ["Early growth and peak", "Later chamber operation"], [6, 20],
    ):
        mask = (h >= start) & (h <= end)
        time = h[mask]
        if start == 0:
            ax.axvspan(0, 3.8, color="#efefef", zorder=0)
            ax.text(1.9, .92, "Training interval", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", color="#505050", fontsize=10)
        for family, color, style, label in curves:
            counts = Y[:, families.index(family), :, 1][:, mask] / 1000
            low, median, high = np.quantile(counts, [.05, .5, .95], axis=0)
            ax.fill_between(time, low, high, color=color, alpha=.15, lw=0)
            ax.plot(time, median, color=color, linestyle=style, lw=2.2, label=label)
        ax.plot(time, obs[mask, 1] / 1000, color="#222222", lw=1.1,
                marker="o", markerfacecolor="white", markeredgecolor="#151515",
                markeredgewidth=1.05, markersize=4.4, markevery=marker_stride,
                label="CLOUD particle counts", zorder=4)
        ax.set(xlim=(time[0], time[-1]), ylim=(0, 24), yticks=range(0, 25, 6),
               ylabel="Particles ≥9.7 nm\n(10³ cm⁻³)")
        ax.set_title(title, loc="left", fontsize=12, weight="semibold")
        ax.grid(axis="y", color=".87", lw=.7)
        ax.set_axisbelow(True)
    axes[-1].set_xlabel("Hours since first size measurement", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend([handles[i] for i in (2, 1, 0)], [labels[i] for i in (2, 1, 0)],
               loc="outside upper center", ncol=3, frameon=False, fontsize=10,
               handlelength=3, columnspacing=1.15)
    fig.savefig(HERE / "warm-ensemble.png", dpi=190)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), layout="constrained", sharey=True)
    subset = [
        "gas_matched",
        "gas_absolute",
        "ozone",
        "peroxide",
        "both",
        "bulk",
        "dimer",
        "measured_clear",
        "gas_matched_clear",
    ]
    display = [
        "OH + SO₂, matched mean",
        "OH + SO₂, absolute",
        "Effective ozone",
        "Effective peroxide",
        "Both effective oxidants",
        "Aqueous pH 2–6",
        "Gas dimers",
        "Clearing sensitivity",
        "OH + SO₂, with clearing*",
    ]
    for ax, period in zip(axes, ["peak", "later"]):
        ax.axvline(0, color="#333333", lw=0.9)
        for i, f in enumerate(subset):
            cmp = "measured_clear" if f == "gas_matched_clear" else "measured"
            q = (
                np.quantile(
                    measures[f, period][:, 1] - measures[cmp, period][:, 1],
                    [0.05, 0.5, 0.95],
                )
                / 1000
            )
            ax.plot([q[0], q[2]], [i, i], lw=2, color="#55819a")
            ax.scatter(q[1], i, s=25, color="#182f47", zorder=4)
        ax.set(
            yticks=range(len(subset)),
            yticklabels=display,
            xlabel="Paired RMSE change (10³ cm⁻³)",
            title="Peak (3.8–6 h)"
            if period == "peak"
            else "Later (6.36–32.97 h)",
        )
        ax.grid(axis="x", alpha=0.2)
    axes[0].invert_yaxis()
    fig.savefig(HERE / "warm-paired-errors.png", dpi=190)
    plt.close(fig)
    comparisons = [
        ("gas_transfer", "Gas transfer, fixed composition"),
        ("finite_hard", "Hard collisions, fixed composition"),
        ("vdw_fixed", "Attractive collisions, fixed composition"),
        ("empirical", "Empirical small-size growth"),
        ("measured", "Attractive collisions, wet salt"),
        ("extended_attraction", "Wet salt, attraction through 19.5 nm"),
        ("ammoniation", "Wet salt, variable ammoniation"),
    ]
    combine, (peak_ax, later_ax) = plt.subplots(
        1, 2, figsize=(10, 4.3), layout="constrained", sharey=True
    )
    peak_mask = (h >= 3.8) & (h <= 6.0001)
    peak_ax.axvline(
        obs[peak_mask, 1].max() / 1000,
        color="#101010",
        ls="--",
        lw=1,
        label="CLOUD peak",
    )
    for row, (f, label) in enumerate(comparisons):
        magnitude = Y[:, families.index(f)][:, peak_mask, 1].max(axis=1)
        for ax, value in (
            (peak_ax, magnitude),
            (later_ax, measures[f, "later"][:, 1]),
        ):
            lo, med, hi = np.quantile(value, [0.05, 0.5, 0.95]) / 1000
            c = "#087f6a" if f == "measured" else "#3b5877"
            ax.plot([lo, hi], [row, row], color=c, lw=2.4)
            ax.scatter(med, row, color=c, s=26)
    peak_ax.set(
        yticks=range(len(comparisons)),
        yticklabels=[v for _, v in comparisons],
        title="Peak concentration ≥9.7 nm",
        xlabel="Number (10³ cm⁻³)",
    )
    later_ax.set(title="Later count error at ≥9.7 nm", xlabel="RMSE (10³ cm⁻³)")
    peak_ax.invert_yaxis()
    peak_ax.legend(frameon=False, fontsize=8)
    for ax in (peak_ax, later_ax):
        ax.grid(axis="x", alpha=0.2)
    combine.savefig(HERE / "warm-growth-controls.png", dpi=190)
    plt.close(combine)


if __name__ == "__main__":
    main()
