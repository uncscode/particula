# %% Replot the saved cloud comparisons. No simulation occurs here.
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt


def rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def figure(folder):
    matrix = rows(folder / "results/matrix-comparisons.csv")
    refined = rows(folder / "refinement/refine-comparisons.csv")
    feedback = rows(folder / "feedback/feedback.csv")
    polished = rows(folder / "cutoff/cutoff-comparisons.csv")
    fit_range = rows(folder / "envelope/envelope-comparisons.csv")
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.8), layout="constrained")
    colors = {"dry": "#884c1f", "wet": "#246587"}
    marker = {"dry": "o", "wet": "s"}
    style = {"dry": "-", "wet": "--"}
    for col, (pair, title) in enumerate(
        [
            ("monoterpenes", "β-ocimene − α-pinene"),
            ("sesquiterpenes", "β-farnesene − β-caryophyllene"),
        ]
    ):
        ax, ak = axes[:, col]
        for rh in ("dry", "wet"):
            selected = sorted(
                [
                    r
                    for r in matrix
                    if r["pair"] == pair
                    and r["rh"] == rh
                    and r["control"] == "primary"
                ],
                key=lambda r: float(r["w"]),
            )
            threshold = [
                float(v["breakeven_acyclic_kappa"]) / float(v["cyclic_kappa"])
                for r in selected
                for v in refined
                if (v["pair"], v["rh"], v["w"]) == (r["pair"], r["rh"], r["w"])
            ]
            xs, ys = (
                np.array([float(r["w"]) for r in selected]),
                np.array(
                    [
                        float(v["gain_pp"])
                        for r in selected
                        for v in polished
                        if (v["pair"], v["rh"], v["w"], v["case"])
                        == (r["pair"], r["rh"], r["w"], "primary")
                    ]
                ),
            )
            label = "Dry aging" if rh == "dry" else "75% RH aging"
            # Offset symbols only to avoid hiding one whisker behind another at a matched ascent.
            positions = xs + (-0.014 if rh == "dry" else 0.014)
            lo, hi = [], []
            for r in selected:
                variants = [
                    float(v["gain_pp"])
                    for v in fit_range
                    if (v["pair"], v["rh"], v["w"]) == (pair, rh, r["w"])
                ]
                variants.append(ys[len(lo)])
                lo.append(min(variants))
                hi.append(max(variants))
            ax.errorbar(
                positions,
                ys,
                yerr=[ys - lo, hi - ys],
                capsize=4,
                linestyle=style[rh],
                color=colors[rh],
                marker=marker[rh],
                markersize=6,
                label=label,
            )
            ak.plot(
                positions,
                threshold,
                linestyle=style[rh],
                color=colors[rh],
                marker=marker[rh],
                markersize=6,
            )
        ax.set_title(title, fontsize=13)
        for a in [ax, ak]:
            a.grid(axis="y", alpha=0.22)
            a.spines[["right", "top"]].set_visible(False)
            a.set_xlim(0.08, 1.12)
            a.set_xticks([0.2, 1.0])
            a.set_xticklabels(["0.2", "1.0"])
        ax.set_ylim(0, 1.08 * max(float(v["gain_pp"]) for v in fit_range))
        ak.axhline(1, color="#414141", ls=":", lw=1.4)
        ak.axhspan(0.06 / 0.17, 0.08 / 0.13, color="#e4e5df", zorder=-1)
        ak.axhline(0.07 / 0.15, color="#555951", ls="-.", lw=1.0)
        ak.text(
            0.6,
            0.485,
            "Whole-tree mixture ratio",
            ha="center",
            fontsize=9,
            color="#42463d",
        )
        ak.set_ylim(0.3, 1.05)
        ak.set_xlabel("Updraft speed (m s⁻¹)")
    axes[0, 0].set_ylabel("Acyclic activation advantage\n(percentage points)")
    axes[1, 0].set_ylabel("Acyclic / cyclic κ needed\nfor equal activation")
    axes[0, 0].legend(fontsize=9, loc="best")
    axes[0, 0].set_xlim(0.06, 1.13)
    fig.savefig(folder / "cloud-activation.png", dpi=180)
    plt.close(fig)

    # Save a table with the numerical contrasts behind the common-air check.
    pairs = {
        "monoterpenes": ("pinene", "ocimene"),
        "sesquiterpenes": ("caryophyllene", "farnesene"),
    }
    summaries = []
    for r in refined:
        pair = r["pair"]
        cyc, acy = pairs[pair]

        def f(carrier, observed):
            return float(
                next(
                    v["fraction_pct"]
                    for v in feedback
                    if (
                        v["pair"],
                        v["rh"],
                        v["w"],
                        v["refined"],
                        v["carrier"],
                        v["observed"],
                    )
                    == (pair, r["rh"], r["w"], "True", carrier, observed)
                )
            )

        effect = float(r["difference_pp"])
        base = f(cyc, acy) - f(cyc, cyc)
        other = f(acy, acy) - f(acy, cyc)
        summaries.append(
            dict(
                pair=pair,
                rh=r["rh"],
                w=r["w"],
                coupled_gain_pp=effect,
                gain_in_cyclic_air_pp=base,
                gain_in_acyclic_air_pp=other,
                share_of_cyclic_air_gain_lost=1 - effect / base,
                refined_threshold_ratio=float(r["breakeven_acyclic_kappa"])
                / float(r["cyclic_kappa"]),
            )
        )
    with (folder / "feedback-comparisons.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path)
    figure(p.parse_args().folder)
