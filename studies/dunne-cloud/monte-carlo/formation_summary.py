"""Validate fits, quantify paired cross-validation and plot saved ensemble outputs."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from formation import (
    ACID,
    ALPHA,
    AMMONIA,
    HERE,
    OUT,
    SPLITS,
    WINDOW,
    J,
    Q,
    T,
    anchors,
    law,
)

from reference import ROOT, cold_condition_label

QS = [0.05, 0.5, 0.95]
COLORS = {"full": "#18577d", "assisted": "#c65c24"}
LABELS = {"full": "Binary allowed", "assisted": "Ammonia required"}


def summary(df, groups, value):
    return (
        df.groupby(groups, observed=True)[value]
        .quantile(QS)
        .unstack()
        .rename(columns={0.05: "p05", 0.5: "median", 0.95: "p95"})
        .reset_index()
    )


def verify():
    data = np.array([(T - 248) / 50, np.log(ACID), np.log(AMMONIA), Q, ALPHA])
    errors = []
    grad = []
    for split in ("223_low", "248_low"):
        for topo in COLORS:
            for shape in ("linear", "curved"):
                for ion in (False, True):
                    p = anchors(split, topo, shape, ion)[0]
                    pred, j, _, _ = law(p, data, topo, shape, ion)
                    path = (
                        OUT / "joint" / f"{split}_{topo}_{shape}.json"
                        if ion
                        else OUT
                        / "fits"
                        / f"{split}_{topo}_{shape}_nominal_neutral_group.json"
                    )
                    run = min(
                        (
                            r
                            for r in json.loads(path.read_text())["runs"]
                            if r["status"] > 0
                        ),
                        key=lambda r: r["cost"],
                    )
                    reference = np.array(
                        run["neutral_predictions" if ion else "prediction"]
                    )
                    keep = WINDOW & SPLITS[split] & (Q == 0)
                    errors.append(abs(np.log(reference) - pred[keep]).max())
                    delta = np.random.default_rng(1995).normal(size=len(p))
                    delta /= np.linalg.norm(delta)
                    h = 0.00001
                    diff = (
                        law(p + h * delta, data, topo, shape, ion)[0]
                        - law(p - h * delta, data, topo, shape, ion)[0]
                    ) / (2 * h)
                    grad.append(
                        np.linalg.norm(diff - j @ delta) / np.linalg.norm(diff)
                    )
    assert max(errors) < 1e-9 and max(grad) < 1e-6
    return {
        "max_abs_difference_in_nominal_log_J_vs_archived_AD": max(errors),
        "max_relative_directional_derivative_error": max(grad),
    }


def main():
    d = pd.read_csv(HERE / "draws.csv")
    s = pd.concat(
        [
            pd.read_csv(HERE / f"solutions{suffix}.csv")
            for suffix in ("", "-ions", "-extension")
        ]
    )
    f = pd.concat(
        [
            pd.read_csv(HERE / f"formation{suffix}.csv")
            for suffix in ("", "-ions", "-extension")
        ]
    )
    assert (
        len(s) == 4096
        and (s.status > 0).all()
        and not s.duplicated(
            ["id", "split", "topology", "sample", "shape"]
        ).any()
    )
    assert np.isfinite(f.predicted_J).all() and (f.predicted_J > 0).all()
    assert np.all(
        f.predicted_J * f.ion_fraction <= Q[f.line.to_numpy() - 2] + 1e-8
    )
    diag = verify()
    diag.update(
        fits=len(s),
        converged=int((s.status > 0).sum()),
        extra_start_improves_cost_1pct=int(
            (s.cost < 0.99 * s.previous_cost).sum()
        ),
        max_relative_cost_improvement=float(
            (1 - s.cost / s.previous_cost).max()
        ),
    )
    (HERE / "formation-checks.json").write_text(
        json.dumps(diag, indent=2) + "\n"
    )
    f["mode"] = np.where(Q[f.line.to_numpy() - 2] == 0, "neutral", "ionized")
    f["absolute_log_error"] = abs(np.log(f.predicted_J / f.adjusted_J))
    score = (
        f.groupby(
            ["block", "id", "sample", "split", "shape", "mode", "topology"]
        )
        .absolute_log_error.median()
        .pipe(np.exp)
        .rename("factor")
        .reset_index()
    )
    score.to_csv(HERE / "holdout-scores.csv", index=False)
    g = ["sample", "split", "shape", "mode", "topology"]
    summary(score, g, "factor").to_csv(HERE / "holdout-bounds.csv", index=False)
    pair = score.pivot(
        index=["block", "id", "sample", "split", "shape", "mode"],
        columns="topology",
        values="factor",
    ).reset_index()
    wins = (
        pair.assign(full_smaller=pair.full < pair.assisted)
        .groupby(["sample", "split", "shape", "mode"])
        .full_smaller.mean()
    )
    wins.to_csv(HERE / "holdout-paired.csv")
    low = f.query(
        "sample=='neutral' and shape=='linear' and split.str.endswith('low')",
        engine="python",
    )
    summary(low, ["line", "topology"], "predicted_J").to_csv(
        HERE / "source-bounds.csv", index=False
    )
    subset = low.query("line==336 and topology=='full'")
    quant = subset.groupby("block").binary_fraction.quantile(QS).unstack()
    quant.to_csv(HERE / "fraction-seed-check.csv")
    cases = pd.read_csv(ROOT / "companion/dunne_kuerten_candidate_edges.csv")
    a = low.merge(cases, left_on="line", right_on="dunne_csv_line")
    a["in_native_range"] = a.predicted_J.between(
        a.kuerten_J_lower, a.kuerten_J_upper
    )
    a.groupby(
        ["line", "topology", "kuerten_fig4_id", "mutually_unique_condition"]
    ).in_native_range.mean().rename("fraction").to_csv(
        HERE / "counterpart-comparison.csv"
    )
    scan = pd.concat(
        [
            pd.read_csv(HERE / "scan.csv"),
            pd.read_csv(HERE / "scan-extension.csv"),
        ]
    )
    summary(scan, ["split", "shape", "topology", "scenario"], "ratio").to_csv(
        HERE / "scan-bounds.csv", index=False
    )
    summary(
        scan, ["split", "shape", "topology", "scenario", "block"], "ratio"
    ).to_csv(HERE / "scan-seed-check.csv", index=False)
    ranks = (
        low.merge(d, on=["block", "id"])
        .groupby(["line", "topology"])[
            [
                "predicted_J",
                "acid",
                "base0",
                "base1",
                "base2",
                "base3",
                "jscale",
            ]
        ]
        .corr(method="spearman")
    )
    ranks.xs("predicted_J", level=-1).drop(columns="predicted_J").to_csv(
        HERE / "source-rank-correlations.csv"
    )

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "figure.dpi": 155,
        }
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.6, 4.7),
        layout="constrained",
        gridspec_kw={"width_ratios": [1.1, 1]},
    )
    order = [366, 332, 334, 344, 336, 338]
    ax = axes[0]
    for idx, topo in enumerate(COLORS):
        bands = (
            low.query("topology==@topo")
            .groupby("line")
            .predicted_J.quantile(QS)
            .unstack()
            .loc[order]
            .to_numpy()
        )
        yy = np.arange(6) + (idx - 0.5) * 0.22
        ax.hlines(
            yy, bands[:, 0], bands[:, 2], color=COLORS[topo], linewidth=2.5
        )
        ax.scatter(
            bands[:, 1], yy, color=COLORS[topo], s=26, label=LABELS[topo]
        )
    ax.scatter(
        J[np.array(order) - 2],
        range(6),
        s=37,
        marker="x",
        c="black",
        label="CLOUD central rate",
        zorder=4,
    )
    ax.set(
        yticks=range(6),
        yticklabels=[cold_condition_label(i) for i in order],
        xscale="log",
        xlabel="Formation rate at 1.7 nm (cm⁻³ s⁻¹)",
        title="Rates on withheld conditions",
        ylabel="Temperature · H₂SO₄ (10⁶ molecules cm⁻³)",
    )
    ax.invert_yaxis()
    ax.legend(fontsize=8, loc="lower left")
    ax = axes[1]
    order2 = [
        "223_low",
        "248_low",
        "223_temperature",
        "248_temperature",
        "278_temperature",
    ]
    for idx, topo in enumerate(COLORS):
        chunk = score.query(
            'topology==@topo and sample=="neutral" and shape=="linear"'
        )
        vals = (
            chunk.groupby("split")
            .factor.quantile(QS)
            .unstack()
            .loc[order2]
            .to_numpy()
        )
        yy = np.arange(5) + (idx - 0.5) * 0.22
        ax.hlines(yy, vals[:, 0], vals[:, 2], color=COLORS[topo], linewidth=2.5)
        ax.scatter(vals[:, 1], yy, s=26, c=COLORS[topo])
    ax.set(
        yticks=range(5),
        yticklabels=[
            "223 K low NH₃",
            "248 K low NH₃",
            "223 K all",
            "248 K all",
            "278 K all",
        ],
        xscale="log",
        xlabel="Median error factor within holdout",
        title="Transfer to a new condition or temperature",
    )
    ax.invert_yaxis()
    for ax in axes:
        ax.grid(axis="x", alpha=0.18)
    fig.savefig(HERE / "formation-bounds.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.1, 4), layout="constrained")
    ax = axes[0]
    for k, (name, g) in enumerate(
        [
            ("Full fits", scan.query('split=="complete"')),
            ("Held-out fits", scan.query('split=="223_low"')),
        ]
    ):
        for n, sc in enumerate(("new_common", "new_differential")):
            for j, top in enumerate(COLORS):
                vals = (
                    g.query(
                        'scenario==@sc and topology==@top and shape=="linear"'
                    )
                    .ratio.quantile(QS)
                    .to_numpy()
                )
                y = 2 * k + n + 0.22 * (j - 0.5)
                ax.hlines(y, vals[0], vals[2], color=COLORS[top], linewidth=2.4)
                ax.scatter(
                    vals[1],
                    y,
                    c=COLORS[top],
                    s=27,
                    label=LABELS[top] if k + n == 0 else None,
                )
    ax.axvline(15.81 / 0.32, c="black", ls=":", label="Archived ratio (49.4)")
    ax.set(
        xscale="log",
        yticks=range(4),
        yticklabels=[
            "Complete, shared",
            "Complete, separate",
            "Held out, shared",
            "Held out, separate",
        ],
        xlabel="Rate at 13.05 / rate at 6.64\nH₂SO₄ in 10⁶ molecules cm⁻³",
        title="223 K repeat at 0.06 pptv NH₃",
    )
    ax.set_ylim(4.4, -0.55)
    ax.legend(fontsize=8, loc="lower left")
    ax = axes[1]
    keys = ["figure45", "all", "primary", "neutral", "ionized"]
    archive = pd.read_csv(HERE / "archive-scores.csv").query(
        'law=="S1_precise"'
    )
    nominal = json.loads((OUT / "reference_metrics.json").read_text())
    vals = (
        archive.groupby("subset")
        .log_r2.quantile(QS)
        .unstack()
        .loc[keys]
        .to_numpy()
    )
    ax.hlines(range(5), vals[:, 0], vals[:, 2], color="#41685c", lw=3)
    ax.scatter(
        vals[:, 1], range(5), color="#41685c", label="Joint calibrations"
    )
    ax.scatter(
        [0.8953]
        + [
            nominal["S1_precise_" + s]["log_r2"]
            for s in ["all", "rh", "neutral", "ionized"]
        ],
        range(5),
        marker="x",
        color="black",
        label="Nominal calibration",
    )
    ax.set(
        xlabel="R² on log₁₀ formation rates",
        yticks=range(5),
        yticklabels=[
            "Fig. S3 selection",
            "All 377",
            "33–43% RH",
            "Neutral, RH window",
            "Ionized, RH window",
        ],
        title="Published law with fixed coefficients",
    )
    ax.set_ylim(5.45, -0.55)
    ax.legend(fontsize=8, loc="lower left")
    for ax in axes:
        ax.grid(axis="x", alpha=0.18)
    fig.savefig(HERE / "design-and-reference.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
