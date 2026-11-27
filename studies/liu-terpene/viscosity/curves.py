"""Plot the independent central predictions and the measured-input sensitivity."""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "radial"))
sys.path.insert(0, str(HERE.parent / "model"))
from chamber import Scenario, predict
from radial import KB, T, inputs, run


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    weights, observed_t, measured = inputs(HERE.parent)
    ts = np.unique(np.r_[np.linspace(0, 1210, 160), observed_t, 15, 200, 1200])
    curves = []
    for law, xi in [("Classical", 1.0), ("Fractional", 0.93)]:
        for case, eta, a in [
            ("fast", 1.4e8, 0.3),
            ("central", 2.1e8, 0.4),
            ("slow", 2.8e8, 0.6),
        ]:
            D = KB * T / (6 * np.pi * a * 1e-9 * 1e-3) * (1e-3 / eta) ** xi
            tt, vfr, _ = run(weights, ts, D, 160)
            curves.append(
                pd.DataFrame(dict(law=law, case=case, time_min=tt, vfr=vfr))
            )
    baseline = predict(weights, "farnesene", "dry", 140, ts, Scenario(), n=5200)
    curves.append(
        pd.DataFrame(
            dict(law="Well mixed", case="central", time_min=ts, vfr=baseline)
        )
    )
    data = pd.concat(curves, ignore_index=True)
    data.to_csv(out / "radial_curves.csv", index=False, float_format="%.11g")
    colors = {"Classical": "#207768", "Fractional": "#AE5725"}
    fig, axes = plt.subplots(
        1, 2, figsize=(9.2, 4.35), layout="constrained", sharey=True
    )
    for ax, limit, title in zip(
        axes, [240, 1240], ["First four hours", "Through the overnight sample"]
    ):
        ax.axhline(1, color="#c5ccd0", lw=0.7)
        for law, color in colors.items():
            wide = data[data.law == law].pivot(
                index="time_min", columns="case", values="vfr"
            )
            ax.fill_between(
                wide.index,
                wide.fast,
                wide.slow,
                color=color,
                alpha=0.15,
                linewidth=0.0,
            )
            ax.plot(
                wide.index,
                wide.central,
                color=color,
                lw=2,
                label=law + " diffusion",
            )
        ax.plot(ts, baseline, "--", color="#677481", lw=1.4, label="Well mixed")
        ax.scatter(
            observed_t,
            measured,
            s=32,
            edgecolors="white",
            facecolors="#181e27",
            zorder=5,
            label="Liu observations",
        )
        ax.set(
            xlim=(0, limit),
            ylim=(0.48, 1.015),
            xlabel="Evaporation time (min)",
            title=title,
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#d8dfe4", lw=0.6)
    axes[0].set_ylabel("Fraction of initial dry volume remaining")
    axes[1].legend(loc="upper right", fontsize=8, frameon=False)
    fig.savefig(out / "viscosity_forecasts.png", dpi=165)
    plt.close(fig)


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--out", type=Path, default=HERE / "results")
    main(cli.parse_args().out)
