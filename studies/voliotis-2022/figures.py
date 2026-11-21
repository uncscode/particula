# %% [markdown]
# Rebuild the empirical comparison and figures from saved data.
# %%
"""Plot the Manchester chamber observations and saved simulation outputs."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from chamber import AP, PRESSURE, RUNS, R
from study import FORCING, OUT, metrics, series

scores = pd.read_csv(OUT / "comparisons.csv")
traces = pd.read_csv(OUT / "trajectories.csv")
envelope = pd.read_csv(OUT / "run5_fit_envelope.csv")
base = scores[scores.test == "reference"].set_index("run")
figdir = OUT / "figures"
figdir.mkdir(exist_ok=True)
colors = {4: "#15667c", 5: "#af4d24", 1: "#576b4d", 6: "#795c8e"}
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlelocation": "left",
        "legend.frameon": False,
        "figure.dpi": 150,
    }
)
rows = []
for run in (4, 5, 1, 6):
    t, y = series(run, "soa_ug_m3")
    m = t <= 18000
    t, y = t[m], y[m]
    ppb, _, _, temp = RUNS[run]
    initial = ppb * PRESSURE * AP / (R * (temp + 273.15))
    reacted = initial * (1 - np.interp(t, *FORCING[run]))
    if run == 4:
        fitted_yield = np.dot(reacted, y) / np.dot(reacted, reacted)
    i = y.argmax()
    rows.append(
        dict(
            run=run,
            observed_peak=float(y[i]),
            peak_hour=t[i] / 3600,
            apparent_reacted_at_peak=reacted[i],
            apparent_suspended_yield_at_peak=y[i] / reacted[i],
            fitted_constant_yield=fitted_yield,
            **metrics(t, fitted_yield * reacted, run),
        )
    )
pd.DataFrame(rows).to_csv(OUT / "empirical_yield.csv", index=False)

fig, ax = plt.subplots(1, 3, figsize=(12, 3.85), layout="constrained")
for run in (4, 5):
    c = colors[run]
    t, y = series(run, "apinene_fraction")
    ax[0].plot(t / 3600, y, color=c, ls="none", marker=".", alpha=0.35, ms=2)
    k, f = FORCING[run]
    ax[0].plot(k / 3600, f, color=c, label=f"Run {run}", lw=2)
ax[0].set(
    title="a   α-Pinene remaining",
    ylabel="Fraction of initial α-pinene",
    xlabel="Hours after lights on",
    xlim=(0, 5.05),
    ylim=(0.25, 1.05),
)
ax[0].legend()
for a, run in zip(ax[1:], (4, 5), strict=True):
    c = colors[run]
    t, y = series(run, "soa_ug_m3")
    a.axvspan(5, 6, color="#eceeed", zorder=-4)
    a.scatter(t / 3600, y, s=10, color="#33373c", alpha=0.7, label="Measured")
    g = traces[(traces.run == run) & (traces.test == "reference")]
    a.plot(g.time_s / 3600, g.soa, color=c, lw=2.3, label="Particula CUDA")
    if run == 5:
        g = traces[(traces.run == run) & (traces.test == "same_seed_number")]
        a.plot(
            g.time_s / 3600,
            g.soa,
            color=c,
            ls="--",
            lw=1.5,
            label="Run 4 seed number",
        )
    a.set(
        title=(
            "b   Run 4 · calibration" if run == 4 else "c   Run 5 · held out"
        ),
        xlabel="Hours after lights on",
        ylabel="Suspended SOA (µg m⁻³)",
        xlim=(-0.05, 6.05),
        ylim=(0, 125),
    )
    a.legend(fontsize=8, loc="lower right")
for a in ax:
    a.grid(axis="y", alpha=0.15)
fig.savefig(figdir / "chamber-transfer.png", dpi=210)
plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(12, 4.1), layout="constrained")
a = ax[0]
for cstar, g in envelope.groupby("cstar"):
    a.scatter(g.run4_rmse, g.model_peak, s=18, label=f"{cstar:g}", alpha=0.65)
a.axhline(base.loc[5, "obs_peak"], c="#af4d24", ls="--", label="Measured peak")
a.set(
    title="a   Fits selected on run 4",
    xlabel="Run 4 RMSE (µg m⁻³)",
    ylabel="Predicted run 5 peak (µg m⁻³)",
)
a.legend(title="C* (µg m⁻³)", fontsize=8, title_fontsize=8)
a = ax[1]
tests = {
    "reference": "Reference",
    "same_gas": "Run 4 gas decay",
    "same_seed_number": "Run 4 seed number",
    "same_number_and_gas": "Both run 4 inputs",
    "counter_ratio_run1": "Counter sensitivity",
    "lower_seed_absorption": "10% absorbing seed",
}
g = scores[(scores.run == 5)].set_index("test")
for i, key in enumerate(tests):
    a.scatter(g.loc[key, "model_peak"], i, c=colors[5], s=35)
a.axvline(base.loc[5, "obs_peak"], c="#af4d24", ls="--")
a.set_yticks(range(len(tests)), tests.values(), fontsize=8)
a.invert_yaxis()
a.set(title="b   Run 5 input controls", xlabel="Predicted peak (µg m⁻³)")
a = ax[2]
x = np.array([1, 4, 5, 6])
for name, marker, color in (
    ("obs_peak", "o", "#30353a"),
    ("model_peak", "s", "#15667c"),
):
    a.scatter(
        range(4),
        base.loc[x, name],
        marker=marker,
        c=color,
        s=45,
        label="Measured" if name == "obs_peak" else "Particula CUDA",
    )
a.set_xticks(range(4), ["1\n309", "4\n155", "5\n155", "6\n103"])
a.set(
    title="c   Transfer across loadings",
    xlabel="Run / initial α-pinene (ppb)",
    ylabel="Peak suspended SOA (µg m⁻³)",
)
a.legend(fontsize=8)
for a in ax:
    a.grid(axis="y", alpha=0.15)
fig.savefig(figdir / "mechanism-tests.png", dpi=210)
plt.close(fig)

# Number concentration provides an independent check on the removal rate
# inferred from fitting organic mass. For the conditioned model it is an input.
nseries = pd.read_csv(OUT / "number_check.csv")
conditioned = pd.read_csv(OUT / "number_trajectories.csv")
fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.8), layout="constrained")
for run in (4, 5):
    g = nseries[nseries.run == run]
    axs[0].plot(
        g.time_s / 3600,
        g.observed_interpolated,
        color=colors[run],
        lw=1.6,
        label=f"Run {run} counter",
    )
    g = conditioned[(conditioned.run == run) & (conditioned.test == "best")]
    axs[1].plot(
        g.time_s / 3600,
        g.soa,
        color=colors[run],
        lw=1.8,
        label=f"Run {run} model",
    )
    t, y = series(run, "soa_ug_m3")
    mask = t <= 18000
    axs[1].scatter(
        t[mask] / 3600,
        y[mask],
        color=colors[run],
        alpha=0.55,
        s=9,
        label="AMS measurements" if run == 4 else None,
    )
g = nseries[nseries.run == 4]
axs[0].plot(
    g.time_s / 3600, g.mass_only, color="0.25", ls="--", label="Mass-only fit"
)
axs[0].set(
    title="a   Number surviving in the chamber",
    xlabel="Hours after lights on",
    ylabel="Fraction of initial number",
)
axs[1].set(
    title="b   Fit with measured number histories",
    xlabel="Hours after lights on",
    ylabel="Suspended SOA (µg m⁻³)",
)
for ax in axs:
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.18)
fig.savefig(figdir / "number-test.png", dpi=210)
plt.close(fig)
