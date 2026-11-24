"""Isolate optical sizing spread, systematic error, and one-at-a-time retrieval changes."""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from instruments import WATER, geometry, profiles, pulse_lookup, fssp, sobol
from simulate import device_areas, corrected, response
from study import ROOT as RUN_DIR, edges

ROOT = Path(__file__).resolve().parent
OPTICS = RUN_DIR / "optics"


def run(out):
    spectra = pd.read_csv(OPTICS / "input-spectra.csv.gz").query(
        'source == "CDA"'
    )
    beads = pd.read_csv(OPTICS / "refined-beads.csv")
    slope = beads.query('case=="nominal" and nm==660').factor_slope_per_um.iloc[
        0
    ]
    previous = pd.read_csv(OPTICS / "attribution.csv").query(
        'case=="nominal" and stage=="refined" and calibration=="frozen" and source=="CDA" and low_um==6 and high_um==13'
    )
    grid = np.sort(spectra.true_um.unique())
    inside = (grid >= 6) & (grid < 13)
    qs = sobol(512)
    z, p, ix = geometry(20, qs)
    radial = np.arange(0, 52, 0.04)
    cache = {nm: pulse_lookup(nm, 80) for nm in (633, 660)}
    all_sizes, budgets = [], []
    for nm in (633, 660):
        field = profiles(grid, nm, WATER, z, radial, 80, "mie")
        cases = {}
        for th in [0.28, 0.40, 0.52]:
            image = device_areas(field, grid, p, qs, ix, 0.04, th, 3)
            cases["I", f"threshold_{th:.2f}" if th != 0.40 else "nominal"] = (
                corrected(image, slope),
                image > 0,
            )
            if th == 0.40:
                cases["I", "uncorrected"] = (image, image > 0)
        for gain in [0.06, 0.12, 0.20]:
            pulse = fssp(grid, nm, 80, qs, 633, gain, cache)
            cases["F", f"pulse_{gain:.2f}" if gain != 0.12 else "nominal"] = (
                pulse,
                pulse > 0,
            )
        for (pr, setting), (raw, trigger) in cases.items():
            band = response(raw, pr, grid, trigger)
            e = edges(pr, "upper")
            valid = (raw >= e[0]) & (raw < e[-1])
            # Restrict the size-scatter calculation to true 6–13 µm drops with a
            # registered diameter. Report omitted fractions alongside these moments.
            take = raw[inside]
            mask = valid[inside]
            v = np.where(mask, take, np.nan)
            mean = np.nanmean(v, axis=1)
            sd = np.nanstd(v, axis=1)
            bias = mean - grid[inside]
            q16, q84 = np.nanquantile(v, [0.16, 0.84], axis=1)
            if setting == "nominal":
                all_sizes.append(
                    pd.DataFrame(
                        dict(
                            probe=pr,
                            nm=nm,
                            true_um=grid[inside],
                            mean_um=mean,
                            sd_um=sd,
                            q16_um=q16,
                            q84_um=q84,
                            registered=mask.mean(axis=1),
                        )
                    )
                )
            for day, daily in spectra.groupby("day"):
                if day == 14:
                    continue
                pop = daily.sort_values("true_um").number_cm3.to_numpy()
                weights = pop[inside] * mask.mean(axis=1)

                def avg(x):
                    return np.average(x, weights=weights)

                var = avg(sd**2)
                b2 = avg(bias**2)
                direct = np.nansum(
                    np.where(mask, (take - grid[inside, None]) ** 2, 0)
                    * pop[inside, None]
                ) / (512 * weights.sum())
                np.testing.assert_allclose(direct, var + b2)
                count = pop @ band.band13
                if setting == "nominal":
                    np.testing.assert_allclose(
                        count,
                        previous.loc[
                            previous.day == str(day), pr.lower() + str(nm)
                        ].iloc[0],
                        atol=1e-7,
                    )
                budgets.append(
                    dict(
                        day=day,
                        probe=pr,
                        nm=nm,
                        setting=setting,
                        reported_band_cm3=count,
                        true_band_cm3=pop[inside].sum(),
                        triggered_fraction=np.average(
                            trigger[inside].mean(axis=1), weights=pop[inside]
                        ),
                        registered_fraction=weights.sum() / pop[inside].sum(),
                        mean_signed_error_um=avg(bias),
                        within_size_scatter_um=np.sqrt(var),
                        size_dependent_bias_rms_um=np.sqrt(b2),
                        total_sizing_rmse_um=np.sqrt(direct),
                    )
                )
    data = pd.concat(all_sizes, ignore_index=True)
    data.to_csv(out / "method-sizing.csv", index=False)
    b = pd.DataFrame(budgets)
    b.to_csv(out / "method-budget.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.8), sharex=True, sharey=True)
    fig.subplots_adjust(
        left=0.095, right=0.975, top=0.73, bottom=0.20, wspace=0.11
    )
    fig.text(
        0.095,
        0.92,
        "How precisely does each probe size the same droplet?",
        fontsize=17,
        weight="bold",
        color="#193340",
    )
    fig.text(
        0.095,
        0.842,
        "Both probes at 633 nm  •  512 simulated observations per true diameter  •  Native calibrations held fixed",
        fontsize=9.5,
    )
    for ax, pr, label, col in zip(
        axes,
        ["F", "I"],
        ["FSSP  ·  forward scattering", "ICEMET  ·  holographic imaging"],
        ["#00698b", "#af4925"],
    ):
        g = data.query("probe==@pr and nm==633")
        ax.fill_between(g.true_um, g.q16_um, g.q84_um, color=col, alpha=0.23)
        ax.plot(g.true_um, g.mean_um, color=col, lw=1.8)
        ax.plot([6, 13], [6, 13], ls="--", lw=1, color="#455663")
        ax.set_title(label, loc="left", fontsize=11, weight="bold")
        ax.set_xlabel("True droplet diameter (µm)")
        ax.set_xlim(6, 13)
        ax.set_xticks([6, 8, 10, 12, 13])
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Reported diameter (µm)")
    fig.text(
        0.095,
        0.048,
        "Solid line: mean registered size. Shading: 16th–84th percentiles of individual droplets. Dashed line: true size.",
        fontsize=9,
    )
    fig.savefig(out / "method-only.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    run(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT)
