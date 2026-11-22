# %%
"""Score individual runs and draw the joint reproduction figures."""

import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from engine import NOX, ORDER, RESULT, TIMES, Engine, observations
from optimize import FAMILIES, PHYSICAL_NAMES, definition

LABELS = {
    "single": "One product",
    "expanded": "Two products + aging",
    "competition": "Two products + competition",
    "single_doses": "One product, fitted doses",
    "expanded_doses": "Two products, fitted doses",
    "competition_doses": "Competition, fitted doses",
    "expanded_counts": "Two products, measured counts",
    "competition_counts": "Competition, measured counts",
    "single_nox": "One product, NOₓ branch",
    "single_nox_five": "Five runs, one NOₓ-dependent product branch",
    "expanded_nox": "Two products, NOₓ branch",
    "competition_nox_counts": "Competition, NOₓ branch + counts",
    "five_doses": "Five runs, fitted doses",
}


def score_curve(y, run):
    """Compare original AMS samples and separately report peak error."""
    t, obs = observations(run)
    keep = (t >= 0) & (t <= 18000)
    t = t[keep]
    obs = obs[keep]
    pred = np.interp(t, TIMES, y)
    peak = obs.max()
    m = y.max()
    rmse = np.sqrt(np.mean((pred - obs) ** 2))
    nrmse = rmse / peak
    err = m / peak - 1
    return dict(
        rmse=rmse,
        nrmse=nrmse,
        peak=m,
        obs_peak=peak,
        peak_rel_error=err,
        peak_time_h=TIMES[y.argmax()] / 3600,
        obs_peak_time_h=t[obs.argmax()] / 3600,
        bias=np.mean(pred - obs),
        pass_example=bool(nrmse <= 0.1 and abs(err) <= 0.15),
    )


if __name__ == "__main__":
    engine = Engine(dt=3)
    fine = Engine(dt=2)
    coarse = Engine(dt=6)
    rows = []
    hist = []
    native = []
    paths = {}
    for family in FAMILIES:
        fit = json.loads((RESULT / (family + ".json")).read_text())
        p = fit["physical"]
        count = definition(family)[1]
        h = engine(p, count)[0]
        hf = fine(p, count)[0]
        hc = coarse(p, count)[0]
        paths[family] = h
        for i, r in enumerate(ORDER):
            rows.append(
                dict(
                    family=family,
                    run=r,
                    in_objective=(i < definition(family)[2]),
                    dose=p[9 + i],
                    refine_6_to_3_abs=np.max(abs(hc[i] - h[i])),
                    refine_3_to_2_abs=np.max(abs(hf[i] - h[i])),
                    **score_curve(h[i], r),
                )
            )
            for sec, oa in zip(TIMES, h[i], strict=True):
                hist.append(dict(family=family, run=r, time_s=sec, soa=oa))
        for file in sorted(
            file
            for device in ("cuda0", "numpy", "cpu")
            for file in RESULT.glob(f"{family}_{device}_*s_18000_split*.csv")
        ):
            meta = json.loads(file.with_suffix(".json").read_text())
            expected = hashlib.sha256(
                np.array(p, dtype="<f8").tobytes()
            ).hexdigest()
            if meta.get("fit_hash") != expected:
                raise ValueError(f"Native replay is stale: {file.name}")
            df = pd.read_csv(file)
            for i, r in enumerate(ORDER):
                target = df[df.run == r].soa.to_numpy()
                native.append(
                    dict(
                        family=family,
                        native_file=file.name,
                        run=r,
                        max_abs_vs_search3=np.max(abs(h[i] - target)),
                        **score_curve(target, r),
                    )
                )
    values = []
    for family in FAMILIES:
        fit = json.loads((RESULT / (family + ".json")).read_text())
        p = fit["physical"]
        odds = (
            p[0]
            / (1 - p[0])
            * (np.array([NOX[r] for r in ORDER]) / 30.0) ** p[16]
        )
        values.append(
            dict(
                family=family,
                **dict(zip(PHYSICAL_NAMES, p, strict=True)),
                **{
                    f"branch_run{r}": q
                    for r, q in zip(ORDER, odds / (1 + odds), strict=True)
                },
            )
        )
    pd.DataFrame(values).to_csv(RESULT / "physical_parameters.csv", index=False)
    # Re-evaluate proxy sensitivity without refitting when run-specific NOx is
    # ambiguous.
    tests = []
    for label, over in (
        ("Table_1", {}),
        ("alternative_run4_15", {4: 15.0}),
        ("alternative_run6_7.8", {6: 7.8}),
    ):
        e = Engine(dt=3, nox=NOX | over)
        for f in ("single_nox", "expanded_nox", "competition_nox_counts"):
            y = e(
                json.loads((RESULT / (f + ".json")).read_text())["physical"],
                definition(f)[1],
            )[0]
            for i, r in enumerate(ORDER[:4]):
                tests.append(
                    dict(
                        family=f,
                        NOx_scenario=label,
                        run=r,
                        **score_curve(y[i], r),
                    )
                )
    pd.DataFrame(tests).to_csv(
        RESULT / "NOx_metadata_sensitivity.csv", index=False
    )
    extra = []
    for hazard in (0.8, 1.0, 1.25):
        e = Engine(dt=3, proxy_hazard=hazard)
        for family in ("five_doses", "single_nox_five"):
            fit = json.loads((RESULT / (family + ".json")).read_text())
            y = e(fit["physical"])[0, -1]
            extra.append(
                dict(
                    family=family,
                    run=2,
                    hazard_multiplier=hazard,
                    **score_curve(y, 2),
                )
            )
    pd.DataFrame(extra).to_csv(
        RESULT / "missing_forcing_sensitivity.csv", index=False
    )
    pd.DataFrame(rows).to_csv(RESULT / "scores.csv", index=False)
    pd.DataFrame(hist).to_csv(RESULT / "trajectories.csv", index=False)
    pd.DataFrame(native).to_csv(RESULT / "native_checks.csv", index=False)
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlelocation": "left",
            "font.size": 10,
            "legend.frameon": False,
        }
    )
    fig, axs = plt.subplots(2, 2, figsize=(9.5, 6.1), layout="constrained")
    for i, (ax, r) in enumerate(zip(axs.flat, ORDER[:4], strict=True)):
        t, y = observations(r)
        use = t <= 18000
        ax.scatter(
            t[use] / 3600, y[use], s=10, c="0.28", alpha=0.7, label="AMS"
        )
        for fam, color, ls, label in [
            ("expanded", "#bf613b", "--", "Shared chemistry, nominal dose"),
            ("expanded_doses", "#11677a", "-", "Shared chemistry, fitted dose"),
            ("single_nox", "#67609d", ":", "NOₓ-linked branch, nominal dose"),
        ]:
            ax.plot(
                TIMES / 3600,
                paths[fam][i],
                color=color,
                ls=ls,
                lw=2,
                label=label,
            )
        loading = {1: 309, 4: 155, 5: 155, 6: 103}[r]
        ax.set(
            title=f"Run {r} · {loading} ppb α-pinene",
            ylabel="Suspended SOA (µg m⁻³)",
            xlabel="Hours after lights on",
        )
        ax.grid(axis="y", alpha=0.14)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2)
    fig.savefig(RESULT / "joint-trajectories.png", dpi=190)
    plt.close(fig)
