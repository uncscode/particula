"""Summarize shared parameter estimates on common observation masks."""

import json
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from global_fit import (
    torch,
    ROOT,
    T,
    PAR,
    HOURS,
    WINDS,
    NAMES,
    write,
    data,
    alternate_data,
    predict,
    masks,
    residual,
    scores,
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def med(y, d):
    return np.exp(np.interp(0.5, np.cumsum(y) / y.sum(), np.log(d)))


def encode(params, names):
    vals = []
    for n in names:
        v = params[n]
        lo, hi, log, _ = PAR[n]
        vals.append(
            np.log(v / lo) / np.log(hi / lo) if log else (v - lo) / (hi - lo)
        )
    return T(vals)


def summarize():
    z = data(5)
    finer = data(1)
    batches = [
        json.loads((ROOT / f"global-{part}.json").read_text())
        for part in [
            "core",
            "stress",
            "extra",
            "combined",
            "measurement",
            "charge",
        ]
    ]
    allrows = [r for b in batches for r in b["results"]]
    stats = []
    points = []
    thresholds = []
    params = []
    validations = []
    pooled = {}
    d = z["d"].numpy()
    for r in allrows:
        key, fold, model = r["case"], r["fold"], r["model"]
        zz = alternate_data(z, r["settings"])
        _, test = masks(model, fold)
        y = np.array(r["prediction"])
        use = np.ones((4, 5), bool) if fold == "all" else test.copy()
        use[:, 0] = False
        out = scores(y, zz, use)
        stats.append(dict(case=key, fold=fold, objective=r["objective"], **out))
        pred = y @ zz["cutmap"].numpy().T
        obs = zz["mass"].numpy()
        if fold.startswith("wind"):
            if key not in pooled:
                pooled[key] = []
            pooled[key].append(out)
        for n, v in r["params"].items():
            params.append(dict(case=key, fold=fold, parameter=n, value=v))
        for w, t in zip(*np.where(use), strict=True):
            observed_shape = zz["shape"][w, 1].numpy()
            points.append(
                dict(
                    case=key,
                    fold=fold,
                    wind=WINDS[w],
                    hours=HOURS[t],
                    total_pred_mg=pred[w, t, 5],
                    total_obs_mg=obs[w, t, 5],
                    middle_pred_mg=pred[w, t, 2] - pred[w, t, 1],
                    middle_obs_mg=obs[w, t, 2] - obs[w, t, 1],
                    d50_pred_um=med(y[w, t], d),
                    d50_obs_um=med(observed_shape, d) if t == 4 else "",
                )
            )
            if fold == "all" or fold.startswith("wind"):
                for i, cut in enumerate([2.5, 10, 20, 30, 40, 63]):
                    thresholds.append(
                        dict(
                            case=key,
                            fold=fold,
                            wind=WINDS[w],
                            hours=HOURS[t],
                            cut_um=cut,
                            observed_mg=obs[w, t, i],
                            predicted_mg=pred[w, t, i],
                            sem_mg=float(zz["sem"][w, t, i]),
                        )
                    )
        if (
            fold == "all"
            and model in ("conditional", "blind", "blind_mixture", "hybrid")
            and key == model
        ):
            names = NAMES[model]
            x = encode(r["params"], names)
            y1 = predict(x, names, model, finer, r["settings"]).numpy()
            replay = predict(x, names, model, z, r["settings"]).numpy()
            out = dict(
                case=key,
                derivative_relative_error=r["sensitivity"][
                    "gradient_relative_error"
                ],
                frozen_replay_max_mg=float(np.max(abs(y - replay))),
                refinement_max_total_mg=float(np.max(abs((y1 - y).sum(-1)))),
                refinement_max_d50_um=float(
                    max(
                        abs(med(a, d) - med(b, d))
                        for a, b in zip(y1[:, 4], y[:, 4], strict=True)
                    )
                ),
            )
            if (
                out["frozen_replay_max_mg"] > 1e-8
                or out["derivative_relative_error"] > 1e-5
            ):
                raise ValueError(out)
            validations.append(out)
        if (
            fold == "wind2"
            and key == model
            and model in ("blind", "conditional", "blind_mixture", "hybrid")
        ):
            # Poison all withheld target cells. Preserve the conditional 30-min input.
            bad = dict(z)
            bad["shape"] = z["shape"].clone()
            bad["mass"] = z["mass"].clone()
            si = 1 if model in ("conditional", "hybrid") else 0
            bad["shape"][1, si:] = bad["shape"][1, si:].flip([-1])
            bad["mass"][1, si:] *= 100
            tr, _ = masks(model, fold)
            x = encode(r["params"], NAMES[model])
            ya = predict(x, NAMES[model], model, z, {})
            yb = predict(x, NAMES[model], model, bad, {})
            if not torch.equal(ya, yb) or not torch.equal(
                residual(ya, z, tr, {}), residual(yb, bad, tr, {})
            ):
                raise ValueError("Withheld observations altered training")
    for key, rr in pooled.items():
        if len(rr) != 4:
            raise ValueError(key)
        stats.append(
            dict(
                case=key,
                fold="wind_average",
                objective="",
                cells=sum(v["cells"] for v in rr),
                **{
                    k: np.mean([v[k] for v in rr])
                    for k in rr[0]
                    if k != "cells"
                },
            )
        )
    # Independent flux-quadrature verification of the smoothed rebound law.
    maxerr = 0
    from scipy.special import ndtr

    for v, s, c in [
        (0.008, 0.04, 0.05),
        (0.03, 0.12, 0.08),
        (0.08, 0.1, 0.02),
        (0.01, 0.001, 0.07),
    ]:
        s2 = s * s + c * c
        mu = v * c * c / s2
        sd = s * c / np.sqrt(s2)
        calc = (
            c
            / np.sqrt(s2)
            * np.exp(-0.5 * v * v / s2)
            * (mu * ndtr(mu / sd) + sd * norm.pdf(mu / sd))
        )
        q = quad(
            lambda w: w
            * np.exp(-0.5 * (w / c) ** 2)
            * norm.pdf((w - v) / s)
            / s,
            0,
            v + 14 * s,
            points=[v],
            epsabs=1e-14,
        )[0]
        maxerr = max(maxerr, abs(calc - q))
    if maxerr > 1e-10:
        raise ValueError(maxerr)
    write("global-scores.csv", stats)
    write("global-comparisons.csv", points)
    write("global-thresholds.csv", thresholds)
    write("global-parameters.csv", params)
    units = {
        "local_ratio": "ratio",
        "normal_sd_over_ustar": "ratio",
        "capture10_ms": "m/s",
        "adhesion_time_min": "min",
        "delivery_lag_min": "min",
        "removal6_per_h": "1/h",
        "air_median_um": "µm",
        "coarse_median_um": "µm",
        "charge_uC_m2": "µC/m²",
    }
    best = []
    for row in params:
        if row["fold"] == "all" and row["case"] in {*NAMES, "charge_inferred"}:
            name = row["parameter"]
            low, high = PAR[name][:2]
            best.append(
                dict(
                    case=row["case"],
                    parameter=name,
                    value=row["value"],
                    unit=units.get(name, "dimensionless"),
                    low=low,
                    high=high,
                    at_bound=bool(np.isclose(row["value"], [low, high]).any()),
                )
            )
    write("global-shared-sets.csv", best)
    meta = dict(
        fits=len(allrows),
        local_starts=sum(
            len(r["starts"]) for r in allrows if NAMES[r["model"]]
        ),
        fixed_evaluations=sum(
            len(r["starts"]) for r in allrows if not NAMES[r["model"]]
        ),
        box_points=sum(r["screen_size"] for r in allrows),
        numerical=validations,
        integral_abs_error_ms=maxerr,
        holdout_contamination_check="passed",
        seconds_by_batch=[b["seconds"] for b in batches],
    )
    (ROOT / "global-checks.json").write_text(json.dumps(meta, indent=2))
    return points, stats


def plot(points):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), layout="constrained")
    for ax, k, ko, title, yl in zip(
        axes,
        ["d50_pred_um", "total_pred_mg"],
        ["d50_obs_um", "total_obs_mg"],
        ["Three-hour median diameter", "Three-hour retained mass"],
        ["Volume median diameter (µm)", "Mass below 63 µm (mg per tray)"],
        strict=True,
    ):
        for i, (model, lab, col, marker) in enumerate(
            [
                ("observed", "Measured", "#20262d", "o"),
                ("conditional", "Shared physics + early tray", "#3074a1", "s"),
                ("power", "Shared growth law + early tray", "#ad532a", "^"),
                ("hybrid", "Shared mass law + modeled size", "#378066", "D"),
                (
                    "blind_mixture",
                    "Two airborne modes, no early tray",
                    "#886293",
                    "v",
                ),
            ]
        ):
            rows = sorted(
                [
                    r
                    for r in points
                    if r["case"] == ("conditional" if i == 0 else model)
                    and r["fold"].startswith("wind")
                    and r["hours"] == 3
                ],
                key=lambda r: r["wind"],
            )
            ax.scatter(
                np.array(WINDS) + (i - 1.5) * 0.13,
                [r[ko if i == 0 else k] for r in rows],
                label=lab,
                color=col,
                marker=marker,
                s=40,
                zorder=3,
            )
        ax.set(
            xticks=WINDS,
            xlabel="Omitted wind speed (m/s)",
            ylabel=yl,
            title=title,
        )
        ax.spines[["right", "top"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(
        "Predictions after refitting on the other three winds",
        fontweight="bold",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside lower center", ncol=2, frameon=False
    )
    fig.savefig(ROOT / "global-holdouts.png", dpi=175)
    plt.close(fig)


if __name__ == "__main__":
    points, stats = summarize()
    plot(points)
