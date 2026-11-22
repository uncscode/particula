# %%
"""Replay the best matched mechanism fits and isolate the isoprene effect."""

import json

import numpy as np
import pandas as pd
from pathway_hybrid import HYBRID
from pathway_refine import MODES
from pathways import DOSES, OUT, Campaign, anchors, check


def switches(mode, p):
    """Return nested channel-removal tests for a fitted mechanism."""
    changes = {}
    if mode in (1, 2, 4):
        changes["no_alpha_product"] = ([10], [0.0])
    if mode in (2, 4, 6, 7):
        changes["no_competition"] = ([12], [0.0])
    if mode in (3, 4, 7):
        changes["no_uptake_vapor"] = ([13], [0.0])
    if mode == 5:
        changes["pooled_partners"] = ([16, 17, 18], p[6:9])
    return changes


def nested_candidates(records):
    """Retain exact submodel fits when a larger search misses the same point."""
    rows = list(records)
    for row in records:
        name = row["model"]
        p = np.array(row["physical"])
        if name == "reference":
            ref = p.copy()
            ref[[10, 12, 13]] = 0.0
            for target in (
                "prompt",
                "auto_competition",
                "reactive_uptake",
                "combined",
                "volatile_diversion",
                "diversion_uptake",
                "partner_yields",
            ):
                nested = (
                    np.r_[ref, ref[6:9]] if target == "partner_yields" else ref
                )
                rows.append(dict(row, model=target, physical=nested.tolist()))
        if name == "reactive_uptake":
            uptake = p.copy()
            uptake[12] = 0.0
            rows.append(
                dict(row, model="diversion_uptake", physical=uptake.tolist())
            )
        if name in ("specified_uptake", "volatile_diversion"):
            if name == "volatile_diversion":
                p[13] = 0.0
            rows.append(
                dict(row, model="diversion_uptake", physical=p.tolist())
            )
        if name in ("auto_competition", "reactive_uptake"):
            p[13 if name == "auto_competition" else 10] = 0.0
            rows.append(dict(row, model="combined", physical=p.tolist()))
    return rows + [
        dict(r, wall_free=True)
        for r in rows
        if r["objective"] == "joint" and not r["wall_free"]
    ]


def main():
    """Write per-run diagnostics, traces, time-step checks and ablations."""
    c = Campaign()
    base, levels, *_ = anchors()
    check(c, base)
    records = sum(
        [
            json.loads((OUT / f).read_text())
            for f in (
                "restarts.json",
                "partner_restarts.json",
                "refinement.json",
                "hybrid.json",
            )
        ],
        [],
    )
    frame = pd.DataFrame(nested_candidates(records))
    best = frame.loc[
        frame.groupby(["model", "objective", "wall_free"]).score.idxmin()
    ]
    metrics, summaries, curves, ablations = [], [], [], []
    perturbations = []
    for row in best.itertuples():
        model, obj, free = row.model, row.objective, row.wall_free
        mode = (MODES | HYBRID)[model][0]
        p = np.array(row.physical)
        score, coarse, _ = c.evaluate(p, True, obj, DOSES, sub=2, pathway=mode)
        np.testing.assert_allclose(score[0], row.score, atol=1e-10, rtol=0)
        _, fine, dose, diag = c.evaluate(
            p, True, obj, DOSES, sub=4, pathway=mode, traces=True
        )
        _, finer, _ = c.evaluate(p, True, obj, DOSES, sub=8, pathway=mode)
        err, d = fine[0], diag[0]
        trace = d[:, 6:606].reshape(len(c.runs), 300, 2)
        if np.any(d < -1e-7) or np.any(np.diff(trace[:, :, 1], axis=1) > 1e-10):
            raise ValueError("Nonphysical product or particle number")
        scalars = dict(
            model=model, objective=obj, wall_free=free, eddy_s_inv=p[9]
        )
        rin = [r[0] for r in c.runs]
        summaries.append(
            dict(
                **scalars,
                mass=err[:, 0].max(),
                count=err[:, 1].max(),
                joint=err.max(),
                median_mass=np.median(err[:, 0]),
                mass_run=rin[err[:, 0].argmax()],
                count_run=rin[err[:, 1].argmax()],
                refinement_change=np.abs(fine - finer).max(),
                coarse_change=np.abs(coarse - fine).max(),
                ledger_abs_ug_m3=d[:, 0].max(),
                peak24=d[rin.index(24), 1],
                within_mass_target=int((err[:, 0] <= 0.10).sum()),
                physical=json.dumps(p.tolist()),
            )
        )
        for j, run in enumerate(rin):
            metrics.append(
                dict(
                    **scalars,
                    run=run,
                    dose=dose[0, j],
                    mass_nrmse=err[j, 0],
                    count_nrmse=err[j, 1],
                    sampled_peak_ug_m3=d[j, 1],
                    observed_peak_ug_m3=c.runs[j][3].max(),
                    end_number_fraction=d[j, 2],
                    cumulative_prompt_or_auto_ug_m3=d[j, 3],
                    cumulative_uptake_ug_m3=d[j, 4],
                    cumulative_soa_produced_ug_m3=d[j, 606],
                    uptake_fraction_of_production=(
                        d[j, 4] / d[j, 606] if d[j, 606] > 1e-10 else np.nan
                    ),
                    alternative_sink_ug_m3=d[j, 5],
                )
            )
            # Export five-minute samples of each fitted trajectory.
            for k in range(4, 300, 5):
                curves.append(
                    dict(
                        **scalars,
                        run=run,
                        time_min=k + 1,
                        soa_ug_m3=trace[j, k, 0],
                        number_fraction=trace[j, k, 1],
                    )
                )
        for label, (indices, values) in switches(mode, p).items():
            altered = p.copy()
            altered[indices] = values
            _, errors, _, peaks = c.evaluate(
                altered, True, obj, DOSES, sub=4, pathway=mode, outcomes=True
            )
            perturbations.append(
                dict(
                    **scalars,
                    change=label,
                    mass=errors[0, :, 0].max(),
                    count=errors[0, :, 1].max(),
                    peak24=peaks[0, rin.index(24), 0],
                )
            )
        if model == "volatile_diversion":
            for name, parent_mode in (("alpha_only", 8), ("cresol_only", 9)):
                _, errors, _, peaks = c.evaluate(
                    p,
                    True,
                    obj,
                    DOSES,
                    sub=4,
                    pathway=parent_mode,
                    outcomes=True,
                )
                perturbations.append(
                    dict(
                        **scalars,
                        change=name,
                        mass=errors[0, :, 0].max(),
                        count=errors[0, :, 1].max(),
                        peak24=peaks[0, rin.index(24), 0],
                    )
                )
            for wall in sorted(set([p[9], base[9], levels[0], levels[-2]])):
                for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
                    test = p.copy()
                    test[9] = wall
                    test[12] *= fraction
                    _, error, _, result = c.evaluate(
                        test,
                        True,
                        obj,
                        DOSES,
                        sub=4,
                        pathway=mode,
                        outcomes=True,
                    )
                    ablations.append(
                        dict(
                            objective=obj,
                            wall_free=free,
                            eddy_s_inv=wall,
                            fitted_strength_fraction=fraction,
                            competition=test[12],
                            mass=error[0, :, 0].max(),
                            count=error[0, :, 1].max(),
                            peak24=result[0, rin.index(24), 0],
                        )
                    )
    pd.DataFrame(perturbations).to_csv(OUT / "ablations.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUT / "comparison.csv", index=False)
    pd.DataFrame(metrics).to_csv(OUT / "by_run.csv", index=False)
    pd.DataFrame(curves).to_csv(
        OUT / "trajectories.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    pd.DataFrame(ablations).to_csv(OUT / "diversion_profile.csv", index=False)


if __name__ == "__main__":
    main()
