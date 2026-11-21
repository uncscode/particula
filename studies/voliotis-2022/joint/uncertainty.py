# %%
"""Cross chemistry and wall mixing on CUDA with bounded precursor doses."""

import json

import numpy as np
import pandas as pd
from scipy.stats import qmc
from uncertainty_profile import DEST, NAMES, anchors
from wall_campaign import OUT, Campaign

SEEDS = (4109, 7753)
N = 256
DOSES = np.linspace(0.85, 1.15, 61)
RESPONSES = ("peak_soa_ug_m3", "five_hour_number_fraction")


def chemistry_design(prior):
    """Sample log delays and linear yields across earlier fitted ranges."""
    transformed = prior[:, :9].copy()
    transformed[:, :3] = np.log(transformed[:, :3])
    low, high = transformed.min(0), transformed.max(0)
    samples = []
    for seed in SEEDS:
        u = qmc.Sobol(9, scramble=True, seed=seed).random_base2(8)
        rows = low + u * (high - low)
        rows[:, :3] = np.exp(rows[:, :3])
        samples.extend(rows)
    return np.asarray(samples)


def shares(y):
    """Partition equal-weight variance into chemistry, wall and interaction."""
    grand = y.mean((0, 1))
    chemistry = y.mean(0, keepdims=True) - grand
    wall = y.mean(1, keepdims=True) - grand
    components = np.stack(
        [
            np.mean(chemistry**2, (0, 1)),
            np.mean(wall**2, (0, 1)),
            np.mean((y - grand - chemistry - wall) ** 2, (0, 1)),
        ]
    )
    variance = y.var((0, 1))
    np.testing.assert_allclose(
        components.sum(0), variance, rtol=1e-10, atol=1e-18
    )
    return np.divide(
        components,
        variance,
        out=np.full_like(components, np.nan),
        where=variance > 1e-20,
    )


def compatible_responses(campaign, passing, control):
    """Yield outcomes at every run dose compatible with the campaign screen."""
    # Each run selects its dose independently at fixed chemistry and wall.
    for dose in DOSES:
        _, dose_errors, _, response = campaign.evaluate(
            passing,
            True,
            "joint",
            np.array([dose]),
            sub=2,
            outcomes=True,
        )
        valid = dose_errors.max(2) <= control
        yield (np.where(valid[:, :, None], response, np.nan))


def main():
    """Save scenario scores, forward envelopes and wall profiles."""
    base, levels, joint, dark, mass = anchors()
    fits = json.loads((DEST / "profile_vectors.json").read_text())
    if any(sum(f["eddy"] == e for f in fits) != 2 for e in levels):
        raise ValueError("Expected two profile vectors at each wall level")
    prior = np.array([r["physical"] for r in joint + dark + mass])
    prof = np.array([r["physical"] for r in fits])
    design = chemistry_design(prior)
    pars = np.vstack([design, prior[:, :9], prof[:, :9]])
    table = pd.DataFrame(pars, columns=NAMES[:9])
    table["source"] = (
        [f"sobol_{s}" for s in SEEDS for _ in range(N)]
        + ["prior_joint"] * len(joint)
        + ["prior_dark"] * len(dark)
        + ["prior_mass"] * len(mass)
        + ["wall_profile"] * len(fits)
    )
    table.to_csv(DEST / "chemistry_design.csv", index_label="scenario")
    c = Campaign()
    runs = [r[0] for r in c.runs]
    shape = (len(levels), len(pars), len(runs), 2)
    nominal, fitted, errors = [np.empty(shape) for _ in range(3)]
    checks = pd.read_csv(OUT / "campaign_scores.csv").query(
        "model == 'constant_brownian' and objective == 'joint'"
    )
    control = checks[["mass_nrmse", "count_nrmse"]].to_numpy().max()
    summary = []
    run_results = []
    admissible = []
    for w, e in enumerate(levels):
        p = np.column_stack((pars, np.full(len(pars), e)))
        for start in range(0, len(p), 64):
            sl = slice(start, start + 64)
            nominal[w, sl] = c.evaluate(
                p[sl], True, "joint", np.array([1.0]), sub=2, outcomes=True
            )[3]
            _, errors[w, sl], _, fitted[w, sl] = c.evaluate(
                p[sl], True, "joint", DOSES, sub=2, outcomes=True
            )
        scores = errors[w].max((1, 2))
        passing = p[scores <= control]
        if len(passing):
            admissible.extend(compatible_responses(c, passing, control))
        idx = scores.argmin()
        best, refined, _, _ = c.evaluate(
            p[idx], True, "joint", DOSES, sub=4, outcomes=True
        )
        _, _, doses = c.evaluate(p[idx], True, "joint", DOSES, sub=2)
        for ri, run in enumerate(runs):
            run_results.append(
                dict(
                    eddy_s_inv=e,
                    run=run,
                    best_scenario=idx,
                    dose=doses[0, ri],
                    mass_nrmse=errors[w, idx, ri, 0],
                    count_nrmse=errors[w, idx, ri, 1],
                    observed_peak_ug_m3=c.runs[ri][3].max(),
                    model_peak_ug_m3=fitted[w, idx, ri, 0],
                    model_five_hour_number_fraction=fitted[w, idx, ri, 1],
                )
            )
        summary.append(
            dict(
                eddy_s_inv=e,
                frozen_chemistry=c.evaluate(
                    np.r_[base[:9], e], True, "joint", DOSES, sub=2
                )[0].item(),
                refit_min=min(r["joint_score"] for r in fits if r["eddy"] == e),
                best_found=scores[idx],
                best_scenario=idx,
                worst_mass=errors[w, idx, :, 0].max(),
                mass_run=runs[errors[w, idx, :, 0].argmax()],
                worst_count=errors[w, idx, :, 1].max(),
                count_run=runs[errors[w, idx, :, 1].argmax()],
                max_error_refinement_change=np.abs(
                    refined[0] - errors[w, idx]
                ).max(),
                refined_joint=best.item(),
                random_passing=int((scores[: 2 * N] <= control).sum()),
                passing_with_anchors=int((scores <= control).sum()),
            )
        )
        print("Crossed wall level", e, flush=True)
    pd.DataFrame(summary).to_csv(DEST / "wall_profile.csv", index=False)
    pd.DataFrame(run_results).to_csv(DEST / "best_by_run.csv", index=False)
    score = errors.max((2, 3))
    screen = score <= control
    if not (
        np.isfinite(nominal).all()
        and (nominal >= 0).all()
        and nominal[:, :, :, 1].max() <= 1 + 1e-12
    ):
        raise ValueError("Invalid forward response")
    if not screen.any():
        raise ValueError("No scenarios meet the comparison threshold")
    forward = nominal[:, : 2 * N]
    fractions = shares(forward)
    discrepancy = np.abs(shares(forward[:, :N]) - shares(forward[:, N:]))
    compatible = np.concatenate(admissible, axis=0)
    compatible_min = np.nanmin(compatible, axis=0)
    compatible_max = np.nanmax(compatible, axis=0)
    sens = []
    for ri, run in enumerate(runs):
        for v, response in enumerate(RESPONSES):
            y = forward[:, :, ri, v]
            accepted = fitted[:, :, ri, v][screen]
            sens.append(
                dict(
                    run=run,
                    outcome=response,
                    design_min=y.min(),
                    design_max=y.max(),
                    screened_min=accepted.min(),
                    screened_max=accepted.max(),
                    all_compatible_doses_min=compatible_min[ri, v],
                    all_compatible_doses_max=compatible_max[ri, v],
                    chemistry_fraction=fractions[0, ri, v],
                    wall_fraction=fractions[1, ri, v],
                    interaction_fraction=fractions[2, ri, v],
                    max_scramble_fraction_difference=discrepancy[
                        :, ri, v
                    ].max(),
                )
            )
    pd.DataFrame(sens).to_csv(DEST / "sensitivity_by_run.csv", index=False)
    pd.DataFrame(
        [
            dict(
                eddy_s_inv=e,
                scenario=i,
                joint=score[w, i],
                mass=errors[w, i, :, 0].max(),
                count=errors[w, i, :, 1].max(),
            )
            for w, e in enumerate(levels)
            for i in range(len(pars))
        ]
    ).to_csv(DEST / "scenario_scores.csv", index=False)
    print("Comparison threshold", control, "minimax", score.min(), flush=True)


if __name__ == "__main__":
    main()
