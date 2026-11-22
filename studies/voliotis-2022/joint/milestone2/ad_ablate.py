"""Delete the strongest refined candidate and refit the reduced network."""

import argparse
import json
from time import perf_counter

import numpy as np
from ad_campaign import CampaignDerivatives, fit_rates_and_doses
from ad_fine import profiles
from ad_shortlist import errors
from baseline_replay import HERE, Model, run, save
from growth import BANK


def limiting_sensitivities(model, x, doses, active, extra):
    """Differentiate the three highest run and observable errors."""
    result = {}
    for name, order in (
        ("forward", active),
        ("reverse", list(reversed(active))),
    ):
        kernel = CampaignDerivatives(model, order, extra, sub=32)
        scores, dose_gradient, rate_gradients = kernel.evaluate(
            x[:16], x[16:], doses
        )
        top = np.argsort(scores.ravel())[::-1][:3]
        rows = []
        for index in top:
            run, obs = np.unravel_index(index, scores.shape)
            rows.append(
                dict(
                    run=int(model.runs[run][0]),
                    observable=("mass", "count")[obs],
                    rmse=float(scores[run, obs]),
                    d_rmse_per_unit_dose=float(dose_gradient[run, obs]),
                    d_rmse_per_unit_rate=rate_gradients[run, obs].tolist(),
                )
            )
        result[name] = rows
    return result


def competing_fluxes(model, candidate, reduced_ids, deleted, refit):
    """Compare transfers using the same reactants under both fitted networks."""
    full = np.asarray(candidate["x"])
    doses = np.asarray(candidate["dose"])
    residual = errors(model, deleted, reduced_ids, doses, 32)
    r, obs = np.unravel_index(residual.argmax(), residual.shape)
    lookup = {
        (BANK[h].a, BANK[h].b, BANK[h].area_scaled) for h in candidate["extra"]
    }
    comparable = [
        h
        for h in candidate["active"]
        if (BANK[h].a, BANK[h].b, BANK[h].area_scaled) in lookup
    ]
    fits = [
        ("child", full, candidate["active"], doses),
        (
            "projected_parent_refit",
            np.asarray(refit["x"]),
            reduced_ids,
            np.asarray(refit["dose"]),
        ),
    ]
    records = []
    for name, x, active, d in fits:
        flux = run(model, x, active, sub=32, doses=d[:, None], diagnostics=1)[
            3
        ][0]
        for h in comparable:
            records.append(
                dict(
                    network=name,
                    channel=BANK[h].key,
                    equation=BANK[h].equation,
                    rate=float(x[16 + h]),
                    gross_at_limiting_run_ug_m3=float(flux[r, h]),
                    max_gross_ug_m3=float(flux[:, h].max()),
                )
            )
    return dict(
        limiting_run_after_deletion=int(model.runs[r][0]),
        limiting_observable=("mass", "count")[obs],
        transfers=records,
    )


def main():
    """Separate the loss from deletion from any gain due to parent refitting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", dest="number", type=int, default=1)
    parser.add_argument(
        "--adjudicated",
        action="store_true",
        help="Save a separate deletion control from the final finer pair",
    )
    args = parser.parse_args()
    decision_file = (
        "ad_decisions.json"
        if args.number == 1
        else f"ad_decisions_{args.number:02d}.json"
    )
    stages = json.loads((HERE / decision_file).read_text())["fits"]
    entrants = [
        (name, pair)
        for name, stage in stages.items()
        if name != "coarse" and (name == "adjudicated") == args.adjudicated
        for pair in stage["pairs"]
    ]
    eligible = [(name, p) for name, p in entrants if p["accepted"]] or entrants
    name, pair = min(
        eligible, key=lambda item: max(item[1]["child"]["scores"][1:])
    )
    source = stages[name]["source"]
    configuration = json.loads((HERE / source).read_text())["config"]
    sub, budget = configuration["sub"], configuration["calls"]
    verify = configuration.get("verify_sub", 32)
    model = Model(BANK)
    candidate = pair["child"]
    full = np.array(candidate["x"])
    doses = np.array(candidate["dose"])
    original = profiles(model, full, candidate["active"], doses, sub, verify)
    deleted = full.copy()
    deleted[16 + np.array(candidate["extra"])] = 0
    reduced_ids = [
        j for j in candidate["active"] if j not in candidate["extra"]
    ]
    frozen = profiles(model, deleted, reduced_ids, doses, sub, verify)
    t = perf_counter()
    fit = fit_rates_and_doses(
        model,
        deleted[:16],
        deleted[16:],
        reduced_ids,
        doses,
        reduced_ids,
        sub=sub,
        max_evaluations=budget,
        tune_physics=[*range(10), 12],
        both_orders=True,
    )
    elapsed = perf_counter() - t
    refitted = profiles(
        model, np.asarray(fit["x"]), reduced_ids, fit["dose"], sub, verify
    )
    np.testing.assert_allclose(
        fit["score"], max(refitted["fixed_scores"][:2]), atol=1e-11, rtol=0
    )
    fit.update(refitted, elapsed_s=elapsed)
    result = dict(
        round=args.number,
        source=source,
        limiting_derivatives=limiting_sensitivities(
            model, full, doses, candidate["active"], candidate["extra"]
        ),
        ranking=(
            "Lowest worst refined score among admitted additions, or "
            "among all fine-tested additions if none admitted."
        ),
        accepted=pair["accepted"],
        deleted=candidate["extra"],
        keys=[BANK[h].key for h in candidate["extra"]],
        same_reactant_transfers=competing_fluxes(
            model, candidate, reduced_ids, deleted, fit
        ),
        fitted_rates=full[16 + np.array(candidate["extra"])].tolist(),
        fine_fit_step_s=60.0 / sub,
        calls=budget,
        resolution=configuration["resolution"],
        original=original,
        frozen_deletion=frozen,
        parent_refit=fit,
        previous_paired_parent_scores=pair["parent"]["scores"],
        frozen_increase=(
            np.array(frozen["fixed_scores"]) - original["fixed_scores"]
        ).tolist(),
        refit_minus_child=(
            np.array(refitted["fixed_scores"]) - original["fixed_scores"]
        ).tolist(),
    )
    prefix = "ad_ablation_adjudicated" if args.adjudicated else "ad_ablation"
    save(f"{prefix}_{args.number:02d}.json", result)
    print(
        result["keys"],
        "frozen deletion",
        result["frozen_increase"],
        "refit minus child",
        result["refit_minus_child"],
        flush=True,
    )


if __name__ == "__main__":
    main()
