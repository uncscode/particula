"""Fit both channel orders at 3.75 s and challenge them at 1.875 s."""

import argparse
import json
from time import perf_counter

import numpy as np
from ad_campaign import fit_rates_and_doses
from ad_shortlist import campaign_seed, errors
from baseline_replay import HERE, Model, run, save
from growth import BANK, HI, LO, initial


def profiles(model, x, ids, doses, sub, verify=32):
    """Native Warp checks with fixed doses and independent 61-dose profiles."""
    fixed, grid = [], []
    for steps, reverse in (
        (sub, False),
        (sub, True),
        (verify, False),
        (verify, True),
    ):
        fixed.append(float(errors(model, x, ids, doses, steps, reverse).max()))
        grid.append(float(run(model, x, ids, sub=steps, reverse=reverse)[0][0]))
    return dict(fixed_scores=fixed, grid_scores=grid)


def fit_both_orders(model, start, doses, active, sub, budget, verify=32):
    """Fit both orders and independently check their native scores."""
    t = perf_counter()
    fit = fit_rates_and_doses(
        model,
        start[:16],
        start[16:],
        active,
        doses,
        active,
        sub=sub,
        max_evaluations=budget,
        tune_physics=[*range(10), 12],
        both_orders=True,
    )
    elapsed = perf_counter() - t
    x, d = np.array(fit["x"]), fit["dose"]
    fit.update(
        profiles(model, x, active, d, sub, verify),
        elapsed_s=elapsed,
        directed_evaluations=2 * fit["evaluations"],
    )
    np.testing.assert_allclose(
        fit["score"], max(fit["fixed_scores"][:2]), atol=1e-11, rtol=0
    )
    return fit


def continued_starts(results, extra):
    """Use the same two shared vectors and doses for a nested pair."""
    seeds = []
    for wanted, label in (([], "best fine parent"), (extra, "best fine child")):
        row = next(r for r in results if r["extra"] == wanted)
        point = min(row["starts"], key=lambda s: s["score"])
        seeds.append((label, np.array(point["x"]), point["dose"]))
    return seeds


def fit_design(args, extras):
    """Resolve the common starts, budgets and saved checkpoints."""
    archived, original_ids = initial()
    if args.number > 1:
        previous = campaign_seed(args.number)
        archived, original_ids = np.asarray(previous["x"]), previous["ids"]
    if args.number == 1:
        coarse = json.loads(
            (HERE / f"ad_fits_{args.number:02d}.json").read_text()
        )
        parent = next(row for row in coarse["results"] if not row["extra"])
        coarse_start = min(
            parent["starts"], key=lambda r: max(r["grid_scores"][1:])
        )
        secondary = coarse_start["start"]
        seeds = [
            ("incumbent", archived, None),
            ("coarse parent", np.array(coarse_start["x"]), None),
        ]
    else:
        shifted = archived.copy()
        shifted[:3] = np.clip(shifted[:3] * 1.01, LO[:3], HI[:3])
        secondary = "1% shift in shared precursor formation times"
        seeds = [
            ("selected lineage", archived, previous["dose"]),
            ("shifted lineage", shifted, previous["dose"]),
        ]
    destination = f"ad_fine_{args.number:02d}.json"
    config = dict(
        calls=args.calls,
        sub=args.sub,
        resolution=[
            "fit forward",
            "fit reverse",
            "1.875s forward",
            "1.875s reverse",
        ],
        coarse_parent_start=secondary,
        fitted_substeps_per_minute=args.sub,
        selected_child_rate_start=[0.0, 0.1],
    )
    if args.verify_sub != 32:
        config["verify_sub"] = args.verify_sub
        config["resolution"][-2:] = [
            f"{60 / args.verify_sub:g}s {order}"
            for order in ("forward", "reverse")
        ]
    if args.number > 1:
        config["lineage_parent"] = {
            key: previous[key] for key in ("ids", "x", "dose")
        }
    if args.continue_pair:
        if len(extras) != 1:
            raise ValueError("Continue exactly one nested pair at a time")
        reference = json.loads((HERE / destination).read_text())
        seeds = continued_starts(reference["results"], extras[0])
        key = "_".join(map(str, extras[0]))
        destination = f"ad_cont_{key}_{args.number:02d}.json"
        config.pop("selected_child_rate_start")
        config["starts"] = [s[0] for s in seeds]
    records, pending = [], {}
    if (HERE / destination).exists():
        last = json.loads((HERE / destination).read_text())
        if last["config"] != config:
            raise ValueError(
                "Existing fit differs in budget, resolution or starting point"
            )
        records, pending = last["results"], last.get("pending", {})
    return original_ids, seeds, config, destination, records, pending


def project_start(model, shared, saved_doses, ids, extras, extra, index, args):
    """Preserve common coordinates and dose under each nested comparison."""
    start = shared.copy()
    doses = (
        run(model, start, ids, sub=args.sub)[2][0]
        if saved_doses is None
        else np.array(saved_doses)
    )
    if index and not args.continue_pair:
        for h in extra:
            start[16 + h] = 0.1
    if not extra and args.continue_pair:
        start[16 + np.array(extras[0])] = 0.0
    return start, doses


def main():
    """Compare finalists and parent under matched fine-fit budgets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        help="Comma-separated channel IDs of a connected candidate",
    )
    parser.add_argument("--calls", type=int, default=180)
    parser.add_argument("--sub", type=int, default=16)
    parser.add_argument("--verify-sub", type=int, default=32)
    parser.add_argument("--round", type=int, default=1, dest="number")
    parser.add_argument(
        "--continue-pair",
        action="store_true",
        help="Use the best parent and projected child as shared starts",
    )
    args = parser.parse_args()
    if args.verify_sub <= args.sub:
        raise ValueError("Validation needs a smaller timestep than training")
    extras = [list(map(int, group.split(","))) for group in args.extra]
    if not extras:
        raise ValueError("At least one finalist is required")
    model = Model(BANK)
    original_ids, seeds, config, destination, records, pending = fit_design(
        args, extras
    )
    for extra in [[], *extras]:
        if any(row["extra"] == extra for row in records):
            continue
        active = sorted(set(original_ids) | set(extra))
        starts = pending["starts"] if pending.get("extra") == extra else []
        for i, (label, shared, saved_doses) in enumerate(seeds):
            if i < len(starts):
                continue
            start, doses = project_start(
                model, shared, saved_doses, original_ids, extras, extra, i, args
            )
            fit = fit_both_orders(
                model,
                start,
                doses,
                active,
                args.sub,
                args.calls,
                args.verify_sub,
            )
            fit["start"] = label
            starts.append(fit)
            print(
                extra,
                label,
                fit["fixed_scores"],
                fit["grid_scores"],
                round(fit["elapsed_s"], 2),
                flush=True,
            )
            # Checkpoint every completed start, including long finer fits.
            save(
                destination,
                dict(
                    config=config,
                    results=records,
                    pending=dict(extra=extra, active=active, starts=starts),
                ),
            )
        records.append(dict(extra=extra, active=active, starts=starts))
        save(destination, dict(config=config, results=records))


if __name__ == "__main__":
    main()
