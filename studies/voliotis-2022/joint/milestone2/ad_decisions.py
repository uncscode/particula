"""Resolve each matched AD fit against its refitted and archived parent."""

import argparse
import json

import numpy as np
from ad_shortlist import campaign_seed, errors
from baseline_replay import HERE, Model, run, save
from growth import BANK, initial


def summarize(row, fine):
    """Select by the training loss and retain both start diagnostics."""
    starts = []
    for record in row["starts"]:
        values = record["fixed_scores" if fine else "scores"]
        starts.append(
            dict(
                start=record["start"],
                training=float(record["score"]),
                fixed_resolution_scores=values,
                grid_resolution_scores=record["grid_scores"],
                evaluations=record["evaluations"],
                status=record["status"],
            )
        )
    idx = min(range(len(starts)), key=lambda i: starts[i]["training"])
    fit = row["starts"][idx]
    return dict(
        extra=row["extra"],
        active=row["active"],
        x=fit["x"],
        dose=fit["dose"],
        best_start=fit["start"],
        starts=starts,
        scores=[
            starts[idx]["training"],
            *starts[idx]["fixed_resolution_scores"][-2:],
        ],
    )


def decide(parent, child, archived, previous_scores):
    """A positive gain must exceed restart and resolution effects."""
    p = np.array(parent["scores"])
    c = np.array(child["scores"])
    a = np.array(archived["scores"])
    restart = abs(
        parent["starts"][0]["training"] - parent["starts"][1]["training"]
    )
    child_restart = abs(
        child["starts"][0]["training"] - child["starts"][1]["training"]
    )
    numerical = max(abs(p[1:] - p[0]).max(), abs(c[1:] - c[0]).max())
    gate = max(float(numerical), restart, child_restart, 1e-10)
    controls = np.minimum(a, p)
    for scores in previous_scores.values():
        controls = np.minimum(controls, scores)
    gain = float(min(controls - c))
    return dict(
        parent=parent,
        child=child,
        per_setting_parent_minus_child=(p - c).tolist(),
        refit_gain=(a - p).tolist(),
        minimum_gain_including_incumbent=gain,
        parent_restart_difference=restart,
        child_restart_difference=child_restart,
        max_restart_difference=max(restart, child_restart),
        max_resolution_difference=numerical,
        acceptance_threshold=gate,
        accepted=bool(gain > gate),
    )


def score_control(model, ids, vector, doses, sub, verify, both_orders):
    """Replay an earlier control on exactly the same comparison grids."""
    x, d = np.asarray(vector), np.asarray(doses)
    train = float(errors(model, x, ids, d, sub).max())
    if both_orders:
        train = max(train, float(errors(model, x, ids, d, sub, True).max()))
    return [
        train,
        *[
            float(errors(model, x, ids, d, verify, rev).max())
            for rev in (False, True)
        ],
    ]


def earlier_solutions(number, ids):
    """Find already solved controls on precisely the current parent topology."""
    donors = {}
    for label, prefix in (
        ("projected deletion parent", "ad_ablation"),
        ("adjudicated deletion parent", "ad_ablation_adjudicated"),
    ):
        deletion = HERE / f"{prefix}_{number:02d}.json"
        if deletion.exists():
            record = json.loads(deletion.read_text())
            pair_file = json.loads((HERE / record["source"]).read_text())
            parent = next(r for r in pair_file["results"] if not r["extra"])
            if set(parent["active"]) == set(ids):
                donors[label] = record["parent_refit"]
    finer = HERE / f"ad_adjudicated_{number - 1:02d}.json"
    if number > 1 and finer.exists():
        for row in json.loads(finer.read_text())["results"]:
            if set(row["active"]) == set(ids):
                donors["previous lineage at finer precision"] = min(
                    row["starts"], key=lambda f: f["score"]
                )
    if number > 1:
        latest = json.loads((HERE / "lineage.json").read_text())[number - 2]
        if set(latest["ids"]) == set(ids):
            donors["latest officially selected parent"] = latest
    return donors


def main():
    """Record comparable decisions from every completed pair."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", dest="number", type=int, default=1)
    args = parser.parse_args()
    model = Model(BANK)
    x, ids = initial()
    if args.number > 1:
        previous = campaign_seed(args.number)
        x, ids = np.asarray(previous["x"]), previous["ids"]
    doses = (
        np.array(previous["dose"])
        if args.number > 1
        else run(model, x, ids)[2][0]
    )
    current_ids = (
        json.loads((HERE / "lineage.json").read_text())[args.number - 2]["ids"]
        if args.number > 1
        else ids
    )
    output = dict(
        round=args.number,
        scores=["training", "validation forward", "validation reverse"],
        selection=(
            "Best training score per topology. Doses fixed across "
            "timesteps and reaction orders."
        ),
        fits={},
    )
    sources = [
        (name, HERE / f"{prefix}_{args.number:02d}.json")
        for name, prefix in (("coarse", "ad_fits"), ("fine", "ad_fine"))
    ]
    sources += [
        (p.stem, p)
        for p in sorted(HERE.glob(f"ad_cont_*_{args.number:02d}.json"))
    ]
    sources.append(
        ("adjudicated", HERE / f"ad_adjudicated_{args.number:02d}.json")
    )
    for name, path in sources:
        if not path.exists():
            continue
        archive = json.loads(path.read_text())
        rows = archive["results"]
        if not rows:
            continue
        fine = name != "coarse"
        sub = archive["config"]["sub"] if fine else 4
        verify = archive.get("config", {}).get("verify_sub", 32)
        design = archive.get("config", {}).get("lineage_parent")
        base_x, base_ids, base_doses = (
            (x, ids, doses)
            if design is None
            else (
                np.array(design["x"]),
                design["ids"],
                np.array(design["dose"]),
            )
        )
        incumbent = dict(
            active=base_ids,
            x=base_x.tolist(),
            dose=base_doses.tolist(),
            scores=score_control(
                model, base_ids, base_x, base_doses, sub, verify, fine
            ),
        )
        parent = summarize(next(row for row in rows if not row["extra"]), fine)
        donors = earlier_solutions(args.number, base_ids)
        if set(base_ids) == set(ids) and (
            not np.array_equal(base_x, x)
            or not np.array_equal(base_doses, doses)
        ):
            donors["original growth seed"] = dict(x=x, dose=doses)
        previous_scores = {
            label: score_control(
                model, base_ids, d["x"], d["dose"], sub, verify, fine
            )
            for label, d in donors.items()
        }
        if name.startswith("ad_cont"):
            previous_scores["original fine parent"] = output["fits"]["fine"][
                "pairs"
            ][0]["parent"]["scores"]
        if "control_scores" in archive:
            previous_scores["strongest incoming parent"] = archive[
                "control_scores"
            ]
        output["fits"][name] = dict(
            additional_parent_control_scores=previous_scores,
            matches_current_lineage=set(base_ids) == set(current_ids),
            validation_step_seconds=60.0 / verify,
            source=path.name,
            archived=incumbent,
            training_step_seconds=60.0 / sub,
            training_orders=["forward", "reverse"] if fine else ["forward"],
            pairs=[
                decide(parent, summarize(row, fine), incumbent, previous_scores)
                for row in rows
                if row["extra"]
            ],
        )
    filename = (
        "ad_decisions.json"
        if args.number == 1
        else f"ad_decisions_{args.number:02d}.json"
    )
    save(filename, output)


if __name__ == "__main__":
    main()
