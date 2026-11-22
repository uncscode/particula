"""Adjudicate paired trials and export one frozen network for size replay."""

import argparse
import json

import numpy as np
from baseline_replay import HERE, Model, packed, run, save
from growth import BANK, growth_seed, initial


def frozen(model, x, ids, dose, sub, reverse=False):
    """Evaluate a fixed fit without adjusting any run dose."""
    return float(
        run(
            model,
            x,
            ids,
            sub=sub,
            doses=np.asarray(dose)[:, None],
            reverse=reverse,
        )[0][0]
    )


def choose(choices, ids):
    """Keep the best qualified network under the two fine channel orders."""

    def fine(row):
        return max(row["qualification_scores"])

    best_parent = min((r for r in choices if r["ids"] == ids), key=fine)
    # Growing a topology must improve on the best available parent. Apply
    # the paired controls first; a bad parent restart must not earn a channel.
    eligible = []
    for row in choices:
        gain = min(
            a - b
            for a, b in zip(
                best_parent["qualification_scores"],
                row["qualification_scores"],
                strict=True,
            )
        )
        row["fine_gain_over_best_parent"] = gain
        passes = len(row["ids"]) <= len(ids) or gain > row["threshold"]
        row["eligible_after_global_control"] = bool(row["eligible"] and passes)
        if row["eligible_after_global_control"]:
            eligible.append(row)
    selected = min(eligible, key=lambda r: (round(fine(r), 12), len(r["ids"])))

    return best_parent, selected


def native_threshold(pair):
    """Include the spread across starts on both sides of a native comparison."""
    return max(
        pair.get("threshold", pair.get("numerical", 0.0)),
        *(
            abs(pair[k]["starts"][0] - pair[k]["starts"][1])
            for k in ("parent", "child")
        ),
    )


def ad_trials(ad, register):
    """Register both sides of every matched gradient-based comparison."""
    decisions = []
    # A dedicated fine-grid adjudication supersedes the lower-resolution
    # admission for the same child while leaving that earlier result visible.
    adjudicated = {
        tuple(pair["child"]["extra"]): pair["accepted"]
        for pair in ad["fits"].get("adjudicated", {}).get("pairs", [])
    }
    for method, group in ad["fits"].items():
        for i, pair in enumerate(group["pairs"]):
            for side in ("parent", "child"):
                fit = pair[side]
                register(
                    f"ad_{method}_{i}_{side}",
                    fit["x"],
                    fit["active"],
                    fit["dose"],
                    side == "parent"
                    or (
                        pair["accepted"]
                        and adjudicated.get(tuple(fit["extra"]), True)
                    ),
                    pair["acceptance_threshold"] if side == "child" else 0.0,
                )
            decisions.append(
                dict(
                    method="ad_" + method,
                    extra=pair["child"]["extra"],
                    accepted=pair["accepted"],
                )
            )

    return decisions


def qualify_choices(model, ad, choices, prior_step):
    """Compare all candidate fits at the finest adjudication resolution."""
    # If a nested pair was adjudicated on a finer grid, compare its child
    # with every eligible, possibly stronger simpler fit at that same grid.
    vetoes = {
        tuple(pair["child"]["active"])
        for pair in ad["fits"].get("adjudicated", {}).get("pairs", [])
        if not pair["accepted"]
    }
    steps = [
        g["validation_step_seconds"] for g in ad["fits"].values() if g["pairs"]
    ]
    qualification_sub = round(60 / min(prior_step, *steps))
    qualified_scores = {}
    for row in choices:
        if tuple(row["ids"]) in vetoes:
            row["eligible"] = False
        row["qualification_scores"] = row["scores"][-2:]
        if qualification_sub != 32:
            key = (tuple(row["ids"]), tuple(row["x"]), tuple(row["dose"]))
            if key not in qualified_scores:
                qualified_scores[key] = [
                    frozen(
                        model,
                        row["x"],
                        row["ids"],
                        row["dose"],
                        qualification_sub,
                        rev,
                    )
                    for rev in (False, True)
                ]
            row["qualification_scores"] = qualified_scores[key]
    return qualification_sub


def native_trials(record, warm, parent_ids, register):
    """Register equal-budget native fits with their original dose grid."""
    sub = round(60 / record["training_seconds"])
    decisions = []
    for method, prefix, pairs in (
        ("finite_difference", "fd_", record["controls"]),
        ("connected_warm_start", "warm_", warm),
    ):
        for pair in pairs:
            add = pair["extra"]
            name = prefix + "_".join(pair["keys"])
            gate = native_threshold(pair)
            gain = (
                pair["gain"]
                if method == "connected_warm_start"
                else pair["minimum_gain"]
            )
            passed = pair["accepted"] and gain > gate
            register(
                name + "_parent",
                pair["parent"]["x"],
                parent_ids,
                profile_sub=sub,
            )
            register(
                name,
                pair["child"]["x"],
                sorted(parent_ids + add),
                eligible=passed,
                threshold=gate,
                profile_sub=sub,
            )
            decisions.append(dict(method=method, extra=add, accepted=passed))
    return decisions


def select(number=1):
    """Keep qualifying growth and compare same-topology refits on fine runs."""
    model = Model(BANK)
    x, ids = initial()
    lineage = (
        json.loads((HERE / "lineage.json").read_text())[: number - 1]
        if number > 1
        else []
    )
    warm_file = HERE / (
        "activation_refits.json"
        if number == 1
        else f"activation_refits_{number:02d}.json"
    )
    warm = json.loads(warm_file.read_text())
    ad_file = HERE / (
        "ad_decisions.json"
        if number == 1
        else f"ad_decisions_{number:02d}.json"
    )
    ad = json.loads(ad_file.read_text())
    choices = []
    evaluated = {}

    def register(
        name,
        vector,
        active,
        dose=None,
        eligible=True,
        threshold=0.0,
        profile_sub=4,
    ):
        vector = np.asarray(vector)
        cache_key = (
            vector.tobytes(),
            tuple(active),
            None if dose is None else tuple(dose),
            profile_sub if dose is None else None,
        )
        if cache_key not in evaluated:
            profile = (
                run(model, vector, active, sub=profile_sub)[2][0]
                if dose is None
                else dose
            )
            scores = [
                frozen(model, vector, active, profile, sub, rev)
                for sub, rev in (
                    (4, False),
                    (16, False),
                    (16, True),
                    (32, False),
                    (32, True),
                )
            ]
            evaluated[cache_key] = list(profile), scores
        dose, scores = evaluated[cache_key]
        choices.append(
            dict(
                name=name,
                ids=active,
                x=vector.tolist(),
                dose=list(dose),
                scores=scores,
                eligible=bool(eligible),
                threshold=float(threshold),
            )
        )

    register("archived", x, ids)
    baseline = choices[0]
    verified_parent = True
    parent_ids = ids
    prior_step = 1.875
    if number > 1:
        prior_step = json.loads(
            (HERE / f"selection_{number - 1:02d}.json").read_text()
        ).get("qualification_step_seconds", 1.875)
        prior = lineage[-1]
        seed = growth_seed(number, prior)
        parent_ids = seed["ids"]
        register("screen_seed", seed["x"], parent_ids, seed["dose"])
        verified_parent = prior["admitted"] and parent_ids == prior["ids"]
        if verified_parent:
            register("previous_verified", prior["x"], parent_ids, prior["dose"])
    record = json.loads((HERE / f"native_round_{number:02d}.json").read_text())
    decisions = native_trials(record, warm, parent_ids, register)
    decisions.extend(ad_trials(ad, register))
    for deletion_file in sorted(HERE.glob(f"ad_ablation*_{number:02d}.json")):
        fit = json.loads(deletion_file.read_text())["parent_refit"]
        register(
            deletion_file.stem + "_parent", fit["x"], parent_ids, fit["dose"]
        )
    qualification_sub = qualify_choices(model, ad, choices, prior_step)
    best_parent, selected = choose(choices, parent_ids)

    def compact(row):
        return {k: v for k, v in row.items() if k not in ("x", "dose")}

    summary = dict(
        score_order=[
            "15s forward",
            "3.75s forward",
            "3.75s reverse",
            "1.875s forward",
            "1.875s reverse",
        ],
        qualification_step_seconds=60 / qualification_sub,
        archive=compact(baseline),
        best_parent=compact(best_parent),
        selected=compact(selected),
        decisions=decisions,
        compared=[compact(r) for r in choices],
    )
    summary["verified_parent"] = verified_parent
    save(f"selection_{number:02d}.json", summary)
    if not verified_parent:
        print("Exploratory screen from superseded parent", selected["name"])
        return
    save(
        "selection.json",
        {
            "round": number,
            **{
                key: summary[key]
                for key in (
                    "score_order",
                    "qualification_step_seconds",
                    "archive",
                    "selected",
                )
            },
        },
    )
    lineage.append(
        dict(
            round=number,
            native_record=f"native_round_{number:02d}.json",
            ids=selected["ids"],
            x=selected["x"],
            dose=selected["dose"],
            selected_fit=selected["name"],
            admitted=bool(set(selected["ids"]) - set(parent_ids)),
        )
    )
    save("lineage.json", lineage)
    save(
        "size_inputs.json",
        {
            "cases": {
                name: packed(BANK, np.asarray(r["x"]), r["ids"], r["dose"])
                for name, r in [("incumbent", baseline), ("selected", selected)]
            }
        },
    )
    print(
        selected["name"], len(selected["ids"]), selected["scores"], flush=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=int, default=1, dest="number")
    select(parser.parse_args().number)
