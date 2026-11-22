"""Resolve a near-threshold chemistry comparison on a finer training grid."""

import argparse
import json

import numpy as np
from ad_fine import fit_both_orders
from ad_shortlist import errors
from baseline_replay import HERE, Model, save
from growth import BANK


def fixed_scores(model, record, ids, sub):
    """Return both frozen-dose native RMSE maxima for one grid."""
    x, d = np.array(record["x"]), record["dose"]
    return [
        float(errors(model, x, ids, d, sub, reverse).max())
        for reverse in (False, True)
    ]


def controls(model, number, extra):
    """Build matched parent and child starts from earlier tested solutions."""
    if extra:
        source = f"ad_fine_{number:02d}.json"
        deleted = list(map(int, extra.split(",")))
        earlier = json.loads((HERE / source).read_text())
        previous = json.loads((HERE / "lineage.json").read_text())[number - 2]
        simpler = [previous]
    else:
        deletion = json.loads(
            (HERE / f"ad_ablation_{number:02d}.json").read_text()
        )
        deleted = deletion["deleted"]
        source = deletion["source"]
        earlier = json.loads((HERE / source).read_text())
        simpler = [deletion["parent_refit"]]
    parent, child = (
        next(r for r in earlier["results"] if r["extra"] == group)
        for group in ([], deleted)
    )
    if extra and set(previous["ids"]) != set(parent["active"]):
        raise ValueError("The finalized parent has a different active topology")
    if extra:
        pinned = earlier["config"]["lineage_parent"]
        simpler.append(pinned)
    starts = (
        parent["starts"]
        if extra
        else [min(parent["starts"], key=lambda entry: entry["score"])]
    )
    best_parent = min(
        [*starts, *simpler],
        key=lambda r: max(fixed_scores(model, r, parent["active"], 32)),
    )
    best_child = min(child["starts"], key=lambda v: v["score"])
    return source, deleted, parent, child, best_parent, best_child


def main():
    """Use identical projected-parent and full-child donors in each topology."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", dest="number", type=int, default=1)
    parser.add_argument("--calls", type=int, default=240)
    parser.add_argument(
        "--extra", help="Round-two finalist from its completed paired fits"
    )
    args = parser.parse_args()
    if args.extra and args.number < 2:
        raise ValueError("Use the first-round deletion control for round one")
    model = Model(BANK)
    source, deleted, parent, child, best_parent, best_child = controls(
        model, args.number, args.extra
    )
    donors = [
        ("strongest simpler fit", best_parent),
        ("best continued child", best_child),
    ]
    config = dict(
        sub=32,
        verify_sub=64,
        calls=args.calls,
        source=source,
        resolution=[
            "1.875s forward",
            "1.875s reverse",
            "0.9375s forward",
            "0.9375s reverse",
        ],
    )
    if args.extra:
        config["donors"] = {
            name: {key: fit[key] for key in ("x", "dose")}
            for name, fit in donors
        }
        config["lineage_parent"] = dict(
            ids=parent["active"], x=best_parent["x"], dose=best_parent["dose"]
        )
    audit_control = [
        max(fixed_scores(model, best_parent, parent["active"], 32)),
        *fixed_scores(model, best_parent, parent["active"], 64),
    ]
    records = []
    destination = f"ad_adjudicated_{args.number:02d}.json"
    pending = {}
    if (HERE / destination).exists():
        saved = json.loads((HERE / destination).read_text())
        if (
            saved["config"] != config
            or saved["control_scores"] != audit_control
        ):
            raise ValueError(
                "Existing adjudication has different controls or budget"
            )
        records, pending = saved["results"], saved.get("pending", {})
    for row in (parent, child):
        if any(group["extra"] == row["extra"] for group in records):
            continue
        starts = (
            pending["starts"] if pending.get("extra") == row["extra"] else []
        )
        for i, (name, donor) in enumerate(donors):
            if i < len(starts):
                continue
            x = np.asarray(donor["x"]).copy()
            if not row["extra"]:
                x[16 + np.asarray(deleted)] = 0.0
            fit = fit_both_orders(
                model,
                x,
                np.array(donor["dose"]),
                row["active"],
                32,
                args.calls,
                64,
            )
            fit.update(
                start=name,
                audit_scores=fixed_scores(model, fit, row["active"], 128),
            )
            starts.append(fit)
            print(
                row["extra"],
                name,
                fit["fixed_scores"],
                fit["audit_scores"],
                flush=True,
            )
            save(
                destination,
                dict(
                    config=config,
                    results=records,
                    control_scores=audit_control,
                    pending=dict(
                        active=row["active"], extra=row["extra"], starts=starts
                    ),
                ),
            )
        records.append(
            dict(active=row["active"], extra=row["extra"], starts=starts)
        )
        save(
            destination,
            dict(config=config, results=records, control_scores=audit_control),
        )


if __name__ == "__main__":
    main()
