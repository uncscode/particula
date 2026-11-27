"""Verify qualification tables and summarize model horizons and uncertainty."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from family_intervals import intervals
from node_transfer import write_table
from projection_study import digest


def table(path):
    """Read a scored panel with its original header."""
    with path.open() as f:
        return list(csv.DictReader(f))


def verify(folder, manifest_sha):
    """Reject missing, unsealed or changed score tables."""
    done = json.loads((folder / "complete.json").read_text())
    if done["manifest_sha256"] != manifest_sha:
        raise ValueError("Scores belong to a different manifest")
    if set(done["sha256"]) != {
        str(f.relative_to(folder)) for f in folder.rglob("*.csv")
    }:
        raise ValueError("Different set of scored tables")
    for name, sha in done["sha256"].items():
        if (
            digest(folder / name) != sha
            or len(table(folder / name)) != done["rows"][name]
        ):
            raise ValueError(f"Changed score table {name}")
    archived = {str(p.relative_to(folder)) for p in folder.rglob("*.npz")}
    if archived != set(done.get("forecast_sha256", {})):
        raise ValueError("Different set of forecast and state archives")
    for name, sha in done.get("forecast_sha256", {}).items():
        if digest(folder / name) != sha:
            raise ValueError(f"Changed family forecasts {name}")
    return sum(done["rows"].values())


def neural_horizons(folder, domain):
    """Read contiguous qualified leads from the recorded scoring decisions."""
    if domain == "pulse":
        r = table(folder / "horizons.csv")
        groups = defaultdict(list)
        for x in r:
            groups[
                (x["checkpoint"], x["initial_frame"], x["observed_history"])
            ].append(x)
        for (model, start, observed), items in groups.items():
            yield dict(
                model=model,
                initial_frame=start,
                observed_history=observed,
                families=items[0]["families"],
                horizon_s=max(
                    [0.0]
                    + [
                        float(i["elapsed_s"])
                        for i in items
                        if i["contiguous_qualified"] == "True"
                    ]
                ),
                evaluated_leads=len(items),
            )
    else:
        for p in sorted(folder.glob("*.npz")):
            if not p.with_suffix(".csv").exists():
                continue
            r = [
                x
                for x in table(p.with_suffix(".csv"))
                if x.get("group") == "elapsed"
            ]
            if r:
                yield dict(
                    model=p.stem,
                    initial_frame=0,
                    observed_history=1,
                    families=r[0]["families"],
                    horizon_s=max(
                        [0.0]
                        + [
                            float(i["time_max_s"])
                            for i in r
                            if i["contiguous"] == "True"
                        ]
                    ),
                    evaluated_leads=len(r),
                )


def bootstrap(folder, names):
    """Read memberwise cold-start forecasts and retain each family cluster."""
    for name in names:
        path = folder / (name.removesuffix(".pt") + ".npz")
        with np.load(path) as a:
            bound = (
                a["pair_uncertainty"]
                if "pair_uncertainty" in a
                else a["uncertainty"].sum(1)
            )
            result = intervals(
                a["forecast_water"],
                a["native_water"],
                bound,
                a["time"],
                a["cluster"],
            )
            yield dict(
                model=name,
                checkpoint_sha256=str(a["checkpoint_sha256"]),
                **result,
            )


def summarize(manifest, scores, out):
    """Produce compact audited outputs from the complete frozen comparison."""
    out.mkdir(parents=True, exist_ok=True)
    m = json.loads(manifest.read_text())
    sha = digest(manifest)
    counts, horizons, errors = [], [], []
    for split in m["splits"]:
        for domain in m["groups"]:
            if split == "challenge" and domain == "stochastic":
                continue
            folder = scores / split / domain
            count = verify(folder, sha)
            counts.append(
                dict(
                    split=split,
                    domain=domain,
                    rows=count,
                    completion_sha256=digest(folder / "complete.json"),
                    manifest_sha256=sha,
                )
            )
            if domain in ("pulse", "multicycle"):
                horizons.extend(
                    dict(split=split, domain=domain, **r)
                    for r in neural_horizons(folder, domain)
                )
                errors.extend(
                    dict(split=split, domain=domain, **r)
                    for r in bootstrap(folder, m["forecasts"])
                )
    write_table(counts, out / "panel-inventory.csv")
    if horizons:
        write_table(horizons, out / "neural-horizons.csv")
        write_table(errors, out / "paired-family-intervals.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in ("manifest", "scores", "out"):
        p.add_argument(k, type=Path)
    summarize(**vars(p.parse_args()))
