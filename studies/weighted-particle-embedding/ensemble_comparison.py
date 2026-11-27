"""Compare forecast errors and mixing-state signals on fixed ensembles."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from ensemble_manifest import checksum, load_frozen

OBSERVABLES = ("number_fraction", "mean_wet_diameter_um", "water_dry_ratio")


def group_scores(rows, native):
    """Use both mixtures at each family/concentration and pair native seeds."""
    values = {}
    member, persistent, sampling, individual = [], [], [], []
    for row in rows:
        key = (
            int(row["family"]),
            int(row["density_multiplier"]),
            int(row["mix"]),
        )
        if key in values:
            raise ValueError("Repeated stochastic evaluation case")
        values[key] = float(row["forecast"])
        member.append(abs(float(row["forecast_minus_mean"])))
        persistent.append(
            abs(float(row["persistence"]) - float(row["native_mean"]))
        )
        sampling.append(float(row["native_standard_error"]))
        individual.append(float(row["mae_over_realizations"]))
    first = rows[0]
    index = OBSERVABLES.index(first["observable"])
    sets = native[
        first["mode"], int(first["slots"]), int(first["native_steps"])
    ]
    signs, pair_error, signal = [], [], []
    keys = sorted({(f, d) for f, d, _ in values})
    for f, d in keys:
        a = sets[f, d, 0, first["leg"]]
        b = sets[f, d, 1, first["leg"]]
        if set(a) != set(b):
            raise ValueError("Paired ensembles require identical seed IDs")
        samples = np.array([b[i][index] - a[i][index] for i in sorted(a)])
        actual = samples.mean()
        se = samples.std(ddof=1) / len(samples) ** 0.5
        if abs(actual) > 2 * se:
            pred = values[f, d, 1] - values[f, d, 0]
            signs.append(bool(pred * actual > 0))
            pair_error.append(abs(pred - actual))
            signal.append(abs(actual))
    return dict(
        members=len(member),
        families=len({f for f, _, _ in values}),
        seeds=int(first["seeds"]),
        member_mae_vs_native_mean=np.mean(member),
        persistence_mae_vs_native_mean=np.mean(persistent),
        mean_native_standard_error=np.mean(sampling),
        mae_vs_realizations=np.mean(individual),
        pairs=len(keys),
        resolved_mixing_pairs=len(signs),
        resolved_sign_accuracy=np.mean(signs) if signs else None,
        paired_error_over_signal=sum(pair_error) / sum(signal)
        if signs
        else None,
    )


def summarize(root, manifest, files, out):
    """Verify common cases, native means and seeds across checkpoints."""
    import json

    config = json.loads(manifest.read_text())
    native = load_frozen(root, manifest, config["slots"])
    digest = checksum(manifest)
    reference = None
    scores = []
    for path in files:
        groups = defaultdict(list)
        cohort = {}
        with path.open() as stream:
            for row in csv.DictReader(stream):
                if row["reference_manifest_sha256"] != digest:
                    raise ValueError("Forecast used another native cohort")
                key = (
                    row["mode"],
                    int(row["slots"]),
                    row["leg"],
                    row["observable"],
                )
                groups[key].append(row)
                record = (
                    *key,
                    row["family"],
                    row["density_multiplier"],
                    row["mix"],
                )
                cohort[record] = (
                    row["native_mean"],
                    row["native_standard_error"],
                    row["seeds"],
                )
        if reference is not None and reference != cohort:
            raise ValueError(
                "Native means or initial conditions differ between models"
            )
        reference = cohort
        for (mode, k, leg, obs), rows in groups.items():
            scores.append(
                dict(
                    checkpoint=rows[0]["checkpoint"],
                    checkpoint_sha256=rows[0]["sha256"],
                    reference_manifest_sha256=digest,
                    mode=mode,
                    slots=k,
                    leg=leg,
                    observable=obs,
                    **group_scores(rows, native),
                )
            )
    with out.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=scores[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(scores)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("manifest", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("files", type=Path, nargs="+")
    summarize(**vars(p.parse_args()))
