"""Audit the archived fits and summarize particle, chamber and design results."""

import csv
import json
from collections import defaultdict
from itertools import product
from pathlib import Path
import numpy as np
from reference import OUT, ROOT, J, Q, ACID, NH3, T, dump_csv


def read(path):
    with (OUT / path).open() as stream:
        return list(csv.DictReader(stream))


def grouped(records, keys):
    groups = defaultdict(list)
    for row in records:
        groups[tuple(row[key] for key in keys)].append(row)
    return groups


def vals(rows, key):
    return np.array([float(row[key]) for row in rows])


def main():
    scores, heldout, sources = [], [], {}
    checks = {}
    for folder in ("fits", "joint"):
        for path in sorted((OUT / folder).glob("*.json")):
            job = json.loads(path.read_text())
            runs = job["runs"]
            completed = [r for r in runs if r["status"] > 0]
            if not completed:
                raise RuntimeError(f"No converged start in {path}")
            best = min(completed, key=lambda r: r["cost"])
            lines = np.array(job["test_lines"], dtype=int)
            observed = J[lines - 2]
            predkey = (
                "prediction" if folder == "fits" else "neutral_predictions"
            )
            pred = np.asarray(best[predkey])
            err = np.log(pred / observed)
            name = path.stem
            near = [r for r in completed if r["cost"] <= 1.02 * best["cost"]]
            scores.append(
                dict(
                    method=folder,
                    config=name,
                    starts=len(runs),
                    converged=len(completed),
                    within_two_percent=len(near),
                    best_cost=best["cost"],
                    best_all_cost=min(r["cost"] for r in runs),
                    gradient_error=job["gradient_error"],
                    n_test=len(lines),
                    median_error_factor=np.exp(np.median(abs(err)))
                    if len(err)
                    else "",
                    geometric_bias=np.exp(np.mean(err)) if len(err) else "",
                )
            )
            for k, line in enumerate(lines):
                heldout.append(
                    dict(
                        method=folder,
                        config=name,
                        line=line,
                        temperature_K=T[line - 2],
                        acid_cm3=ACID[line - 2] * 1e6,
                        ammonia_pptv=NH3[line - 2],
                        observed_J=observed[k],
                        predicted_J=pred[k],
                        binary_fraction=best["binary_fraction"][k],
                        near_min_J=min(r[predkey][k] for r in near),
                        near_max_J=max(r[predkey][k] for r in near),
                    )
                )
                if (
                    folder == "fits"
                    and "_low_" in name
                    and name.endswith("_linear_nominal_neutral_group")
                ):
                    sources[(str(line), name.split("_")[2])] = pred[k]
    dump_csv(OUT / "fit_scores.csv", scores)
    dump_csv(OUT / "heldout.csv", heldout)
    checks["fit_starts"] = sum(s["starts"] for s in scores)
    checks["fit_starts_converged"] = sum(s["converged"] for s in scores)
    checks["configurations_with_unfinished_start_below_best_converged"] = sum(
        s["best_all_cost"] < s["best_cost"] for s in scores
    )
    checks["max_relative_AD_directional_error"] = max(
        s["gradient_error"] for s in scores
    )
    assert checks["max_relative_AD_directional_error"] < 1e-6
    parts = read("cuda/particles.csv") + read("cuda_attraction/particles.csv")
    checks["sampled_trajectories"] = sum(int(r["tracks"]) for r in parts)
    checks["max_mean_number_closure_error"] = max(
        abs(
            float(r["survival10"])
            + sum(float(r[k]) for k in ("wall_loss", "dil_loss", "coag_loss"))
            - 1
        )
        for r in parts
    )
    assert all(
        0 <= float(r["survival10"]) <= float(r["survival32"]) <= 1
        for r in parts
    )
    assert all(
        float(r["survival10"]) == 1
        for r in parts
        if r["scenario"] == "no_losses"
    )
    oracle = {}
    for size, key in [(2.9, "survival32"), (9.7, "survival10")]:
        analytic = np.exp(
            -1800 * (0.001 * np.log(size / 1.4) + 9.6e-5 * (size - 1.4))
        )
        estimate = np.mean(
            vals([r for r in parts if r["scenario"] == "constant_oracle"], key)
        )
        oracle[key] = dict(expected=analytic, mean=estimate)
        assert abs(estimate / analytic - 1) < 0.01
    checks["constant_growth_oracle"] = oracle
    flux = []
    for (line, scenario), rows in grouped(parts, ("line", "scenario")).items():
        for size, k in [(3.2, "survival32"), (10.0, "survival10")]:
            s = vals(rows, k)
            for label in ["archived", "full", "assisted"]:
                source = (
                    J[int(line) - 2]
                    if label == "archived"
                    else sources.get((line, label))
                )
                if source is None:
                    continue
                flux.append(
                    dict(
                        line=line,
                        scenario=scenario,
                        source=label,
                        source_J17=source,
                        threshold_mobility_nm=size,
                        mean_survival=s.mean(),
                        sd_survival_across_seeds=s.std(ddof=1),
                        mean_J=source * s.mean(),
                        min_seed_J=source * s.min(),
                        max_seed_J=source * s.max(),
                        expected_count_cm3_per_hour=source * s.mean() * 3600,
                    )
                )
    dump_csv(OUT / "chamber_flux.csv", flux)
    pop = read("population.csv")
    summaries = []
    keys = (
        "line",
        "source",
        "coag",
        "dt_s",
        "particles_per_source_hour",
        "window_end_h",
    )
    for group, rows in grouped(pop, keys).items():
        rec = dict(zip(keys, group))
        for k in ("J32", "J10", "mean_N32_cm3", "mean_N10_cm3"):
            a = vals(rows, k)
            rec.update(
                {
                    k + "_mean": a.mean(),
                    k + "_seed_min": a.min(),
                    k + "_seed_max": a.max(),
                }
            )
        summaries.append(rec)
    dump_csv(OUT / "population_windows.csv", summaries)
    checks["population_runs"] = len(pop) // 3
    checks["maximum_population_relative_species_error"] = max(
        abs(vals(pop, "relative_species_error"))
    )
    observed = {
        tuple(r[k] for k in keys[2:]) + (r["seed"],): r
        for r in pop
        if r["line"] == "336" and r["source"] == "observed"
    }
    for r in pop:
        if r["source"] == "equal_total":
            target = observed[tuple(r[k] for k in keys[2:]) + (r["seed"],)]
            assert all(r[k] == target[k] for k in r if k != "source")
    pairs = list(
        csv.DictReader(
            (ROOT / "rate_sources/figure_s3_plotted_pairs.csv").open()
        )
    )
    obs, pred = [np.log10(vals(pairs, k)) for k in list(pairs[0])[1:]]
    checks["S3_log_R2"] = 1 - sum((obs - pred) ** 2) / sum(
        (obs - obs.mean()) ** 2
    )
    assert len(pairs) == 355 and round(checks["S3_log_R2"], 2) == 0.90
    assert len(pop) == 576 and len(parts) == 1152 and len(scores) == 80
    assert checks["maximum_population_relative_species_error"] < 1e-10
    assert checks["max_mean_number_closure_error"] < 1e-10
    (OUT / "numerical_checks.json").write_text(
        json.dumps(checks, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
