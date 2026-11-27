"""Check transfer on independent schedules and shared pulse prefixes."""

import argparse
import csv
from pathlib import Path

import numpy as np
from multicycle_score import summarize


def sources(folder):
    """Map a native shard's complete family identities to its path."""
    result = {}
    for path in sorted(folder.glob("families-*.npz")):
        with np.load(path) as archive:
            group = tuple(archive["ids"])
        if group in result:
            raise ValueError("Repeated family cohort")
        result[group] = path
    return result


def check_prefix(short, long):
    """Compare every common native node, reservoir reset and observation."""
    a = sources(short / "data/multicycle/validation")
    b = sources(long / "data/multicycle/validation")
    if not a or set(a) != set(b):
        raise ValueError("Repeated-pulse cohorts differ")
    for group, path in a.items():
        with np.load(path) as left, np.load(b[group]) as right:
            for key in (
                "ids",
                "counts",
                "volume",
                "density",
                "steps",
                "families",
            ):
                np.testing.assert_array_equal(left[key], right[key])
            for key in ("time", "water", "gas", "force", "mass", "lengths"):
                x = left[key]
                np.testing.assert_array_equal(x, right[key][: len(x)])
            np.testing.assert_array_equal(
                left["levels"], right["levels"][:, : left["levels"].shape[1]]
            )
    return sum(map(len, a))


def horizon(t, pred, truth, bound):
    """Count all earlier resolved comparisons when reporting a horizon."""
    last = 0.0
    for i in range(1, len(t)):
        result = summarize(pred[:, :, i], truth[:, :, i], bound[:, :, i])
        if result["qualified"] is None:
            continue
        if not result["qualified"]:
            break
        last = float(t[i]) * 1000
    return last


def table(root, blocks, out):
    """Score pooled and per-schedule errors and test forecast prefixes."""
    rows = []
    previous = {}
    previous_folder = None
    for block in blocks:
        folder = root / "repeated" / f"b{block}"
        if previous_folder:
            check_prefix(previous_folder, folder)
        now = {}
        for path in sorted(
            (root / "recurrence-transfer" / f"b{block}").glob("*.npz")
        ):
            with np.load(path) as d:
                ids, clusters, time, pred, ref, error = (
                    d[k]
                    for k in (
                        "ids",
                        "cluster",
                        "time",
                        "forecast_water",
                        "native_water",
                        "uncertainty",
                    )
                )
                sha = d["checkpoint_sha256"].item()
                if sha in previous:
                    prior_ids, prior_t, prior_pred = previous[sha]
                    np.testing.assert_array_equal(ids, prior_ids)
                    np.testing.assert_allclose(
                        time[: len(prior_t)], prior_t, atol=1e-12, rtol=0
                    )
                    np.testing.assert_allclose(
                        pred[:, :, : len(prior_t)],
                        prior_pred,
                        atol=5e-10,
                        rtol=0,
                    )
                groups = np.unique(clusters)
                horizons = [
                    horizon(
                        time,
                        pred[clusters == g],
                        ref[clusters == g],
                        error[clusters == g],
                    )
                    for g in groups
                ]
                rows.append(
                    dict(
                        model=path.stem,
                        sha256=sha,
                        plateaus=block * 4,
                        families=len(ids),
                        schedule_patterns=len(groups),
                        forecast_interval_ms=time[-1] * 1000,
                        pooled_horizon_ms=horizon(time, pred, ref, error),
                        worst_schedule_ms=min(horizons),
                        median_schedule_ms=np.median(horizons),
                        member_mae_mg_m3=np.abs(pred - ref).mean() * 1e6,
                        terminal_member_mae_mg_m3=np.abs(
                            pred[:, :, -1] - ref[:, :, -1]
                        ).mean()
                        * 1e6,
                    )
                )
                now[sha] = (ids, time, pred)
        if not now or (previous and set(now) != set(previous)):
            raise ValueError(
                f"Incomplete matched model panel for block count {block}"
            )
        previous = now
        previous_folder = folder
    with out.open("w", newline="") as fp:
        wr = csv.DictWriter(fp, fieldnames=rows[0], lineterminator="\n")
        wr.writeheader()
        wr.writerows(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--blocks", type=int, nargs="+", default=[1, 2, 3])
    table(**vars(p.parse_args()))
