"""Screen the two established mechanisms against paired diameter and volume readings."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "transport"))
from four_precursors import (
    COLUMNS,
    KINDS,
    SETTINGS,
    SIZES,
    Solver,
    parameters,
    ref,
    simulate,
)


def compatible(pred, readings, mask, spectra_only=False):
    """Find one allowed initial modal diameter for the entire trajectory.

    For each sample, take the span of the Figure 4 marker-reading interval and
    the S4 interval conditional on that initial diameter. The lower and upper
    constraints on the shared denominator can then be intersected analytically.
    """
    subset = readings.iloc[np.flatnonzero(mask)]
    v = pred[:, mask]
    root = np.cbrt(v)
    lower = subset.dt_low.to_numpy() / root
    upper = subset.dt_high.to_numpy() / root
    if not spectra_only:
        lower = np.where(v < subset.figure_low.to_numpy(), lower, -np.inf)
        upper = np.where(v > subset.figure_high.to_numpy(), upper, np.inf)
    min_d = np.maximum(subset.d0_low.iloc[0], lower.max(axis=1))
    max_d = np.minimum(subset.d0_high.iloc[0], upper.min(axis=1))
    return min_d <= max_d


def run(lib, bounds, out):
    start = time.monotonic()
    solver = Solver(lib)
    fits = pd.read_csv(ROOT / "model/results/fits.csv")
    all_reads = pd.read_csv(bounds)
    models = parameters()
    rows = []
    scores = []
    held = []
    for name, species in all_reads.groupby("precursor"):
        paired = species.query(
            "status=='paired' and resolution=='half_bin'"
        ).sort_values("time_min")
        times = paired.spectrum_min.to_numpy()
        day = times <= 300
        night = times > 900
        both = day | night
        base = ref.CYCLIC[name]
        # Keep the exact parameter grid used for the previous main figure.
        for setting in SETTINGS:
            sc = ref.SCENARIOS[setting]
            f = (
                fits.query("precursor==@base and rh=='wet' and model==@setting")
                .sort_values("cstar_ug_m3")
                .fitted_fraction.to_numpy()
            )
            f /= f.sum()
            for kind, pars in zip(KINDS, models):
                pars = pars[
                    (
                        (pars[:, 0] >= -22)
                        & (pars[:, 0] <= -15)
                        & (pars[:, 1] <= 16)
                    )
                    | np.isinf(pars[:, 0])
                ]
                native = solver.run(
                    f, name, ref.SIZE[name], times, pars, 650, sc
                )
                ids = np.unique(np.linspace(0, len(pars) - 1, 4).astype(int))
                cp = simulate(
                    f,
                    name,
                    ref.SIZE[name],
                    times,
                    pd.DataFrame(pars[ids], columns=COLUMNS),
                    650,
                    sc,
                    "wet",
                )
                assert np.max(abs(cp - native[ids])) < 1e-9
                groups = {}
                for label, mode in [
                    ("pixels_only", "pixels_only"),
                    ("half_bin", "half_bin"),
                    ("full_bin", "full_bin"),
                    ("spectra_only", "half_bin"),
                ]:
                    reading = species.query(
                        "resolution==@mode and status=='paired'"
                    ).sort_values("time_min")
                    assert np.array_equal(times, reading.spectrum_min)
                    accepted_day = compatible(
                        native, reading, day, label == "spectra_only"
                    )
                    accepted = compatible(
                        native, reading, both, label == "spectra_only"
                    )
                    assert np.all(~accepted | accepted_day)
                    groups[label] = accepted
                    scores.append(
                        (
                            name,
                            setting,
                            kind,
                            label,
                            len(pars),
                            int(day.sum()),
                            int(night.sum()),
                            accepted_day.sum(),
                            accepted.sum(),
                        )
                    )
                primary = groups["half_bin"]
                excluded = species.query(
                    "resolution=='half_bin' and (status!='paired' or (spectrum_min>300 and spectrum_min<=900)) and time_min>5"
                )
                # Sensitivity: use the paired intermediate point as well. For omitted
                # peaks, transfer the largest paired daytime relative reading margins.
                all_times = compatible(
                    native, paired, np.ones(len(times), bool)
                )
                pday = paired.iloc[np.flatnonzero(day)]
                rel_lower = (pday.read_low / pday.observed).min()
                rel_upper = (pday.read_high / pday.observed).max()
                if len(excluded):
                    test = solver.run(
                        f,
                        name,
                        ref.SIZE[name],
                        excluded.time_min.to_numpy(),
                        pars,
                        650,
                        sc,
                    )
                    for j, point in enumerate(excluded.itertuples()):
                        if point.status != "paired":
                            all_times &= (
                                test[:, j] >= point.observed * rel_lower
                            ) & (test[:, j] <= point.observed * rel_upper)
                        if primary.any():
                            held.append(
                                (
                                    name,
                                    setting,
                                    kind,
                                    point.time_min,
                                    point.observed,
                                    point.status,
                                    test[primary, j].min(),
                                    test[primary, j].max(),
                                )
                            )
                assert np.all(~all_times | primary)
                groups["all_times"] = all_times
                scores.append(
                    (
                        name,
                        setting,
                        kind,
                        "all_times",
                        len(pars),
                        sum(day),
                        sum(night),
                        np.nan,
                        all_times.sum(),
                    )
                )
                chosen = np.flatnonzero(
                    np.logical_or.reduce(list(groups.values()))
                )
                for d in SIZES:
                    result = (
                        solver.run(
                            f,
                            name,
                            d,
                            [30.0, 200.0, 600.0, 1200.0],
                            pars[chosen],
                            650,
                            sc,
                        )
                        if len(chosen)
                        else np.empty((0, 4))
                    )
                    for label, accepted in groups.items():
                        vals = result[accepted[chosen]]
                        for j, t in enumerate((30, 200, 600, 1200)):
                            low = (
                                float(vals[:, j].min()) if len(vals) else np.nan
                            )
                            high = (
                                float(vals[:, j].max()) if len(vals) else np.nan
                            )
                            rows.append(
                                (
                                    name,
                                    setting,
                                    kind,
                                    label,
                                    d,
                                    t,
                                    len(vals),
                                    low,
                                    high,
                                )
                            )
        print(name, solver.trajectories, flush=True)
    out.mkdir(parents=True, exist_ok=True)
    for filename, data, columns in [
        (
            "model_ranges",
            rows,
            "precursor setting mechanism resolution diameter_nm time_min accepted low high",
        ),
        (
            "screen_counts",
            scores,
            "precursor setting mechanism resolution trials daytime_points night_points accepted_day accepted_joint",
        ),
        (
            "unscored_markers",
            held,
            "precursor setting mechanism time_min measured status low high",
        ),
    ]:
        pd.DataFrame(data, columns=columns.split()).to_csv(
            out / (filename + ".csv"), index=False, float_format="%.10g"
        )
    (out / "runtime.json").write_text(
        json.dumps(
            dict(
                cuda_trajectories=solver.trajectories,
                elapsed_s=time.monotonic() - start,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--library", type=Path, required=True)
    p.add_argument(
        "--bounds", type=Path, default=HERE / "results/read_bounds.csv"
    )
    p.add_argument("--out", type=Path, default=HERE / "results")
    a = p.parse_args()
    run(a.library, a.bounds, a.out)
