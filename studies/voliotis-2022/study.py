# %% [markdown]
# Fit the five-hour suspended-organic trajectory in Manchester run 4, freeze
# the fit, and transfer it to runs 5, 1 and 6 using each run's gas forcing.
# %%
"""Calibrate a reduced mechanism and test transfer across chamber runs."""

import argparse
import dataclasses
import itertools
import json
import platform
import subprocess
import time
from importlib import metadata
from pathlib import Path
from shutil import which

import numpy as np
import pandas as pd
import particula
import scipy
import warp as wp
from chamber import RUNS, Case, replay
from scipy.optimize import isotonic_regression

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
DATA = pd.read_csv(HERE / "data/observations.csv")
HORIZON = 18000
METRICS = [
    "soa",
    "vapor",
    "vapor_to_wall",
    "oa_to_wall",
    "precursor_to_wall",
    "precursor",
]
SETTINGS = ["cstar", "accommodation", "seed_access"]


def series(run, kind):
    """Return original timestamps and values for one observable."""
    values = DATA[(DATA.run == run) & (DATA.kind == kind)].sort_values("time_s")
    return values.time_s.to_numpy(), values.value.to_numpy()


def forcing_knots():
    """Build monotone precursor histories at the nominal GC-MS cadence."""
    knots = np.arange(0, HORIZON + 1, 1200)
    forcing = {}
    for run in RUNS:
        timestamps, measured = series(run, "apinene_fraction")
        residual = np.interp(knots, timestamps, measured)
        residual[0] = 1
        residual = isotonic_regression(
            np.clip(residual, 1e-6, 1), increasing=False
        ).x
        forcing[run] = (knots, residual)
    return forcing


FORCING = forcing_knots()


def metrics(t, y, run):
    """Compare model and AMS at the original sampling times."""
    ts, obs = series(run, "soa_ug_m3")
    good = (ts >= 0) & (ts <= t[-1])
    obs = obs[good]
    pred = np.interp(ts[good], t, y)
    diff = pred - obs
    return dict(
        rmse=float(np.mean(diff**2) ** 0.5),
        mae=float(np.mean(abs(diff))),
        bias=float(np.mean(diff)),
        obs_peak=float(obs.max()),
        model_peak=float(y.max()),
        model_peak_h=float(t[np.argmax(y)] / 3600),
        obs_peak_h=float(ts[good][np.argmax(obs)] / 3600),
        n=int(len(obs)),
    )


def training():
    """Fit run 4 alone and freeze the lowest-RMSE parameter set."""
    fractions = np.arange(0.1, 0.71, 0.05)
    cases = [
        Case(4, f, v, p, lag)
        for f, v, p, lag in itertools.product(
            fractions, (0.0, 0.6, 1.8), (0.12, 0.24, 0.42), (5.0, 25.0)
        )
    ]
    rows = []
    groups = itertools.product((1.0, 10.0, 100.0), (0.03, 0.3))
    for cstar, accommodation in groups:
        start = time.monotonic()
        t, hist, checks = replay(
            cases, FORCING, cstar=cstar, accommodation=accommodation
        )
        for i, case in enumerate(cases):
            rows.append(
                dict(
                    **dataclasses.asdict(case),
                    cstar=cstar,
                    accommodation=accommodation,
                    seed_access=1.0,
                    **metrics(t, hist[:, i, 0], 4),
                    **checks,
                )
            )
        print(
            "Training group",
            cstar,
            accommodation,
            round(time.monotonic() - start, 1),
            "s",
            flush=True,
        )
    frame = pd.DataFrame(rows).sort_values("rmse", kind="stable")
    frame.to_csv(OUT / "run4_calibration.csv", index=False)
    frame.iloc[0].to_json(
        OUT / "frozen_run4_fit.json", indent=2, double_precision=15
    )


def case_from(row, run):
    """Transfer the calibrated parameters to a specified chamber run."""
    return Case(
        run,
        **{
            key: float(row[key])
            for key in (
                "fraction",
                "vapor_wall_h",
                "particle_wall_h",
                "lag_min",
            )
        },
    )


def settings(row):
    """Read the three common condensation settings."""
    return {key: float(row[key]) for key in SETTINGS}


def evaluate():
    """Replay the frozen fit and numerical and physical controls."""
    fit = json.loads((OUT / "frozen_run4_fit.json").read_text())
    params = settings(fit)
    fitted = [case_from(fit, run) for run in (4, 5, 1, 6)]
    scores, traces = [], []

    def batch(label, cases, *, forcing=FORCING, **opts):
        config = params | opts
        t, h, checks = replay(cases, forcing, **config)
        for i, c in enumerate(cases):
            scores.append(
                dict(
                    test=label,
                    **dataclasses.asdict(c),
                    **config,
                    **checks,
                    **metrics(t, h[:, i, 0], c.run),
                )
            )
            for j, secs in enumerate(t):
                traces.append(
                    dict(
                        test=label,
                        run=c.run,
                        time_s=secs,
                        **dict(zip(METRICS, h[j, i], strict=True)),
                    )
                )
        return h

    base = batch("reference", fitted)
    pair = fitted[:2]
    batch(
        "no_vapor_wall",
        [dataclasses.replace(c, vapor_wall_h=0.0) for c in pair],
    )
    batch(
        "no_particle_wall",
        [dataclasses.replace(c, particle_wall_h=0.0) for c in pair],
    )
    batch(
        "precursor_wall",
        [dataclasses.replace(c, precursor_wall_s=2.24e-5) for c in pair],
    )
    batch("lower_seed_absorption", pair, seed_access=0.1)
    batch(
        "same_seed_number",
        [
            fitted[0],
            dataclasses.replace(
                fitted[1], number_factor=RUNS[4][2] / RUNS[5][2]
            ),
        ],
    )
    # Run 1 has both counters. Transfer the lights-on WCPC:DMPS ratio only as
    # an instrument sensitivity because the count ratio can vary with size.
    batch(
        "counter_ratio_run1",
        [dataclasses.replace(fitted[1], number_factor=10500 / 20699.1)],
    )
    same_gas = FORCING | {5: FORCING[4]}
    batch("same_gas", [fitted[1]], forcing=same_gas)
    batch(
        "same_number_and_gas",
        [dataclasses.replace(fitted[1], number_factor=RUNS[4][2] / RUNS[5][2])],
        forcing=same_gas,
    )
    for factor in (0.85, 1.15):
        batch(
            f"injection_{factor}",
            [dataclasses.replace(c, injection=factor) for c in pair],
        )
    batch("size_5", pair, slots=5)
    batch("size_11", pair, slots=11)
    refined = batch("refined", pair, dt=6, interval=180)
    fine = batch("fine", pair, dt=3, interval=180)
    cpu = batch("cpu", pair, device="numpy")
    pd.DataFrame(scores).to_csv(OUT / "comparisons.csv", index=False)
    pd.DataFrame(traces).to_csv(OUT / "trajectories.csv", index=False)
    numerical = dict(
        cpu_cuda_max_abs_ug_m3=float(abs(base[:, :2, 0] - cpu[:, :, 0]).max()),
        refinement_12s_to_6s_max_abs_ug_m3=float(
            abs(base[:, :2, 0] - refined[::2, :, 0]).max()
        ),
        refinement_6s_to_3s_max_abs_ug_m3=float(
            abs(refined[:, :, 0] - fine[:, :, 0]).max()
        ),
    )
    (OUT / "numerical_checks.json").write_text(
        json.dumps(numerical, indent=2) + "\n"
    )


def envelope():
    """Evaluate the near-best run-4 fits against run 5."""
    # This descriptive tolerance has no probability interpretation.
    grid = pd.read_csv(OUT / "run4_calibration.csv")
    selected = grid[grid.rmse <= grid.rmse.min() + 2].copy()
    rows = []
    for _, group in selected.groupby(SETTINGS):
        runs = [case_from(row, 5) for _, row in group.iterrows()]
        t, h, checks = replay(runs, FORCING, **settings(group.iloc[0]))
        for i, (_, row) in enumerate(group.iterrows()):
            rows.append(
                dict(
                    **dataclasses.asdict(runs[i]),
                    **settings(row),
                    run4_rmse=row.rmse,
                    **metrics(t, h[:, i, 0], 5),
                    **checks,
                )
            )
    pd.DataFrame(rows).to_csv(OUT / "run5_fit_envelope.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=["train", "evaluate", "envelope", "all"]
    )
    args = parser.parse_args()
    start = time.monotonic()
    for name, action in (
        ("train", training),
        ("evaluate", evaluate),
        ("envelope", envelope),
    ):
        if args.stage in (name, "all"):
            action()
    record = dict(
        stage=args.stage,
        seconds=time.monotonic() - start,
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        particula=particula.__version__,
        source=str(Path(particula.__file__).resolve()),
        warp=metadata.version("warp-lang"),
        gpu=wp.get_device("cuda:0").name,
        head=subprocess.check_output(  # noqa: S603 -- fixed Git metadata query.
            [which("git"), "rev-parse", "HEAD"], cwd=HERE, text=True
        ).strip(),
    )
    (OUT / f"runtime_{args.stage}.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
