"""Compare two conditional evaporation models at new sizes for all four precursors."""

import argparse
import ctypes
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from internal_supply import ROOT, ref, simulate

SIZES = (60, 90, 120, 140, 165, 220)
SETTINGS = (
    "primary",
    "low_accommodation",
    "zero_curvature",
    "high_tension",
    "humid_water",
)
KINDS = ("evolving_D", "long_lived_fraction")
COLUMNS = ["log10_D0", "loss_sensitivity", "long_lived_fraction"]


def parameters(pilot=False):
    # Shared dense grid for all cases. Include a no-resistance control in both families.
    stride = 4 if pilot else 1
    ds = np.arange(-440, -299, stride) / 20
    gs = np.arange(0, 257, stride) / 16
    d, g = np.meshgrid(ds, gs)
    transport = np.column_stack((d.ravel(), g.ravel(), np.zeros(d.size)))
    # Broader exploratory grid checks sensitivity to the original fitted-parameter limits.
    broad_d, broad_g = np.meshgrid(np.arange(-120, -64) / 5, np.arange(97) / 4)
    outside = (broad_d < -22) | (broad_d > -15) | (broad_g > 16)
    broad = np.column_stack(
        (broad_d[outside], broad_g[outside], np.zeros(outside.sum()))
    )
    transport = np.vstack([transport, [np.inf, 0.0, 0.0], broad])
    frac = np.arange(0, 1281, stride) * 0.000625
    pool = np.column_stack(
        (np.full_like(frac, np.inf), np.zeros_like(frac), frac)
    )
    return transport, pool


class Solver:
    def __init__(self, lib):
        self.solve = ctypes.CDLL(str(lib.resolve())).solve
        F = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        I = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
        self.solve.argtypes = [
            F,
            F,
            F,
            ctypes.c_int,
            F,
            I,
            ctypes.c_int,
            ctypes.c_int,
            F,
        ]
        self.solve.restype = ctypes.c_int
        self.trajectories = 0

    def run(self, weights, name, diameter, times, params, steps, scenario):
        times = np.asarray(times)
        grid = np.unique(
            np.r_[0.0, np.geomspace(0.0001, max(times), steps), times]
        )
        events = np.full(len(grid) - 1, -1, np.int32)
        events[np.searchsorted(grid, times) - 1] = np.arange(len(times))
        assert np.all(times > 0)
        rho, water, organic = ref.particle_properties(name, "wet", scenario)
        r = diameter * 0.5e-9
        kr = (
            ref.par.particles.get_kelvin_radius(
                scenario.sigma, rho, ref.MW, ref.TEMP
            )
            if scenario.sigma
            else 0.0
        )
        conditions = np.asarray(
            [
                r,
                4 * np.pi * r**3 * rho / 3,
                water,
                organic,
                ref.MFP,
                ref.GAS_D,
                kr,
                scenario.alpha,
            ]
        )
        inputs = params.copy(order="C")
        inputs[:, 0] = 10.0 ** (-inputs[:, 0])
        values = np.empty((len(params), len(times)))
        err = self.solve(
            conditions,
            np.ascontiguousarray(weights),
            inputs,
            len(params),
            60 * np.diff(grid),
            events,
            len(events),
            len(times),
            values,
        )
        if err:
            raise RuntimeError(f"CUDA runtime error {err}")
        assert np.isfinite(values).all() and np.all(
            (values > 0) & (values < 1.0000001)
        )
        self.trajectories += len(values)
        return values


def run(lib, out, pilot):
    solver = Solver(lib)
    start = time.monotonic()
    fits = pd.read_csv(ROOT / "model/results/fits.csv")
    obs = pd.read_csv(ROOT / "sources/observations.csv").query("rh=='wet'")
    models = parameters(pilot)
    boxes, checks = [], []
    for name, group in obs.groupby("precursor"):
        measured = ref.SIZE[name]
        times = group.time_min.to_numpy()
        observed = group.vfr.to_numpy()
        family = ref.CYCLIC[name]
        for setting in ("primary",) if pilot else SETTINGS:
            sc = ref.SCENARIOS[setting]
            w = (
                fits.query(
                    "model==@setting and precursor==@family and rh=='wet'"
                )
                .sort_values("cstar_ug_m3")
                .fitted_fraction.to_numpy()
            )
            w /= w.sum()
            for kind, params in zip(KINDS, models):
                pred = solver.run(w, name, measured, times, params, 420, sc)
                # Verify independent Python/Particula integration at identical grid points.
                indices = np.unique(
                    np.linspace(0, len(params) - 1, 5).astype(int)
                )
                oracle = simulate(
                    w,
                    name,
                    measured,
                    times,
                    pd.DataFrame(params[indices], columns=COLUMNS),
                    420,
                    sc,
                    "wet",
                )
                shift = np.max(abs(oracle - pred[indices]))
                assert shift < 1e-9, (name, setting, kind, shift)
                day = np.sqrt(
                    np.mean(
                        (pred[:, times <= 300] - observed[times <= 300]) ** 2,
                        axis=1,
                    )
                )
                night = np.max(
                    abs(pred[:, times > 900] - observed[times > 900]), axis=1
                )
                logd, g, frac = params.T
                original = (
                    (logd >= -22) & (logd <= -15) & (g <= 16)
                ) | np.isinf(logd)
                if kind == "evolving_D":
                    coarse = (
                        np.isclose(
                            logd * 5,
                            np.round(np.where(np.isfinite(logd), logd, 0) * 5),
                        )
                        | np.isinf(logd)
                    ) & np.isclose(g * 4, np.round(g * 4))
                else:
                    coarse = np.isclose(frac * 200, np.round(frac * 200))
                ranges = []
                for resolution, domain in [
                    ("main", original),
                    ("coarse", original & coarse),
                    ("expanded", np.ones(len(params), bool)),
                ]:
                    best = day[domain].min()
                    for tolerance in (0.025, 0.05):
                        selected = (
                            domain & (day <= best + 0.01) & (night <= tolerance)
                        )
                        ranges.append((resolution, tolerance, selected, best))
                # Include near-threshold coarse settings even if its best differs slightly.
                IDs = np.flatnonzero(
                    np.logical_or.reduce([r[2] for r in ranges])
                )
                local = params[IDs]
                max_dt = 0.0
                for size in SIZES:
                    sample_times = [30.0, 200.0, 600.0, 1200.0]
                    trajectories = (
                        solver.run(w, name, size, sample_times, local, 500, sc)
                        if len(IDs)
                        else np.zeros((0, 4))
                    )
                    for resolution, tol, selected, best in ranges:
                        active = np.flatnonzero(selected[IDs])
                        for j, age in enumerate(sample_times):
                            vals = trajectories[:, j]
                            lo = hi = np.nan
                            if len(active):
                                extrema = active[
                                    [
                                        np.argmin(vals[active]),
                                        np.argmax(vals[active]),
                                    ]
                                ]
                                lo, hi = vals[extrema]
                                if (
                                    resolution == "main"
                                    and tol == 0.025
                                    and age == 1200.0
                                ):
                                    fine = solver.run(
                                        w,
                                        name,
                                        size,
                                        sample_times,
                                        local[extrema],
                                        1500,
                                        sc,
                                    )[:, j]
                                    max_dt = max(
                                        max_dt,
                                        float(np.max(abs(fine - [lo, hi]))),
                                    )
                            boxes.append(
                                (
                                    name,
                                    family,
                                    measured,
                                    setting,
                                    kind,
                                    resolution,
                                    tol,
                                    size,
                                    age,
                                    len(active),
                                    best,
                                    lo,
                                    hi,
                                )
                            )
                assert max_dt < 0.001, (name, setting, kind, max_dt)
                checks.append((name, setting, kind, len(params), shift, max_dt))
        print(name, solver.trajectories, flush=True)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        boxes,
        columns="precursor baseline native_nm setting mechanism grid night_tolerance diameter_nm time_min accepted best_day_rmse low high".split(),
    ).to_csv(out / "species_size_ranges.csv", index=False, float_format="%.9g")
    pd.DataFrame(
        checks,
        columns="precursor setting mechanism candidates cpu_gpu_max_abs max_time_refinement".split(),
    ).to_csv(out / "species_size_checks.csv", index=False, float_format="%.9g")
    (out / "species_size_runtime.json").write_text(
        json.dumps(
            dict(
                trajectories=solver.trajectories,
                elapsed_seconds=time.monotonic() - start,
                pilot=pilot,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pilot", action="store_true")
    args = p.parse_args()
    run(args.library, args.out, args.pilot)
