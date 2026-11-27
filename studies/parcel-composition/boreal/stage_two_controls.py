"""Measure reference sensitivity to sampling, step length and event stream."""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from campaign import read
from model import step
from restart import _check_identity, activation, relative
from stage_two import cohort_sizes, optical, spectrum
from stage_two_design import REGIME, SEEDS, TIMES, Case, design, initialize


def verify_identity(root):
    """Re-evolve uncompressed checkpoints with their saved random state."""
    for seed in SEEDS:
        case = Case(seed, 0, 0, 0)
        folder = root / case.key
        _, _, forcing = initialize(case)
        for start in (3600, 10800):
            boxes, rng, _ = read(folder / f"{start}.npz")
            for t in range(start, 21600, case.dt):
                step(boxes, rng, forcing, REGIME, t, case.dt)
            full, reference_rng, _ = read(folder / "21600.npz")
            _check_identity(boxes, full, rng, reference_rng)


def compare(root, optpath):
    """Compare concentrations and size spectra under reference refinements."""
    opt = optical(optpath)
    rows = []
    for c in design():
        if c.primary:
            continue
        control = replace(c, n=25000, dt=75, stream=0)
        traces = [
            pd.read_csv(root / i.key / "trace.csv").set_index("seconds")
            for i in (control, c)
        ]
        kind = (
            "population"
            if c.n != control.n
            else ("step" if c.dt != control.dt else "stream")
        )
        for t in TIMES:
            ref, _, _ = read(root / control.key / f"{t}.npz")
            other, _, _ = read(root / c.key / f"{t}.npz")
            for j, (b, r) in enumerate(zip(other, ref, strict=True)):
                rmse, sup = activation(r.dry, b.dry)
                af = opt.areas(r.dry, r.water)[:, 0].sum() / control.n
                ab = opt.areas(b.dry, b.water)[:, 0].sum() / c.n
                f, s = (tr.loc[t] for tr in traces)
                pref, p = (tr.loc[:t, "drops_cm3"].max() for tr in traces)
                lref, liquid = (tr.loc[:t, "water_g_m3"].max() for tr in traces)
                rows.append(
                    dict(
                        kind=kind,
                        seed=c.seed,
                        acid=c.acid,
                        exchange=c.exchange,
                        lift=c.lift,
                        seconds=t,
                        parcel=j,
                        ref_born_cm3=f.born / (2 * control.n / 1000),
                        born_cm3=s.born / (2 * c.n / 1000),
                        survival_pp=100
                        * abs(s.reached / s.born - f.reached / f.born)
                        if s.born and f.born
                        else np.nan,
                        ref_drops=pref,
                        drops=p,
                        ref_lwc=lref,
                        lwc=liquid,
                        act_rmse=rmse,
                        act_max=sup,
                        dry_abs=relative(ab, af),
                        **spectrum(cohort_sizes(r), cohort_sizes(b)),
                    )
                )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--optics", default=".artifacts/boreal/optics.npz")
    args = parser.parse_args()
    verify_identity(args.root)
    compare(args.root, args.optics).to_csv(
        args.output, index=False, float_format="%.10g"
    )
