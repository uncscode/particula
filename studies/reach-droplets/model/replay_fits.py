"""Refit the four representative histories and the leading sectional families."""

import argparse
from assimilation import run_case
from fit import HERE, fit
from native_spectrum import replay


def histories(run):
    for split, limit in (("early", 45), ("hold", 38)):
        fit(run, split=split, obj="native", width=1, iters=limit)
        replay(run, split=split)
    run_case(run, 1, 576)


def ranked(run):
    fit(run, obj="native", width=1, iters=38)
    run_case(run, 1, 18)
    fit(run, cadence=0.0, iters=36)
    start = HERE / "final-fits" / f"{run}-18-2-0.05-joint-0.5-hold-bins-0.0" / "fit.json"
    fit(run, knot=-1, cadence=0.0, dt=0.0125, iters=75, source=start)
    dense = HERE / "final-fits" / f"{run}-18-2-0.0125-joint--1-hold-bins-0.0" / "fit.json"
    fit(run, knot=-1, cadence=0.0, dt=0.0125, iters=180, source=dense, tag="-continued")
    fit(run, kind="entry", obj="native", width=1, iters=60)
    fit(run, kind="transport-time", iters=48)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("group", choices=["histories", "ranked"])
    parser.add_argument("run", choices=["forced", "unforced"])
    args = parser.parse_args()
    {"histories": histories, "ranked": ranked}[args.group](args.run)
