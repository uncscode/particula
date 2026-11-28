"""Repeat the additional paired configurations of the 44-case distribution screen."""

import argparse

from assimilation import run_case
from fit import HERE, fit
from native_spectrum import replay
from replay_fits import histories, ranked


def additional(run):
    for kind, knot, obj, cadence, limit in [
        ("standard", 0, "bins", 0.05, 42),
        ("collisions", 0.5, "bands", 0.05, 42),
        ("collisions", 0.5, "bins", 0.05, 42),
        ("joint", 0.5, "bins", 0.05, 42),
        ("joint", 0, "bins", 0.05, 36),
        ("environment", 0.5, "bins", 0.05, 36),
        ("joint", 0.125, "bins", 0.05, 36),
        ("joint", 0.5, "bins", 0.25, 36),
        ("joint", 0.5, "shape", 0.05, 48),
    ]:
        fit(run, kind=kind, knot=knot, obj=obj, cadence=cadence, iters=limit)
    for kind in ("joint", "standard"):
        fit(run, kind=kind, mesh=108, iters=35)
    for kind, objective, limit in [
        ("standard", "native", 38),
        ("collisions", "native", 38),
        ("smooth-pairs", "native", 65),
        ("reservoir", "native", 60),
        ("joint", "tail4", 55),
        ("joint", "tail16", 55),
    ]:
        fit(run, kind=kind, obj=objective, width=1, iters=limit)
    fit(run, kind="smooth-pairs", obj="native", width=1, iters=65, seed=1.0, tag="-start1")
    fit(run, obj="native", width=1, dt=0.025, iters=38)
    for mesh in (18, 108, 576):
        for gain in (0.25, 0.5, 1):
            run_case(run, gain, mesh)
    base = HERE / "final-fits" / f"{run}-18-2-0.05-joint-0.5-hold-bins-0.0" / "fit.json"
    fit(run, cadence=0.0, dt=0.0125, iters=75, source=base)
    fit(run, cadence=0.0, dt=0.0125, iters=75, source=base, shared_endpoints=True, tag="-matched")
    for kind, objective in (
        ("smooth-pairs", "native"),
        ("entry", "native"),
        ("standard", "native"),
        ("joint", "tail4"),
    ):
        replay(run, kind=kind, objective=objective)
    replay(run, mesh=1944, n=262144)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", choices=["forced", "unforced"])
    args = parser.parse_args()
    histories(args.run)
    ranked(args.run)
    additional(args.run)
