"""Follow target-particle activation after a dry intrusion."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from campaign import RESULTS, Case, build_cuda, call_gpu, prepare, unpack, write


def main():
    """Compare tagged counts before and after entrainment at six heights."""
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--refine", action="store_true")
    args = cli.parse_args()
    bins, nodes, dt = (64, 2049, 0.05) if args.refine else (48, 1025, 0.1)
    RESULTS.mkdir(parents=True, exist_ok=True)
    func = build_cuda()
    stages = [25, 49, 55, 65, 80, 100]
    cases = [
        Case(regime=r, environment=incoming, w=w, entr=e, gas=1.0, rh_ext=0.90)
        for r, incoming in [("marine", "urban"), ("urban", "marine")]
        for w in (0.2, 1.0)
        for e in (0.0, 0.15)
    ]
    if args.refine:
        cases = [c for c in cases if c.w == 0.2 and c.entr > 0]
    records = []
    population = []
    t0 = time.monotonic()
    for case in cases:
        for height in stages:
            bounds = {}
            for _ in range(3):
                prep = prepare(case, bins, nodes, dt, bounds)
                steps = round(height / case.w / dt)
                prep = (*prep[:3], np.ascontiguousarray(prep[3][:steps]))
                # The kernel archives 100 samples, so pad neither time nor air.
                # These heights at 1 m/s can give 250/490/550 steps; call_gpu
                # accepts a custom history length dividing the exact horizon.
                air, parts, _ = call_gpu(
                    func, [prep], nodes, steps, nh=int(np.gcd(100, steps))
                )
                res, groups, bounds = unpack(
                    case, prep, air[0], parts[0], bins, expected_height=height
                )
            identity = dict(
                resident=case.regime,
                incoming=case.environment,
                updraft_ms=case.w,
                entrained_fraction=case.entr,
                height_m=height,
                bins=bins,
                dt=dt,
            )
            records.append({**identity, **res})
            population.extend([{**identity, **r} for r in groups])
    prefix = RESULTS / ("milestones_refined" if args.refine else "milestones")
    write(prefix.with_suffix(".csv"), records)
    write(Path(str(prefix) + "_populations.csv"), population)
    prefix.with_suffix(".json").write_text(
        json.dumps(
            dict(cases=len(records), rounds=3, seconds=time.monotonic() - t0),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
