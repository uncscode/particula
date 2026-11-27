"""Independent course validation and fixed-forcing particle-count sweep."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from redraw_checkpoint import DEFAULT_RECIPE
from redraw_design import DT, FORCING, WINDOWS, reference
from redraw_run import worker
from stage_two_design import Case

SEEDS = (9059, 9199, 9341, 9479)
COURSE_SIZES = (5000, 10000, 25000, 50000)
SCALE_SIZES = (1000, *COURSE_SIZES, 100000)
RECIPES = (DEFAULT_RECIPE, (10, 32, 1))


def course_cases():
    """Pair four new seeds and forcing levels across particle counts."""
    return [
        Case(seed, *levels, n=n, dt=DT)
        for n in COURSE_SIZES
        for seed, levels in zip(SEEDS, FORCING, strict=True)
    ]


def scale_cases():
    """Hold forcing, seed and concentration fixed when measuring size costs."""
    return [Case(SEEDS[0], *FORCING[0], n=n, dt=DT) for n in SCALE_SIZES]


def main():
    """Run the command-line study stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--optics", default=".artifacts/boreal/optics.npz")
    args = parser.parse_args()
    cases = course_cases()
    with ProcessPoolExecutor(args.workers) as pool:
        refs = {
            case: pool.submit(reference, (args.root, case))
            for case in dict.fromkeys([*cases, *scale_cases()])
        }
        forecasts = []
        for case in cases:
            refs[case].result()
            for start in WINDOWS:
                job = (
                    args.root,
                    args.root / "course",
                    case,
                    start,
                    RECIPES,
                    args.optics,
                )
                forecasts.append(pool.submit(worker, job))
        for future in [*refs.values(), *forecasts]:
            future.result()


if __name__ == "__main__":
    main()
