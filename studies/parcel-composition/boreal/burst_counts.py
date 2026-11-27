"""Resolve the birth-count law in the first 75 s after the precursor pulse."""

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from campaign import audit, inventory
from model import step
from redraw_burst import SEEDS, START
from redraw_checkpoint import DEFAULT_RECIPE, draw_box, fit_box, load, serialize
from redraw_design import DT, FORCING
from stage_two_design import REGIME, Case, initialize

FRESH = (12007, 12109, 12227, 12347, 12457, 12569, 12689, 12799)
N = 50000
DRAWS = 192
FIVE_SECOND_DRAWS = 32
SPEC = DEFAULT_RECIPE
FINE = (10723, FRESH[0])


class BirthStream:
    """Record the event-generator state at each formation draw."""

    def __init__(self, rng, isolated=None):
        """Optionally route formation through its own generator."""
        self.rng = rng
        self.positions = []
        self.isolated = isolated

    def __getattr__(self, name):
        """Forward all other random events without changing their draws."""
        return getattr(self.rng, name)

    def poisson(self, mean):
        """Distinguish scalar formation draws from vector collision draws."""
        if np.ndim(mean) == 0:
            self.positions.append(self.rng.bit_generator.state.copy())
            if self.isolated is not None:
                return self.isolated.poisson(mean)
        return self.rng.poisson(mean)


def advance(boxes, rng, forcing, dt, isolated=None):
    """Record count and integrated formation rate in each parcel."""
    stream = BirthStream(rng, isolated)
    before = np.array([b.born for b in boxes])
    expected = np.zeros(2)
    for t in range(START, START + DT, dt):
        step(boxes, stream, forcing, REGIME, t, dt)
        expected += [b.jrate * dt * forcing["volume"] for b in boxes]
    return (
        np.array([b.born for b in boxes]) - before,
        expected,
        stream.positions,
    )


def history(job):
    """Compare paired and independent repeats from one onset checkpoint."""
    seed, levels, folder, draws, steps = job
    case = Case(seed, *levels, n=N, dt=DT)
    boxes, saved, forcing = initialize(case)
    initial = inventory(boxes)
    source = np.zeros(8)
    for t in range(0, START, DT):
        source += step(boxes, saved, forcing, REGIME, t, DT)
    audit(boxes, initial, N, source)
    fields = [fit_box(b, *SPEC) for b in boxes]
    packed, _ = load(serialize(fields, case, forcing, START, saved, SPEC))
    rows = []
    for dt in steps:
        original, rate, _ = advance(
            copy.deepcopy(boxes), copy.deepcopy(saved), forcing, dt
        )
        for draw in range(draws):
            key = [seed, N, START, 7501, draw]
            if dt == 5:
                key.append(dt)
            event, sample, independent, birth = [
                np.random.default_rng([*key, tag]) for tag in range(4)
            ]
            new = [draw_box(b, sample) for b in packed]
            full, lam, loc = advance(
                copy.deepcopy(boxes), copy.deepcopy(event), forcing, dt
            )
            reduced, mu, pos = advance(
                copy.deepcopy(new), copy.deepcopy(event), forcing, dt
            )
            repeat, other_rate, _ = advance(
                copy.deepcopy(boxes), independent, forcing, dt
            )
            if dt == DT:
                np.testing.assert_array_equal(lam, rate)
                np.testing.assert_array_equal(mu, rate)
                np.testing.assert_array_equal(other_rate, rate)
            # Isolate formation random numbers from random consumption by
            # coagulation. A common dedicated generator drives both layouts.
            if dt == DT and seed == 10723:
                a, _, _ = advance(
                    copy.deepcopy(boxes),
                    copy.deepcopy(event),
                    forcing,
                    dt,
                    copy.deepcopy(birth),
                )
                b, _, _ = advance(new, event, forcing, dt, birth)
                np.testing.assert_array_equal(a, b)
            active = np.flatnonzero(rate > 0)
            for i in range(2):
                same = None
                if dt == DT and i in active:
                    k = list(active).index(i)
                    np.testing.assert_equal(len(pos), len(active))
                    np.testing.assert_equal(len(loc), len(active))
                    same = loc[k] == pos[k]
                rows.append(
                    dict(
                        seed=seed,
                        dt=dt,
                        draw=draw,
                        parcel=i,
                        original=original[i],
                        full=full[i],
                        independent=repeat[i],
                        redraw=reduced[i],
                        full_mean=lam[i],
                        redraw_mean=mu[i],
                        same_rng=same,
                    )
                )
    dest = folder / f"{seed}.csv"
    pd.DataFrame(rows).to_csv(dest, index=False, float_format="%.15g")
    print(f"Completed {seed}", flush=True)


def main():
    """Generate all primary and time-subdivision trials."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--five-second", action="store_true")
    args = p.parse_args()
    args.folder.mkdir(parents=True, exist_ok=True)
    seeds = SEEDS + FRESH
    jobs = [
        (
            s,
            f,
            args.folder,
            FIVE_SECOND_DRAWS if args.five_second else DRAWS,
            (5,) if args.five_second else ((DT, 25) if s in FINE else (DT,)),
        )
        for s, f in zip(seeds, FORCING * 4, strict=True)
    ]
    if args.smoke:
        history(
            (
                10723,
                FORCING[1],
                args.folder,
                2,
                (5,) if args.five_second else (DT, 25),
            )
        )
    else:
        with ProcessPoolExecutor(args.workers) as pool:
            list(pool.map(history, jobs))


if __name__ == "__main__":
    main()
