"""Test paired draws from one checkpoint across measurement operators."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from campaign import audit, inventory, read
from model import KAPPA, sat, step
from redraw_checkpoint import DEFAULT_RECIPE, fit_box, load, serialize
from redraw_run import (
    _checked_draw,
    errors,
    observe,
    optics,
    percentage,
    physical,
)
from response_course import ASCENT, CASES, SHORT
from response_optics import Scattering
from stage_two_design import REGIME, initialize

BUDGET = 192
LONG_BUDGET = 12
FAMILIES = {
    "activation": {"ccn_rmse_pp": 2, "ccn_max_pp": 5},
    "cloud": {"drop_pct": 10, "lwc_pct": 10},
    "chemistry": {"organic_pct": 5, "gas_pp": 1, "transfer_pct": 10},
    "absorption": {"dry_abs_pct": 5, "wet_abs_pct": 5},
    "scattering": {"dry_sca_pct": 5, "wet_sca_pct": 5},
    "counter": {"channel_pct": 10},
}


def observe_state(boxes, opt, scat, peaks, base, source):
    """Measure each parcel, including a narrow dry counter channel."""
    o = observe(boxes, opt, peaks, base, source)
    scattering = []
    channel = []
    for box in boxes:
        total = box.dry.sum(1)
        bc = box.dry[:, 3]
        channel.append(np.count_nonzero((total >= 150**3) & (total < 160**3)))
        scattering.append(
            [
                scat.area(*scat.coordinates(total, bc, water)).sum()
                for water in (0.0, box.water)
            ]
        )
    o["scattering"] = np.array(scattering)
    o["channel"] = np.array(channel)
    return o


def run(
    boxes, rng, forcing, start, duration, opt, scat, initial_obs=None, dt=75
):
    """Diagnose snapshot cloud state and track peaks on every step."""
    box = boxes[0]
    water_ratio = box.qt / sat(box.temperature)
    active = (box.dry.sum(1) + box.water > 1000**3) & (
        box.water > 1.05 * 49 * (box.dry @ KAPPA)
    )
    drops = active.sum() / forcing["volume"] if water_ratio >= 0.99 else 0.0
    peaks = np.array(
        [drops, box.water.sum() * np.pi / 6 * 1e-15 / forcing["volume"]]
    )
    base = dict(
        gas=sum(b.gas[5:] for b in boxes),
        born=sum(b.born for b in boxes),
        reached=sum(b.reached for b in boxes),
        pending=sum(b.pending.sum() for b in boxes),
    )
    source = np.zeros(8)
    if initial_obs is None:
        initial_obs = observe_state(boxes, opt, scat, peaks, base, source)
        initial_obs["current"] = peaks.copy()
    initial = inventory(boxes)
    if duration % dt:
        raise ValueError("Duration must end on an integration step")
    for lag in range(0, duration, dt):
        source += step(boxes, rng, forcing, REGIME, start + lag, dt)
        peaks = np.maximum(peaks, [boxes[0].droplets, boxes[0].liquid])
    audit(boxes, initial, 50000, source)
    return [initial_obs, observe_state(boxes, opt, scat, peaks, base, source)]


def measurements(o, volume):
    """Retain trial measurements for marginal distribution checks."""
    p = physical(o, volume)
    for i, k in enumerate(("a", "b")):
        p["sca_" + k], p["wet_sca_" + k] = o["scattering"][i] * 1e-6 / volume
        p["wet_abs_" + k] = o["area"][i, 1] * 1e-6 / volume
        p["channel_" + k] = o["channel"][i] / volume
    return p


@lru_cache(maxsize=2)
def tables(root):
    """Share both qualified optical tables between jobs."""
    return optics(str(root / "optics.npz")), Scattering(
        root / "response-scattering.npz"
    )


def worker(job):
    """Compare fresh pairs and independent repeats for one checkpoint."""
    root, out, c, start, n = job
    dest = out / f"{c.seed}-{start}.csv.gz"
    if dest.exists():
        return
    original, saved, stamp = read(
        root / "response-course" / c.key / f"{start}.npz"
    )
    if stamp != start:
        raise ValueError("Checkpoint time does not match forecast start")
    _, _, forcing = initialize(c)
    opt, scat = tables(root)
    spec = DEFAULT_RECIPE
    payload = serialize(
        [fit_box(b, *spec) for b in original], c, forcing, start, saved, spec
    )
    packed, header = load(payload)
    decoded_forcing = header["forcing"]
    for field in forcing:
        np.testing.assert_equal(decoded_forcing[field], forcing[field])
    duration = 1800 if start == ASCENT else 75
    cached = None
    rows = []
    for trial in range(n):
        stream = [20261127, c.seed, start, trial]
        event = np.random.default_rng([*stream, 0])
        ref = run(
            deepcopy(original),
            deepcopy(event),
            forcing,
            start,
            duration,
            opt,
            scat,
            cached,
        )
        cached = ref[0]
        redraw = _checked_draw(
            packed, np.random.default_rng([*stream, 1]), original
        )
        comparisons = {
            "redraw": run(
                redraw, event, decoded_forcing, start, duration, opt, scat
            )
        }
        if trial < 12:
            comparisons["repeat"] = run(
                deepcopy(original),
                np.random.default_rng([*stream, 2]),
                forcing,
                start,
                duration,
                opt,
                scat,
                cached,
            )
        for name, path in comparisons.items():
            for lag, o, r in zip((0, duration), path, ref, strict=True):
                err = errors(o, r)
                err["dry_sca_pct"], err["wet_sca_pct"] = percentage(
                    o["scattering"], r["scattering"]
                ).max(0)
                err["channel_pct"] = percentage(
                    o["channel"], r["channel"]
                ).max()
                row = dict(
                    seed=c.seed, start=start, lead=lag, trial=trial, method=name
                )
                for family, gates in FAMILIES.items():
                    row[family] = max(
                        err[k] / limit for k, limit in gates.items()
                    )
                row.update(err)
                row.update(measurements(o, forcing["volume"]))
                row.update(
                    {
                        "ref_" + k: v
                        for k, v in measurements(r, forcing["volume"]).items()
                    }
                )
                rows.append(row)
    tmp = dest.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(
        tmp, index=False, compression="gzip", float_format="%.12g"
    )
    tmp.replace(dest)
    print(c.seed, start, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    jobs = [
        (a.root, a.out, c, t, BUDGET if t in SHORT else LONG_BUDGET)
        for t in (*SHORT, ASCENT)
        for c in CASES
    ]
    if a.smoke:
        jobs = [(*j[:-1], 1) for j in (jobs[0], jobs[16], jobs[32])]
    with ProcessPoolExecutor(a.workers) as pool:
        list(pool.map(worker, jobs))
