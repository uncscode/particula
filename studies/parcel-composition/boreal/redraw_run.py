"""Forecast from randomized compact checkpoints and score four endpoint
families.
"""

import argparse
import copy
import json
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from campaign import audit, inventory, read
from metrics import critical
from model import KAPPA, SS, sat, step
from redraw_checkpoint import draw_box, fit_box, load, serialize
from redraw_design import CANDIDATES, DRAWS, LEADS, TEST, TRAIN, WINDOWS, cases
from stage_two_design import REGIME, initialize

from optics import Optics

GATES = {
    "chemistry": {"organic_pct": 5, "gas_pp": 1, "transfer_pct": 10},
    "cloud": {"ccn_rmse_pp": 2, "ccn_max_pp": 5, "drop_pct": 10, "lwc_pct": 10},
    "nucleation": {
        "birth_pct": 10,
        "reach_pp": 5,
        "new_count_pct": 10,
        "new_cdf_pp": 5,
    },
    "absorption": {"dry_abs_pct": 5, "wet_abs_pct": 5},
}
FAMILIES = tuple(GATES)


@lru_cache
def optics(path):
    """Share a qualified optical table between jobs within each process."""
    return Optics(path)


def observe(boxes, opt, peaks, base, source):
    """Record individual-parcel optics, activation and chemical inventories."""
    dry = np.stack([b.dry.sum(0) for b in boxes])
    gas = np.stack([b.gas for b in boxes])
    born = sum(b.born for b in boxes) - base["born"]
    reached = sum(b.reached for b in boxes) - base["reached"]
    young = [np.cbrt(b.dry[~b.old].sum(1)) for b in boxes]
    new_count = np.array([len(x) for x in young])
    under = np.array(
        [
            [np.count_nonzero(x <= cut) for cut in (2.0, 3.0, 5.0, 10.0)]
            for x in young
        ]
    )
    return dict(
        new_count=new_count,
        under=under,
        area=np.stack([opt.areas(b.dry, b.water).sum(0)[:2] for b in boxes]),
        sc=[np.sort(critical(b.dry)) for b in boxes],
        dry=dry,
        gas=gas,
        transfer=base["gas"] + source[5:] - gas[:, 5:].sum(0),
        births=born,
        reach=reached / (base["pending"] + born)
        if base["pending"] + born
        else 0.0,
        peaks=peaks.copy(),
        current=np.array([boxes[0].droplets, boxes[0].liquid]),
    )


def trajectory(boxes, rng, forcing, c, start, opt):
    """Advance a half-hour and check all conserved inventories at its end."""
    initial = inventory(boxes)
    base = dict(
        gas=sum(b.gas[5:] for b in boxes),
        born=sum(b.born for b in boxes),
        reached=sum(b.reached for b in boxes),
        pending=sum(b.pending.sum() for b in boxes),
    )
    source = np.zeros(8)
    peaks = np.zeros(2)
    b = boxes[0]
    wet = b.dry.sum(1) + b.water
    active = (wet > 1000.0**3) & (b.water > 1.05 * 49 * (b.dry @ KAPPA))
    drops = (
        active.sum() / forcing["volume"]
        if b.qt / sat(b.temperature) >= 0.99
        else 0.0
    )
    liquid = b.water.sum() * np.pi / 6 * 1e-15 / forcing["volume"]
    observed = observe(boxes, opt, np.array([drops, liquid]), base, source)
    observed["current"] = observed["peaks"].copy()
    out = [observed]
    for lead in range(c.dt, 1801, c.dt):
        source += step(boxes, rng, forcing, REGIME, start + lead - c.dt, c.dt)
        peaks = np.maximum(peaks, [boxes[0].droplets, boxes[0].liquid])
        if lead in LEADS:
            out.append(observe(boxes, opt, peaks, base, source))
    audit(boxes, initial, c.n, source)
    return out


def percentage(pred, ref):
    """Treat an unexpected nonzero event against zero as a gate failure."""
    p, r = np.broadcast_arrays(pred, ref)
    return 100 * np.divide(
        abs(p - r), r, out=np.where(p == r, 0.0, np.inf), where=r != 0
    )


def curve(a, b):
    """Channel RMSE and exact maximum over the original supersaturation
    range.
    """

    def diff(grid, side):
        return 100 * (
            np.searchsorted(b, grid, side) / len(b)
            - np.searchsorted(a, grid, side) / len(a)
        )

    channels = diff(SS, "right")
    jumps = np.unique(b[(b >= SS[0]) & (b <= SS[-1])])
    return np.sqrt(np.mean(channels**2)), max(
        abs(channels).max(),
        abs(diff(jumps, "right")).max(initial=0),
        abs(diff(jumps[jumps > SS[0]], "left")).max(initial=0),
    )


def mixture(samples):
    """Equal weight per realization even when stochastic parcel counts
    differ.
    """
    x = np.concatenate(samples)
    w = np.concatenate(
        [np.full(len(s), 1 / (len(samples) * len(s))) for s in samples]
    )
    order = np.argsort(x, kind="stable")
    return x[order], np.r_[0.0, w[order].cumsum()]


def ensemble(observations):
    """Average physical observables and the full CCN curves of ten draws."""
    result = []
    for batch in zip(*observations, strict=True):
        o = {
            k: np.mean([b[k] for b in batch], axis=0)
            for k in batch[0]
            if k != "sc"
        }
        o["mixtures"] = [mixture([b["sc"][i] for b in batch]) for i in range(2)]
        result.append(o)
    return result


def ensemble_ccn(o, r):
    """Score weighted mean CCN curves on settings and all interior jumps."""
    ref = r.get("mixtures")
    if ref is None:
        ref = [mixture([s]) for s in r["sc"]]
    errors = []
    for (a, aw), (b, bw) in zip(ref, o["mixtures"], strict=True):

        def diff(grid, side, a=a, aw=aw, b=b, bw=bw):
            return 100 * (
                bw[np.searchsorted(b, grid, side)]
                - aw[np.searchsorted(a, grid, side)]
            )

        ch = diff(SS, "right")
        jumps = np.unique(b[(b >= SS[0]) & (b <= SS[-1])])
        errors.append(
            [
                np.sqrt(np.mean(ch**2)),
                max(
                    abs(ch).max(),
                    abs(diff(jumps, "right")).max(initial=0),
                    abs(diff(jumps[jumps > SS[0]], "left")).max(initial=0),
                ),
            ]
        )
    return np.max(errors, axis=0)


def errors(o, r):
    """Return the worst parcel value for every predeclared error metric."""
    ccn = (
        ensemble_ccn(o, r)
        if "mixtures" in o
        else np.array(
            [curve(x, y) for x, y in zip(r["sc"], o["sc"], strict=True)]
        ).max(0)
    )
    optical = percentage(o["area"], r["area"]).max(0)
    peak = percentage(o["peaks"], r["peaks"])
    active = [0, 1, 5, 6, 7]
    chemgas = (
        100
        * abs(o["gas"][:, active] - r["gas"][:, active])
        / (r["dry"] + r["gas"])[:, active]
    )
    a = o["new_count"][:, None]
    b = r["new_count"][:, None]
    fa = np.divide(o["under"], a, out=np.zeros((2, 4)), where=a > 0)
    fb = np.divide(r["under"], b, out=np.zeros((2, 4)), where=b > 0)
    return dict(
        new_count_pct=percentage(a, b).max(),
        new_cdf_pp=100 * abs(fa - fb).max(),
        dry_abs_pct=optical[0],
        wet_abs_pct=optical[1],
        ccn_rmse_pp=ccn[0],
        ccn_max_pp=ccn[1],
        drop_pct=peak[0],
        lwc_pct=peak[1],
        birth_pct=percentage(o["births"], r["births"]).item(),
        reach_pp=100 * abs(o["reach"] - r["reach"]),
        organic_pct=percentage(o["dry"][:, 5:], r["dry"][:, 5:]).max(),
        transfer_pct=percentage(o["transfer"], r["transfer"]).max(),
        gas_pp=chemgas.max(),
    )


def passed(err):
    """Apply each family gate once to both parcels jointly."""
    return {
        f: all(err[k] <= limit for k, limit in req.items())
        for f, req in GATES.items()
    }


def physical(o, volume):
    """Retain selected trajectory observables in their physical units."""
    return dict(
        abs_a=o["area"][0, 0] * 1e-6 / volume,
        abs_b=o["area"][1, 0] * 1e-6 / volume,
        ccn_a=100 * np.searchsorted(o["sc"][0], 0.3, "right") / len(o["sc"][0]),
        ccn_b=100 * np.searchsorted(o["sc"][1], 0.3, "right") / len(o["sc"][1]),
        drops_cm3=o["current"][0],
        lwc_g_m3=o["current"][1],
        peak_drops=o["peaks"][0],
        births=o["births"],
        reach=100 * o["reach"],
        organic_transfer=o["transfer"].sum(),
    )


def recipe_name(spec):
    """Identify a grid, row cap and within-row density."""
    return "-".join(map(str, spec))


def _checked_draw(data, redraw, original):
    boxes = [draw_box(b, redraw) for b in data]
    for b, r in zip(boxes, original, strict=True):
        np.testing.assert_allclose(b.dry.sum(0), r.dry.sum(0), rtol=2e-11)
        np.testing.assert_allclose(b.water.sum(), r.water.sum(), rtol=2e-11)
        np.testing.assert_equal(
            [len(b.old), b.old.sum(), b.pending.sum()],
            [len(r.old), r.old.sum(), r.pending.sum()],
        )
    return boxes


def _save_ensembles(batches, reference, c, start, out):
    ensemble_rows = []
    averaged = {name: ensemble(sample) for name, sample in batches.items()}
    for name, sample in averaged.items():
        for comparison, ref in [
            ("original", reference),
            ("full ensemble", averaged["full"]),
        ]:
            for lead, o, r in zip(LEADS, sample, ref, strict=True):
                err = errors(o, r)
                ensemble_rows.append(
                    dict(
                        seed=c.seed,
                        n=c.n,
                        start=start,
                        draw=0,
                        recipe=name,
                        lead=lead,
                        comparison=comparison,
                        **err,
                        **passed(err),
                    )
                )
    pd.DataFrame(ensemble_rows).to_csv(
        out / f"{start}-ensemble.csv", index=False, float_format="%.10g"
    )


def worker(job):
    """Evaluate all redraws at one checkpoint with common downstream streams."""
    root, out, c, start, recipes, table = job
    out = out / c.key
    out.mkdir(exist_ok=True, parents=True)
    dest = out / f"{start}.csv.gz"
    if dest.exists():
        return
    original, saved, _ = read(root / c.key / f"{start}.npz")
    _, _, forcing = initialize(c)
    opt = optics(table)
    reference = trajectory(
        copy.deepcopy(original), copy.deepcopy(saved), forcing, c, start, opt
    )
    full_bytes = len(
        serialize([vars(b) for b in original], c, forcing, start, saved, None)
    )
    enc = {}
    for spec in recipes:
        packed = [fit_box(b, *spec) for b in original]
        payload = serialize(packed, c, forcing, start, saved, spec)
        name = recipe_name(spec)
        if c.seed == TEST[0] and c.n == 10000 and start == 10800:
            (out / f"{name}.npz").write_bytes(payload)
        recovered, _ = load(payload)
        enc[name] = (recovered, len(payload))
    rows = []
    batches = (
        {k: [] for k in ["full", *enc]}
        if out.parent.name in ("test", "fine")
        else {}
    )
    for draw in range(DRAWS):
        seed = [c.seed, c.n, start, draw]
        event = np.random.default_rng(np.random.SeedSequence([*seed, 1]))
        full = trajectory(
            copy.deepcopy(original),
            copy.deepcopy(event),
            forcing,
            c,
            start,
            opt,
        )
        for name in ["full", *enc]:
            if name == "full":
                result = full
                size = full_bytes
                count = sum(len(b.dry) for b in original)
            else:
                data, size = enc[name]
                count = sum(len(b["count"]) for b in data)
                redraw = np.random.default_rng(
                    np.random.SeedSequence([*seed, 2])
                )
                boxes = _checked_draw(data, redraw, original)
                result = trajectory(
                    boxes, copy.deepcopy(event), forcing, c, start, opt
                )
            if batches:
                batches[name].append(result)
            for lead, o, r, f in zip(
                LEADS, result, reference, full, strict=True
            ):
                err = errors(o, r)
                paired = passed(errors(o, f))
                rows.append(
                    dict(
                        seed=c.seed,
                        n=c.n,
                        start=start,
                        draw=draw,
                        recipe=name,
                        lead=lead,
                        rows=count,
                        stored_bytes=size,
                        full_bytes=full_bytes,
                        particles=sum(len(b.dry) for b in original),
                        **err,
                        **passed(err),
                        **{f"paired_{k}": v for k, v in paired.items()},
                        **physical(o, forcing["volume"]),
                        **{
                            "ref_" + k: v
                            for k, v in physical(r, forcing["volume"]).items()
                        },
                    )
                )
    if batches:
        _save_ensembles(batches, reference, c, start, out)
    tmp = dest.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(
        tmp, index=False, compression="gzip", float_format="%.10g"
    )
    tmp.replace(dest)
    print(c.key, start, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("phase", choices=("train", "test"))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--selection", type=Path)
    p.add_argument("--optics", default=".artifacts/boreal/optics.npz")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    training = a.phase == "train"
    choices = (
        CANDIDATES
        if training
        else [tuple(json.loads(a.selection.read_text())["recipe"])]
    )
    jobs = [
        (a.root, a.root / a.phase, c, t, choices, a.optics)
        for c in cases(TRAIN if training else TEST)
        for t in WINDOWS
    ]
    with ProcessPoolExecutor(a.workers) as pool:
        list(pool.map(worker, jobs[:1] if a.smoke else jobs))
