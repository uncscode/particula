"""Compare parcel-specific dry bins and instrument cutoffs on matched draws."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from adaptive_grid import MODES, choose_edges, fixed_pinned_edges
from campaign import audit, inventory, read
from metrics import EDGES
from model import step
from redraw_checkpoint import DEFAULT_RECIPE, fit_box, load, serialize
from redraw_run import GATES, _checked_draw, errors, percentage
from response_course import ASCENT, CASES, SHORT
from response_pairs import FAMILIES, run, tables
from stage_two_design import REGIME, initialize

HERE = Path(__file__).resolve().parent
ARMS = ("fixed-pinned", *MODES)
SHORT_DRAWS = 192
BULK_DRAWS = 24
ASCENT_DRAWS = 6


def channel(boxes):
    """Count both parcels using identical lower-inclusive diameter cutoffs."""
    numbers = []
    for b in boxes:
        v = b.dry.sum(1)
        numbers.append(np.count_nonzero((v >= 150**3) & (v < 160**3)))
    return np.array(numbers)


def pack(boxes, saved, c, forcing, start, arm):
    """Charge actual serialized edges and verify bin inventories."""
    if arm == "fixed":
        edges = [EDGES[10]] * len(boxes)
    elif arm == "fixed-pinned":
        edges = [fixed_pinned_edges()] * len(boxes)
    else:
        edges = [choose_edges(b, arm) for b in boxes]
    data = [
        fit_box(b, size_edges=e if arm != "fixed" else None)
        for b, e in zip(boxes, edges, strict=True)
    ]
    recipe = (
        DEFAULT_RECIPE
        if arm == "fixed"
        else dict(base=DEFAULT_RECIPE, edges=arm)
    )
    payload = serialize(data, c, forcing, start, saved, recipe)
    data, header = load(payload)
    records = []
    for i, (b, e, rows) in enumerate(zip(boxes, edges, data, strict=True)):
        if arm != "fixed":
            np.testing.assert_array_equal(rows["dry_edges_nm"], e)
        d = np.cbrt(b.dry.sum(1))
        which = np.searchsorted(e, d, "right")
        labels = rows["label"][:, 0]
        counts = rows["count"]
        m = rows["mean"]
        for sign in (-1, 1):
            endpoint = m + sign * rows["delta"]
            low = np.r_[0.0, e][labels]
            high = np.r_[e, np.inf][labels]
            diam = np.cbrt(endpoint[:, :8].sum(1))
            if np.any(diam < low - 1e-10) or np.any(diam > high + 1e-10):
                raise ValueError("Redraw support crosses stored bin")
        np.testing.assert_array_equal(
            np.bincount(labels, counts, minlength=len(e) + 1),
            np.bincount(which, minlength=len(e) + 1),
        )
        for j in np.unique(which):
            idx = which == j
            chosen = labels == j
            inventory_ref = np.r_[b.dry[idx].sum(0), b.water[idx].sum()]
            np.testing.assert_allclose(
                counts[chosen] @ m[chosen], inventory_ref, rtol=2e-11
            )
        for k, edge in enumerate(e):
            records.append(
                dict(
                    parcel=i,
                    edge_index=k,
                    edge_nm=edge,
                    rows=len(m),
                    archive_bytes=len(payload),
                )
            )
    return data, header["forcing"], records


def family_errors(o, r):
    """Score the same bulk and cohort endpoints as the pre-existing test."""
    e = errors(o, r)
    e["dry_sca_pct"], e["wet_sca_pct"] = percentage(
        o["scattering"], r["scattering"]
    ).max(0)
    e["drop_pct"], e["lwc_pct"] = np.maximum(
        [e["drop_pct"], e["lwc_pct"]], percentage(o["current"], r["current"])
    )
    return {
        f: max(e[k] / limit for k, limit in gates.items())
        for f, gates in {**FAMILIES, "nucleation": GATES["nucleation"]}.items()
        if f != "counter"
    }


def archived_count(row, prefix, volume):
    """Convert the archived concentration back to its integer count."""
    return np.rint(
        row[[prefix + "channel_a", prefix + "channel_b"]].to_numpy(float)
        * volume
    ).astype(int)


def forecast(data, par, original, stream, start, duration, opt, scat, ref, n):
    """Run one redraw, saving full process scores on the designated subset."""
    boxes = _checked_draw(data, np.random.default_rng([*stream, 1]), original)
    event = np.random.default_rng([*stream, 0])
    saved_counts = channel(boxes)
    if ref is None:
        if duration != 75:
            raise ValueError("Reduced diagnostics support one step only")
        initial = inventory(boxes)
        source = step(boxes, event, par, REGIME, start, duration)
        audit(boxes, initial, n, source)
        scores = [{}, {}]
    else:
        path = run(boxes, event, par, start, duration, opt, scat)
        scores = [family_errors(o, r) for o, r in zip(path, ref, strict=True)]
    return [saved_counts, channel(boxes)], scores


def fixed_forecast(baselines, volume, full):
    """Read matching legacy counts and family scores from the original trial."""
    nums = [archived_count(row, "", volume) for row in baselines]
    fields = [f for f in (*FAMILIES, "nucleation") if f != "counter"]
    scores = [row[fields].to_dict() for row in baselines] if full else [{}, {}]
    return nums, scores


def worker(job):
    """Fit each grid once and forecast on paired streams."""
    root, out, c, start, smoke = job
    dest = out / f"{c.seed}-{start}-pairs.csv.gz"
    if dest.exists():
        return
    original, saved, t = read(root / "response-course" / c.key / f"{start}.npz")
    if t != start:
        raise ValueError("Wrong reference snapshot")
    _, _, forcing = initialize(c)
    opt, scat = tables(root)
    # The previous study saved the full and fixed counts on these exact streams.
    use = pd.read_csv(HERE / "response-pairs.csv.gz")
    fixed = use[
        (use.seed == c.seed) & (use.start == start) & (use.method == "redraw")
    ].set_index(["trial", "lead"])
    duration = 1800 if start == ASCENT else 75
    total = ASCENT_DRAWS if start == ASCENT else SHORT_DRAWS
    total = min(total, 1) if smoke else total
    stored = {}
    edge_records = []
    for arm in ("fixed", *ARMS):
        data, forcing_read, ed = pack(original, saved, c, forcing, t, arm)
        stored[arm] = (data, forcing_read)
        edge_records.extend(
            dict(seed=c.seed, start=t, method=arm, **v) for v in ed
        )
    results = []
    cached = None
    for trial in range(total):
        stream = [20261127, c.seed, start, trial]
        full = trial < BULK_DRAWS
        if full:
            ref = run(
                deepcopy(original),
                np.random.default_rng([*stream, 0]),
                forcing,
                start,
                duration,
                opt,
                scat,
                cached,
            )
            cached = ref[0]
        ref_path = ref if full else None
        baselines = [fixed.loc[trial, lead] for lead in (0, duration)]
        target = [
            archived_count(baseline, "ref_", forcing["volume"])
            for baseline in baselines
        ]
        if full:
            np.testing.assert_array_equal(
                target, [r["channel"] for r in ref_path]
            )
        for arm in ("fixed", *ARMS):
            if arm == "fixed":
                nums, scored = fixed_forecast(
                    baselines, forcing["volume"], full
                )
            else:
                data, par = stored[arm]
                nums, scored = forecast(
                    data,
                    par,
                    original,
                    stream,
                    t,
                    duration,
                    opt,
                    scat,
                    ref_path,
                    c.n,
                )
            for lead, num, truth, fields in zip(
                (0, duration), nums, target, scored, strict=True
            ):
                results.append(
                    dict(
                        seed=c.seed,
                        start=t,
                        trial=trial,
                        lead=lead,
                        method=arm,
                        a=num[0],
                        b=num[1],
                        full_a=truth[0],
                        full_b=truth[1],
                        counter_pct=percentage(num, truth).max(),
                        **fields,
                    )
                )
    pd.DataFrame(edge_records).to_csv(
        out / f"{c.seed}-{start}-edges.csv.gz", index=False
    )
    tmp = dest.with_suffix(".tmp")
    pd.DataFrame(results).to_csv(
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
        (a.root, a.out, c, t, a.smoke) for c in CASES for t in (*SHORT, ASCENT)
    ]
    with ProcessPoolExecutor(a.workers) as pool:
        list(pool.map(worker, jobs[:1] if a.smoke else jobs))
