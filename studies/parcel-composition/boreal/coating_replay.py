"""Locate signed absorption errors and replay locally refined BC checkpoints."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from campaign import audit, inventory, read
from coating_split import SHELL, refine
from metrics import EDGES
from model import step
from restart import activation
from stage_two import optical, reduce
from stage_two_design import REGIME, TIMES, design, initialize

CORE = np.array([50.0, 80.0, 120.0, 180.0, 250.0])
OUT = Path(__file__).parent
STARTS = (3600, 10800)


def cells(b, areas):
    """Sum additive optical area, BC volume, shell thickness and count."""
    keep = b.dry[:, 3] > 0
    v = b.dry[keep]
    d = np.cbrt(v.sum(1))
    c = np.cbrt(v[:, 3])
    shape = (len(EDGES[10]) + 1, len(CORE) + 1, len(SHELL) + 1)
    ix = np.ravel_multi_index(
        (
            np.searchsorted(EDGES[10], d, "right"),
            np.searchsorted(CORE, c, "right"),
            np.searchsorted(SHELL, d / c, "right"),
        ),
        shape,
    )
    return np.stack(
        [
            np.bincount(ix, weights=w, minlength=np.prod(shape)).reshape(shape)
            for w in (np.ones_like(d), v[:, 3], areas[keep, 0], (d - c) / 2)
        ],
        axis=-1,
    )


def compare(b, r, opt):
    """Resolve the contributions from BC inventory and specific absorption."""
    ar = opt.areas(r.dry, r.water)
    ab = opt.areas(b.dry, b.water)
    aa, af = ab[:, 0].sum(), ar[:, 0].sum()
    mb, mr = b.dry[:, 3].sum(), r.dry[:, 3].sum()
    # Signed terms sum exactly to the total relative error.
    mass = 100 * (mb / mr - 1)
    shape = 100 * (aa / af - mb / mr)
    rmse, sup = activation(r.dry, b.dry)
    # Condensed inventories measure the state of each organic class.
    cond = b.dry[:, 5:].sum(0) / r.dry[:, 5:].sum(0) - 1
    gas = (b.gas[5:] - r.gas[5:]) / np.maximum(
        r.gas[5:] + r.dry[:, 5:].sum(0), 1e-30
    )
    row = dict(
        ref_area=af,
        area=aa,
        signed_pct=mass + shape,
        bc_mass_pp=mass,
        efficiency_pp=shape,
        act_rmse=rmse,
        act_max=sup,
        wet_abs_pct=100 * (ab[:, 1].sum() / ar[:, 1].sum() - 1),
    )
    for j, name in enumerate(("svoc", "lvoc", "hom")):
        row[f"cond_{name}_pct"] = 100 * cond[j]
        row[f"gas_{name}_pp"] = 100 * gas[j]
    return row, ab, ar


def _reduce(b, method, budget, targets):
    if method == "repeat":
        return len(b.dry)
    if method == "baseline":
        return reduce(b, "rank")
    rule, extra = {"paired": ("local", 8), "paired4": ("local", 4)}.get(
        method, (method, budget)
    )
    return reduce(
        b,
        "rank",
        refine=lambda state, blocks: refine(
            state, blocks, rule, extra, targets
        ),
    )


def run(job):
    """Advance both restarts and retain each scored comparison."""
    root, dest, case, method, budget, targets, save_cells, optpath = job
    folder = root / case.key
    out = dest / case.key
    out.mkdir(exist_ok=True, parents=True)
    ident = f"{method}-{budget}"
    path = out / f"{ident}.csv"
    if path.exists():
        return
    opt = optical(optpath)
    _, _, par = initialize(case)
    scores = []
    frames = []
    trace = pd.read_csv(folder / "trace.csv").set_index("seconds")
    for start in STARTS:
        boxes, rng, _ = read(folder / f"{start}.npz")
        initial = inventory(boxes)
        source = np.zeros(8)
        base_born = sum(b.born for b in boxes)
        if method in ("repeat", "paired", "paired4"):
            rng = np.random.default_rng(
                np.random.SeedSequence([case.seed, start, 9000 + budget])
            )
        nrows = [_reduce(b, method, budget, targets) for b in boxes]
        peak = np.zeros(2)
        for end in range(start, 21601, case.dt):
            if end > start:
                source += step(boxes, rng, par, REGIME, end - case.dt, case.dt)
                peak = np.maximum(peak, [boxes[0].droplets, boxes[0].liquid])
            if end not in TIMES:
                continue
            full, _, _ = read(folder / f"{end}.npz")
            audit(boxes, initial, case.n, source)
            born = sum(b.born for b in boxes)
            ref_born = sum(b.born for b in full)
            survival = sum(b.reached for b in boxes) / born if born else np.nan
            ref_surv = (
                sum(b.reached for b in full) / ref_born if ref_born else np.nan
            )
            rp = trace.loc[
                start + case.dt : end, ["drops_cm3", "water_g_m3"]
            ].max()
            for i, (b, r) in enumerate(zip(boxes, full, strict=True)):
                row, ab, ar = compare(b, r, opt)
                scores.append(
                    dict(
                        **asdict(case),
                        method=method,
                        budget=budget,
                        start=start,
                        end=end,
                        parcel=i,
                        rows=nrows[i],
                        births=born - base_born,
                        ref_births=ref_born - base_born,
                        survival_pp=100 * (survival - ref_surv),
                        drops=peak[0],
                        ref_drops=rp.iloc[0],
                        lwc=peak[1],
                        ref_lwc=rp.iloc[1],
                        **row,
                    )
                )
                if save_cells:
                    obs, pred = cells(r, ar), cells(b, ab)
                    ix = np.where((obs[..., 0] + pred[..., 0]) > 0)
                    sub = pd.DataFrame(
                        np.c_[obs[ix], pred[ix]],
                        columns=[
                            f"{k}_{v}"
                            for k in ("ref", "pred")
                            for v in ("n", "bc", "area", "thickness_sum")
                        ],
                    )
                    sub = sub.assign(
                        start=start,
                        end=end,
                        parcel=i,
                        dry_bin=ix[0],
                        core_bin=ix[1],
                        shell_bin=ix[2],
                        signed_pp=100
                        * (pred[ix][:, 2] - obs[ix][:, 2])
                        / row["ref_area"],
                    )
                    np.testing.assert_allclose(
                        sub.signed_pp.sum(), row["signed_pct"], atol=1e-10
                    )
                    frames.append(sub)
    if save_cells:
        pd.concat(frames).to_csv(
            out / f"{ident}-cells.csv.gz", index=False, float_format="%.11g"
        )
    tmp = path.with_suffix(".tmp")
    pd.DataFrame(scores).to_csv(tmp, index=False, float_format="%.11g")
    tmp.replace(path)
    print(case.key, ident, flush=True)


def failures():
    """Identify the original six histories from the published scores."""
    names = ["seed", "acid", "exchange", "lift"]
    h = pd.read_csv(OUT / "stage-two-histories.csv")
    sel = h.query(
        "phase=='restart' & method=='rank' & n==25000 & dt==75 "
        "& stream==0 & dry_abs>5"
    )
    return set(sel[names].itertuples(index=False, name=None))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("dest", type=Path)
    p.add_argument("--methods", nargs="+", default=["baseline-0"])
    p.add_argument(
        "--scope", choices=["six", "others", "all", "controls"], default="six"
    )
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--target-file", type=Path)
    p.add_argument("--optics", default=".artifacts/boreal/optics.npz")
    a = p.parse_args()
    keys = failures()
    target = pd.read_csv(a.target_file) if a.target_file else None
    selected = [
        c
        for c in design()
        if (not c.primary if a.scope == "controls" else c.primary)
        and (a.scope != "six" or (c.seed, c.acid, c.exchange, c.lift) in keys)
        and (
            a.scope != "others"
            or (c.seed, c.acid, c.exchange, c.lift) not in keys
        )
    ]
    jobs = []
    for c in selected:
        for spec in a.methods:
            method, k = spec.split("-")
            jobs.append(
                (
                    a.root,
                    a.dest,
                    c,
                    method,
                    int(k),
                    target,
                    method in ("baseline", "shell")
                    and c.primary
                    and (c.seed, c.acid, c.exchange, c.lift) in keys,
                    a.optics,
                )
            )
    with ProcessPoolExecutor(a.workers) as pool:
        list(pool.map(run, jobs))
