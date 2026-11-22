"""Grow one nested network with paired refits and resolution controls."""

import argparse
import json
from dataclasses import asdict
from functools import partial
from time import perf_counter

import numpy as np
from baseline_replay import DOSES, HERE, OUT, Model, run, save
from proposals import bank_and_motifs
from scipy.optimize import linprog

BANK, MOTIFS = bank_and_motifs()
CAP = np.array([25.0 if r.b >= 0 else 5.0 for r in BANK])
LO = np.r_[
    [1 / 12] * 3, [0.0] * 6, 2.818, 0, 0, 0, 0, 0, 0, np.zeros(len(BANK))
]
HI = np.r_[[4.0] * 3, [2.0] * 6, 16.5515, 1, 300, 100, 1, 0.1, 5, CAP]


def initial():
    """Load the archived joint parent and pad new rates with zeros."""
    d = json.loads((OUT / "continued.json").read_text())["family_evidence"][
        "joint"
    ]
    x = np.zeros(16 + len(BANK))
    x[: len(d["fit"]["x"])] = d["fit"]["x"]
    return x, list(d["active"])


def score(model, x, ids, **kwargs):
    """Return the worst normalized mass or count error."""
    return float(run(model, x, ids, **kwargs)[0][0])


def screen(model, x, ids, sub=4):
    """Profile four positive rates and all 61 feasible doses per experiment."""
    rows = []
    additions = sorted(
        {tuple(sorted(set(m) - set(ids))) for m in MOTIFS} - {()}
    )
    for extra in additions:
        if len(ids) + len(extra) > 1000:
            continue
        active = sorted(ids + list(extra))
        starts = np.tile(x, (4, 1))
        rates = [0.01, 0.1, 0.5, 2.0]
        for h in extra:
            starts[:, 16 + h] = rates
        scores, errors, doses, _ = run(model, starts, active, sub=sub)
        best = int(np.argmin(scores))
        _, _, _, flux = run(
            model,
            starts[best],
            active,
            doses=doses[best, :, None],
            sub=sub,
            diagnostics=1,
        )
        # Confirm actual substrate supply in at least one campaign run.
        idx = np.argmin(scores)
        rows.append(
            dict(
                extra=extra,
                keys=[BANK[h].key for h in extra],
                rate=rates[idx],
                score=float(scores[idx]),
                mean_mass=float(errors[idx, :, 0].mean()),
                gross=float(flux[0][:, list(extra)].max(axis=0).min()),
            )
        )
    return sorted(
        rows, key=lambda row: (row["score"], row["mean_mass"], row["extra"])
    )


def refit(model, x, ids, slots, start, cycles, sub=4):
    """Use equal numbers of finite-difference probes and line-search calls."""
    replay = partial(run, model, sub=sub)
    ix = np.r_[np.arange(10), 12, 16 + np.array(slots)].astype(int)
    ix_live = np.array([j < 16 or j - 16 in ids for j in ix])
    calls = 0
    x = x.copy()
    # Same small shift in the shared starting chemistry for both nested fits.
    if start:
        x[:3] = np.clip(x[:3] * 1.01, LO[:3], HI[:3])
    # Anchor includes the unshifted incumbent in both feasible sets upstream.
    for iteration in range(cycles):
        _, err, dose, _ = replay(x, ids)
        calls += len(DOSES)
        ref = err[0, :, :2].ravel()
        span = HI[ix] - LO[ix]
        scale = np.minimum(span * 0.2, 0.18 * abs(x[ix]) + 0.012 * span) / (
            1 + iteration * 0.25
        )
        delta = (
            np.where(x[ix] + 1e-4 * scale <= HI[ix], 1.0, -1.0) * 1e-4 * scale
        )
        probes = np.tile(x, (len(ix), 1))
        probes[np.arange(len(ix)), ix] += delta * ix_live
        values = replay(probes, ids, doses=dose[0, :, None])[1][
            :, :, :2
        ].reshape(len(ix), -1)
        calls += len(ix)
        slopes = ((values - ref) / (delta / scale)[:, None]).T
        # A perturbation of the local profiled dose can offset shared chemistry.
        d = dose[0]
        ds = np.where(d > 1.1499, -1e-5, 1e-5)
        err_d = replay(x, ids, doses=(d + ds)[:, None])[1][0, :, :2]
        calls += 1
        sd = 0.025
        dose_grad = (
            np.repeat(np.eye(len(d)), 2, axis=0)
            * ((err_d - err[0, :, :2]) / ds[:, None]).ravel()[:, None]
            * sd
        )
        g = np.column_stack([slopes, dose_grad])
        lo = np.maximum(-1, np.r_[(LO[ix] - x[ix]) / scale, (0.85 - d) / sd])
        hi = np.minimum(1, np.r_[(HI[ix] - x[ix]) / scale, (1.15 - d) / sd])
        inert = np.max(abs(g), axis=0) < 1e-10
        lo[inert] = hi[inert] = 0
        sol = linprog(
            np.r_[np.zeros(len(lo)), 1],
            A_ub=np.c_[g, -np.ones(len(g))],
            b_ub=-ref,
            bounds=list(zip(lo, hi, strict=True)) + [(0, None)],
            method="highs",
        )
        if not sol.success:
            raise RuntimeError(sol.message)
        levels = np.array([0.0, 0.04, 0.15, 0.4, 0.75, 1.0])
        points = np.tile(x, (len(levels), 1))
        points[:, ix] = np.clip(
            x[ix] + np.outer(levels, sol.x[: len(ix)] * scale), LO[ix], HI[ix]
        )
        scores = replay(points, ids)[0]
        calls += len(points) * len(DOSES)
        x = points[np.argmin(scores)]
    return x, calls


def values(model, x, ids, sub=4):
    """Evaluate central and fine trajectories in both channel orders."""
    return [
        score(model, x, ids, sub=sub),
        score(model, x, ids, sub=32),
        score(model, x, ids, sub=32, reverse=True),
    ]


def matched(model, x, ids, extra, warm_rate, cycles, shared=(), sub=4):
    """Compare nested refits with matched starts and call budgets."""
    child_ids = sorted(ids + list(extra))
    models = {}
    for label, active in (("parent", ids), ("child", child_ids)):
        reps = []
        for restart in (0, 1):
            seed = x.copy()
            if restart:
                for h in shared:
                    seed[16 + h] = max(seed[16 + h], warm_rate)
            if label == "child" and restart:
                seed[16 + np.array(extra)] = warm_rate
            fitted, n = refit(
                model, seed, active, child_ids, restart, cycles, sub=sub
            )
            reps.append((score(model, fitted, active, sub=sub), fitted, n))
        win = min(reps, key=lambda a: a[0])
        models[label] = dict(
            x=win[1].tolist(),
            scores=values(model, win[1], active, sub=sub),
            starts=[r[0] for r in reps],
            per_start_calls=win[2],
        )
    p, c = models["parent"], models["child"]
    gain = min(a - b for a, b in zip(p["scores"], c["scores"], strict=True))
    numerical = max(
        abs(a[0] - v) for a in (p["scores"], c["scores"]) for v in a[1:]
    )
    variation = max(abs(v["starts"][0] - v["starts"][1]) for v in (p, c))
    accepted = gain > max(numerical, variation, 1e-10)
    return dict(
        extra=list(extra),
        parent=p,
        child=c,
        minimum_gain=gain,
        numerical=numerical,
        variation=variation,
        accepted=bool(accepted),
    )


def growth_seed(number, previous):
    """Keep the exact incumbent used when a subsequent screen began."""
    name = f"growth_seed_{number:02d}.json"
    path = HERE / name
    if path.exists():
        return json.loads(path.read_text())
    seed = {
        k: previous[k] for k in ("ids", "x", "dose", "selected_fit", "admitted")
    }
    save(name, seed)
    return seed


def grow(number=1, max_seconds=6900, shortlist=8, cycles=5, sub=4):
    """Screen and compare matched refits until the growth rule stops."""
    start = perf_counter()
    model = Model(BANK)
    x, ids = initial()
    save(
        "bank.json",
        [dict(asdict(r), id=i, upper=CAP[i]) for i, r in enumerate(BANK)],
    )
    rnd = number
    line = (
        []
        if rnd == 1
        else json.loads((HERE / "lineage.json").read_text())[: rnd - 1]
    )
    if rnd > 1:
        seed = growth_seed(rnd, line[-1]) if len(line) == rnd - 1 else {}
        if not seed.get("admitted"):
            raise ValueError(
                "A verified prior admission is required to grow the lineage"
            )
        ids = seed["ids"]
        x = np.array(seed["x"])

    scanned = screen(model, x, ids, sub=sub)
    save(f"screen_{rnd:02}.json", scanned)
    shortlist_rows = [s for s in scanned if s["gross"] > 1e-9][:shortlist]
    # Include the best mean-mass challenger despite ties in joint score.
    supplied = [s for s in scanned if s["gross"] > 1e-9]
    if supplied:
        mass_best = min(supplied, key=lambda r: r["mean_mass"])
        if mass_best not in shortlist_rows:
            shortlist_rows.append(mass_best)
    save(f"shortlist_{rnd:02}.json", shortlist_rows)
    controls = []
    for s in shortlist_rows:
        if perf_counter() - start >= max_seconds:
            break
        extra = list(s["extra"])
        child_ids = sorted(ids + extra)
        # An exact zero-rate child must reproduce all run-level diagnostics.
        parent_zero = run(model, x, ids, sub=sub)[1]
        child_zero = run(model, x, child_ids, sub=sub)[1]
        np.testing.assert_allclose(parent_zero, child_zero, rtol=0, atol=1e-10)
        pair = matched(model, x, ids, extra, s["rate"], cycles, sub=sub)
        controls.append(dict(keys=s["keys"], **pair))
        print(
            rnd,
            s["keys"],
            pair["minimum_gain"],
            pair["accepted"],
            flush=True,
        )
    save(
        f"native_round_{rnd:02d}.json",
        dict(
            round=rnd,
            training_seconds=60 / sub,
            screened=len(scanned),
            shortlisted=len(shortlist_rows),
            budget_exhausted=len(controls) != len(shortlist_rows),
            seconds_available=max_seconds,
            optimization_cycles=cycles,
            controls=controls,
            elapsed=perf_counter() - start,
        ),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float, default=6900)
    ap.add_argument("--round", type=int, default=1, dest="number")
    ap.add_argument("--sub", type=int, default=4)
    ap.add_argument("--shortlist", type=int, default=8)
    ap.add_argument("--cycles", type=int, default=5)
    args = ap.parse_args()
    grow(args.number, args.seconds, args.shortlist, args.cycles, args.sub)
