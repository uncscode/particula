"""Sequentially fit 40 product-aging motifs and matched chamber controls."""

import json

import numpy as np
from bank import BANK, CAP, DOSES, OUT, PARENTS, TRIALS, manifest
from chain_checks import starting
from scipy.optimize import linprog

from gpu import Model

OBJECTIVES = ("mass", "joint")
LOW = np.r_[
    [1 / 12] * 3, [0.0] * 6, 2.818, 0, 0, 0, 0, 0, 0, np.zeros(len(BANK))
]
HIGH = np.r_[[4.0] * 3, [2.0] * 6, 16.5515, 1, 300, 100, 1, 0.1, 5, CAP]
KEYS = [r.key for r in BANK]


def save(name, value):
    """Write an atomic JSON checkpoint."""
    path = OUT / name
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, allow_nan=False, indent=1) + "\n")
    tmp.replace(path)


def evaluate(model, obj, points, doses=DOSES, sub=4, flux=False):
    """Run only channels with nonzero rates in the batch."""
    a = np.atleast_2d(points)
    k = a[:, 16:]
    ids = np.flatnonzero(np.any(k > 0, axis=0)).tolist()
    return model.run(a[:, :16], k, ids, doses, obj, sub=sub, diagnostics=flux)


def fit_state(model, obj, points):
    """Exactly compare candidate states on the 61-dose search grid."""
    points = np.atleast_2d(points)
    scores, errors, doses, _ = evaluate(model, obj, points)
    i = np.lexsort((errors[:, :, 0].mean(1), scores))[0]
    return dict(
        x=points[i].tolist(), dose=doses[i].tolist(), score=float(scores[i])
    )


def propose(model, obj, st, ids, radius=1.0):
    """Bounded minimax linearization including independent per-run doses."""
    x = np.array(st["x"])
    d = np.array(st["dose"])
    ix = np.array(
        list(range(9))
        + ([9] if obj == "joint" else [])
        + [12]
        + [16 + i for i in ids]
    )
    span = HIGH[ix] - LOW[ix]
    scale = radius * np.minimum(span * 0.25, 0.25 * abs(x[ix]) + 0.025 * span)
    delta = np.where(x[ix] + 1e-3 * scale <= HIGH[ix], 1, -1) * 1e-3 * scale
    probes = np.tile(x, (len(ix) + 1, 1))
    probes[1:, ix] += np.diag(delta)
    nobs = 2 if obj == "joint" else 1
    y = evaluate(model, obj, probes, d[:, None])[1][:, :, :nobs]
    ref = y[0].ravel()
    grad = (y[1:].reshape(len(ix), -1) - ref).T / (delta / scale)
    dstep = np.where(d > 1.1499, -1e-5, 1e-5)
    dy = evaluate(model, obj, x, (d + dstep)[:, None])[1][0, :, :nobs] - y[0]
    ds = 0.025 * radius
    dg = (
        np.repeat(np.eye(len(d)), nobs, axis=0)
        * (dy / dstep[:, None]).ravel()[:, None]
        * ds
    )
    lo = np.maximum(-1, np.r_[(LOW[ix] - x[ix]) / scale, (0.85 - d) / ds])
    hi = np.minimum(1, np.r_[(HIGH[ix] - x[ix]) / scale, (1.15 - d) / ds])
    # Fix locally invisible coordinates. Multi-edge starts below activate
    # intermediate chains whose individual derivatives vanish at zero.
    g = np.column_stack([grad, dg])
    inert = np.max(abs(g), axis=0) < 1e-9
    lo[inert] = hi[inert] = 0
    result = linprog(
        np.r_[np.zeros(len(lo)), 1.0],
        A_ub=np.column_stack([g, -np.ones(len(ref))]),
        b_ub=-ref,
        bounds=list(zip(lo, hi, strict=True)) + [(0, None)],
        method="highs",
    )
    if not result.success:
        raise RuntimeError(result.message)
    pts = np.tile(x, (5, 1))
    pts[:, ix] = np.clip(
        x[ix]
        + np.outer([0, 0.06, 0.20, 0.55, 1.0], result.x[: len(ix)] * scale),
        LOW[ix],
        HIGH[ix],
    )
    return pts


def seeded(st, family, motif, expanded):
    """Start whole chains together and give dormant ancestors a source."""
    x = np.array(st["x"])
    parent = KEYS.index(PARENTS[family][0]) + 16
    pts = [x]
    for k in (0.03, 0.3, 1.5):
        p = x.copy()
        if expanded:
            for i in motif:
                p[16 + i] = max(p[16 + i], k)
        pts.append(p.copy())
        p[parent] = max(p[parent], k * (0.2 if BANK[parent - 16].b >= 0 else 1))
        pts.append(p)
    return np.unique(pts, axis=0)


def local_fit(m, obj, start, ids, family, motif, expanded):
    """Profile coordinated starts, then refit incumbent chemistry and doses."""
    starts = seeded(start, family, motif, expanded)
    anchor = fit_state(m, obj, starts)
    # Optimize one nonzero precursor/product seed even when the first step is
    # a retained phase relabeling with zero bulk signal.
    warm = starts[-1]
    if expanded:
        warm = np.array(start["x"])
        for j in motif:
            warm[16 + j] = max(warm[16 + j], 0.25)
        parent = KEYS.index(PARENTS[family][0])
        warm[16 + parent] = max(warm[16 + parent], 0.06)
    bridge = fit_state(m, obj, warm)
    batch = np.vstack(
        [propose(m, obj, anchor, ids), propose(m, obj, bridge, ids, 0.5)]
    )
    return fit_state(m, obj, batch)


def main():
    """Fit both objectives and their matched controls for each round."""
    OUT.mkdir(parents=True, exist_ok=True)
    spec = manifest()
    existing = OUT / "manifest.json"
    if existing.exists() and json.loads(existing.read_text()) != json.loads(
        json.dumps(spec)
    ):
        raise ValueError("Saved settings differ from the current reaction bank")
    save("manifest.json", spec)
    m = Model(BANK)
    old = list(range(20))
    active = old.copy()
    fits = {obj: fit_state(m, obj, starting(obj)) for obj in OBJECTIVES}
    static = fits.copy()
    save("baseline.json", fits)
    for n, trial in enumerate(TRIALS, 1):
        file = f"iteration_{n:02}.json"
        new_ids = sorted(set(active + trial["ids"]))
        if (OUT / file).exists():
            rec = json.loads((OUT / file).read_text())
            if rec["trial"] != trial or rec["active"] != new_ids:
                raise ValueError(f"Inconsistent saved reaction set in {file}")
            fits, static, active = rec["fits"], rec["static"], rec["active"]
            continue
        prev = fits.copy()
        controls = {}
        for obj in OBJECTIVES:
            args = (trial["family"], trial["ids"])
            static[obj] = local_fit(m, obj, static[obj], old, *args, False)
            controls[obj] = local_fit(m, obj, prev[obj], active, *args, False)
            exp = local_fit(m, obj, prev[obj], new_ids, *args, True)
            fits[obj] = min(
                [exp, controls[obj], static[obj], prev[obj]],
                key=lambda s: s["score"],
            )
        save(
            file,
            dict(
                iteration=n,
                trial=trial,
                active=new_ids,
                before=prev,
                controls=controls,
                fits=fits,
                static=static,
            ),
        )
        active = new_ids
        print(
            n,
            trial["family"],
            trial["motif"],
            *(round(100 * fits[o]["score"], 5) for o in OBJECTIVES),
            flush=True,
        )


if __name__ == "__main__":
    main()
