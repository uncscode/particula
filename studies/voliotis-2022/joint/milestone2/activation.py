"""Screen connected routes to dormant intermediates from the same incumbent."""

import argparse
import json

import numpy as np
from baseline_replay import FAMILIES, HERE, PARENTS, Model, run, save
from growth import BANK, growth_seed, initial, matched
from particula.dynamics.chemistry.discovery import enumerate_networks

KEY = {r.key: i for i, r in enumerate(BANK)}


def material_path(extra, x):
    """Identify a parent-specific supply route for each intermediate."""
    chain = set(extra)
    for h in extra:
        a = BANK[h].a
        if a < 11:
            continue
        family = FAMILIES[(a - 11) % 5]
        parent = max((KEY[k] for k in PARENTS[family]), key=lambda j: x[16 + j])
        chain.add(parent)
        if a >= 16:
            chain.add(KEY[f"{family}_oxidize" if a < 21 else f"{family}_split"])
    return sorted(chain)


def check_connected(chain):
    """Use SiMBA reachability on the effective source/destination graph."""
    # Half-mass fragments consume two bookkeeping units to make two daughters.
    rows = []
    for i in chain:
        c = BANK[i]
        row = np.zeros(31, dtype=int)
        if c.split >= 0:
            row[c.a] -= 2
            row[c.split] += 1
        else:
            row[c.a] -= 1
        if c.b >= 0:
            row[c.b] -= 1
        row[c.dest] += 1
        rows.append(tuple(row.tolist()))
    search = enumerate_networks(
        rows,
        [j < 9 for j in range(31)],
        [],
        max_steps=len(rows),
        max_trials=2 ** len(rows),
        max_candidates=2 ** len(rows),
    )
    return tuple(sorted(set(rows))) in search.networks


def main(number=1, sub=4, finalists=3, cycles=5):
    """Fit connected motifs with warm upstream channels."""
    x, ids = initial()
    suffix = "" if number == 1 else f"_{number:02d}"
    if number > 1:
        record = json.loads((HERE / "lineage.json").read_text())[number - 2]
        seed = growth_seed(number, record)
        ids, x = seed["ids"], np.array(seed["x"])
    model = Model(BANK)
    rows = json.loads((HERE / f"screen_{number:02d}.json").read_text())
    distinct = {}
    for row in rows:
        if row["gross"] >= 0.01:
            continue
        path = material_path(row["extra"], x)
        if tuple(path) in distinct or not check_connected(path):
            continue
        extra = sorted(set(path) - set(ids))
        if not extra:
            continue
        # Shared upstream seeds are present in both sides of the comparison.
        shared = sorted(set(path) & set(ids))
        starts = np.tile(x, (4, 1))
        rate_grid = np.array([0.03, 0.15, 0.6, 2.0])
        for h in shared:
            starts[:, 16 + h] = np.maximum(starts[:, 16 + h], rate_grid)
        base, _, _, _ = run(model, starts, ids, sub=sub)
        child = starts.copy()
        for h in extra:
            child[:, 16 + h] = rate_grid
        active = sorted(ids + extra)
        scores, _, doses, _ = run(model, child, active, sub=sub)
        best = int(np.argmax(base - scores))
        _, err, _, f = run(
            model,
            child[best],
            active,
            sub=sub,
            doses=doses[best, :, None],
            diagnostics=1,
        )
        peak = f[0][:, path].max(axis=0)
        distinct[tuple(path)] = dict(
            path=path,
            extra=extra,
            shared=shared,
            keys=[BANK[h].key for h in path],
            rate=rate_grid[best],
            paired_seed_gain=float(base[best] - scores[best]),
            score=float(scores[best]),
            gross_min=float(peak.min()),
            mass=float(err[0, :, 0].mean()),
        )
    rows = sorted(
        distinct.values(),
        key=lambda row: (-row["paired_seed_gain"], row["score"]),
    )
    save(f"activation_screen{suffix}.json", rows)
    checks = []
    for row in [r for r in rows if r["gross_min"] > 1e-9][:finalists]:
        extra, shared = row["extra"], row["shared"]
        active = sorted(ids + extra)
        np.testing.assert_allclose(
            run(model, x, ids, sub=sub)[1],
            run(model, x, active, sub=sub)[1],
            rtol=0,
            atol=1e-10,
        )
        pair = matched(
            model, x, ids, extra, row["rate"], cycles, shared=shared, sub=sub
        )
        checks.append(
            dict(
                **row,
                parent=pair["parent"],
                child=pair["child"],
                gain=pair["minimum_gain"],
                threshold=max(pair["numerical"], pair["variation"]),
                accepted=pair["accepted"],
            )
        )
        save(f"activation_refits{suffix}.json", checks)
        print(
            row["keys"], checks[-1]["gain"], checks[-1]["threshold"], flush=True
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=int, default=1, dest="number")
    parser.add_argument("--sub", type=int, default=4)
    parser.add_argument("--finalists", type=int, default=3)
    parser.add_argument("--cycles", type=int, default=5)
    args = parser.parse_args()
    main(args.number, args.sub, args.finalists, args.cycles)
