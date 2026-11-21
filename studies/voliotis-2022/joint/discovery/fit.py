"""Fit, look up, admit and refit one chamber reaction per round."""

import json
import time

import numpy as np
from directions import compete
from reactions import CANDIDATES, catalog
from smoke import DOSES, OUT, start

from gpu import Model

CAT = catalog()
UPPER = np.array([x["upper"] for x in CAT])
OBJECTIVES = ("mass", "joint")


def save(path, value):
    """Write a complete checkpoint using an atomic rename."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def encode(p, k, output, i=0):
    """Store one fitted vector and its per-run errors and doses."""
    score, err, dose, _ = output
    return dict(
        p=p.tolist(),
        k=k.tolist(),
        score=float(score[i]),
        dose=dose[i].tolist(),
        worst_mass=float(err[i, :, 0].max()),
        worst_count=float(err[i, :, 1].max()),
        mean_mass=float(err[i, :, 0].mean()),
        per_run=err[i, :, :].tolist(),
    )


def best(output):
    """Select by the worst trajectory; use mean mass error for exact ties."""
    return np.lexsort((output[1][:, :, 0].mean(1), output[0]))[0]


def candidates(model, incumbent, ids, remaining):
    """Screen unused equations against the current shared mass solution."""
    p = np.array(incumbent["p"])
    k = np.array(incumbent["k"])
    rows = []
    fractions = np.array([0.002, 0.01, 0.05, 0.2, 0.55, 1.0])
    for h in remaining:
        trials = np.tile(k, (len(fractions), 1))
        trials[:, h] = fractions * UPPER[h]
        rows.extend(trials)
    rates = np.asarray(rows)
    # Reprofile on a coarse dose grid before shortlisting a rate per equation.
    coarse = model.run(p, rates, ids + remaining, DOSES[::10], "mass", sub=1)
    pool = [k]
    for j, _ in enumerate(remaining):
        sl = slice(j * len(fractions), (j + 1) * len(fractions))
        order = np.lexsort((coarse[1][sl, :, 0].mean(1), coarse[0][sl]))
        pool.extend(rates[sl][order[:2]])
    pool = np.asarray(pool)
    fine = model.run(p, pool, ids + remaining, DOSES, "mass", sub=2)
    scores = []
    for j, h in enumerate(remaining):
        left = 1 + 2 * j
        ix = left + best((fine[0][left : left + 2], fine[1][left : left + 2]))
        scores.append(
            dict(
                key=CANDIDATES[h].key,
                index=h,
                rate=float(pool[ix, h]),
                score=float(fine[0][ix]),
                mean_mass=float(fine[1][ix, :, 0].mean()),
                response=float(
                    np.abs(fine[1][ix, :, 0] - fine[1][0, :, 0]).max()
                ),
            )
        )
    # Prefer observable nonzero rates if minimax candidates agree to 1e-7.
    return sorted(
        scores,
        key=lambda r: (
            round(r["score"], 7),
            round(r["mean_mass"], 7),
            -r["response"],
        ),
    )


def profile(model, obj, state, ids, h):
    """Fit the proposed rate on a dense grid including the exact null."""
    p = np.array(state["p"])
    k = np.array(state["k"])
    grid = np.r_[0.0, np.geomspace(0.0002, 1.0, 21)] * UPPER[h]
    rates = np.tile(k, (len(grid), 1))
    rates[:, h] = grid
    out = model.run(p, rates, ids + [h], DOSES, obj, sub=2)
    i = best(out)
    return encode(p, rates[i], out, i), [
        dict(rate=float(x), score=float(y))
        for x, y in zip(grid, out[0], strict=True)
    ]


def refine(model, obj, state, ids, seed):
    """Use equal-budget local proposals for the incumbent and expanded set."""
    rng = np.random.default_rng(seed)
    current = state
    # Shared delays, six source yields, mixing in joint fits and diversion.
    base_indices = list(range(9)) + ([9] if obj == "joint" else []) + [12]
    plo = np.array(
        [1 / 12] * 3 + [0.0] * 6 + [2.818, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    phi = np.array(
        [4.0] * 3 + [2.0] * 6 + [16.5515, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0]
    )
    for scale in (0.18, 0.055):
        p = np.array(current["p"])
        k = np.array(current["k"])
        paths = np.tile(k, (48, 1))
        params = np.tile(p, (48, 1))
        for j in range(1, 48):
            # Include coordinated moves and one-coordinate moves, keep the
            # incumbent unchanged in the first row of every candidate batch.
            if j % 3 == 0 and ids:
                h = ids[(j // 3 - 1) % len(ids)]
                paths[j, h] = np.clip(
                    k[h] + rng.normal() * scale * (0.035 * UPPER[h] + k[h]),
                    0.0,
                    UPPER[h],
                )
            elif j % 3 == 1:
                a = base_indices[(j // 3) % len(base_indices)]
                width = (0.01 + abs(p[a])) * scale
                params[j, a] = np.clip(
                    p[a] + rng.normal() * width, plo[a], phi[a]
                )
            else:
                for a in base_indices:
                    params[j, a] = np.clip(
                        p[a] + rng.normal() * scale * (abs(p[a]) + 0.0001),
                        plo[a],
                        phi[a],
                    )
                if ids:
                    paths[j, ids] = np.clip(
                        k[ids]
                        + rng.normal(size=len(ids))
                        * scale
                        * (k[ids] + 0.01 * UPPER[ids]),
                        0.0,
                        UPPER[ids],
                    )
        quick = model.run(params, paths, ids, DOSES[::10], obj, sub=1)
        ix = np.unique(np.r_[0, np.argsort(quick[0])[:5]])
        out = model.run(params[ix], paths[ix], ids, DOSES, obj, sub=2)
        i = best(out)
        current = encode(params[ix[i]], paths[ix[i]], out, i)
    if seed >= 9676:
        for _ in range(2):
            cp, ck, out, i = compete(model, obj, current, ids, [], DOSES)[0]
            current = encode(cp, ck, out, i)
    return current


def pad(states):
    """Extend early checkpoints with zero rates for later candidates."""
    for st in states.values():
        st["k"] += [0.0] * (len(CAT) - len(st["k"]))
    return states


def propose_round(model, states, ids, tried, attempt, proposed, seed):
    """Fit each unused candidate before recording the selected equation."""
    excluded = {0, 1, 2, 23, 24, 25} if attempt >= 3 else set()
    available = len(CAT) if attempt >= 3 else 26
    remaining = [
        j for j in range(available) if j not in tried and j not in excluded
    ]
    shadow = {
        obj: refine(model, obj, states[obj], ids, seed) for obj in OBJECTIVES
    }
    ranked = candidates(model, shadow["mass"], ids, remaining)
    compensated = {}
    if attempt >= 3:
        for h, (cp, ck, out, i) in zip(
            remaining,
            compete(model, "mass", shadow["mass"], ids, remaining, DOSES),
            strict=True,
        ):
            compensated[h] = encode(cp, ck, out, i)
        for row in ranked:
            st = compensated[row["index"]]
            if st["score"] < row["score"]:
                row.update(
                    score=st["score"],
                    mean_mass=st["mean_mass"],
                    rate=st["k"][row["index"]],
                )
        ranked.sort(
            key=lambda r: (
                round(r["score"], 10),
                round(r["mean_mass"], 8),
                -r["response"],
            )
        )
    h = ranked[0]["index"]
    fitted = {}
    profiles = {}
    for obj in OBJECTIVES:
        fitted[obj], profiles[obj] = profile(model, obj, shadow[obj], ids, h)
    if h in compensated and compensated[h]["score"] < fitted["mass"]["score"]:
        fitted["mass"] = compensated[h]
    save(
        proposed,
        dict(
            round=attempt,
            index=h,
            key=CAT[h]["key"],
            equation=CAT[h]["equation"],
            hypothesis=CAT[h]["hypothesis"],
            prior=states,
            null=shadow,
            profiles=profiles,
            initial_fit=fitted,
            screen=ranked,
        ),
    )


def await_decision(evidence_source, decision_file, h, number):
    """Verify that the paper review belongs to the fitted equation."""
    if evidence_source is not None and not decision_file.exists():
        saved = json.loads((evidence_source / decision_file.name).read_text())
        if saved["key"] != CAT[h]["key"]:
            raise ValueError("Archived match differs from the new proposal")
        save(decision_file, saved)
    while not decision_file.exists():
        time.sleep(2)
    evidence = json.loads(decision_file.read_text())
    if (
        evidence["round"] != number
        or evidence["key"] != CAT[h]["key"]
        or not evidence["url"].startswith("https://")
    ):
        raise ValueError("Mismatched reaction evidence")
    return evidence


def main(output=OUT, evidence_source=None):
    """Run the sequential search with a literature decision between rounds."""
    output.mkdir(parents=True, exist_ok=True)
    save(output / "catalog.json", CAT)
    model = Model()
    origin = {}
    for obj in OBJECTIVES:
        p, k, out = start(model, obj)
        origin[obj] = encode(p, k, out)
    save(output / "baseline.json", origin)
    states = origin
    ids = []
    tried = []
    for attempt in range(1, len(CAT) + 1):
        if len(ids) == 20:
            break
        stem = f"{attempt:02d}"
        final = output / f"round_{stem}.json"

        if final.exists():
            rec = json.loads(final.read_text())
            tried.append(rec["index"])
            if rec["accepted"]:
                ids.append(rec["index"])
            states = pad(rec["fits"])
            continue
        seed = 9673 + attempt
        proposed = output / f"proposal_{stem}.json"
        if not proposed.exists():
            propose_round(model, states, ids, tried, attempt, proposed, seed)
        proposal = json.loads(proposed.read_text())
        h = proposal["index"]
        print("Fit ready", attempt, proposal["key"], flush=True)
        decision_file = output / f"decision_{stem}.json"
        evidence = await_decision(evidence_source, decision_file, h, attempt)
        tried.append(h)
        states = pad(proposal["null"])
        if evidence["accepted"]:
            ids.append(h)
            states = {
                obj: refine(
                    model, obj, pad(proposal["initial_fit"])[obj], ids, seed
                )
                for obj in OBJECTIVES
            }
        save(
            final,
            dict(
                round=attempt,
                index=h,
                key=CAT[h]["key"],
                accepted=evidence["accepted"],
                active_indices=ids.copy(),
                fits=states,
            ),
        )
        print(
            "Completed",
            attempt,
            proposal["key"],
            evidence["accepted"],
            states["mass"]["score"],
            states["joint"]["score"],
            flush=True,
        )
    if len(ids) != 20:
        raise RuntimeError(
            "Candidate bank exhausted before twenty supported rounds"
        )


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--evidence-source", type=Path)
    args = parser.parse_args()
    main(args.output, args.evidence_source)
