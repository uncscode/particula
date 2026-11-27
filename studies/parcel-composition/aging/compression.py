"""Nested, inventory-conserving representatives and size-conditioned entropy."""

import heapq

import numpy as np
from physics import DENSITY, EDGES, SS, responses
from scipy.special import entr


def node(rows, features):
    """Choose the median split along the feature with most scatter."""
    score = features[rows].var(axis=0) * len(rows)
    axis = np.argmax(score)
    return -score[axis], axis


def tree(vol, features, limit, protect_bc):
    """Return cell counts and mean species volumes, and the split sequence."""
    counts, means, splits, heap = [], [], [], []

    def add(rows):
        idx = len(counts)
        counts.append(len(rows))
        means.append(vol[rows].mean(axis=0))
        if len(rows) > 1:
            priority, axis = node(rows, features)
            heapq.heappush(heap, (priority, idx, axis, rows))
        return idx

    add(np.arange(len(vol)))
    while heap and len(splits) + 1 < limit:
        _, parent, axis, rows = heapq.heappop(heap)
        soot = vol[rows, 3] > 0
        if protect_bc and parent == 0 and np.any(soot) and not np.all(soot):
            left, right = rows[soot], rows[~soot]
        else:
            mid = len(rows) // 2
            order = np.argpartition(features[rows, axis], mid)
            left, right = rows[order[:mid]], rows[order[mid:]]
        splits.append((parent, add(left), add(right)))
    return np.array(counts), np.stack(means), splits


def entropy_contribution(vol):
    """Return dry mass times particle entropy, with a common volume factor."""
    mass = vol * DENSITY
    weight = mass.sum(axis=1)
    return weight * entr(mass / weight[:, None]).sum(axis=1)


def chi_endmembers(vol, optics):
    """Responses for fully internal and fully external states at fixed sizes."""
    d3 = vol.sum(axis=1)
    fractions = vol.sum(axis=0) / d3.sum()
    recipes = [fractions, *np.eye(len(fractions))]
    values = []
    for recipe in recipes:
        critical, area = responses(d3[:, None] * recipe, optics)
        curve = np.searchsorted(np.sort(critical), SS, side="right")
        values.append(np.r_[curve, area.sum()])
    return [values[0], fractions @ np.asarray(values[1:])]


def evaluate(vol, optics, limit, protect_bc):
    """Score every integer cap using additive changes at each tree split."""
    n = len(vol)
    diam = np.cbrt(vol.sum(axis=1))
    group = np.searchsorted(EDGES[1:-1], diam)
    count = np.bincount(group, minlength=12)
    critical, area = responses(vol, optics)
    reference = np.searchsorted(np.sort(critical), SS, side="right") / n
    cross = area.mean()
    # Per-bin split features are log diameter in bin-width units and
    # the four dry volume fractions. No optical/activation outputs enter.
    features = np.column_stack(
        [
            np.log(diam) / np.log(EDGES[1] / EDGES[0]),
            vol / vol.sum(axis=1, keepdims=True),
        ]
    )
    chi = np.full(12, np.nan)
    approx_chi = np.full((limit, 12), np.nan)
    summed = np.zeros((limit, len(SS) + 1))
    nrep = np.zeros(limit)
    bases = np.zeros((12, 2, len(SS) + 1))
    bulk_entropy = np.zeros(12)
    for b in np.flatnonzero(count):
        part, feat = vol[group == b], features[group == b]
        species = (part * DENSITY).sum(axis=0)
        mass = species.sum()
        bulk_entropy[b] = entr(species / mass).sum()
        divisor = np.expm1(bulk_entropy[b])
        bases[b] = chi_endmembers(part, optics)
        chi[b] = np.expm1(entropy_contribution(part).sum() / mass) / divisor
        weights, means, splits = tree(part, feat, limit, protect_bc)
        c, a = responses(means, optics)
        totals = weights[:, None] * np.column_stack(
            [c[:, None] <= SS, a, entropy_contribution(means), means]
        )
        increments = np.zeros((limit, totals.shape[1]))
        increments[0] = totals[0]
        if splits:
            parent, left, right = np.array(splits).T
            increments[1 : len(splits) + 1] = (
                totals[left] + totals[right] - totals[parent]
            )
        accum = increments.cumsum(axis=0)
        np.testing.assert_allclose(
            accum[:, -4:],
            np.broadcast_to(part.sum(axis=0), (limit, 4)),
            rtol=1e-10,
            atol=1e-8,
        )
        summed += accum[:, : len(SS) + 1]
        approx_chi[:, b] = np.expm1(accum[:, len(SS) + 1] / mass) / divisor
        nrep += np.minimum(np.arange(1, limit + 1), len(part))
    active = count > 0
    errors = (summed[:, :-1] / n - reference) * 100
    scores = np.column_stack(
        [
            np.sqrt((errors**2).mean(axis=1)),
            abs(errors).max(axis=1),
            100 * abs(summed[:, -1] / n - cross) / cross,
            np.sqrt(
                ((approx_chi[:, active] - chi[active]) ** 2)
                @ (count[active] / n)
            ),
            nrep,
        ]
    )
    # At 1,000 particles/cm³, mean area in nm² times 0.001 gives Mm^-1.
    return (
        scores,
        chi,
        count,
        reference,
        cross * 1e-3,
        approx_chi,
        summed[:, :-1] / n,
        bases / n,
        bulk_entropy,
    )
