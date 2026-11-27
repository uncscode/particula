"""Subdivide existing representative groups using BC core and coating data."""

import numpy as np
from metrics import EDGES

SHELL = np.array([1.25, 1.5, 2.0, 3.0])


def _best_cut(idx, eligible, core, ratio, optical):
    """Choose a core or shell/core cut within one existing row."""
    if len(idx) < 2 or not eligible[idx].any():
        return None
    candidate = None
    for coordinate in (core, ratio):
        order = idx[np.argsort(coordinate[idx], kind="stable")]
        for cut in sorted({len(idx) // 2, 3 * len(idx) // 4}):
            if cut == 0 or coordinate[order[cut - 1]] == coordinate[order[cut]]:
                continue
            a, b = order[:cut], order[cut:]
            delta = optical[a].mean(0) - optical[b].mean(0)
            gain = (
                len(a)
                * len(b)
                / len(idx)
                * np.dot(delta, delta)
                * eligible[idx].mean()
            )
            if candidate is None or gain > candidate[0]:
                candidate = (gain, a, b)
    return candidate


def _shell_cut(idx, eligible, v, total, coats):
    """Reduce the BC volume assigned to the wrong coating class in one row."""
    if not eligible[idx].any() or len(idx) < 2:
        return None

    def displaced(part):
        bc = v[part].sum()
        if not bc:
            return 0.0
        tier = np.searchsorted(SHELL, np.cbrt(total[part].sum() / bc), "right")
        return v[part][coats[part] != tier].sum()

    current = displaced(idx)
    best = None
    for cut in range(1, len(SHELL) + 1):
        left = coats[idx] < cut
        if left.all() or not left.any():
            continue
        a, b = idx[left], idx[~left]
        gain = current - displaced(a) - displaced(b)
        if gain > 0 and (best is None or gain > best[0]):
            best = (gain, a, b)
    return best


def _resolve_selected(blocks, eligible):
    # Diagnostic ceiling on the checkpoint resolution in the chosen cells.
    refined = []
    for idx in blocks:
        keep = eligible[idx]
        if keep.any():
            refined.extend(np.split(idx[keep], keep.sum()))
        if (~keep).any():
            refined.append(idx[~keep])
    return refined


def refine(state, blocks, method, budget, targets):
    """Spend a per-parcel budget by splitting existing BC-bearing groups."""
    d = np.cbrt(state[:, :8].sum(1))
    v = state[:, 3]
    core = np.cbrt(v)
    ratio = np.divide(d, core, out=np.zeros_like(d), where=core > 0)
    bins = np.searchsorted(EDGES[10], d, "right")
    coats = np.searchsorted(SHELL, ratio, "right")
    selected = np.zeros(len(v), bool)
    for target in targets.itertuples():
        selected |= (bins == target.dry_bin) & (coats == target.shell_bin)
    eligible = (v > 0) & selected
    if method == "outside":
        eligible = (v > 0) & ~np.isin(bins, targets.dry_bin)
    if method == "exact":
        return _resolve_selected(blocks, eligible)

    # Use a between-group sum of squares for log shell/core and BC volume;
    # the latter weights carriers according to variance in transported BC.
    optical = np.column_stack((v, v * np.log(np.maximum(ratio, 1.0))))

    def candidate(idx):
        if method == "shell":
            return _shell_cut(idx, eligible, v, total, coats)
        return _best_cut(idx, eligible, core, ratio, optical)

    total = state[:, :8].sum(1)
    blocks = list(blocks)
    scores = [candidate(idx) for idx in blocks]
    for _ in range(budget):
        gains = [s[0] if s is not None else 0 for s in scores]
        j = np.argmax(gains)
        if gains[j] <= 0:
            break
        _, a, b = scores[j]
        blocks[j] = a
        blocks.append(b)
        scores[j] = candidate(a)
        scores.append(candidate(b))
    return blocks
