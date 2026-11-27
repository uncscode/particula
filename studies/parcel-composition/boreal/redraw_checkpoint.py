"""Independent constrained draws from a portable joint size/composition
checkpoint.
"""

import io
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from coating_split import refine
from metrics import EDGES, make_tree, prepare
from model import KAPPA, Box
from stage_two import rank_sections

TARGETS = pd.read_csv(Path(__file__).with_name("coating-targets.csv"))
PARTICLE = ("dry", "water", "old", "pending")
DEFAULT_RECIPE = (10, 16, 1)  # Bins per decade, base row cap, joint spread.


def groups(box, grid, cap, size_edges=None):
    """Allocate rows within dry size, keeping each occupied physical class."""
    state, features, *_ = prepare(box, grid)
    d = np.cbrt(box.dry.sum(1))
    boundaries = EDGES[grid] if size_edges is None else size_edges
    size = np.searchsorted(boundaries, d, "right")
    growing = box.water > 1.05 * 49 * (box.dry @ KAPPA) + 100
    keys, inv, counts = np.unique(
        np.column_stack(
            (size, growing, box.old, box.pending, box.dry[:, 3] > 0)
        ),
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    quotas = np.ones(len(keys), int)
    for j in np.unique(keys[:, 0]):
        ix = np.flatnonzero(keys[:, 0] == j)
        for _ in range(min(cap, counts[ix].sum()) - len(ix)):
            priority = np.where(
                quotas[ix] < counts[ix], counts[ix] / (1 + quotas[ix]), -1
            )
            quotas[ix[np.argmax(priority)]] += 1
    edges = np.r_[0, counts.cumsum()].astype(np.int32)
    slots = np.argsort(inv, kind="stable").astype(np.int32)
    _, meta, ordered = make_tree(
        (state, features, np.zeros(len(d), np.int32), edges, slots), cap, 0
    )
    chosen = (meta[:, 1] <= quotas[meta[:, 0]]) & (
        quotas[meta[:, 0]] < meta[:, 2]
    )
    blocks = [
        ordered[start : start + n]
        for g, _, _, start, n in meta[chosen]
        if keys[g, 2]
    ]
    for g in np.flatnonzero(keys[:, 2] == 0):
        blocks.extend(
            rank_sections(slots[edges[g] : edges[g + 1]], d, quotas[g])
        )
    return (
        state,
        features,
        keys[inv],
        refine(state, blocks, "local", 4, TARGETS),
    )


def direction(values, features, center, key, grid, size_edges=None):
    """Fit one correlated fluctuation, limiting support to the physical
    class.
    """
    if len(values) < 2:
        return np.zeros(9)
    axis = features[:, np.argmax(features.var(0))]
    dev = axis - axis.mean()
    sd = np.sqrt(np.mean(dev**2))
    if sd < 1e-14:
        return np.zeros(9)
    v = np.sqrt(3) * (dev @ (values - center)) / (len(dev) * sd)
    # For z uniform on [-1,1], constrain both endpoints by linear inequalities.
    amplitudes = list(abs(v))
    margins = list(center)
    i, wet, _, pending, _ = key
    volume = center[:8].sum()
    delta = v[:8].sum()
    edges = EDGES[grid] if size_edges is None else size_edges
    if i > 0:
        amplitudes.append(abs(delta))
        margins.append(volume - edges[i - 1] ** 3)
    if i < len(edges):
        amplitudes.append(abs(delta))
        margins.append(edges[i] ** 3 - volume)
    if pending:
        amplitudes.append(abs(delta))
        margins.append(1000 - volume)
    water = center[8] - 1.05 * 49 * (center[:8] @ KAPPA) - 100
    amplitudes.append(abs(v[8] - 1.05 * 49 * (v[:8] @ KAPPA)))
    margins.append(water if wet else -water)
    amp = np.asarray(amplitudes)
    scale = np.min(np.asarray(margins)[amp > 0] / amp[amp > 0], initial=1.0)
    return v * np.clip(scale, 0, 1) * (1 - 1e-10)


def fit_box(
    box,
    grid=DEFAULT_RECIPE[0],
    cap=DEFAULT_RECIPE[1],
    spread=DEFAULT_RECIPE[2],
    *,
    size_edges=None,
):
    """Pack number, ancestry, dry inventories and water without original
    slots.
    """
    if size_edges is not None:
        size_edges = np.asarray(size_edges, dtype=float)
        if (
            size_edges.shape != EDGES[grid].shape
            or not np.isfinite(size_edges).all()
            or size_edges[0] <= 0
            or (np.diff(size_edges) <= 0).any()
        ):
            raise ValueError("Invalid parcel dry-size boundaries")
    state, feat, labels, blocks = groups(box, grid, cap, size_edges)
    means = []
    vectors = []
    keys = []
    counts = []
    for idx in blocks:
        values = state[idx]
        center = values[0] + (values - values[0]).mean(0)
        key = labels[idx[0]]
        keys.append(key)
        counts.append(len(idx))
        means.append(center)
        if spread:
            vectors.append(
                direction(values, feat[idx], center, key, grid, size_edges)
            )
    data = dict(
        mean=np.asarray(means),
        count=np.array(counts, np.int32),
        label=np.asarray(keys, np.int32),
    )
    if size_edges is not None:
        data["dry_edges_nm"] = size_edges.copy()
    if spread:
        data["delta"] = np.asarray(vectors)
    for name, value in vars(box).items():
        if name not in PARTICLE:
            data[name] = np.asarray(value)
    np.testing.assert_allclose(
        data["count"] @ data["mean"], state.sum(0), rtol=2e-11
    )
    np.testing.assert_equal(
        data["count"] @ data["label"][:, 3], box.pending.sum()
    )
    return data


def draw_box(data, rng):
    """Create a new particle list using balanced random within-row
    deviations.
    """
    count = data["count"]
    rows = np.repeat(np.arange(len(count)), count)
    state = data["mean"][rows].copy()
    if "delta" in data:
        offset = np.repeat(np.r_[0, count.cumsum()[:-1]], count)
        pos = np.arange(len(rows)) - offset
        pair = (pos % 2 == 0) & (pos + 1 < count[rows])
        deviation = (
            rng.uniform(-1, 1, size=pair.sum())[:, None]
            * data["delta"][rows[pair]]
        )
        ix = np.flatnonzero(pair)
        state[ix] += deviation
        state[ix + 1] -= deviation
    order = rng.permutation(len(rows))
    labels = data["label"][rows[order]]
    other = {
        k: (v.copy() if v.ndim else v.item())
        for k, v in data.items()
        if k not in ("delta", "count", "label", "mean", "dry_edges_nm")
    }
    return Box(
        dry=state[order, :8].copy(),
        water=state[order, 8].copy(),
        old=labels[:, 2].astype(bool),
        pending=labels[:, 3].copy(),
        **other,
    )


def serialize(particles, case, forcing, start, rng, recipe):
    """Serialize comparable parcel and forcing state for packed or full
    records.
    """
    buf = io.BytesIO()
    fields = {
        f"p{i}_{k}": v for i, b in enumerate(particles) for k, v in b.items()
    }
    meta = dict(
        version=1,
        case=asdict(case),
        seconds=start,
        recipe=recipe,
        forcing={k: np.asarray(v).tolist() for k, v in forcing.items()},
        rng=rng.bit_generator.state,
    )
    np.savez_compressed(buf, header=json.dumps(meta), **fields)
    return buf.getvalue()


def load(payload):
    """Read a packed checkpoint independently of any reference particle
    arrays.
    """
    with np.load(io.BytesIO(payload), allow_pickle=False) as z:
        boxes = [
            {k[3:]: z[k] for k in z.files if k.startswith(f"p{i}_")}
            for i in (0, 1)
        ]
        return boxes, json.loads(z["header"].item())
