"""Serialize a representative restart including exact member lineage & slot
order.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from campaign import read
from metrics import compress, make_tree, prepare
from model import Box, init, step


def encode(path, source, seed, regime, grid=10, cap=16, rule=2):
    """Store representative means, exact multiplicities and restart state."""
    boxes, rng, t = read(source)
    n, dt, _ = map(int, (source.parent / "done").read_text().split())
    _, _, forcing = init(seed, n)
    header = dict(
        time=t,
        dt=dt,
        regime=regime,
        seed=seed,
        n=n,
        grid=grid,
        cap=cap,
        rule=rule,
        forcing={k: np.asarray(v).tolist() for k, v in forcing.items()},
        rng=rng.bit_generator.state,
    )
    data = {"header": json.dumps(header)}
    for i, b in enumerate(boxes):
        prefix = f"p{i}_"
        centers, meta, order = make_tree(prepare(b, grid), cap, rule)
        use = (meta[:, 1] <= cap) & (meta[:, 2] > cap)
        selected = meta[use]
        data[prefix + "centers"] = centers[use]
        data[prefix + "bin_count"] = selected[:, [0, 4]].copy()
        slots = np.empty(
            len(b.dry), np.uint16 if len(selected) <= 65536 else np.uint32
        )
        for j, (_, _, _, lo, size) in enumerate(selected):
            slots[order[lo : lo + size]] = j
        data[prefix + "slots"] = slots
        data[prefix + "old_bits"] = np.packbits(b.old)
        data[prefix + "pending"] = b.pending
        data.update(
            {
                prefix + k: v
                for k, v in vars(b).items()
                if k not in ("dry", "water", "old", "pending")
            }
        )
    np.savez_compressed(path, **data)
    decoded, stream, meta = decode(path)
    np.testing.assert_equal(stream.bit_generator.state, rng.bit_generator.state)
    for b, other in zip(boxes, decoded, strict=True):
        compress(b, grid, cap, rule)
        for k in vars(b):
            np.testing.assert_array_equal(getattr(b, k), getattr(other, k))


def decode(path):
    """Expand a portable checkpoint in its saved random-event slot order."""
    with np.load(path) as z:
        h = json.loads(z["header"].item())
        boxes = []
        for i in range(2):
            a = {k[3:]: z[k] for k in z.files if k.startswith(f"p{i}_")}
            idx = a.pop("slots")
            np.testing.assert_array_equal(
                np.bincount(idx), a.pop("bin_count")[:, 1]
            )
            v = a.pop("centers")[idx]
            a["old"] = np.unpackbits(a.pop("old_bits"), count=len(v)).astype(
                bool
            )
            boxes.append(
                Box(
                    dry=np.ascontiguousarray(v[:, :8]),
                    water=v[:, 8].copy(),
                    **{k: x.item() if not x.ndim else x for k, x in a.items()},
                )
            )
        rng = np.random.default_rng()
        rng.bit_generator.state = h["rng"]
        return boxes, rng, h


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--source", type=Path)
    a = p.parse_args()
    if a.source:
        encode(
            a.checkpoint,
            a.source,
            int(a.source.parent.parent.name),
            a.source.parent.name,
        )
    else:
        boxes, rng, h = decode(a.checkpoint)
        for t in range(h["time"], 21600, h["dt"]):
            step(boxes, rng, h["forcing"], h["regime"], t, h["dt"])
        print(
            h["seed"],
            h["regime"],
            [len(b.dry) for b in boxes],
            [b.droplets for b in boxes],
        )
