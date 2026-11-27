"""Verify shared trajectory prefixes before scoring longer pulse schedules."""

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
import torch
from node_compare import Physical, checkpoints
from node_fit import load, metrics_for_series
from quadrature_validation import load_reference
from repeated_validation import check_prefix


def source_digest(folder):
    """Identify every exact input shard by name and contents."""
    h = hashlib.sha256()
    for f in sorted(
        (folder / "data/multicycle/validation").glob("families-*.npz")
    ):
        h.update(f.name.encode())
        h.update(hashlib.sha256(f.read_bytes()).digest())
    return h.hexdigest()


def physical_configs(values):
    """Parse radius counts and substeps for each physical control."""
    out = []
    for v in values:
        n, s = (int(token) for token in v.split("x"))
        if n < 2 or s < 1:
            raise ValueError(v)
        out.append(
            (
                f"physical-n{n}-s{s}",
                n,
                "extrapolated",
                s,
                0,
                Physical("extrapolated"),
            )
        )
    return out


def write_table(rows, output):
    """Publish a complete evidence table atomically."""
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(dict.fromkeys(k for row in rows for k in row)),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(output)


@torch.inference_mode()
def forecast(root, store, files, controls, blocks):
    """Replay the same checkpoints on every length and retain the outputs."""
    models, _ = checkpoints(files, False)
    digests = {
        f.stem: hashlib.sha256(f.read_bytes()).hexdigest() for f in files
    }
    rows = []
    previous = {}
    prior = None
    for block in blocks:
        folder = root / "repeated" / f"b{block}"
        if prior is not None:
            check_prefix(prior, folder)
        digest = source_digest(folder)
        out = store / f"b{block}"
        out.mkdir(parents=True, exist_ok=True)
        data = {}
        start = len(rows)
        for name, n, scheme, s, update, model in models + physical_configs(
            controls
        ):
            if n not in data:
                data[n] = load(folder, "multi", "validation", n, refined=False)
            d = data[n]
            args = [d[k] for k in ("mass", "number", "gas", "force")]
            w, g = model.rollout(*args, s)
            if not (
                torch.isfinite(w).all()
                and torch.isfinite(g).all()
                and w.min() >= 0
                and g.min() >= 0
            ):
                raise ValueError(f"Invalid inventories for {name}")
            old = previous.get(name)
            shared = 0
            if old is not None:
                ids, t, water, gas = old
                shared = t.shape[1]
                for left, right in (
                    (ids, d["ids"]),
                    (t, d["time"][:, :shared]),
                    (water, w[:, :shared]),
                    (gas, g[:, :shared]),
                ):
                    torch.testing.assert_close(left, right, atol=0, rtol=0)
            previous[name] = (d["ids"], d["time"], w, g)
            checkpoint = digests.get(name, "")
            np.savez_compressed(
                out / f"{name}.npz",
                ids=d["ids"][::2],
                time=d["time"][::2],
                water=w,
                gas=g,
                input_sha256=digest,
                checkpoint_sha256=checkpoint,
                step=update,
            )
            rows.append(
                dict(
                    model=name,
                    forecast_sha256=hashlib.sha256(
                        (out / f"{name}.npz").read_bytes()
                    ).hexdigest(),
                    plateaus=4 * block,
                    families=len(w) // 2,
                    shared_observations_per_box=shared,
                    verified_water_and_gas_values=2 * shared * len(w),
                    radius_nodes=n,
                    substeps=s,
                    scheme=scheme,
                    update=update,
                    input_sha256=digest,
                    checkpoint_sha256=checkpoint,
                )
            )
        write_table(rows[start:], out / "manifest.csv")
        prior = folder
    return rows


def score(root, store, blocks):
    """Align saved predictions to the refined reference and score them."""
    rows = []
    panel = None
    for block in blocks:
        folder = root / "repeated" / f"b{block}"
        digest = source_digest(folder)
        _, ids, time, pair, bound, member, _ = load_reference(
            folder, "multicycle", "validation"
        )
        ref = dict(
            time=torch.from_numpy(time.repeat(2, 0)),
            water=torch.from_numpy(member.reshape(-1, member.shape[-1])),
            ref=(pair, bound),
        )
        archive = store / f"b{block}"
        with (archive / "manifest.csv").open() as fp:
            manifest = list(csv.DictReader(fp))
        identities = {
            (e["model"], e["checkpoint_sha256"], e["update"]) for e in manifest
        }
        if (
            not identities
            or len(identities) != len(manifest)
            or (panel is not None and identities != panel)
        ):
            raise ValueError("Incomplete matched model panel")
        panel = identities
        for entry in manifest:
            file = archive / (entry["model"] + ".npz")
            if (
                hashlib.sha256(file.read_bytes()).hexdigest()
                != entry["forecast_sha256"]
            ):
                raise ValueError(f"Changed forecast {file}")
            with np.load(file) as d:
                if d["input_sha256"].item() != digest:
                    raise ValueError(f"Changed native inputs for {file}")
                np.testing.assert_array_equal(d["ids"], ids)
                np.testing.assert_array_equal(d["time"], time)
                metrics = metrics_for_series(
                    torch.from_numpy(d["water"]),
                    torch.from_numpy(d["gas"]),
                    ref,
                    True,
                )
                for pattern, result in metrics.items():
                    rows.append(
                        dict(
                            model=file.stem,
                            plateaus=4 * block,
                            schedule_reset_ms=pattern,
                            input_sha256=digest,
                            checkpoint_sha256=d["checkpoint_sha256"].item(),
                            update=d["step"].item(),
                            **result,
                        )
                    )
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["forecast", "score"])
    p.add_argument("root", type=Path)
    p.add_argument("store", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--files", type=Path, nargs="*", default=[])
    p.add_argument(
        "--controls", nargs="*", default=["3x1", "3x2", "4x2", "5x6"]
    )
    p.add_argument("--blocks", type=int, nargs="+", default=[1, 2])
    a = p.parse_args()
    torch.set_num_threads(1)
    if a.action == "forecast":
        records = forecast(a.root, a.store, a.files, a.controls, a.blocks)
    else:
        records = score(a.root, a.store, a.blocks)
    write_table(records, a.out)
    print(a.out, len(records))
