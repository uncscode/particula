"""Fit conservative node corrections from paired bulk water histories."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from moment_quadrature import decode, encode
from node_correction import NodeCorrection
from quadrature import restore_arrays
from quadrature_validation import frames_by_time, load_reference
from qualification_rank import score


def folders(root, key):
    """Locate the paired histories and their independent refined validation."""
    if key == "multi":
        return root, root / "data/multicycle", "multicycle"
    base = root / "small-transfer" / key
    return base, base / "data", "pulse"


def load(root, key, split, nodes, *, refined=True):
    """Compress only the supplied initial particle records for each history."""
    base, folder, mode = folders(root, key)
    blocks = []
    for file in sorted((folder / split).glob("families-*.npz")):
        with np.load(file) as source:
            mass = source["mass0"] if "mass0" in source else source["mass"][0]
            if nodes == 0:
                counts, per_particle = source["counts"], mass
            else:
                code, totals = encode(mass, source["counts"], nodes)
                shape, _ = decode(code.astype(np.float32), totals, nodes)
                counts, per_particle = restore_arrays(shape, totals)
            blocks.append(
                (
                    per_particle,
                    counts / source["volume"],
                    source["gas"][0],
                    source["force"].swapaxes(0, 1),
                    source["water"].T,
                    np.repeat(source["time"][None], len(mass), 0),
                    np.repeat(source["ids"], 2),
                )
            )
    if not blocks:
        raise ValueError(f"No histories in {folder / split}")
    data = dict(
        zip(
            ("mass", "number", "gas", "force", "water", "time", "ids"),
            [
                torch.from_numpy(np.concatenate(x))
                for x in zip(*blocks, strict=True)
            ],
            strict=True,
        )
    )
    if not refined:
        return data
    if split == "validation" and key == "multi":
        _, ids, times, pair, bound, members, _ = load_reference(
            base, mode, split
        )
        np.testing.assert_array_equal(data["ids"][::2], ids)
        np.testing.assert_array_equal(data["time"][::2], times)
        data["water"] = torch.from_numpy(members.reshape(-1, members.shape[-1]))
        data["ref"] = (pair, bound)
    elif split == "validation":
        records = json.loads((base / "validation-refinement.json").read_text())[
            "records"
        ]
        np.testing.assert_array_equal(
            data["ids"][::2], [r["id"] for r in records]
        )
        members = np.stack([r["member_water"] for r in records])
        np.testing.assert_array_equal(
            data["water"], members.reshape(-1, members.shape[-1])
        )
        pair = np.stack([r["pair_contrast"] for r in records])
        np.testing.assert_array_equal(pair, members[:, 1] - members[:, 0])
        data["ref"] = pair, np.stack([r["reference_error"] for r in records])
    return data


@torch.no_grad()
def evaluate(model, data, substeps, schedules=False):
    """Score complete free forecasts, including the weakest duration pattern."""
    prediction, gas = model.rollout(
        data["mass"], data["number"], data["gas"], data["force"], substeps
    )
    return metrics_for_series(prediction, gas, data, schedules)


def metrics_for_series(prediction, gas, data, schedules=False):
    """Compare a supplied forecast with the refined paired histories."""
    if (
        not torch.isfinite(prediction).all()
        or not torch.isfinite(gas).all()
        or gas.min() < 0
        or prediction.min() < 0
    ):
        raise ValueError("Invalid forecast inventory")
    pair, bound = data["ref"]
    times = data["time"][::2].numpy()
    unique, inverse = np.unique(times, axis=0, return_inverse=True)
    groups = [np.ones(len(times), dtype=bool)] + [
        inverse == k for k in range(len(unique))
    ]
    scores = []
    for group in groups:
        mask = np.repeat(group, 2)
        elapsed, frames = frames_by_time(times[group])
        target = np.take_along_axis(pair[group], frames, axis=1)
        uncertainty = np.take_along_axis(bound[group], frames, axis=1)
        p = prediction[mask]
        indices = torch.arange(p.shape[1]).expand_as(p)
        scores.append(
            score(
                [(indices, p, None)],
                {"water": data["water"][mask]},
                dict(
                    frames=frames,
                    time=elapsed,
                    target=target,
                    mask=abs(target) > 4 * uncertainty,
                ),
            )
        )
        delta = (p - data["water"][mask]).abs().numpy()
        aligned = np.take_along_axis(
            delta, np.repeat(frames, 2, axis=0), axis=1
        )
        resolved = (abs(target) > 4 * uncertainty)[:, elapsed > 0]
        scores[-1].update(
            families=int(group.sum()),
            resolved_pairs_times=int(resolved.sum()),
            member_mae_mg_m3=1e6 * aligned[:, elapsed > 0].mean(),
            terminal_member_mae_mg_m3=1e6 * delta[:, -1].mean(),
        )
    scores[0]["worst_schedule_s"] = min(s["horizon_s"] for s in scores[1:])
    if schedules:
        labels = ["all"] + [
            ",".join(f"{t * 1000:.6g}" for t in row[:-1][np.diff(row) == 0])
            for row in unique
        ]
        return dict(zip(labels, scores, strict=True))
    return scores[0]


def objective(model, d, rng, batches, window, substeps):
    """Differentiate a paired forecast window after causal warmup."""
    boxes = (
        rng.integers(len(d["mass"]) // 2, size=batches)[:, None] * 2
        + np.arange(2)
    ).ravel()
    mass, number, gas, force, target = (
        d[k][boxes] for k in ("mass", "number", "gas", "force", "water")
    )
    length = min(window, force.shape[1])
    start = (
        0
        if rng.random() < 0.25
        else int(rng.integers(force.shape[1] - length + 1))
    )
    static = model.static(mass, number)
    memory = model.start(number)
    water = mass[..., 0]
    series = []
    for i in range(start + length):
        with torch.set_grad_enabled(i >= start):
            f = force[:, i]
            for _ in range(substeps):
                water, gas, memory = model.step(
                    water, gas, number, static, f[:, 0] / substeps, memory
                )
            gas = torch.where(f[:, 3] > 0, f[:, 4], gas)
            if i >= start:
                series.append((water * number).sum(1))
    forecast = torch.stack(series, 1)
    true = target[:, start + 1 : start + 1 + length]
    # Scale member and mixing-state errors by conserved dry mass.
    dry = (mass[..., 1:].sum(-1) * number).sum(1).clamp_min(1e-20)
    diff = (forecast - true) / dry[:, None]
    return diff.abs().mean() + 2 * (diff[1::2] - diff[::2]).abs().mean()


def run(
    root,
    out,
    nodes,
    hidden,
    memory,
    scheme,
    substeps,
    updates,
    batch,
    window,
    seed,
    keys,
    mixture=False,
    physical_clock=False,
):
    """Record initial and fitted scores against the same fixed references."""
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    train = [load(root, k, "train", nodes) for k in keys]
    validation = [load(root, k, "validation", nodes) for k in keys]
    for a, b in zip(train, validation, strict=True):
        if np.intersect1d(a["ids"], b["ids"]).size:
            raise ValueError("Overlapping training and validation IDs")
    config = dict(
        nodes=nodes,
        hidden=hidden,
        memory=memory,
        scheme=scheme,
        substeps=substeps,
        seed=seed,
        keys=keys,
        mixture=mixture,
        physical_clock=physical_clock,
    )
    model = NodeCorrection(hidden, memory, scheme, mixture, physical_clock)
    opt = torch.optim.AdamW(model.parameters(), lr=0.002)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for step in range(updates + 1):
        if step:
            d = train[(step - 1) % len(train)]
            loss = objective(model, d, rng, batch, window, substeps)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            opt.step()
        if step in {0, updates // 4, updates // 2, updates}:
            for k, d in zip(keys, validation, strict=True):
                row = dict(
                    step=step,
                    domain=k,
                    mixture=mixture,
                    physical_clock=physical_clock,
                    nodes=nodes,
                    memory=memory,
                    scheme=scheme,
                    substeps=substeps,
                    seed=seed,
                    parameters=sum(p.numel() for p in model.parameters()),
                    **evaluate(model, d, substeps),
                )
                rows.append(row)
                print(json.dumps(row), flush=True)
            with out.with_suffix(".csv.tmp").open("w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=rows[0], lineterminator="\n"
                )
                writer.writeheader()
                writer.writerows(rows)
            out.with_suffix(".csv.tmp").replace(out.with_suffix(".csv"))
            torch.save(
                dict(config=config, step=step, model=model.state_dict()),
                out.with_suffix(".pt.tmp"),
            )
            out.with_suffix(".pt.tmp").replace(out.with_suffix(".pt"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--nodes", type=int, default=3)
    p.add_argument("--physical-clock", action="store_true")
    p.add_argument("--mixture", action="store_true")
    p.add_argument("--hidden", type=int, default=24)
    p.add_argument("--memory", type=int, default=0)
    p.add_argument(
        "--scheme", choices=["implicit", "extrapolated"], default="implicit"
    )
    p.add_argument("--substeps", type=int, default=1)
    p.add_argument("--updates", type=int, default=300)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--window", type=int, default=12)
    p.add_argument("--seed", type=int, default=2101)
    p.add_argument("--keys", nargs="+", default=["multi", "2ms", "20ms"])
    run(**vars(p.parse_args()))
