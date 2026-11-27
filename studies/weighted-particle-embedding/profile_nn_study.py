"""Learn compact particle mixtures for a physical water-transfer solver."""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from conditional_geometry import (
    baseline,
    fit_geometry,
    geometry,
    shape_from_profile,
)
from joint_fit import atomic_save
from learned_profile import ProfileCodec, particle_inputs, section_features
from moment_quadrature import encode
from multicycle_storage import forcing, inputs
from node_fit import metrics_for_series
from projected_moments import context
from projection_study import digest, save_prediction
from quadrature import restore_arrays
from quadrature_validation import load_reference
from reduced_exchange import rollout
from torch.nn import functional


def prepare(root, folder, n):
    """Read initial particles from both disjoint training cohorts."""
    directory = [
        root / "data/multicycle/train",
        root / "small-transfer/2ms/data/train",
    ]
    parts = []
    for base in directory:
        for path in sorted(base.glob("families-*.npz")):
            with np.load(path) as d:
                mass = d["mass0"] if "mass0" in d else d["mass"][0]
                counts = d["counts"]
                code, t = encode(mass, counts, n)
                core, profile = geometry(code, t, n)
                x, w = particle_inputs(mass, counts)
                parts.append((core, profile, t, x, w, d["ids"]))
    max_slots = max(p[3].shape[1] for p in parts)
    padded = []
    for core, profile, t, x, w, ids in parts:
        extra = max_slots - x.shape[1]
        padded.append(
            (
                core,
                profile,
                t,
                np.pad(x, ((0, 0), (0, extra), (0, 0))),
                np.pad(w, ((0, 0), (0, extra))),
                ids,
            )
        )
    core, profile, t, x, w, ids = [
        np.concatenate(v) for v in zip(*padded, strict=True)
    ]
    if len(ids) != len(np.unique(ids)):
        raise ValueError("Overlapping training families")
    linear = fit_geometry(core, profile, t, n)
    conditioned = (profile - baseline(core, t)).reshape(len(t), -1)
    normalized = (conditioned - linear["center"]) / linear["scale"]
    known = context(core, t)[:, 1:]
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        folder / f"train-n{n}.npz",
        x=x,
        weights=w,
        known=known.astype(np.float32),
        profile=normalized.astype(np.float32),
        ids=ids,
        center=linear["center"],
        scale=linear["scale"],
        axes=linear["axes"],
    )
    print(len(ids), len(x), max_slots)


def train(folder, n, width, kind, hidden, updates, seed):
    """Fit the common profile targets for a fixed number of updates."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    datafile = folder / f"train-n{n}.npz"
    with np.load(datafile) as d:
        k = width - (2 * n - 2)
        if not 0 < k <= len(d["axes"]):
            raise ValueError("Invalid conditional width")
        known = torch.from_numpy(d["known"])
        profile = torch.from_numpy(d["profile"])
        x = torch.from_numpy(d["x"])
        weight = torch.from_numpy(d["weights"])
        model = ProfileCodec(
            d["axes"][:k].copy(),
            known.mean(0).numpy(),
            known.std(0).clamp_min(1e-7).numpy(),
            kind,
            hidden,
        )
        center = torch.from_numpy(d["center"].copy())
        scale = torch.from_numpy(d["scale"].copy())
    sections = (
        torch.cat(
            [
                section_features(a, b)
                for a, b in zip(x.split(128), weight.split(128), strict=True)
            ]
        )
        if kind == "sections"
        else None
    )
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    filename = folder / f"{kind}-n{n}-w{width}-h{hidden}-u{updates}-s{seed}.pt"
    for step in range(1, updates + 1):
        idx = (
            2 * rng.integers(len(x) // 2, size=32)[:, None] + np.arange(2)
        ).ravel()
        ctx = known[idx]
        target = profile[idx]
        z = model.encode(
            ctx,
            x[idx],
            weight[idx],
            target,
            sections[idx] if sections is not None else None,
        )
        pred = model.decode(z, ctx)
        member_loss = functional.smooth_l1_loss(pred, target)
        loss = member_loss + 0.5 * functional.smooth_l1_loss(
            pred[1::2] - pred[::2], target[1::2] - target[::2]
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 5, error_if_nonfinite=True
        )
        opt.step()
        if step == updates // 2:
            opt.param_groups[0]["lr"] *= 0.35
    atomic_save(
        dict(
            model=model.state_dict(),
            kind=kind,
            width=width,
            degree=n,
            hidden=hidden,
            seed=seed,
            training_sha256=digest(datafile),
            step=updates,
            center=center,
            scale=scale,
        ),
        filename,
    )
    print(filename, loss.item(), flush=True)


@torch.inference_mode()
def evaluate(
    root,
    folder,
    checkpoints,
    archive,
    out,
    substeps,
    mode="multicycle",
    split="validation",
    predictions=None,
):
    """Reload each initial record and compare independent trajectories."""
    files, ids, times, pair, bound, ref, ref_level = load_reference(
        root, mode, split
    )
    common, mass, counts, _ = inputs(root, files, ids, mode, split)
    args = particle_inputs(mass, counts)
    x, weights = [torch.from_numpy(a) for a in args]
    truth = dict(
        time=torch.from_numpy(np.repeat(times, 2, axis=0)),
        water=torch.from_numpy(ref.reshape(-1, ref.shape[-1])),
        ref=(pair, bound),
    )
    rows = []
    archive.mkdir(exist_ok=True, parents=True)
    for path in checkpoints:
        cp = torch.load(path, map_location="cpu", weights_only=True)
        state = cp["model"]
        n = cp["degree"]
        datafile = folder / f"train-n{n}.npz"
        with np.load(datafile) as training:
            if (
                np.intersect1d(ids, training["ids"]).size
                or digest(datafile) != cp["training_sha256"]
            ):
                raise ValueError("Training/validation identity mismatch")
        model = ProfileCodec(
            state["axes"],
            state["context_mean"],
            state["context_std"],
            cp["kind"],
            cp["hidden"],
        )
        model.load_state_dict(state)
        model.eval()
        code, t = encode(mass, counts, n)
        core = np.delete(code[:, : 2 * n - 1], 2, axis=1)
        known = torch.from_numpy(context(core, t)[:, 1:]).float()
        normalized = None
        if model.kind == "profile":
            _, teacher = geometry(code, t, n)
            local = (teacher - baseline(core, t)).reshape(len(t), -1)
            normalized = (
                (torch.from_numpy(local) - cp["center"]) / cp["scale"]
            ).float()
        z = model.encode(known, x, weights, normalized)
        file = archive / (path.stem + ".npz")
        np.savez_compressed(
            file,
            size=core.astype(np.float32),
            conditional=z.numpy().astype(np.float16),
            totals=t,
            model_sha256=digest(path),
            **common,
        )
        with np.load(file) as a:
            if a["model_sha256"].item() != digest(path):
                raise ValueError("Changed shared network")
            core = a["size"].astype(np.float64)
            t = a["totals"]
            known = torch.from_numpy(context(core, t)[:, 1:]).float()
            z = torch.from_numpy(a["conditional"]).float()
            p = (
                model.decode(z, known).double() * cp["scale"] + cp["center"]
            ).numpy().reshape(-1, n, 5) + baseline(core, t)
            shape, adjustments = shape_from_profile(core, p, t, n)
            count, nodes = restore_arrays(shape, t)
            gas = np.repeat(a["gas_levels_kg_m3"][:, 0], 2)
            f = forcing(
                a["lengths"], a["gas_levels_kg_m3"], float(a["interval_s"])
            )
            native_inputs = [
                torch.from_numpy(v)
                for v in (nodes, count / a["volume_m3"], gas, f)
            ]
        w, g = rollout(*native_inputs, substeps, True)
        save_prediction(predictions, path.stem, ids, times, w, g, file, path)
        for pattern, result in metrics_for_series(w, g, truth, True).items():
            rows.append(
                dict(
                    model=path.stem,
                    kind=model.kind,
                    seed=cp["seed"],
                    updates=cp["step"],
                    width=cp["width"],
                    reset_times_ms=pattern,
                    archive_bytes=file.stat().st_size,
                    member_reference=ref_level,
                    checkpoint_bytes=path.stat().st_size,
                    archive_sha256=digest(file),
                    corrected_boxes=int((adjustments > 0).sum()),
                    **result,
                )
            )
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["prepare", "train", "evaluate"])
    p.add_argument("root", type=Path)
    p.add_argument("folder", type=Path)
    p.add_argument("--degree", type=int, default=5)
    p.add_argument("--width", type=int, default=16)
    p.add_argument(
        "--kind", choices=["direct", "profile", "sections"], default="direct"
    )
    p.add_argument("--hidden", type=int, default=48)
    p.add_argument("--updates", type=int, default=2400)
    p.add_argument("--seed", type=int, default=2201)
    p.add_argument("--substeps", type=int, default=6)
    p.add_argument("--files", type=Path, nargs="+")
    p.add_argument("--archive", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument(
        "--mode", choices=["pulse", "multicycle"], default="multicycle"
    )
    p.add_argument("--split", default="validation")
    a = p.parse_args()
    torch.set_num_threads(1)
    if a.action == "prepare":
        prepare(a.root, a.folder, a.degree)
    elif a.action == "train":
        train(a.folder, a.degree, a.width, a.kind, a.hidden, a.updates, a.seed)
    else:
        evaluate(
            a.root,
            a.folder,
            a.files,
            a.archive,
            a.out,
            a.substeps,
            a.mode,
            a.split,
        )
