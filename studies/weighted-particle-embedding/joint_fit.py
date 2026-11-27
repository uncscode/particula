"""Tune the weighted particle map against complete conservative pulse
forecasts.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from benchmark import load_model
from embedding import load_split
from forecast import read_encoder


def atomic_save(payload, path):
    """Publish a complete checkpoint by replacing a sibling temporary file."""
    tmp = path.with_suffix(".pending")
    torch.save(payload, tmp)
    tmp.replace(path)


def prep(root, split, encoder, device, extra=False):
    """Prepare the fixed two-leg experiment without using future states as
    inputs.
    """
    folder = (
        root / "data/pulse-extra" / split if extra else root / "data" / split
    )
    d = load_split(folder, device, grid=encoder.grid)
    schedules = []
    reset_values = []
    order = []
    for path in sorted(folder.glob("families-*.npz")):
        with np.load(path) as archive:
            time = archive["time"]
            boundary = np.flatnonzero(np.diff(time) == 0) + 1
            if len(boundary) != 1:
                raise ValueError(
                    "Joint two-leg fitting requires a single vapor reset"
                )
            gas = archive["gas"]
            specs = json.loads(str(archive["families"]))
            wet = np.repeat([f["wet_rh"] for f in specs], 2)
            dry = np.repeat([f["dry_rh"] for f in specs], 2)
            imposed = gas[0] / wet * dry
            np.testing.assert_allclose(
                imposed, gas[boundary[0]], rtol=1e-13, atol=1e-18
            )
            reset_values.extend(imposed)
            order.extend(np.repeat(archive["ids"], 2))
            schedules.append(time)
    for time in schedules[1:]:
        np.testing.assert_array_equal(schedules[0], time)
    np.testing.assert_array_equal(d["ids"].cpu().numpy(), order)
    d["time"] = torch.as_tensor(
        schedules[0], device=device, dtype=torch.float64
    )
    d["reset"] = torch.as_tensor(
        reset_values, device=device, dtype=torch.float64
    )
    return d


def join(groups):
    """Join disjoint families on a single common physical schedule."""
    if not all(torch.equal(groups[0]["time"], v["time"]) for v in groups):
        raise ValueError("Joint fit requires a common observation schedule")
    return {
        k: v if k == "time" else torch.cat([g[k] for g in groups])
        for k, v in groups[0].items()
    }


def cycle(encoder, model, x, counts, p, reset, time):
    """Predict every frame of both closed legs from one encoded observation."""
    n = len(p)
    samples = len(time) // 2
    process = p.new_tensor([0.0, 1.0]).expand(n, -1)
    z = (encoder.encode(x, counts) - model.z_mean) / model.z_std
    zs = []
    ps = []
    for leg, ages in enumerate(time.split(samples)):
        if leg:
            p = torch.cat([p[:, :4], reset[:, None]], 1)
        zz, pp = model.propagate(z, p, process, (ages - ages[0]).expand(n, -1))
        zs.append(zz)
        ps.append(pp)
        z, p = zz[:, -1], pp[:, -1]
    physical = torch.cat(ps, 1)
    shape = torch.cat(zs, 1) * model.z_std + model.z_mean
    diagnostics = encoder.diagnose(shape, physical)
    return shape, physical, diagnostics


def uncertainties(root, ids, device):
    """Align pilot reference refinements by family, independently of file
    order.
    """
    ref = {
        row["id"]: row
        for row in json.loads(
            (root / "validation-refinement.json").read_text()
        )["records"]
    }
    ids = ids[::2].tolist()
    return tuple(
        torch.tensor(
            np.stack([ref[key][k] for key in ids]),
            device=device,
            dtype=torch.float64,
        )
        for k in ("pair_contrast", "reference_error")
    )


def evaluate(model, enc, d, ref, bound):
    """Apply the resolved-signal criteria at every validation observation."""
    with torch.inference_mode():
        _, p, diag = cycle(
            enc,
            model,
            d["x"][:, 0],
            d["counts"][:, 0],
            d["physical"][:, 0],
            d["reset"],
            d["time"],
        )
        pred = p[1::2, :, 1] - p[::2, :, 1]
        mask = ref.abs() > 4 * bound
        error = (pred - ref).abs()
        relative = (error * mask).sum(0) / (ref.abs() * mask).sum(0).clamp_min(
            1e-30
        )
        signs = ((pred * ref > 0) & mask).sum(0) / mask.sum(0).clamp_min(1)
        passed = (relative <= 0.25) & (signs >= 0.9)
        passed[0] = True
        contiguous = int(passed.int().cumprod(0).sum() - 1)
        water = (p[..., 1] - d["physical"][..., 1]).abs()
        return dict(
            contiguous_frames=contiguous,
            contiguous_seconds=float(d["time"][contiguous]),
            qualified_observations=int(passed[1:].sum()),
            sign=signs.tolist(),
            relative=relative.tolist(),
            wet_mae_mg_m3=float(error[:, len(d["time"]) // 2 - 1].mean() * 1e6),
            dry_mae_mg_m3=float(error[:, -1].mean() * 1e6),
            member_mae_mg_m3=float(water.mean() * 1e6),
            diagnostic_nrmse=float(
                ((diag - d["diag"]) / enc.diag_std).square().mean().sqrt()
            ),
        )


def fit(args):
    """Fine-tune the encoder and dynamics with separate learning rates."""
    device = args.device
    torch.manual_seed(args.seed)
    model, info = load_model(args.root / args.start, device)
    if info["kind"] != "relaxation":
        raise ValueError("Joint cycle training requires closed-leg relaxation")
    enc = read_encoder(args.root / info["encoder"], device)
    train = join(
        [
            prep(args.root, "train", enc, device),
            prep(args.root, "train", enc, device, True),
        ]
    )
    valid = prep(args.root, "validation", enc, device)
    ref, bound = uncertainties(args.root, valid["ids"], device)
    dry = train["physical"][..., 2:4].sum(-1)
    true = train["physical"][..., 1] / dry
    typical = (
        (true[1::2] - true[::2]).square().mean(0).sqrt().clamp_min(0.00035)
    )
    opt = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": args.lr},
            {"params": enc.phi.parameters(), "lr": args.lr / 8},
            {"params": enc.decoder.parameters(), "lr": args.lr / 4},
        ],
        weight_decay=0.00005,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, args.steps, eta_min=args.lr / 80
    )
    best = None
    history = []
    name = f"joint-cycle-w{info['width']}-s{args.seed}"
    started = time.monotonic()
    for i in range(args.steps):
        family = torch.randint(len(dry) // 2, (args.batch // 2,), device=device)
        rows = torch.stack([family * 2, family * 2 + 1], 1).flatten()
        z, p, diag = cycle(
            enc,
            model,
            train["x"][rows, 0],
            train["counts"][rows, 0],
            train["physical"][rows, 0],
            train["reset"][rows],
            train["time"],
        )
        truth = train["physical"][rows]
        err = (p[..., 1] - truth[..., 1]) / dry[rows]
        raw = err[1::2] - err[::2]
        pair = (truth[1::2, :, 1] - truth[::2, :, 1]) / dry[rows[::2]]
        local = raw / torch.maximum(pair.abs(), 0.10 * typical).clamp_min(1e-5)

        def huber(v):
            return torch.nn.functional.smooth_l1_loss(v, torch.zeros_like(v))

        # A random true snapshot per trajectory anchors the diagnostic decoder.
        frame = torch.randint(len(train["time"]), (len(rows),), device=device)
        native_z = enc.encode(
            train["x"][rows, frame], train["counts"][rows, frame]
        )
        native_diag = enc.diagnose(
            native_z, truth[torch.arange(len(rows), device=device), frame]
        )
        observed = (
            ((native_diag - train["diag"][rows, frame]) / enc.diag_std)
            .square()
            .mean()
        )
        ahead = ((diag - train["diag"][rows]) / enc.diag_std).square().mean()
        loss = (
            2 * err.square().mean()
            + 0.25 * huber(raw / typical)
            + 0.12 * huber(local)
            + 0.25 * observed
            + 0.1 * ahead
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(enc.parameters()) + list(model.parameters()), 3
        )
        opt.step()
        sched.step()
        if (i + 1) % args.every == 0 or i + 1 == args.steps:
            metrics = evaluate(model, enc, valid, ref, bound)
            rank = (
                metrics["contiguous_frames"],
                metrics["qualified_observations"],
                -float(np.mean(metrics["relative"][1:]))
                - 0.05 * metrics["diagnostic_nrmse"],
            )
            history.append(dict(step=i + 1, validation=metrics))
            if best is None or rank > best:
                best = rank
                identity = f"{name}-encoder-{i + 1}.pt"
                atomic_save(
                    dict(
                        width=info["width"],
                        weights=enc.state_dict(),
                        grid=enc.grid,
                        representation=enc.representation,
                        calibration="training only",
                        parent=info["encoder"],
                    ),
                    args.root / identity,
                )
                atomic_save(
                    dict(
                        **{
                            k: info[k]
                            for k in (
                                "width",
                                "kind",
                                "kelvin",
                                "terms",
                                "hidden",
                            )
                        },
                        equilibrium_knots=info.get("equilibrium_knots", 0),
                        rate_scaling=info.get("rate_scaling", "pilot"),
                        weights=model.state_dict(),
                        encoder=identity,
                        encoder_sha256=hashlib.sha256(
                            (args.root / identity).read_bytes()
                        ).hexdigest(),
                        seed=args.seed,
                        step=i + 1,
                        parent=args.start,
                        validation=metrics,
                    ),
                    args.root / (name + ".pt"),
                )
            print(name, i + 1, rank, flush=True)
    (args.root / (name + ".json")).write_text(
        json.dumps(
            dict(history=history, elapsed_s=time.monotonic() - started),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("start")
    p.add_argument("--device", default="cpu")
    p.add_argument("--lr", type=float, default=0.0003)
    for k, v in (
        ("steps", 96000),
        ("seed", 711),
        ("batch", 128),
        ("every", 1000),
    ):
        p.add_argument("--" + k, type=int, default=v)
    fit(p.parse_args())
