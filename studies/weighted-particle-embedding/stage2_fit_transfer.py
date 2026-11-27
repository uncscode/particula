"""Fit conservative vapor operators to particle-resolved trajectories."""

import argparse
import csv
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from stage2_transfer import PARENT, Transfer
from vapor_properties import materials


def dataset(path, device):
    """Load the node histories, physical targets and family identifiers."""
    with np.load(path) as z:
        return {
            k: torch.as_tensor(z[k], device=device)
            for k in ("m", "number", "levels", "mask", "q", "g", "t", "ids")
        }


def model_width(kind, budget):
    # Count actual weights for this architecture, including GRU biases.
    """Choose the width closest to a common trainable-parameter budget."""
    sizes = []
    for w in range(2, 129):
        model = Transfer(materials(2), kind, w)
        sizes.append(
            (abs(sum(p.numel() for p in model.parameters()) - budget), w)
        )
    return min(sizes)[1]


def errors(q, g, truth, reference, weight):
    """Compute particle, bulk aerosol and gas errors on the current step."""
    scale = truth.clamp_min(1e-7)
    node = (q - truth) / scale
    bulk = (q * weight[..., None]).sum(1)
    true_bulk = (truth * weight[..., None]).sum(1)
    wet = (bulk - true_bulk) / true_bulk.clamp_min(1e-11)
    gas = (g - reference) / reference.clamp_min(1e-11)
    return node, wet, gas


def loss_terms(node, wet, gas):
    # Unit error equals the required relative tolerance. Log-compression
    # permits optimization across transient defects spanning several decades.
    """Scale node and phase errors to the requested physical tolerances."""

    def penalty(x):
        return torch.log1p((x * 1000).square())

    bulk = torch.cat((penalty(wet), penalty(gas)), -1)
    return (
        0.25 * penalty(node).mean() + bulk.mean() + 0.3 * bulk.amax(-1).mean()
    )


def objective(net, d, rng, batch, window, substeps, device):
    """Supervise multistep states and integrated particle-level flux."""
    batches, frames = d["q"].shape[:2]
    ids = torch.as_tensor(rng.integers(batches, size=batch), device=device)
    leg_frames = frames // d["levels"].shape[1]
    if rng.random() < 0.55:
        start = min(
            int(rng.integers(d["levels"].shape[1])) * leg_frames,
            frames - window - 1,
        )
    else:
        start = int(rng.integers(frames - window))
    static = net.static(d["m"][ids], d["number"][ids])
    q, g = d["q"][ids, start], d["g"][ids, start]
    h = net.start(d["number"][ids])
    losses = []
    for k in range(start + 1, start + window + 1):
        if k % leg_frames == 0:
            leg = k // leg_frames
            g = torch.where(d["mask"][ids, leg], d["levels"][ids, leg], g)
            continue
        q0 = q
        dt = (d["t"][ids, k] - d["t"][ids, k - 1]) / substeps
        for _ in range(substeps):
            q, g, h = net.step(q, g, static, dt, h)
        n, w, v = errors(q, g, d["q"][ids, k], d["g"][ids, k], static[3])
        # Supervise the integrated particle-level flux as well as phase states.
        delta = d["q"][ids, k] - d["q"][ids, k - 1]
        inc = (q - q0 - delta) / (
            abs(delta) + 0.02 * d["q"][ids, k].clamp_min(1e-7)
        )
        if net.active_lanes is not None:
            n, w, v, inc = (x[..., net.active_lanes] for x in (n, w, v, inc))
        losses.append(
            loss_terms(n, w, v) + 0.03 * torch.log1p((inc * 10).square()).mean()
        )
    if not losses:
        return next(net.parameters()).sum() * 0.0
    return torch.stack(losses).mean()


@torch.inference_mode()
def validate(net, d, substeps):
    """Score uninterrupted forecasts on independent development families."""
    m, n = d["m"], d["number"]
    if not torch.equal(d["t"], d["t"][:1].expand_as(d["t"])):
        raise ValueError("Validation needs common observation times")
    wet, gas = net.rollout(
        m, n, d["levels"], d["t"][0].cpu().numpy(), substeps, d["mask"]
    )
    weight = n[..., None]
    wb = (wet * weight[:, None]).sum(2)
    dry = m[..., -2:].sum(-1)
    true = (d["q"] * (n * dry)[:, None, :, None]).sum(2)
    ew, eg = (
        abs(wb - true) / true.clamp_min(1e-11),
        abs(gas - d["g"]) / d["g"].clamp_min(1e-11),
    )
    if net.active_lanes is not None:
        ew, eg = ew[..., net.active_lanes], eg[..., net.active_lanes]
    e = torch.cat((ew, eg), -1)
    return (
        float(e.max()),
        float(torch.quantile(e.flatten(), 0.99)),
        float((e < 0.001).double().mean()),
    )


def initial_transfer(args, width):
    """Bind the chosen dynamics, active species and initial head weights."""
    if args.radau:
        from stage2_radau import Radau

        net = Radau(materials(2), *args.radau, args.kind, width).to(args.device)
    else:
        net = Transfer(materials(2), args.kind, width).to(args.device)
    if args.active != "all":
        net.active_lanes = torch.arange(2, device=args.device) == (
            0 if args.active == "water" else 1
        )
    if args.warm:
        parent = torch.load(
            PARENT, map_location=args.device, weights_only=False
        )["head"]
        (net.head.shared if args.kind == "water" else net.head).load_state_dict(
            parent
        )
    return net


def run(args):
    """Fit a transfer head and save the best validation checkpoint."""
    torch.set_num_threads(1)
    d = dataset(args.train, args.device)
    if args.augment:
        extra = dataset(args.augment, args.device)
        d = {k: torch.cat((v, extra[k])) for k, v in d.items()}
    valid = dataset(args.valid, args.device)
    if np.intersect1d(d["ids"].cpu(), valid["ids"].cpu()).size:
        raise ValueError("Family leakage between training and validation")
    width = args.width or model_width(args.kind, args.budget)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    net = initial_transfer(args, width)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-5)
    start = perf_counter()
    best = (np.inf, np.inf, np.inf)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    parameters = sum(p.numel() for p in net.parameters())
    for update in range(args.updates + 1):
        if update:
            opt.zero_grad()
            loss = objective(
                net, d, rng, args.batch, args.window, args.substeps, args.device
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            if not torch.isfinite(loss):
                raise ArithmeticError("Nonfinite fitting loss")
        if update % args.eval_every == 0 or update == args.updates:
            worst, q99, passes = validate(net, valid, args.substeps)
            # Exact donor exhaustion can give identical worst errors for every
            # fit. Break numerical ties on tail error and qualified coverage.
            metric = (round(worst, 8), round(q99, 8), -passes)
            row = dict(
                update=update,
                worst_relative=worst,
                q99_relative=q99,
                fraction_passing=passes,
                training_loss=float(loss.detach()) if update else 0.0,
                elapsed_s=perf_counter() - start,
                parameters=parameters,
            )
            rows.append(row)
            if metric < best:
                best = metric
                cp = dict(
                    kind=args.kind,
                    width=width,
                    head=net.head.state_dict(),
                    seed=args.seed,
                    update=update,
                    parameters=parameters,
                    validation=row,
                    source_sha256=hashlib.sha256(
                        args.train.read_bytes()
                    ).hexdigest(),
                    substeps=args.substeps,
                    window=args.window,
                    batch=args.batch,
                    radau=args.radau,
                    active=args.active,
                    augmentation_sha256=hashlib.sha256(
                        args.augment.read_bytes()
                    ).hexdigest()
                    if args.augment
                    else "",
                )
                temp = args.output.with_suffix(".tmp")
                torch.save(cp, temp)
                temp.replace(args.output)
            with args.output.with_suffix(".csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            print(args.kind, update, worst, q99, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("train", "valid", "output"):
        p.add_argument(k, type=Path)
    p.add_argument(
        "--kind",
        choices=["ff", "gru", "attention", "causal", "water", "mixture"],
        required=True,
    )
    p.add_argument("--warm", action="store_true")
    p.add_argument("--augment", type=Path)
    p.add_argument(
        "--active", choices=["all", "water", "organic"], default="all"
    )
    p.add_argument("--radau", nargs=2, type=int)
    p.add_argument("--device", default="cpu")
    p.add_argument("--lr", type=float, default=0.002)
    for k, v in (
        ("updates", 600),
        ("eval-every", 100),
        ("seed", 2081),
        ("budget", 1500),
        ("width", 0),
        ("batch", 16),
        ("window", 3),
        ("substeps", 1),
    ):
        p.add_argument("--" + k, type=int, default=v)
    run(p.parse_args())
