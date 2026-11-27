"""Fit and checkpoint weighted encoders with reconstruction and response
losses.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from embedding import WeightedEncoder, errors_by_pair, load_split


def flat(data):
    """Combine the box and observation axes for snapshot minibatches."""
    return {k: v.flatten(0, 1) for k, v in data.items() if k != "ids"}


def loss(model, data, idx, predictive):
    """Measure reconstruction and known-forcing response on a minibatch."""
    x, weight, p, diag, f, y = (
        data[k][idx]
        for k in ("x", "counts", "physical", "diag", "forcing", "future")
    )
    state = model.state(model.encode(x, weight), p)
    diagnostics = model.decoder(state)
    ldiag = (
        (diagnostics - (diag - model.diag_mean) / model.diag_std)
        .square()
        .mean()
    )
    # The control diagnostic probe learns on detached reconstruction-only
    # embeddings.
    response = model.response(
        torch.cat([state if predictive else state.detach(), f], -1)
    )
    lresp = (
        (response - (y - model.future_mean) / model.future_std).square().mean()
    )
    return ldiag, lresp


@torch.no_grad()
def evaluate(model, data):
    """Score individual trajectories and paired response on unseen families."""
    z = model.encode(data["x"], data["counts"])
    state = model.state(z, data["physical"])
    pred = (
        model.response(torch.cat([state, data["forcing"]], -1))
        * model.future_std
        + model.future_mean
    )
    y = data["future"]
    diagnostics = (
        model.decoder(state) - (data["diag"] - model.diag_mean) / model.diag_std
    )
    return dict(
        wet=errors_by_pair(pred[:, 0, 0], y[:, 0, 0]),
        full=errors_by_pair(pred[:, 0, 1], y[:, 0, 1]),
        diagnostic_nrmse=float(diagnostics.square().mean().sqrt()),
        normalized_future_rmse=float(
            ((pred - y) / model.future_std).square().mean().sqrt()
        ),
    )


def fit(root, width, steps, predictive, seed, device):
    """Train the requested width and retain the best validation checkpoint."""
    torch.manual_seed(seed)
    train = load_split(root / "data/train", device)
    valid = load_split(root / "data/validation", device)
    if set(train["ids"].tolist()) & set(valid["ids"].tolist()):
        raise ValueError("Initial-condition family occurs in both splits")
    model = WeightedEncoder(width).to(device)
    model.calibrate(train)
    train = flat(train)
    opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0001)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, steps, eta_min=0.00015
    )
    best = float("inf")
    name = f"encoder-w{width}-p{int(predictive)}-s{seed}"
    history = []
    start = time.monotonic()
    for i in range(steps):
        idx = torch.randint(len(train["x"]), (256,), device=device)
        a, b = loss(model, train, idx, predictive)
        opt.zero_grad()
        (a + b).backward()
        opt.step()
        schedule.step()
        if (i + 1) % 200 == 0 or i + 1 == steps:
            metrics = evaluate(model, valid)
            score = metrics["diagnostic_nrmse"] + (
                metrics["wet"]["pair_abs_error_over_mean_abs_signal"]
                if predictive
                else 0.0
            )
            history.append(dict(step=i + 1, **metrics))
            if score < best:
                best = score
                torch.save(
                    dict(
                        width=width,
                        weights=model.state_dict(),
                        seed=seed,
                        step=i + 1,
                        predictive=predictive,
                        validation=metrics,
                    ),
                    root / (name + ".pt"),
                )
    record = dict(
        name=name,
        steps=steps,
        elapsed_s=time.monotonic() - start,
        history=history,
    )
    (root / (name + ".json")).write_text(json.dumps(record, indent=2) + "\n")
    print(name, history[-1], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--widths", nargs="+", type=int, default=[8, 16, 32])
    p.add_argument("--seed", type=int, default=20261125)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    for w in args.widths:
        for predictive in (True, False):
            fit(args.root, w, args.steps, predictive, args.seed, args.device)
