"""Scale water transfer by dry inventory and explicitly train on paired
contrasts.
"""

import argparse
import json
from pathlib import Path

import torch
from embedding import WeightedEncoder, errors_by_pair, load_split


def dry_mass(data):
    """Return the dry material concentration for each observation."""
    return data["physical"][..., 2:4].sum(-1, keepdim=True).float()


def normalize(data):
    """Express supervised water transfers per unit dry aerosol mass."""
    data["future"] = data["future"] / dry_mass(data).clamp_min(1e-20)
    return data


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
    target = data["future"]
    error = (pred - target) / model.future_std
    pred, target = pred * dry_mass(data), target * dry_mass(data)
    diagnostic = (
        model.decoder(state) - (data["diag"] - model.diag_mean) / model.diag_std
    )
    return dict(
        wet=errors_by_pair(pred[:, 0, 0], target[:, 0, 0]),
        full=errors_by_pair(pred[:, 0, 1], target[:, 0, 1]),
        diagnostic_nrmse=float(diagnostic.square().mean().sqrt()),
        normalized_future_rmse=float(error.square().mean().sqrt()),
    )


def fit(root, width, steps, seed, device):
    """Train the requested width and retain the best validation checkpoint."""
    torch.manual_seed(seed)
    train = normalize(load_split(root / "data/train", device))
    valid = normalize(load_split(root / "data/validation", device))
    model = WeightedEncoder(width).to(device)
    model.calibrate(train)
    n, t = train["x"].shape[:2]
    # Initial contrasts on drying can be much smaller than individual responses.
    # Use separate ensemble RMS scales for each target and observation time.
    contrasts = train["future"][1::2] - train["future"][::2]
    scales = contrasts.square().mean(0).sqrt().clamp_min(0.003)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0015)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, steps, eta_min=0.00008
    )
    name = f"paired-w{width}-s{seed}"
    history = []
    best = float("inf")
    for i in range(steps):
        member = 2 * torch.randint(n // 2, (128,), device=device)
        obs = torch.randint(t, (128,), device=device)
        obs[:32] = 0
        row = torch.stack((member, member + 1))
        x, counts, p, d, f, y = (
            train[k][row, obs]
            for k in ("x", "counts", "physical", "diag", "forcing", "future")
        )
        state = model.state(model.encode(x, counts), p)
        response = (
            model.response(torch.cat([state, f], -1)) * model.future_std
            + model.future_mean
        )
        pair_error = ((response[1] - response[0]) - (y[1] - y[0])) / scales[obs]
        direct = ((response - y) / model.future_std).square().mean()
        diag = (
            (model.decoder(state) - (d - model.diag_mean) / model.diag_std)
            .square()
            .mean()
        )
        objective = (
            diag
            + direct
            + 0.25
            * torch.nn.functional.smooth_l1_loss(
                pair_error, torch.zeros_like(pair_error)
            )
        )
        opt.zero_grad()
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        opt.step()
        sched.step()
        if (i + 1) % 500 == 0 or i + 1 == steps:
            metrics = evaluate(model, valid)
            history.append(dict(step=i + 1, **metrics))
            score = (
                metrics["diagnostic_nrmse"]
                + metrics["wet"]["pair_abs_error_over_mean_abs_signal"]
                + metrics["full"]["pair_abs_error_over_mean_abs_signal"] * 0.3
            )
            if score < best:
                best = score
                torch.save(
                    dict(
                        width=width,
                        seed=seed,
                        step=i + 1,
                        validation=metrics,
                        weights=model.state_dict(),
                        target_units="kg_water/kg_dry",
                    ),
                    root / (name + ".pt"),
                )
    (root / (name + ".json")).write_text(json.dumps(history, indent=2) + "\n")
    print(name, history[-1], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--widths", type=int, nargs="+", default=[8, 16, 32])
    p.add_argument("--seed", type=int, default=2081)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    for w in a.widths:
        fit(a.root, w, a.steps, a.seed, a.device)
