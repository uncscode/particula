"""Fit multi-step latent forecasts and keep full-validation rollout metrics."""

import argparse
import json
import time
from pathlib import Path

import torch
from embedding import errors_by_pair
from forecast import MODES, Dynamics, read_encoder, trajectories


def norms(pred, truth, dry):
    """Report absolute and dry-inventory-normalized water errors."""
    delta = (pred - truth).abs()
    return dict(
        mae=float(delta.mean()),
        p95=float(torch.quantile(delta.flatten(), 0.95)),
        dry_normalized_mae=float((delta / dry).mean()),
    )


def predict(model, data, persistence=False, start=0, observed=1):
    """Initialize from observations and roll forward under known forcing."""
    z = (data["z"] - model.z_mean) / model.z_std
    p = data["p"]
    force = data["force"]
    if persistence:
        steps = force.shape[1] - start
        guess = p[:, start : start + 1].expand(-1, steps + 1, -1).clone()
        for i in range(steps):
            f = force[:, start + i]
            guess[:, i + 1, 4] = torch.where(
                f[:, 3] > 0, f[:, 4], guess[:, i, 4]
            )
        return z[:, start : start + 1].expand(-1, steps + 1, -1), guess
    begin = max(0, start - observed + 1)
    history = model.features(
        z[:, begin:start], p[:, begin:start], force[:, begin:start]
    )
    return model.rollout(z[:, start], p[:, start], force[:, start:], history)


@torch.no_grad()
def evaluate(
    model, encoder, data, mode, persistence=False, start=0, observed=1
):
    """Score individual trajectories and paired response on unseen families."""
    model.eval()
    z, p = predict(model, data, persistence, start, observed)
    true_p = data["p"][:, start:]
    true_z = data["z"][:, start:]
    physical_z = z * model.z_std + model.z_mean
    diag = encoder.diagnose(physical_z, p)
    oracle = encoder.diagnose(true_z, true_p)
    truth = data["diag"][:, start:]
    dry = true_p[..., 2] + true_p[..., 3]
    dscale = encoder.diag_std
    mid = data["p"].shape[1] // 2 - 1 - start
    result = dict(
        water_kg_m3=norms(p[..., 1], true_p[..., 1], dry),
        number_fraction_mae=float(
            ((p[..., 0] - true_p[..., 0]) / data["p"][:, 0:1, 0]).abs().mean()
        ),
        decoded_oracle_nrmse=float(
            ((oracle - truth) / dscale).square().mean().sqrt()
        ),
        rollout_diagnostic_nrmse=float(
            ((diag - truth) / dscale).square().mean().sqrt()
        ),
        added_diagnostic_nrmse=float(
            ((diag - oracle) / dscale).square().mean().sqrt()
        ),
        shape_rmse=float(
            ((physical_z - true_z) / model.z_std).square().mean().sqrt()
        ),
        number_increase=float(torch.diff(p[..., 0]).clamp_min(0).max()),
        dry_drift_kg_m3=float((p[..., 2:4] - true_p[:, 0:1, 2:4]).abs().max()),
        water_budget_kg_m3=float(
            torch.where(
                data["force"][:, start:, 3] > 0,
                0.0,
                torch.diff(p[..., 1] + p[..., 4], dim=1),
            )
            .abs()
            .max()
        ),
        minimum_inventory=float(p[..., 1:].min()),
    )
    endpoints = [(1, "one"), (mid, "wet_end"), (-1, "dry_end")]
    if mode == "multicycle":
        first_reset = (data["force"][:, start:, 3] > 0).int().argmax(1)
        endpoints = [(1, "one"), (first_reset, "wet_end"), (-1, "dry_end")]
    for step, name in endpoints:
        if isinstance(step, int) and (step == 0 or step < -1):
            continue
        if isinstance(step, torch.Tensor):
            boxes = torch.arange(len(p), device=p.device)
            q, true = p[boxes, step], true_p[boxes, step]
        else:
            q, true = p[:, step], true_p[:, step]
        entry = dict(
            water_mae_mg_m3=float((q[:, 1] - true[:, 1]).abs().mean() * 1e6),
            number_fraction_mae=float(
                ((q[:, 0] - true[:, 0]) / data["p"][:, 0, 0]).abs().mean()
            ),
        )
        if mode != "coagulation":
            entry.update(
                errors_by_pair(
                    q[:, 1] - p[:, 0, 1], true[:, 1] - true_p[:, 0, 1]
                )
            )
        result[name] = entry
    model.train()
    return result


def validation_score(met, paired):
    """Rank free rollouts across process regimes on the validation set."""
    score = sum(
        v["water_kg_m3"]["dry_normalized_mae"]
        + v["number_fraction_mae"]
        + 0.1 * v["shape_rmse"]
        for v in met.values()
    )
    if paired and "pulse" in met:
        score += (
            0.5 * met["pulse"]["wet_end"]["pair_abs_error_over_mean_abs_signal"]
        )
        score += (
            0.1 * met["pulse"]["dry_end"]["pair_abs_error_over_mean_abs_signal"]
        )
    return score


def check_family_splits(train, valid):
    """Keep all seeds, concentrations and mixtures of a family together."""
    for m in train:
        if set(train[m]["id"].tolist()) & set(valid[m]["id"].tolist()):
            raise RuntimeError("Initial-condition family leaked between splits")


def build_model(
    width,
    kind,
    history,
    hidden,
    substeps,
    target_parameters,
    thermal_transfer=False,
):
    """Choose a hidden width for a common trainable parameter budget."""
    from continuous import ContinuousODE

    def construct(size):
        if kind == "continuous":
            return ContinuousODE(width, size, substeps)
        return Dynamics(
            width, kind, history, size, thermal_transfer=thermal_transfer
        )

    if not target_parameters:
        return construct(hidden), hidden
    stride = 4 if kind == "transformer" else 1
    lo, hi = 1, 1

    def parameters(n):
        with torch.device("meta"):
            return sum(p.numel() for p in construct(n * stride).parameters())

    while parameters(hi) < target_parameters:
        hi *= 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if parameters(mid) < target_parameters:
            lo = mid
        else:
            hi = mid
    actual = (
        min((lo, hi), key=lambda n: abs(parameters(n) - target_parameters))
        * stride
    )
    if kind == "transformer":
        deviation = target_parameters - parameters(actual // stride)
        feedforward = max(1, 2 * actual + round(deviation / (4 * actual + 2)))
        return Dynamics(
            width, kind, history, actual, feedforward, thermal_transfer
        ), actual
    return construct(actual), actual


def warm_start(args, model, width, hidden):
    """Restore a fit with matching features and dynamics dimensions."""
    if args.initialize:
        prior = torch.load(
            args.root / args.initialize,
            map_location=args.device,
            weights_only=True,
        )
        identity = (
            prior["encoder"],
            prior["kind"],
            prior["width"],
            prior["hidden"],
            prior.get("history", 1),
        )
        if identity != (args.encoder, args.kind, width, hidden, args.history):
            raise ValueError(
                "Warm start requires matching features and architecture"
            )
        model.load_state_dict(prior["weights"])


def train(args):
    """Fit on trajectory windows and select on full free-running histories."""
    root = args.root
    device = args.device
    torch.manual_seed(args.seed)
    enc = read_encoder(root / args.encoder, device)
    enc.requires_grad_(False)
    roots = {
        m: args.collision_root
        if args.collision_root and m in ("coagulation", "coupled")
        else root
        for m in args.modes
    }
    train = {
        m: trajectories(roots[m], "train", m, enc, device) for m in args.modes
    }
    valid = {
        m: trajectories(roots[m], "validation", m, enc, device)
        for m in args.modes
    }
    check_family_splits(train, valid)
    width = train[args.modes[0]]["z"].shape[-1]
    model, hidden = build_model(
        width,
        args.kind,
        args.history,
        args.hidden,
        args.substeps,
        args.target_parameters,
        args.thermal_transfer,
    )
    model = model.to(device)
    parameter_count = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    all_z = torch.cat([d["z"].flatten(0, 1) for d in train.values()])
    model.z_mean.copy_(all_z.mean(0))
    model.z_std.copy_(all_z.std(0).clamp_min(0.01))
    warm_start(args, model, width, hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, args.steps, eta_min=args.lr / 20
    )
    name = f"{args.tag}-{args.kind}-h{args.history}-w{width}-s{args.seed}"
    history = []
    contrast_scale = {}
    for mode, data in train.items():
        water_ratio = data["p"][..., 1] / data["p"][..., 2:4].sum(-1)
        contrast_scale[mode] = (
            (water_ratio[1::2] - water_ratio[::2])
            .square()
            .mean(0)
            .sqrt()
            .clamp_min(0.002)
        )
    best = float("inf")
    started = time.monotonic()
    for i in range(args.steps):
        mode = args.modes[i % len(args.modes)]
        visit = i // len(args.modes)
        d = train[mode]
        n, t = d["p"].shape[:2]
        horizon = min(args.horizon if visit % 4 == 3 else 1 + visit % 3, t - 1)
        idx = 2 * torch.randint(n // 2, (args.batch // 2,), device=device)
        idx = torch.stack([idx, idx + 1], -1).flatten()
        start = int(torch.randint(t - horizon, ()).item()) if visit % 5 else 0
        if args.paired and visit % 5 == 0:
            start = 0
            horizon = t - 1 if visit % 10 == 0 else t // 2 - 1
        z = (d["z"][idx] - model.z_mean) / model.z_std
        p = d["p"][idx]
        f = d["force"][idx]
        begin = max(0, start - args.history + 1) if visit % 3 else start
        prior = model.features(
            z[:, begin:start], p[:, begin:start], f[:, begin:start]
        )
        zh, ph = model.rollout(
            z[:, start], p[:, start], f[:, start : start + horizon], prior
        )
        target = p[:, start + 1 : start + horizon + 1]
        dry = target[..., 2:4].sum(-1)
        water = (ph[:, 1:, 1] - target[..., 1]) / dry
        number = (ph[:, 1:, 0] - target[..., 0]) / p[:, 0:1, 0]
        shape = zh[:, 1:] - z[:, start + 1 : start + horizon + 1]
        objective = (
            shape.square().mean()
            + water.square().mean()
            + 4 * number.square().mean()
        )
        if mode == "pulse":
            pair = water[1::2] - water[::2]
            objective = objective + 2 * pair.square().mean()
        if args.paired and mode != "coagulation":
            contrast_error = water[1::2] - water[::2]
            normalized = (
                contrast_error
                / contrast_scale[mode][start + 1 : start + horizon + 1]
            )
            objective = objective + 0.18 * torch.nn.functional.smooth_l1_loss(
                normalized, torch.zeros_like(normalized)
            )
        if args.refinement_loss and visit % 4 == 0:
            # Compare solutions of the identical vector field at one interval.
            local_force = f[:, start : start + 1]
            model.substeps = 2 * args.substeps
            with torch.no_grad():
                fine_z, fine_p = model.rollout(
                    z[:, start], p[:, start], local_force
                )
            model.substeps = args.substeps
            shape_discrepancy = (zh[:, 1] - fine_z[:, 1]).square().mean()
            water_discrepancy = (
                ((ph[:, 1, 1] - fine_p[:, 1, 1]) / dry[:, 0]).square().mean()
            )
            objective = objective + args.refinement_loss * (
                shape_discrepancy + water_discrepancy
            )
        opt.zero_grad()
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
        opt.step()
        sched.step()
        if (i + 1) % args.evaluate_every == 0 or i + 1 == args.steps:
            met = {m: evaluate(model, enc, v, m) for m, v in valid.items()}
            score = validation_score(met, args.paired)
            if not torch.isfinite(torch.tensor(score)):
                raise FloatingPointError(
                    f"Nonfinite validation {name} at {i + 1}"
                )
            history.append(dict(step=i + 1, score=score, validation=met))
            if score < best:
                best = score
                torch.save(
                    dict(
                        weights=model.state_dict(),
                        width=width,
                        kind=args.kind,
                        hidden=hidden,
                        trainable_parameters=parameter_count,
                        feedforward=getattr(model, "feedforward", None),
                        refinement_loss=args.refinement_loss,
                        thermal_transfer=args.thermal_transfer,
                        initialize=args.initialize,
                        history=args.history,
                        substeps=args.substeps,
                        step=i + 1,
                        encoder=args.encoder,
                        modes=args.modes,
                        collision_root=str(args.collision_root)
                        if args.collision_root
                        else None,
                        seed=args.seed,
                        validation=met,
                    ),
                    root / (name + ".pt"),
                )
            print(name, i + 1, "validation score", score, flush=True)
    baseline = {m: evaluate(model, enc, v, m, True) for m, v in valid.items()}
    (root / (name + ".json")).write_text(
        json.dumps(
            dict(
                history=history,
                trainable_parameters=parameter_count,
                hidden=hidden,
                matched_parameter_target=args.target_parameters,
                persistence=baseline,
                elapsed_s=time.monotonic() - started,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument(
        "kind",
        choices=["ode", "residual", "recurrent", "transformer", "continuous"],
    )
    p.add_argument(
        "--modes",
        choices=(*MODES, "multicycle"),
        nargs="+",
        default=list(MODES),
    )
    p.add_argument("--encoder", default="paired-w8-s2081.pt")
    p.add_argument("--tag", default="joint")
    p.add_argument("--initialize")
    p.add_argument("--collision-root", type=Path)
    p.add_argument("--thermal-transfer", action="store_true")
    p.add_argument("--paired", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--target-parameters", type=int, default=0)
    p.add_argument("--refinement-loss", type=float, default=0.0)
    for key, value in (
        ("seed", 77),
        ("substeps", 2),
        ("history", 1),
        ("hidden", 64),
        ("steps", 2000),
        ("batch", 64),
        ("horizon", 8),
        ("evaluate-every", 500),
    ):
        p.add_argument("--" + key, type=int, default=value)
    args = p.parse_args()
    if args.refinement_loss and args.kind != "continuous":
        p.error("Refinement loss requires a continuous vector field")
    if args.thermal_transfer and args.kind == "continuous":
        p.error("The continuous ODE has its own equilibrium closure")
    train(args)
