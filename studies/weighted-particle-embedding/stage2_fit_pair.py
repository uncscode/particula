"""Fit exchange-symmetric collision kernels at matched capacity."""

import argparse
import csv
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from stage2_pair import PairNet, load_pair


def read(file, device):
    """Load native pair-rate labels and their live-particle masks."""
    with np.load(file) as data:
        return {
            k: torch.as_tensor(data[k], device=device)
            for k in ("r", "mass", "charge", "active", "K", "family", "i", "j")
        }


def errors(net, d, indices, rate_weighted=False):
    """Compute log pair-rate error and relative total collision-hazard error."""
    r, m, z, active, k = [
        d[name][indices] for name in ("r", "mass", "charge", "active", "K")
    ]
    i, j = d["i"], d["j"]
    h = net.start(r[:, 0])
    logs, totals = [], []
    for t in range(r.shape[1]):
        live = active[:, t, i] & active[:, t, j]
        pred, h = net(r[:, t], m[:, t], z[:, t], active[:, t], i, j, h)
        err = (pred[live] / k[:, t][live]).log()
        if rate_weighted:
            mean_rate = (k[:, t] * live).sum(-1) / live.sum(-1).clamp_min(1)
            err = (
                err
                * (k[:, t] / mean_rate[:, None].clamp_min(1e-300))[live].sqrt()
            )
        logs.append(err)
        truth = (k[:, t] * live).sum(-1)
        totals.append(pred.sum(-1) / truth - 1)
    return torch.cat(logs), torch.cat(totals)


@torch.inference_mode()
def evaluate(net, data):
    """Measure validation tails for individual pairs and whole populations."""
    errors_all, totals = [], []
    for ids in torch.arange(len(data["r"]), device=data["r"].device).split(32):
        log_pair_error, t = errors(net, data, ids)
        errors_all.append(log_pair_error.expm1().abs())
        totals.append(t.abs())
    e = torch.cat(errors_all)
    t = torch.cat(totals)
    return dict(
        worst_pair=float(e.max()),
        p99_pair=float(torch.quantile(e, 0.99)),
        mean_pair=float(e.mean()),
        worst_hazard=float(t.max()),
        p99_hazard=float(torch.quantile(t, 0.99)),
        mean_hazard=float(t.mean()),
    )


def initial_pair(a):
    """Construct a budget-matched pair model or freeze its neutral backbone."""
    if a.neutral:
        parent = load_pair(a.neutral)
        kind, width = parent.kind, parent.width
    else:
        kind = a.kind
        options = [
            (
                abs(
                    sum(p.numel() for p in PairNet(kind, w).parameters())
                    - a.budget
                ),
                w,
            )
            for w in range(2, 96)
        ]
        width = a.width or min(options)[1]
    torch.manual_seed(a.seed)
    net = PairNet(kind, width, a.charged).to(a.device)
    if a.neutral:
        missing, extras = net.load_state_dict(parent.state_dict(), strict=False)
        if extras or any(not k.startswith("charge_head") for k in missing):
            raise ValueError("Neutral checkpoint differs from charged backbone")
        for k, p in net.named_parameters():
            p.requires_grad_(k.startswith("charge_head"))
    return net, kind, width


def run(a):
    """Fit a neutral backbone or its signed-charge extension."""
    torch.set_num_threads(1)
    train, valid = read(a.train, a.device), read(a.valid, a.device)
    if np.intersect1d(train["family"].cpu(), valid["family"].cpu()).size:
        raise ValueError("Family leakage")
    net, kind, width = initial_pair(a)
    params = [p for p in net.parameters() if p.requires_grad]
    count = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-6)
    rng = np.random.default_rng(a.seed)
    best = np.inf
    start = perf_counter()
    rows = []
    a.output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(a.train.read_bytes()).hexdigest()
    for it in range(a.updates + 1):
        if it:
            ids = torch.as_tensor(
                rng.integers(len(train["r"]), size=a.batch), device=a.device
            )
            e, t = errors(net, train, ids, a.rate_weighted)
            loss = (
                e.square().mean()
                + 0.1
                * e.abs().topk(max(1, e.numel() // 100)).values.square().mean()
                + 0.5 * t.square().mean()
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 2.0)
            opt.step()
            if it in (a.updates // 2, 3 * a.updates // 4):
                for g in opt.param_groups:
                    g["lr"] *= 0.4
        if it % a.eval_every == 0 or it == a.updates:
            row = dict(
                update=it,
                elapsed_s=perf_counter() - start,
                trainable=count,
                **evaluate(net, valid),
            )
            rows.append(row)
            metric = (
                row["p99_hazard"]
                + 0.25 * row["p99_pair"]
                + 0.05 * row["worst_pair"]
            )
            if metric < best:
                best = metric
                cp = dict(
                    state=net.state_dict(),
                    kind=kind,
                    width=width,
                    charged=a.charged,
                    seed=a.seed,
                    validation=row,
                    source_sha256=digest,
                    update=it,
                    neutral=a.neutral.name if a.neutral else "",
                )
                temp = a.output.with_suffix(".tmp")
                torch.save(cp, temp)
                temp.replace(a.output)
            with a.output.with_suffix(".csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            print(kind, it, row["p99_pair"], row["p99_hazard"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("train", "valid", "output"):
        p.add_argument(k, type=Path)
    p.add_argument(
        "--kind", choices=["ff", "gru", "set-attention", "causal"], default="ff"
    )
    p.add_argument("--rate-weighted", action="store_true")
    p.add_argument("--charged", action="store_true")
    p.add_argument("--neutral", type=Path)
    p.add_argument("--device", default="cpu")
    p.add_argument("--lr", type=float, default=0.003)
    for k, v in (
        ("updates", 1800),
        ("batch", 16),
        ("eval-every", 200),
        ("budget", 1500),
        ("width", 0),
        ("seed", 3301),
    ):
        p.add_argument("--" + k, type=int, default=v)
    run(p.parse_args())
