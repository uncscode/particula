"""Learn the dimensionless single-particle jump distance for Brownian pairs."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from coupled_events import KT, MFP_AIR, MU_AIR
from stage2_accuracy import write_rows
from torch import nn


def target(logx):
    """Evaluate Fuchs' jump-distance ratio with a rationalized numerator."""
    x = np.exp(np.asarray(logx, dtype=np.longdouble))
    u = np.hypot(1, x)
    p = 1 + x
    ratio = (2 * (p * p + p * u + u * u) / (3 * (p + u)) - 1) / x
    return ratio.astype(np.float64)


class JumpDistance(nn.Module):
    """Predict a bounded jump distance once per live particle."""

    def __init__(self, width):
        """Construct an asymptotically anchored scalar neural correction."""
        super().__init__()
        self.width = width
        self.kind = "jump-distance"
        self.offset = nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.shape = nn.Sequential(
            nn.Linear(1, width), nn.Tanh(), nn.Linear(width, 1)
        ).double()
        nn.init.zeros_(self.shape[-1].weight)
        nn.init.zeros_(self.shape[-1].bias)

    def ratio(self, x):
        """Return the positive ratio of jump distance to mean path length."""
        logx = x.log()
        z = self.shape((logx / 3).unsqueeze(-1).to(self.offset.dtype)).squeeze(
            -1
        )
        return 0.5 + 0.5 * torch.sigmoid(logx + self.offset + z)

    @staticmethod
    def start(r):
        """Return a memoryless pair state compatible with the split driver."""
        return r.new_zeros((*r.shape, 0))

    def forward(self, r, mass, z, active, i, j, h):
        """Assemble symmetric positive pair rates from predicted distances."""
        r = torch.where(active, r, 80e-9)
        mass = torch.where(active, mass, 2e-18)
        kn = MFP_AIR / r
        diff = (
            KT
            * (1 + kn * (1.257 + 0.4 * torch.exp(-1.1 / kn)))
            / (6 * torch.pi * MU_AIR * r)
        )
        speed = (8 * KT / (torch.pi * mass)).sqrt()
        length = 8 * diff / (torch.pi * speed)
        g = length * self.ratio(length / (2 * r))
        rad = r[:, i] + r[:, j]
        d = diff[:, i] + diff[:, j]
        denom = rad / (rad + torch.hypot(g[:, i], g[:, j])) + 4 * d / (
            rad * torch.hypot(speed[:, i], speed[:, j])
        )
        return (4 * torch.pi * d * rad / denom) * (
            active[:, i] & active[:, j]
        ), h


def evaluate(net, low, high, num, offset):
    """Measure interpolated and extrapolated jump-ratio errors."""
    t = np.linspace(low, high, num) + offset
    exact = target(t)
    with torch.no_grad():
        est = net.ratio(torch.from_numpy(np.exp(t))).numpy()
    err = abs(est / exact - 1)
    return dict(
        worst_ratio_pct=100 * err.max(),
        p99_ratio_pct=100 * np.quantile(err, 0.99),
        rms_ratio_pct=100 * np.sqrt(np.mean(err**2)),
        min_ratio=est.min(),
        max_ratio=est.max(),
        monotonic_violations=int((np.diff(est) < 0).sum()),
    )


def fit(a):
    """Fit on dimensionless training nodes and select on an interlaced grid."""
    a.folder.mkdir(exist_ok=True, parents=True)
    t = np.linspace(-7, 6, 512)
    x, y = torch.from_numpy(np.exp(t)), torch.from_numpy(target(t))
    rows = []
    for width in a.widths:
        for seed in a.seeds:
            torch.manual_seed(seed)
            model = JumpDistance(width)
            opt = torch.optim.Adam(model.parameters(), lr=0.018)
            tic = perf_counter()
            best = np.inf
            for step in range(a.steps + 1):
                opt.zero_grad()
                rel = (model.ratio(x) / y - 1) * 1000
                loss = (
                    rel.square().mean()
                    + 0.15 * rel.square().topk(24).values.mean()
                )
                loss.backward()
                opt.step()
                if step in (a.steps // 2, 3 * a.steps // 4):
                    for g in opt.param_groups:
                        g["lr"] *= 0.25
                if step % 200 == 0 or step == a.steps:
                    val = evaluate(model, -7, 6, 1024, 0.00003)
                    if val["worst_ratio_pct"] < best:
                        best = val["worst_ratio_pct"]
                        torch.save(
                            dict(
                                kind=model.kind,
                                width=width,
                                seed=seed,
                                charged=False,
                                state=model.state_dict(),
                                update=step,
                                parameters=sum(
                                    p.numel() for p in model.parameters()
                                ),
                                validation=val,
                            ),
                            a.folder / f"jump-w{width}-s{seed}.pt",
                        )
            cp = torch.load(
                a.folder / f"jump-w{width}-s{seed}.pt", weights_only=False
            )
            model.load_state_dict(cp["state"])
            for lower, upper, domain in (
                (-7, 6, "interpolation"),
                (-12, 11, "expanded"),
            ):
                rows.append(
                    dict(
                        width=width,
                        seed=seed,
                        parameters=cp["parameters"],
                        selected_update=cp["update"],
                        domain=domain,
                        logx_min=lower,
                        logx_max=upper,
                        seconds=perf_counter() - tic,
                        **evaluate(model, lower, upper, 4001, 0.00007),
                    )
                )
            write_rows(a.output, rows)
            print(width, seed, best, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--widths", nargs="+", type=int, default=[4, 12, 32])
    p.add_argument("--seeds", nargs="+", type=int, default=[901, 902, 903])
    p.add_argument("--steps", type=int, default=4800)
    torch.set_num_threads(1)
    fit(p.parse_args())
