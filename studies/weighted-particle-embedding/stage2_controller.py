"""Predict the benefit of short ODE corrections from the current state."""

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from stage2_accuracy import write_rows
from stage2_fit_transfer import dataset
from stage2_transfer import restore
from torch import nn
from vapor_properties import materials


def features(net, q, g, state, dt, age):
    """Return box features in the same order during fitting and deployment."""
    w = state[3]
    live = state[5]
    rate = net.linearization(q, g, state, rate_only=True)
    wet = (w[..., None] * q).sum(1)
    flux = dt[:, None] * (w[..., None] * rate).sum(1)
    activity = (dt[:, None, None] * rate.abs() / q.clamp_min(1e-7)) * live[
        ..., None
    ]
    n = (w / state[0]).sum(-1)
    dry_r = (3 * state[2] / (4 * torch.pi)).pow(1 / 3) * 1e9
    present = live.sum(-1).clamp_min(1)
    node_size = (dry_r * live).sum(-1) / present
    frac = state[4]
    avg = (frac * live).sum(-1) / present
    spread = (((frac - avg[:, None]).square() * live).sum(-1) / present).sqrt()
    idx = slice(1, None) if q.shape[-1] > 1 else slice(0, 1)
    a = flux / wet.clamp_min(1e-11)
    b = flux / g.clamp_min(1e-11)
    state_vars = (
        torch.log1p(dt * 1000),
        torch.log1p(age),
        torch.log1p(n * 1e-14),
        torch.log1p(dt * n * 2e-15),
        torch.log1p(node_size),
        spread,
        (g / net.sat)[:, 0],
        (g / net.sat)[:, idx].mean(-1),
        torch.asinh(a[:, 0]),
        torch.asinh(b[:, 0]),
        torch.asinh(a[:, idx].abs().amax(-1)),
        torch.asinh(b[:, idx].abs().amax(-1)),
        torch.log1p(activity[:, :, 0].amax(-1)),
        torch.log1p(activity[:, :, idx].amax((-1, -2))),
    )
    return torch.stack(state_vars, -1)


class Benefit(nn.Module):
    """Estimate the future benefit of replacing one neural step."""

    def __init__(self, width=32):
        """Allocate the feature normalization and scalar regression layers."""
        super().__init__()
        self.register_buffer("mean", torch.zeros(14, dtype=torch.float64))
        self.register_buffer("scale", torch.ones(14, dtype=torch.float64))
        self.fit = nn.Sequential(
            nn.Linear(14, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        ).double()

    def forward(self, x):
        """Predict log-transformed improvement in the local mass forecast."""
        return self.fit((x - self.mean) / self.scale).squeeze(-1)


def load_controller(file, device):
    """Restore fitted weights and validation-calibrated budget thresholds."""
    cp = torch.load(file, map_location="cpu", weights_only=False)
    net = Benefit(cp["width"])
    net.load_state_dict(cp["state"])
    return net.to(device), cp["thresholds"]


@torch.inference_mode()
def labels(d, transfer, substeps, horizon, batch):
    """Compare two causal continuations from a common training state."""
    model = restore(materials(2), transfer)
    allx, ally = [], []
    for start in range(0, len(d["m"]), batch):
        ss = slice(start, start + batch)
        m, n = d["m"][ss], d["number"][ss]
        ctx = model.static(m, n)
        t, q, g = d["t"][ss], d["q"][ss], d["g"][ss]
        nf = q.shape[1] // d["levels"].shape[1]
        for k in range(q.shape[1] - 1):
            if (k + 1) % nf == 0:
                continue
            dt = t[:, k + 1] - t[:, k]
            x = features(
                model,
                q[:, k],
                g[:, k],
                ctx,
                dt,
                dt.new_full(dt.shape, float(k % nf)),
            )
            # Both branches start from exactly the same available state.
            # During supervision the second branch gets just one ODE endpoint.
            paths = [
                (q[:, k], g[:, k], model.start(n)),
                (q[:, k + 1], g[:, k + 1], model.start(n)),
            ]
            loss = [torch.zeros_like(dt), torch.zeros_like(dt)]
            for lead in range(horizon):
                j = k + lead + 1
                if j >= q.shape[1] or j // nf != k // nf:
                    break
                for branch in (0, 1):
                    qq, gg, hh = paths[branch]
                    if not (branch == 1 and lead == 0):
                        for _ in range(substeps):
                            qq, gg, hh = model.step(
                                qq, gg, ctx, dt / substeps, hh
                            )
                    paths[branch] = qq, gg, hh
                    bulk = (ctx[3][..., None] * q[:, j]).sum(1)
                    e1 = (
                        (ctx[3][..., None] * (qq - q[:, j])).sum(1).abs()
                        / bulk.clamp_min(1e-11)
                    ).amax(-1)
                    e2 = ((gg - g[:, j]).abs() / g[:, j].clamp_min(1e-11)).amax(
                        -1
                    )
                    loss[branch] = torch.maximum(
                        loss[branch], torch.maximum(e1, e2)
                    )
            allx.append(x)
            ally.append(torch.stack((loss[0], loss[1]), -1))
    return torch.cat(allx), torch.cat(ally)


def run(a):
    """Fit the correction selector and measure its validation recall."""
    torch.set_num_threads(1)
    a.cache.mkdir(exist_ok=True, parents=True)
    sets = []
    for file in (a.train, a.augment, a.valid):
        cache = a.cache / (
            f"{file.stem}-{Path(a.transfer).stem}-k{a.substeps}"
            f"-h{a.horizon}.npz"
        )
        if not cache.exists():
            x, y = labels(
                dataset(file, "cpu"), a.transfer, a.substeps, a.horizon, a.batch
            )
            np.savez_compressed(cache, x=x.numpy(), y=y.numpy())
        with np.load(cache) as data:
            sets.append(tuple(torch.tensor(data[k]) for k in ("x", "y")))
    tx, ty = (torch.cat((sets[0][j], sets[1][j])) for j in (0, 1))
    vx, vy = sets[2]

    def target(y):
        return torch.log1p(1000 * (y[:, 0] - y[:, 1]).clamp_min(0))

    torch.manual_seed(a.seed)
    net = Benefit(a.width)
    net.mean.copy_(tx.mean(0))
    net.scale.copy_(tx.std(0).clamp_min(0.05))
    opt = torch.optim.AdamW(net.parameters(), lr=0.002)
    y = target(ty)
    for _ in range(a.updates):
        idx = torch.randint(len(y), (256,))
        pred = net(tx[idx])
        loss = torch.nn.functional.smooth_l1_loss(pred, y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.inference_mode():
        pred = net(vx).numpy()
    observed = target(vy).numpy()
    thresholds = {}
    records = []
    for every in (4, 8, 16, 32, 64):
        threshold = float(np.quantile(pred, 1.0 - 1 / every))
        thresholds[every] = threshold
        selected = pred >= threshold
        target_top = observed >= np.quantile(observed, 1.0 - 1 / every)
        records.append(
            dict(
                every=every,
                threshold=threshold,
                validation_intervals=len(pred),
                rank_correlation=float(spearmanr(pred, observed).statistic),
                recall_top_benefit=float(np.mean(selected[target_top])),
                median_selected_free_error_pct=100
                * float(vy[selected, 0].median()),
                selected_observed_benefit_pct=100
                * float((vy[selected, 0] - vy[selected, 1]).mean()),
                calibration_violations=int(
                    ((vy[:, 0].numpy() >= 0.001) & ~selected).sum()
                ),
            )
        )
    a.out.parent.mkdir(exist_ok=True, parents=True)
    torch.save(
        dict(
            width=a.width,
            state=net.state_dict(),
            thresholds=thresholds,
            transfer=Path(a.transfer).stem,
            substeps=a.substeps,
            horizon=a.horizon,
            seed=a.seed,
        ),
        a.out,
    )
    write_rows(a.out.with_suffix(".csv"), records)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("train", "augment", "valid", "cache", "out"):
        p.add_argument(key, type=Path)
    p.add_argument("--transfer", default="parent")
    for key, val in (
        ("substeps", 8),
        ("horizon", 4),
        ("batch", 96),
        ("width", 32),
        ("updates", 3000),
        ("seed", 17281),
    ):
        p.add_argument("--" + key, type=int, default=val)
    run(p.parse_args())
