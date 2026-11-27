"""Learn recursive corrections for hidden volatile allocation."""

import argparse
from pathlib import Path

import numpy as np
import torch
from history_encoder import observed
from projection_study import digest
from torch import nn


def evaluate(a, response):
    """Return the two-observation surface and its allocation Jacobian."""
    xy = (a / 0.9).unbind(-1)
    coords = []
    deriv = []
    for x in xy:
        z = torch.ones_like(x)
        coords.append(torch.stack((z, x, x * x, x * x * x), -1))
        deriv.append(torch.stack((0 * z, z, 2 * x, 3 * x * x), -1) / 0.9)

    def product(left, right):
        return (left[:, :, None] * right[:, None]).flatten(1)

    basis = torch.stack(
        (
            product(*coords),
            product(deriv[0], coords[1]),
            product(coords[0], deriv[1]),
        ),
        1,
    )
    calculated = (basis @ response.flatten(2)).reshape(
        len(a), 3, *response.shape[2:]
    )
    return calculated[:, 0], calculated[:, 1:].permute(0, 2, 3, 1)


class Estimate(nn.Module):
    """Mix a physical inverse step, an instantaneous correction and a GRU."""

    def __init__(self, kind, features, width=18):
        """Share one estimator across successive updates of the same state."""
        super().__init__()
        if kind not in ("physical", "ff", "gru", "mixture"):
            raise ValueError("Unknown hydration inverse")
        self.kind = kind
        self.width = width
        self.register_buffer("center", torch.zeros(features))
        self.register_buffer("scale", torch.ones(features))
        self.register_buffer(
            "floor", torch.tensor([0.001, 0.0001], dtype=torch.float64)
        )
        size = features + 11
        if kind in ("ff", "mixture"):
            self.instant = nn.Sequential(
                nn.Linear(size, width), nn.SiLU(), nn.Linear(width, 2)
            )
            nn.init.zeros_(self.instant[-1].weight)
            nn.init.zeros_(self.instant[-1].bias)
        if kind in ("gru", "mixture"):
            self.cell = nn.GRUCell(size, width)
            self.map = nn.Linear(width, 2)
            nn.init.zeros_(self.map.weight)
            nn.init.zeros_(self.map.bias)
        if kind == "mixture":
            self.gate = nn.Linear(size, 3)

    def forward(self, static, x, response, sigma, iterations):
        """Fit the measured past with bounded, shared parameter updates."""
        # The last observation is the exactly retained current inventory.
        obs = x[:, :-1].double()
        resp = response[:, :, :-1].double()
        err = (self.floor.square() + sigma[:, None].double().square()).sqrt()
        a = obs.new_zeros((len(obs), 2))
        h = static.new_zeros((len(static), self.width))
        state = (static - self.center) / self.scale
        ident = torch.eye(2, device=x.device, dtype=obs.dtype)
        for _ in range(iterations):
            pred, j = evaluate(a, resp)
            raw = obs - pred
            r = raw / err[:, None]
            j = (j / err[:, None, :, None]).flatten(1, 2)
            gram = j.transpose(1, 2) @ j
            # Normal prior width 0.55 in bounded, dimensionless slope units.
            penalty = 1 / 0.55**2
            grad = (j.transpose(1, 2) @ r.flatten(1)[..., None])[
                ..., 0
            ] - penalty * a
            step = torch.linalg.solve(gram + ident * penalty, grad[..., None])[
                ..., 0
            ].clamp(-0.55, 0.55)
            base = a + step
            if self.kind != "physical":
                diag = gram.diagonal(dim1=-2, dim2=-1)
                corr = gram[:, 0, 1] / diag.prod(-1).clamp_min(1e-24).sqrt()
                signals = torch.cat(
                    (
                        a,
                        step,
                        diag.log1p() / 12,
                        corr[:, None],
                        r.mean(1).asinh(),
                        r.square().mean(1).clamp_min(1e-20).sqrt().asinh(),
                    ),
                    -1,
                )
                feat = torch.cat((state, signals.float()), -1)
                if self.kind in ("ff", "mixture"):
                    instant = 0.35 * self.instant(feat).tanh()
                if self.kind in ("gru", "mixture"):
                    h = self.cell(feat, h)
                    recurrent = 0.35 * self.map(h).tanh()
                if self.kind == "mixture":
                    weights = self.gate(feat).softmax(-1)
                    corr = (
                        weights[:, 1:2] * instant + weights[:, 2:3] * recurrent
                    )
                else:
                    corr = instant if self.kind == "ff" else recurrent
                base = base + corr
            a = base.clamp(-0.90, 0.90)
        return a.float()


def load(record, surface):
    """Bind the physical sensitivity surface to the same aerosol families."""
    with np.load(record) as d, np.load(surface) as f:
        if str(f["input_sha256"]) != digest(record):
            raise ValueError("The physical response belongs to another cohort")
        np.testing.assert_array_equal(f["ids"], d["ids"])
        static, x = observed(d)
        return (
            static,
            x,
            torch.from_numpy(f["coeff"]),
            torch.tensor(d["coef"], dtype=torch.float32),
            f["interpolation_mae"],
        )


def setup(kind, static, floor, budget):
    """Match the learned estimator sizes and use training-only calibration."""
    size = static.shape[-1]

    def count(w):
        return sum(p.numel() for p in Estimate(kind, size, w).parameters())

    width = (
        min(range(4, 129), key=lambda w: abs(count(w) - budget))
        if kind != "physical"
        else 1
    )
    config = dict(kind=kind, features=size, width=width)
    model = Estimate(**config)
    model.center.copy_(static.mean(0))
    model.scale.copy_(static.std(0).clamp_min(1e-7))
    model.floor.copy_(torch.as_tensor(floor).clamp_min(1e-5))
    return model, config


def fit(record, surface, out, kinds, seed, updates, batch, iterations, budget):
    """Train on unseen-family prediction of known hydration coefficients."""
    static, x, resp, target, floor = load(record, surface)
    out.mkdir(parents=True, exist_ok=True)
    for kind in kinds:
        torch.manual_seed(seed)
        gen = torch.Generator().manual_seed(seed + 234)
        net, config = setup(kind, static, floor, budget)
        opt = (
            torch.optim.AdamW(net.parameters(), lr=0.001)
            if kind != "physical"
            else None
        )
        for _ in range(updates if opt else 0):
            i = torch.randint(len(resp), (batch,), generator=gen)
            j = (2 * i[:, None] + torch.arange(2)).flatten()
            u = torch.rand(len(j), generator=gen)
            sigma = torch.where(
                u < 0.25, 0.0, 10 ** (-3.3 + 3.3 * (u - 0.25) / 0.75)
            )
            eps = torch.randn(x[j].shape, generator=gen) * sigma[:, None, None]
            eps[:, -1] = 0
            a = net(
                static[j],
                x[j] + eps,
                resp[i].repeat_interleave(2, 0),
                sigma,
                iterations,
            )
            loss = (a - target[j]).square().mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                net.parameters(), 1.0, error_if_nonfinite=True
            )
            opt.step()
        trained = updates if opt else 0
        path = out / f"recursive-{kind}-n{len(resp)}-u{trained}-s{seed}.pt"
        tmp = path.with_suffix(".tmp")
        torch.save(
            dict(
                model=net.state_dict(),
                config=config,
                seed=seed,
                iterations=iterations,
                training_sha256=digest(record),
                sensitivity_sha256=digest(surface),
                parameters=sum(p.numel() for p in net.parameters()),
            ),
            tmp,
        )
        tmp.replace(path)
        print(path.name, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for f in ("record", "surface", "out"):
        p.add_argument(f, type=Path)
    p.add_argument(
        "--kinds",
        nargs="+",
        choices=["physical", "ff", "gru", "mixture"],
        default=["physical", "ff", "gru", "mixture"],
    )
    for name, value in (
        ("seed", 4401),
        ("updates", 6000),
        ("batch", 32),
        ("iterations", 3),
        ("budget", 3200),
    ):
        p.add_argument("--" + name, type=int, default=value)
    torch.set_num_threads(1)
    fit(**vars(p.parse_args()))
