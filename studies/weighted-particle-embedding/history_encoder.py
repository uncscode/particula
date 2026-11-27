"""Infer unresolved volatile allocation from recent bulk measurements."""

import argparse
from pathlib import Path

import numpy as np
import torch
from dual_materials import SATURATED
from history_population import HISTORY
from projection_study import digest
from torch import nn


def observed(d):
    """Return only the present dry state and the gas/particle uptake history."""
    fixed = d["fixed"]
    dry = fixed.sum(-1)
    w = dry * d["number"]
    scale = w.sum(1, keepdims=True)
    radius = np.cbrt(3 * dry / (1400 * 4 * np.pi)) * 1e6
    static = np.c_[
        np.log(radius),
        fixed[..., 1] / dry,
        w / scale,
        np.log(scale),
        d["mean"],
        d["previous"] / SATURATED,
    ]
    # The final stored inventory is shared exactly by the two members.
    bulk = d["history"][:, -1:]
    history = (d["history"] - bulk) * 1e6
    return torch.tensor(static, dtype=torch.float32), torch.tensor(
        history, dtype=torch.float32
    )


class Recovery(nn.Module):
    """Predict two bounded allocation coefficients for the known dry nodes."""

    def __init__(self, kind, width, frames, features):
        """Use a direct, recurrent or mixed conditional decoder."""
        super().__init__()
        self.kind = kind
        self.frames = frames
        self.register_buffer("center", torch.zeros(features))
        self.register_buffer("scale", torch.ones(features))
        self.register_buffer("deltas", torch.ones(2))
        self.project = nn.Sequential(nn.Linear(features, width), nn.SiLU())
        if kind in ("instant", "ff", "mixture"):
            size = width if kind == "instant" else width + 2 * frames
            self.head = nn.Sequential(
                nn.Linear(size, width), nn.SiLU(), nn.Linear(width, 2)
            )
        if kind in ("gru", "mixture"):
            self.cell = nn.GRUCell(2, width)
            self.readout = nn.Linear(width, 2)
        if kind == "mixture":
            self.gate = nn.Linear(width + 2, 3)

    def forward(self, static, history):
        """Condition both allocations on measurements ending at the restart."""
        base = self.project((static - self.center) / self.scale)
        hist = history[:, -self.frames :] / self.deltas
        if self.kind == "instant":
            return self.head(base).tanh()
        if self.kind in ("ff", "mixture"):
            direct = self.head(torch.cat((base, hist.flatten(1)), -1)).tanh()
            if self.kind == "ff":
                return direct
        hidden = base
        for obs in hist.unbind(1):
            hidden = self.cell(obs, hidden)
        recurrent = self.readout(hidden).tanh()
        if self.kind == "gru":
            return recurrent
        weights = self.gate(torch.cat((base, hist[:, 0]), -1)).softmax(-1)
        return weights[:, 1:2] * direct + weights[:, 2:3] * recurrent


def create(kind, frames, size, budget):
    """Match the number of trainable parameters across temporal designs."""

    def number(w):
        return sum(
            p.numel() for p in Recovery(kind, w, frames, size).parameters()
        )

    width = min(range(4, 129), key=lambda w: abs(number(w) - budget))
    config = dict(kind=kind, width=width, frames=frames, features=size)
    return Recovery(**config), config


def train(record, out, frames, kinds, seed, updates, batch, noise, budget):
    """Fit under paired family sampling and training-only scaling."""
    with np.load(record) as a:
        static, hist = observed(a)
        target = torch.tensor(a["coef"], dtype=torch.float32)
        np.testing.assert_array_equal(static[::2].numpy(), static[1::2].numpy())
    out.mkdir(parents=True, exist_ok=True)
    for k in frames:
        for kind in kinds:
            if kind == "instant" and k != frames[0]:
                continue
            torch.manual_seed(seed)
            model, config = create(kind, k, static.shape[-1], budget)
            model.center.copy_(static.mean(0))
            model.scale.copy_(static.std(0).clamp_min(1e-8))
            model.deltas.copy_(
                hist[:, -max(2, k) :]
                .square()
                .mean((0, 1))
                .sqrt()
                .clamp_min(1e-6)
            )
            opt = torch.optim.AdamW(model.parameters(), lr=0.002)
            gen = torch.Generator().manual_seed(seed + 171)
            for _ in range(updates):
                i = torch.randint(len(static) // 2, (batch,), generator=gen)
                ix = (i[:, None] * 2 + torch.arange(2)).flatten()
                local = hist[ix]
                eps = torch.randn(local.shape, generator=gen) * noise
                eps[:, -1] = 0
                pred = model(static[ix], local + eps)
                loss = (pred - target[ix]).square().mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, error_if_nonfinite=True
                )
                opt.step()
            file = out / f"{kind}-h{k}-noise{noise:g}-s{seed}.pt"
            tmp = file.with_suffix(".tmp")
            torch.save(
                dict(
                    model=model.state_dict(),
                    config=config,
                    seed=seed,
                    updates=updates,
                    batch=batch,
                    noise_mg_m3=noise,
                    training_sha256=digest(record),
                    parameters=sum(p.numel() for p in model.parameters()),
                ),
                tmp,
            )
            tmp.replace(file)
            print(file.name, float(loss.detach()), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("record", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--frames", type=int, nargs="+", default=[2, 4, HISTORY])
    p.add_argument(
        "--kinds",
        nargs="+",
        choices=["instant", "ff", "gru", "mixture"],
        default=["instant", "ff", "gru", "mixture"],
    )
    for key, value in (
        ("seed", 4201),
        ("updates", 2400),
        ("batch", 64),
        ("budget", 3200),
    ):
        p.add_argument("--" + key, type=int, default=value)
    p.add_argument("--noise", type=float, default=0.0)
    torch.set_num_threads(1)
    train(**vars(p.parse_args()))
