"""Learn a conservative correction from two time-step resolutions."""

import argparse
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from material_bank import materials
from stage2_accuracy import write_rows
from stage2_fit_transfer import dataset, objective, validate
from stage2_transfer import Transfer, restore
from torch import nn


class Defect(Transfer):
    """Bound the correction using two trial resolutions."""

    def __init__(self, spec, base, kind, width):
        """Bind the physical stepper and the correction readout."""
        super().__init__(spec)
        self.base = restore(spec, base)
        self.richardson = 1 / 7 if base == "radau-2-2" else 1 / 15
        self.flavor = kind
        self.memory_width = width if kind == "gru" else 0
        self.cell = nn.GRUCell(13, width).double() if kind == "gru" else None
        self.readout = None
        if kind in ("ff", "gru"):
            self.readout = nn.Sequential(
                nn.Linear(13 + width if self.cell else 13, width),
                nn.SiLU(),
                nn.Linear(width, width),
                nn.SiLU(),
                nn.Linear(width, 1),
            ).double()
            nn.init.zeros_(self.readout[-1].weight)
            nn.init.zeros_(self.readout[-1].bias)

    def step(self, q, gas, state, dt, memory):
        """Debit gas for the transfer accepted across both trials."""
        coarse, cg, _ = self.base.step(q, gas, state, dt, memory)
        half, hg, _ = self.base.step(q, gas, state, dt / 2, memory)
        fine, fg, _ = self.base.step(half, hg, state, dt / 2, memory)
        defect = fine - coarse
        gain = self.richardson
        if self.readout is not None:
            scale = q.abs() + fine.abs() + 1e-5

            def spread(v):
                return v.expand_as(q)

            shape = state[2].clamp_min(1e-40)
            total_w = state[3].sum(1)
            wfrac = state[3] / total_w[:, None].clamp_min(1e-32)
            xs = (
                torch.log1p(q),
                torch.log1p(fine),
                spread(state[4][..., None]),
                spread(torch.log(shape / 1e-21)[..., None] / 8),
                spread((gas / self.sat)[:, None]),
                spread((fg / self.sat)[:, None]),
                (half - q) / scale,
                (fine - half) / scale,
                defect / scale,
                spread(((fg - cg) / self.sat)[:, None]),
                spread(
                    torch.log(dt.clamp_min(1e-15) / 0.005)[:, None, None] / 4
                ),
                spread(wfrac[..., None]),
                spread(torch.log1p(total_w[:, None] / self.sat)[:, None]),
            )
            x = torch.stack(xs, -1).clamp(-20, 20)
            if self.cell is not None:
                new = self.cell(
                    x.flatten(0, -2), memory.flatten(0, -2)
                ).reshape_as(memory)
                memory = torch.where(state[5][..., None, None], new, memory)
                x = torch.cat((x, memory), -1)
            gain = gain + 4 * self.readout(x).squeeze(-1).tanh()
        qn, gn = self.balance(fine, fg, state[3], gain * defect)
        return qn, gn, memory


def train(a):
    """Validate uninterrupted ordinary and collector forecasts."""
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    data = [dataset(p, "cpu") for p in a.train]
    valid = [dataset(p, "cpu") for p in a.valid]
    model = Defect(materials(2), a.base, a.kind, a.width)
    weights = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr) if weights else None
    start = perf_counter()
    best = None
    rows = []
    for it in range(a.updates + 1):
        if it and opt:
            loss = objective(
                model,
                data[it % len(data)],
                rng,
                a.batch,
                a.window,
                a.steps,
                "cpu",
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
            if not torch.isfinite(loss):
                raise ArithmeticError("Non-finite defect training loss")
            if it == 3 * a.updates // 4:
                opt.param_groups[0]["lr"] *= 0.35
        if it % a.eval_every and it != a.updates:
            continue
        with torch.no_grad():
            metrics = [validate(model, d, a.steps) for d in valid]
        worst = max(x[0] for x in metrics)
        p99 = max(x[1] for x in metrics)
        frac = min(x[2] for x in metrics)
        rank = (round(worst, 6), p99, -frac)
        row = dict(
            update=it,
            parameters=weights,
            base=a.base,
            kind=a.kind,
            width=a.width,
            window=a.window,
            steps=a.steps,
            max_across_sets_pct=100 * worst,
            tail_quantile_pct=100 * p99,
            lower_fraction_passing=frac,
            seconds=perf_counter() - start,
        )
        for j, r in enumerate(metrics):
            row[f"validation_{j}_maximum_pct"] = 100 * r[0]
        rows.append(row)
        if best is None or rank < best:
            cp = dict(
                defect=True,
                base=a.base,
                kind=a.kind,
                width=a.width,
                readout=model.readout.state_dict() if model.readout else None,
                cell=model.cell.state_dict() if model.cell else None,
                selected=row,
                seed=a.seed,
                training_sha256={
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in a.train
                },
            )
            torch.save(cp, a.out)
            best = rank
        write_rows(a.out.with_suffix(".csv"), rows)
        print(it, worst, p99, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    for name in ("train", "valid"):
        p.add_argument("--" + name, type=Path, nargs="+", required=True)
    p.add_argument("--base", default="radau-2-2")
    p.add_argument("--kind", choices=["fixed", "ff", "gru"], default="ff")
    for k, v in (
        ("updates", 3000),
        ("width", 28),
        ("window", 3),
        ("batch", 24),
        ("steps", 1),
        ("eval-every", 150),
        ("seed", 12873),
    ):
        p.add_argument("--" + k, type=int, default=v)
    p.add_argument("--lr", type=float, default=0.001)
    torch.set_num_threads(1)
    train(p.parse_args())
