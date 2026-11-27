"""Fit conservative vapor corrections on moment-matched aerosols."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
from continuous_pulse import box
from multicycle_storage import forcing
from node_fit import metrics_for_series
from node_transfer import write_table
from projection_study import digest
from reduced_exchange import rollout
from tail_quadrature import decode, encode


def features(mass, counts):
    """Measure particle size and composition conditional on dry size."""
    dry = mass[..., 1:].sum(-1)
    f = np.divide(mass[..., 2], dry, out=np.zeros_like(dry), where=dry > 0)
    radius = np.cbrt(3 * dry / (4 * np.pi * 1400)) * 1e6
    out = np.zeros((*f.shape, 5))
    for b, (r, c, n) in enumerate(zip(radius, f, counts, strict=True)):
        active = n > 0
        if not active.any():
            continue
        rid = np.round(np.log(r[active]), 10)
        unique, ix = np.unique(rid, return_inverse=True)
        weight = n[active]
        total = np.bincount(ix, weights=weight, minlength=len(unique))
        mean = np.bincount(ix, weights=weight * c[active]) / total
        centered = c[active] - mean[ix]
        std = np.sqrt(np.bincount(ix, weights=weight * centered**2) / total)
        z = np.divide(
            centered, std[ix], out=np.zeros_like(centered), where=std[ix] > 1e-7
        )
        out[b, active] = np.column_stack(
            (np.log(r[active]), 2 * c[active] - 1, 2 * mean[ix] - 1, std[ix], z)
        )
    return out


def prepare(folder, output, fine):
    """Cache causal physical predictions and independent training labels."""
    output.mkdir(parents=True, exist_ok=True)
    for p in sorted(folder.glob("families-*.npz")):
        dest = output / p.name
        if dest.exists():
            with np.load(dest) as a:
                if (
                    a["sha256"].item() != digest(p)
                    or bool(a["continuous"]) != fine
                ):
                    raise ValueError(
                        "Wrong source or target for cached examples"
                    )
            continue
        with np.load(p) as a:
            m, n = a["mass"][0], a["counts"]
            t = a["time"]
            lengths = np.array([len(t) // 2 - 1] * 2)
            ends = [0, len(t) // 2]
            levels = a["gas"][ends].T
            np.testing.assert_array_equal(levels[::2], levels[1::2])
            f = forcing(np.tile(lengths, (len(n) // 2, 1)), levels[::2], t[1])
            volume = float(a["volume"])
            rec = encode(m, n, 8, 2)
            reduced, number = decode(rec)
            w, g = rollout(
                *[
                    torch.from_numpy(np.ascontiguousarray(x))
                    for x in (reduced, number / volume, levels[:, 0], f)
                ],
                12,
                True,
                True,
            )
            bulk = (
                (w * torch.from_numpy(number / volume)[:, None]).sum(-1).numpy()
            )
            saved = dict(
                x=features(m, n).astype(np.float32),
                counts=n,
                dry=(m[..., 1:].sum(-1) * n).sum(1) / volume,
                time=t,
                ids=a["ids"],
                force=f,
                base=bulk,
                base_gas=g.numpy(),
                water=a["water"].T,
                sha256=digest(p),
                continuous=fine,
            )
            if fine:
                saved["water"] = np.stack(
                    [
                        box(mi, ni, volume, t, li, 1e-9)[0]
                        for mi, ni, li in zip(m, n, levels, strict=True)
                    ]
                )
            tmp = dest.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, **saved)
            tmp.replace(dest)


def data(folder, expected):
    """Require complete groups and a common time axis."""
    arrays = {}
    for src in sorted(folder.glob("*.npz")):
        with np.load(src) as a:
            for k in (
                "x",
                "counts",
                "dry",
                "time",
                "ids",
                "force",
                "base",
                "base_gas",
                "water",
            ):
                v = a[k]
                if k == "time":
                    v = np.repeat(v[None], len(a["x"]), axis=0)
                arrays.setdefault(k, []).append(v)
    out = {
        k: torch.from_numpy(np.concatenate(items))
        for k, items in arrays.items()
    }
    if len(out["ids"]) != expected or len(np.unique(out["ids"])) != expected:
        raise ValueError("Incomplete or repeated paired training families")
    return out


class TailCorrection(torch.nn.Module):
    """Combine a weighted particle embedding and a bounded correction."""

    def __init__(self, width, encoder, head):
        """Select the particle map and temporal correction at a common width."""
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.width = width
        self.particle = (
            torch.nn.Sequential(
                torch.nn.Linear(5, 32),
                torch.nn.SiLU(),
                torch.nn.Linear(32, width),
            )
            if encoder == "learned"
            else None
        )
        static = (
            width if encoder == "learned" else (6 if encoder == "fourth" else 5)
        )
        inputs = static + 7
        self.cell = (
            torch.nn.GRUCell(inputs, 16) if head in ("gru", "mixture") else None
        )
        self.instant = (
            torch.nn.Sequential(
                torch.nn.Linear(inputs, 32),
                torch.nn.SiLU(),
                torch.nn.Linear(32, 1),
            )
            if head in ("ff", "mixture")
            else None
        )
        self.recurrent = (
            torch.nn.Sequential(
                torch.nn.Linear(inputs + 16, 24),
                torch.nn.SiLU(),
                torch.nn.Linear(24, 1),
            )
            if self.cell
            else None
        )
        self.gate = torch.nn.Linear(inputs, 3) if head == "mixture" else None
        for net in (self.instant, self.recurrent):
            if net is not None:
                torch.nn.init.zeros_(net[-1].weight)
                torch.nn.init.zeros_(net[-1].bias)

    def encode(self, x, n):
        """Pool a shared map over particles using their counts."""
        if self.particle is not None:
            mapped = self.particle(x)
        else:
            size, f, mean, spread, z = x.unbind(-1)
            vals = [size, mean, spread, f.square(), f.pow(3)]
            if self.encoder == "fourth":
                vals.append(z.pow(4))
            mapped = torch.stack(vals, -1)
        weight = n / n.sum(-1, keepdim=True).clamp_min(1e-30)
        return (weight[:, :, None] * mapped.double()).sum(1).float()

    def forward(self, x, n, dry, force, base, base_gas):
        """Conserve gas plus particle water along the corrected forecast."""
        return self.advance(self.encode(x, n), dry, force, base, base_gas)

    def advance(self, z, dry, force, base, base_gas):
        """Forecast from a stored shape record and prescribed future forcing."""
        water, g = base[:, 0], base_gas[:, 0]
        series, gases = [water], [g]
        memory = z.new_zeros((len(z), 16))
        age = torch.zeros_like(g)
        for i, frame in enumerate(force.unbind(1)):
            h = frame[:, 0]
            change = base[:, i + 1] - base[:, i]
            inputs = torch.cat(
                (
                    z,
                    torch.stack(
                        (
                            base[:, i] / dry.clamp_min(1e-25),
                            base_gas[:, i] / 0.017,
                            change / dry.clamp_min(1e-25),
                            g / 0.017,
                            (water - base[:, i]) / dry.clamp_min(1e-25) * 100,
                            h / 0.0025,
                            age / 0.04,
                        ),
                        dim=-1,
                    ).float(),
                ),
                dim=-1,
            )
            if self.cell is not None:
                memory = torch.where(
                    h[:, None] > 0, self.cell(inputs, memory), memory
                )
                radial = (
                    self.recurrent(torch.cat((inputs, memory), dim=-1))
                    .squeeze(-1)
                    .tanh()
                )
            if self.instant is not None:
                instant = self.instant(inputs).squeeze(-1).tanh()
            if self.head == "mixture":
                route = self.gate(inputs).softmax(-1)
                result = route[:, 1] * instant + route[:, 2] * radial
            elif self.head == "gru":
                result = radial
            else:
                result = instant
            delta = (
                change + result.double() * dry * 0.002 * (h / 0.0025)
            ).clamp(min=-water, max=g)
            water = water + delta
            g = torch.where(frame[:, 3] > 0, frame[:, 4], g - delta)
            age = torch.where(frame[:, 3] > 0, 0, age + h)
            series.append(water)
            gases.append(g)
        return torch.stack(series, 1), torch.stack(gases, 1)


def train(
    root,
    out,
    width,
    encoder,
    head,
    updates,
    seed,
    train_families,
    train_folder="train",
):
    """Compare native held-out contrasts after end-to-end paired fitting."""
    train_data = data(root / train_folder, train_families)
    val = data(root / "validation", 64)
    coarse = data(root / "validation64", 64)
    np.testing.assert_array_equal(coarse["ids"], val["ids"])
    if np.intersect1d(train_data["ids"], val["ids"]).size:
        raise ValueError("Data leakage")
    delta = (coarse["water"] - val["water"]).abs()
    target = val["water"][1::2] - val["water"][::2]
    truth = dict(
        time=val["time"],
        water=val["water"],
        ref=(target.numpy(), (delta[1::2] + delta[::2]).numpy()),
    )
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = TailCorrection(width, encoder, head)
    opt = torch.optim.AdamW(model.parameters(), lr=0.002)
    keys = ("x", "counts", "dry", "force", "base", "base_gas")
    rows = []
    out.parent.mkdir(parents=True, exist_ok=True)
    positive = train_data["force"][0, :, 0] > 0
    for step in range(updates + 1):
        if step:
            ix = (
                2 * rng.integers(train_families, size=24)[:, None]
                + np.arange(2)
            ).ravel()
            pred, _ = model(*(train_data[k][ix] for k in keys))
            err = 1e6 * (pred - train_data["water"][ix])[:, 1:][:, positive]
            contrast = (
                1e6
                * (
                    train_data["water"][ix][1::2] - train_data["water"][ix][::2]
                )[:, 1:][:, positive]
            )
            relative = (err[1::2] - err[::2]) / contrast.abs().clamp_min(0.02)
            loss = (
                torch.log1p(err.abs() / 0.1).mean()
                + 2 * torch.log1p(relative.abs()).mean()
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            opt.step()
        if step in {0, updates // 4, updates // 2, updates}:
            with torch.no_grad():
                w, g = model(*(val[k] for k in keys))
                total = w + g
                forcing = val["force"]
                external = torch.where(
                    forcing[:, :, 3] > 0, forcing[:, :, 4] - g[:, :-1], 0.0
                )
                torch.testing.assert_close(
                    total.diff(dim=1), external, rtol=1e-12, atol=2e-14
                )
                shape = model.encode(val["x"], val["counts"])
                stats = metrics_for_series(w, g, truth, True)["all"]
                rows.append(
                    dict(
                        step=step,
                        seed=seed,
                        encoder=encoder,
                        head=head,
                        width=width,
                        parameters=sum(p.numel() for p in model.parameters()),
                        max_pair_shape_difference=(shape[1::2] - shape[::2])
                        .abs()
                        .max()
                        .item(),
                        **stats,
                    )
                )
                weights = out.with_suffix(".pt")
                pending = weights.with_suffix(".tmp")
                torch.save(
                    dict(
                        model=model.state_dict(),
                        encoder=encoder,
                        head=head,
                        width=width,
                        seed=seed,
                        step=step,
                        train_ids_sha256=hashlib.sha256(
                            train_data["ids"].numpy().tobytes()
                        ).hexdigest(),
                    ),
                    pending,
                )
                pending.replace(weights)
                write_table(rows, out.with_suffix(".csv"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="action", required=True)
    a = sub.add_parser("prepare")
    for k in ("folder", "output"):
        a.add_argument(k, type=Path)
    a.add_argument("--fine", action="store_true")
    a = sub.add_parser("train")
    for k in ("root", "out"):
        a.add_argument(k, type=Path)
    a.add_argument("--width", type=int, default=16)
    a.add_argument(
        "--encoder", choices=("learned", "fourth", "three"), default="learned"
    )
    a.add_argument(
        "--head", choices=("ff", "gru", "mixture"), default="mixture"
    )
    a.add_argument("--updates", type=int, default=1200)
    a.add_argument("--seed", type=int, default=2801)
    a.add_argument("--train-families", type=int, default=512)
    a.add_argument("--train-folder", default="train")
    torch.set_num_threads(1)
    args = vars(p.parse_args())
    {"prepare": prepare, "train": train}[args.pop("action")](**args)
