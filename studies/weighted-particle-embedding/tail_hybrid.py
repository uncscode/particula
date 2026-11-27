"""Learn a single within-size particle statistic for a three-class decoder."""

import argparse
from pathlib import Path

import numpy as np
import torch
from multicycle_storage import forcing
from node_transfer import write_table
from projection_study import digest
from tail_compact import nodes, restore, shape, store
from tail_study import physical, read, score


def inputs(mass, counts, code):
    """Locate each particle in its dry-size section and normalize its weight."""
    dry = mass[:, :, 1:].sum(-1)
    groups = np.zeros(counts.shape, dtype=np.int64)
    z2 = np.zeros_like(dry)
    weight = np.zeros_like(dry)
    bins = code.shape[1]
    for i in range(len(dry)):
        active = counts[i] > 0
        if not active.any():
            continue
        d = dry[i, active]
        axis = np.log(d)
        group = np.zeros(len(d), dtype=int)
        if bins > 1 and np.ptp(axis) >= 1e-12:
            group = (
                np.floor((axis - axis.min()) / np.ptp(axis) * bins + 1e-10)
                .astype(int)
                .clip(0, bins - 1)
            )
        w = counts[i, active] * d
        weight[i, active] = (
            w / np.bincount(group, weights=w, minlength=bins)[group]
        )
        groups[i, active] = group
        mean = code[i, group, 3]
        sd = code[i, group, 4] / 2
        z = np.divide(
            mass[i, active, 2] / d - mean,
            sd,
            out=np.zeros_like(sd),
            where=sd > 1e-6,
        )
        z2[i, active] = z * z
    return z2, weight, groups


class TailStatistic(torch.nn.Module):
    """Choose a positive even function of within-size composition deviations."""

    def __init__(self, kind):
        """Build a power, feed-forward or soft mixture particle map."""
        super().__init__()
        self.kind = kind
        if kind != "mlp":
            self.logit = torch.nn.Parameter(
                torch.tensor(-1.0, dtype=torch.float64)
            )
        if kind != "power":
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(1, 24), torch.nn.SiLU(), torch.nn.Linear(24, 1)
            ).double()
        if kind == "mixture":
            self.gate = torch.nn.Linear(1, 1).double()

    def forward(self, z2, weight, group, code):
        """Pool one learned scalar per size and infer the tail probability."""
        if self.kind != "mlp":
            exponent = 1 + 2 * self.logit.sigmoid()
            power = z2.clamp_min(1e-30).pow(exponent)
        if self.kind != "power":
            x = torch.log1p(z2)[:, :, None]
            mlp = z2 * self.mlp(x).squeeze(-1).clamp(-8, 8).exp()
        if self.kind == "power":
            value = power
        elif self.kind == "mlp":
            value = mlp
        else:
            gate = self.gate(x).squeeze(-1).sigmoid()
            value = gate * power + (1 - gate) * mlp
        pooled = torch.zeros_like(code[:, :, 0]).scatter_add(
            1, group, value * weight
        )
        sd = code[:, :, 4] / 2
        skew = (2 * code[:, :, 5] - 1) / sd.clamp_min(1e-6)
        # Physical value is at least one. A straight-through clamp lets a
        # fitted map recover if its predicted fourth moment starts too low.
        inverse = pooled - skew.square()
        inverse = inverse + (inverse.clamp_min(1) - inverse).detach()
        return torch.where(sd > 1e-6, inverse.reciprocal(), 0.0)


def build(folder, targets, out, bins):
    """Measure native water sensitivity to each section's tail probability."""
    out.mkdir(parents=True, exist_ok=True)
    for p in sorted(folder.glob("*.npz")):
        cp = targets / p.name
        if not cp.is_file():
            raise ValueError("Missing continuous training label")
        dest = out / p.name
        if dest.exists():
            continue
        with np.load(p) as a, np.load(cp) as t:
            if t["sha256"].item() != digest(p) or not t["continuous"]:
                raise ValueError("Unmatched training source")
            m, n = a["mass"][0], a["counts"]
            code, totals = shape(m, n, bins)
            z, w, g = inputs(m, n, code)
            gas = a["gas"][0]
            vol = float(a["volume"])
            length = len(a["time"])
            take = np.array(
                [
                    1,
                    2,
                    4,
                    8,
                    12,
                    length // 2 - 1,
                    length // 2 + 1,
                    length // 2 + 2,
                    length // 2 + 4,
                    length // 2 + 8,
                    length // 2 + 12,
                    length - 1,
                ]
            )
            # Local one-sided derivatives keep probability on the simplex.
            variants = np.broadcast_to(code, (bins + 1, *code.shape)).copy()
            for i in range(bins):
                variants[i + 1, :, i, -1] *= np.exp(-0.005)
            mass, num, _ = nodes(
                variants.reshape(-1, bins, 7), np.tile(totals, (bins + 1, 1))
            )
            force = np.tile(t["force"], (bins + 1, 1, 1))
            water, _ = physical(
                mass, num, np.tile(gas, bins + 1), force, vol, 12
            )
            water = (
                water.numpy()[:, take].reshape(bins + 1, len(code), -1) * 1e6
            )
            deriv = ((water[0] - water[1:]) / 0.005).transpose(1, 2, 0)
            tmp = dest.with_suffix(".tmp.npz")
            np.savez_compressed(
                tmp,
                code=code,
                z=z,
                weight=w,
                group=g,
                teacher=t["water"][:, take] * 1e6,
                base=water[0],
                jacobian=deriv,
                ids=a["ids"],
                source_sha256=digest(p),
            )
            tmp.replace(dest)


def training(folder):
    """Read complete paired fitting inputs in family order."""
    arrays = {}
    for file in sorted(folder.glob("*.npz")):
        with np.load(file) as a:
            for k in (
                "code",
                "z",
                "weight",
                "group",
                "teacher",
                "base",
                "jacobian",
                "ids",
            ):
                arrays.setdefault(k, []).append(a[k])
    return {k: torch.from_numpy(np.concatenate(v)) for k, v in arrays.items()}


def fit(folder, out, kind, strength, seed, updates, families):
    """Fit conditional shape and local water-response targets."""
    data = training(folder)
    if len(data["ids"]) != families or len(np.unique(data["ids"])) != families:
        raise ValueError("Incomplete sensitivity training families")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = TailStatistic(kind)
    opt = torch.optim.Adam(
        model.parameters(), lr=0.015 if kind == "power" else 0.003
    )
    for step in range(updates):
        ix = (
            2 * rng.integers(families, size=48)[:, None] + np.arange(2)
        ).ravel()
        code = data["code"][ix]
        pred = model(data["z"][ix], data["weight"][ix], data["group"][ix], code)
        delta = torch.log(pred.clamp_min(1e-8)) - torch.log(
            code[:, :, -1].clamp_min(1e-8)
        )
        valid = code[:, :, 4] > 2e-6
        geometry = delta[valid].square().mean()
        est = data["base"][ix] + (data["jacobian"][ix] * delta[:, None, :]).sum(
            -1
        )
        target = data["teacher"][ix]
        residual = est - target
        diff = (residual[1::2] - residual[::2]) / (
            target[1::2] - target[::2]
        ).abs().clamp_min(0.02)
        loss = geometry + strength * (
            torch.log1p(residual.abs() / 0.1).mean()
            + torch.log1p(diff.abs()).mean()
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 2.0, error_if_nonfinite=True
        )
        opt.step()
        if step in (updates // 2, updates * 3 // 4):
            opt.param_groups[0]["lr"] *= 0.3
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(
            kind=kind,
            weights=model.state_dict(),
            strength=strength,
            seed=seed,
            train_ids=data["ids"].numpy().tolist(),
            updates=updates,
        ),
        out,
    )


@torch.no_grad()
def evaluate(
    root, checkpoints, folder, out, prefix, bins, bits, split="validation"
):
    """Reload packed shapes and compare free water forecasts."""
    parent = root / "composition-tails"
    d = read(parent / f"{prefix}128" / split)
    c = read(parent / f"{prefix}64" / split)
    np.testing.assert_array_equal(d["ids"], c["ids"])
    if len(d["ids"]) != 64:
        raise ValueError("Incomplete validation")
    code, totals = shape(d["m"], d["n"], bins)
    z, weight, group = inputs(d["m"], d["n"], code)
    err = abs(d["w"] - c["w"])
    times = np.broadcast_to(d["time"], (64, len(d["time"]))).copy()
    truth = dict(
        time=torch.from_numpy(times.repeat(2, 0)),
        water=torch.from_numpy(d["w"]),
        ref=(d["w"][1::2] - d["w"][::2], err[1::2] + err[::2]),
    )
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for cp in checkpoints:
        config = torch.load(cp, map_location="cpu", weights_only=True)
        if np.intersect1d(config["train_ids"], d["ids"]).size:
            raise ValueError("Overlapping families")
        model = TailStatistic(config["kind"])
        model.load_state_dict(config["weights"])
        probability = model(
            *map(torch.from_numpy, (z, weight, group, code))
        ).numpy()
        learned = code.copy()
        learned[:, :, -1] = probability
        for k in bits:
            path = folder / f"{cp.stem}-p{k}.npz"
            store(
                path,
                learned,
                totals,
                k,
                ids=d["ids"],
                levels=d["levels"][::2],
                time=d["time"],
                volume=d["volume"],
                density=d["density"],
                model_sha256=digest(cp),
            )
            with np.load(path) as a:
                mass, n, clipped, record = restore(a)
                levels = a["levels"]
                steps = (len(d["time"]) - 2) // 2
                force = forcing(np.full((64, 2), steps), levels, a["time"][1])
                w, g = physical(
                    mass,
                    n,
                    levels[:, 0].repeat(2),
                    force,
                    float(a["volume"]),
                    12,
                )
                metrics = score(
                    cp.stem,
                    path,
                    cp.stat().st_size,
                    w,
                    g,
                    truth,
                    times,
                    record,
                    digest(cp),
                )
                rows.append(
                    dict(
                        seed=config["seed"],
                        kind=config["kind"],
                        strength=config["strength"],
                        bits=k,
                        clipped_nodes=clipped,
                        learned_parameters=sum(
                            p.numel() for p in model.parameters()
                        ),
                        mean_tail_probability_error=np.abs(
                            probability - code[:, :, -1]
                        ).mean(),
                        **metrics,
                    )
                )
    write_table(rows, out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(required=True, dest="action")
    a = sub.add_parser("build")
    for k in ("folder", "targets", "out"):
        a.add_argument(k, type=Path)
    a.add_argument("--bins", type=int, default=6)
    a = sub.add_parser("fit")
    for k in ("folder", "out"):
        a.add_argument(k, type=Path)
    a.add_argument(
        "--kind", choices=("power", "mlp", "mixture"), default="power"
    )
    a.add_argument("--strength", type=float, default=0.01)
    a.add_argument("--seed", type=int, default=2901)
    a.add_argument("--updates", type=int, default=2400)
    a.add_argument("--families", type=int, default=512)
    a = sub.add_parser("evaluate")
    for k in ("root", "folder", "out"):
        a.add_argument(k, type=Path)
    a.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    a.add_argument(
        "--split",
        choices=("validation", "test", "challenge"),
        default="validation",
    )
    a.add_argument("--prefix", default="refinement-s")
    a.add_argument("--bins", type=int, default=6)
    a.add_argument("--bits", type=int, nargs="+", default=[8, 12])
    args = vars(p.parse_args())
    torch.set_num_threads(1)
    {"build": build, "fit": fit, "evaluate": evaluate}[args.pop("action")](
        **args
    )
