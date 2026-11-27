"""Fit water and organic conductances inside conservative vapor collocation."""

import argparse
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from particula.particles import get_vapor_transition_correction
from scipy.optimize import least_squares
from stage2_accuracy import write_rows
from stage2_lowrank import LowRankROS4
from stage2_radau import Radau
from torch import nn


class Conductance(nn.Module):
    """Map Knudsen number and accommodation to a positive transfer factor."""

    def __init__(self, kind, width):
        """Allocate a rational or flexible dimensionless function."""
        super().__init__()
        self.kind = kind
        if kind == "rational":
            self.coeff = nn.Parameter(torch.zeros(3, dtype=torch.float64))
        else:
            self.layer = nn.Linear(3, width).double()
            self.output = nn.Linear(width, 1).double()
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def feature_factor(self, kn, a):
        """Evaluate a fitted feature head without computing its radial slope."""
        u = kn / (kn + a)
        v = kn / (kn + 1)
        features = torch.stack(
            (2 * u - 1, 2 * v - 1, torch.log(a).expand_as(u) / 3), -1
        )
        h = self.layer(features).tanh()
        y = self.output(h).squeeze(-1)
        beta = (a / (a + kn)) * torch.exp(u * y)
        return beta, u, v, h, y

    def forward(self, kn, alpha):
        """Return the conductance and its radial logarithmic slope."""
        a = alpha.clamp_min(1e-12)
        if self.kind == "rational":
            c, d, e = self.coeff.exp()
            lin = (c / a + d) * kn
            quad = e / a * kn.square()
            den = 1 + lin + quad
            beta = (1 + kn) / den
            power = 1 - kn / (1 + kn) + (lin + 2 * quad) / den
        else:
            beta, u, v, h, y = self.feature_factor(kn, a)
            du = u * (1 - u)
            dv = v * (1 - v)
            slope = (1 - h.square()) * (
                2 * du[..., None] * self.layer.weight[:, 0]
                + 2 * dv[..., None] * self.layer.weight[:, 1]
            )
            dy = (slope * self.output.weight[0]).sum(-1)
            power = 1 + u - du * y - u * dy
        return torch.where(alpha > 0, beta, 0.0), power


class LearnedTransport:
    """Use the same kinetic head and derivative in each implicit solver."""

    def bind_heads(self, spec, cp):
        """Restore the fitted process functions for a particular vapor bank."""
        self.units = nn.ModuleDict()
        self.single_precision = cp.get("precision") == "float32"
        for label, weights in cp["heads"].items():
            unit = Conductance(cp["kind"], cp["width"])
            unit.load_state_dict(weights)
            self.units[label] = unit.float() if self.single_precision else unit
        self.register_buffer(
            "water_lane", torch.tensor([n == "water" for n in spec.names])
        )

    def transport(self, kn):
        """Evaluate each material through its selected kinetic head."""
        dtype = kn.dtype
        x = kn.float() if self.single_precision else kn
        a = self.alpha.float() if self.single_precision else self.alpha
        if "shared" in self.units:
            values = self.units["shared"](x, a)
        else:
            water = self.units["water"](x, a)
            organic = self.units["organic"](x, a)
            values = tuple(
                torch.where(self.water_lane, w, o)
                for w, o in zip(water, organic, strict=True)
            )
        return tuple(value.to(dtype) for value in values)

    def transport_gain(self, rr, dry, live):
        """Evaluate the learned conductance and its wet-radius derivative."""
        beta, power = self.transport(self.path / rr)
        return 4 * torch.pi * rr * self.diffusion * beta / dry[
            ..., None
        ] * live[..., None], power

    def linearization(self, q, gas, state, rate_only=False):
        """Differentiate the fitted particle flux and reservoir gain."""
        dry, base_moles, base_vol, _, _, active = state
        volume = base_vol + dry * (q / self.density).sum(-1)
        radius = (3 * volume / (4 * torch.pi)).pow(1 / 3)
        beta, power = self.transport(self.path / radius[..., None])
        gain = (
            4
            * torch.pi
            * radius[..., None]
            * self.diffusion
            * beta
            / dry[..., None]
            * active[..., None]
        )
        mol = base_moles + (q / self.molar).sum(-1)
        x = q / (self.molar * mol[..., None])
        saturation = self.sat * torch.exp(self.kelvin / radius[..., None])
        drive = gas[:, None] - saturation * x
        rate = gain * drive
        if rate_only:
            return rate
        dr = dry[..., None] / (
            4 * torch.pi * radius[..., None].square() * self.density
        )
        dx = (self.identity / self.molar - x[..., None] / self.molar) / mol[
            ..., None, None
        ]
        dx_sat = saturation[..., None] * (
            dx
            - (x * self.kelvin / radius[..., None].square())[..., None]
            * dr[..., None, :]
        )
        local = gain[..., None] * (
            (drive * power / radius[..., None])[..., None] * dr[..., None, :]
            - dx_sat
        )
        return rate, local, gain, drive, radius


class KineticRadau(LearnedTransport, Radau):
    """Advance the learned conductances with three-stage Radau IIA."""

    def __init__(self, spec, cp):
        """Bind the finite-reservoir collocation solver and process weights."""
        super().__init__(spec, 3, 4)
        self.bind_heads(spec, cp)


class KineticROS4(LearnedTransport, LowRankROS4):
    """Advance identical conductances with the rank-two Rosenbrock solver."""

    def __init__(self, spec, cp):
        """Bind the low-rank material solver and process weights."""
        super().__init__(spec)
        self.bind_heads(spec, cp)


def examples(seed, count, domain):
    """Draw fresh native kinetic labels without sampling an aerosol family."""
    rng = np.random.default_rng(seed)
    x = np.exp(rng.uniform(-6, 6, count))
    low, high = (0.08, 0.55) if domain == "water" else (0.015, 1.0)
    a = np.exp(rng.uniform(np.log(low), np.log(high), count))
    y = get_vapor_transition_correction(x, a)
    return tuple(torch.from_numpy(r) for r in (x, a, y))


def fit_features(model, x, a, y, seed):
    """Fit the output weights of a fixed tanh layer by reweighted regression."""
    rng = np.random.default_rng([19571, seed])
    with torch.no_grad():
        model.layer.weight.copy_(
            torch.from_numpy(rng.normal(0, 1.4, (model.layer.out_features, 3)))
        )
        model.layer.bias.copy_(
            torch.from_numpy(rng.uniform(-2, 2, model.layer.out_features))
        )
        u, v = x / (x + a), x / (x + 1)
        features = torch.stack((2 * u - 1, 2 * v - 1, torch.log(a) / 3), -1)
        h = model.layer(features).tanh()
        design = (
            u[:, None] * torch.cat((torch.ones_like(u[:, None]), h), 1)
        ).numpy()
        target = torch.log(y / (a / (a + x))).numpy()
        weights = np.ones(len(y))
        for _ in range(6):
            coef = np.linalg.lstsq(
                weights[:, None] * design, weights * target, rcond=2e-12
            )[0]
            error = abs(design @ coef - target)
            weights = np.sqrt(
                1
                + np.minimum(200.0, error / max(np.quantile(error, 0.9), 1e-14))
                ** 2
            )
        model.output.bias.copy_(torch.tensor([coef[0]], dtype=torch.float64))
        model.output.weight.copy_(torch.from_numpy(coef[1:][None]))
    return 6


def train_head(kind, width, domain, seed, steps):
    """Fit one transition function under a log-relative error objective."""
    torch.manual_seed(seed)
    train = examples(seed, 8192, domain)
    x, a, y = train
    model = Conductance(kind, width)
    if kind == "rational":
        xx, aa, yy = [z.numpy() for z in train]

        def residual(p):
            c, d, e = np.exp(p)
            return np.log(
                (1 + xx) / (1 + (c / aa + d) * xx + e * xx**2 / aa) / yy
            )

        result = least_squares(
            residual, np.zeros(3), ftol=1e-14, xtol=1e-14, gtol=1e-14
        )
        with torch.no_grad():
            model.coeff.copy_(torch.from_numpy(result.x))
        updates = result.nfev
    elif kind == "features":
        updates = fit_features(model, x, a, y, seed)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=0.012)
        for step in range(steps):
            idx = torch.randint(len(x), (384,))
            est = model(x[idx], a[idx])[0]
            err = (est / y[idx]).log() * 100
            loss = (
                err.square().mean() + 0.1 * err.square().topk(20).values.mean()
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step in (steps // 2, 3 * steps // 4):
                for g in opt.param_groups:
                    g["lr"] *= 0.35
        polish = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=240,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-12,
            tolerance_change=1e-14,
        )

        def closure():
            polish.zero_grad()
            err = (model(x[::4], a[::4])[0] / y[::4]).log() * 100
            loss = (
                err.square().mean() + 0.05 * err.square().topk(96).values.mean()
            )
            loss.backward()
            return loss

        polish.step(closure)
        updates = steps + int(polish.state[next(model.parameters())]["n_iter"])
    xv, av, yv = examples(seed + 8765, 8192, domain)
    with torch.no_grad():
        pred, power = model(xv, av)
        err = abs(pred / yv - 1)
        return model, dict(
            domain=domain,
            updates=updates,
            train_points=len(x),
            validation_points=len(xv),
            max_validation_pct=100 * err.max().item(),
            p99_validation_pct=100 * torch.quantile(err, 0.99).item(),
            min_conductance=pred.min().item(),
            max_conductance=pred.max().item(),
            min_radial_slope=power.min().item(),
            max_radial_slope=power.max().item(),
        )


def fit(a):
    """Archive selected process heads and every local validation result."""
    a.folder.mkdir(exist_ok=True, parents=True)
    rows = []
    for kind in a.kind:
        for width in [0] if kind == "rational" else a.width:
            for split in a.split:
                heads = {}
                records = []
                tic = perf_counter()
                for j, domain in enumerate(
                    ["water", "organic"] if split == "separate" else ["shared"]
                ):
                    model, r = train_head(
                        kind, width, domain, a.seed + 19 * j, a.steps
                    )
                    heads[domain] = model.state_dict()
                    records.append(
                        dict(
                            kind=kind,
                            width=width,
                            split=split,
                            parameters=sum(
                                p.numel() for p in model.parameters()
                            ),
                            **r,
                        )
                    )
                name = f"conductance-{kind}-w{width}-{split}"
                out = a.folder / (name + ".pt")
                torch.save(
                    dict(
                        conductance=True,
                        kind=kind,
                        width=width,
                        split=split,
                        heads=heads,
                        seed=a.seed,
                    ),
                    out,
                )
                digest = hashlib.sha256(out.read_bytes()).hexdigest()
                rows.extend(
                    dict(
                        name=name,
                        sha256=digest,
                        seconds=perf_counter() - tic,
                        **r,
                    )
                    for r in records
                )
                write_rows(a.table, rows)
                print(
                    name,
                    max(r["max_validation_pct"] for r in records),
                    flush=True,
                )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument("table", type=Path)
    p.add_argument(
        "--kind",
        nargs="+",
        choices=["rational", "mlp", "features"],
        default=["rational", "mlp"],
    )
    p.add_argument("--width", nargs="+", type=int, default=[8, 24])
    p.add_argument(
        "--split",
        nargs="+",
        choices=["shared", "separate"],
        default=["shared", "separate"],
    )
    p.add_argument("--seed", type=int, default=979)
    p.add_argument("--steps", type=int, default=4000)
    torch.set_num_threads(1)
    fit(p.parse_args())
