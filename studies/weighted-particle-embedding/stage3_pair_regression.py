"""Fit a small neural or trilinear charged collision law at fixed density."""

import argparse
import csv
import math
from pathlib import Path
from time import perf_counter

import torch
from coupled_events import KT
from particula.util import constants as pc
from stage2_pair import COULOMB_LENGTH, charged_kernel
from torch.nn import functional

# Coordinates are log(smaller radius / nm), log(radius ratio) and asinh(phi).
LOW = (math.log(2.5), 0.0, -math.asinh(183.0))
HIGH = (math.log(800.0), math.log(240.0), math.asinh(4096.0))
RHO = 1400.0
ZERO_PHI = math.log(1e-80)
PHYSICAL_LENGTH = pc.ELEMENTARY_CHARGE_VALUE**2 / (
    4 * math.pi * pc.ELECTRIC_PERMITTIVITY * KT
)


def physical_coordinates(u):
    """Invert normalized coordinates to geometry and electrostatic energy."""
    x = torch.lerp(u.new_tensor(LOW), u.new_tensor(HIGH), u)
    r = 1e-9 * x[:, 0].exp()
    return r, r * x[:, 1].exp(), x[:, 2].sinh()


def label(u):
    """Evaluate the dimensional native-formula teacher on independent pairs."""
    small, big, phi = physical_coordinates(u)
    radii = torch.stack((small, big), 1)
    mass = (4 * math.pi / 3 * RHO) * radii**3
    # Only the charge product enters the physical equation. Integer products
    # are used during all particle forecasts and independent native audits.
    charge = torch.stack(
        (torch.ones_like(phi), -phi * (small + big) / COULOMB_LENGTH), 1
    )
    ids = torch.arange(2, device=u.device)
    target = charged_kernel(radii, mass, charge, ids[:1], ids[1:])[:, 0]
    return torch.log(target.clamp_min(1e-300) / 1e-15) - correction(phi)


def correction(phi):
    """Remove dominant activation suppression and attractive enhancement."""
    return phi.clamp_max(0.0) + torch.log1p(abs(phi))


def coordinates(r, z, i, j):
    """Map permutation-symmetric particle pairs to normalized coordinates."""
    ri, rj = r[i], r[j]
    lo, hi = torch.minimum(ri, rj), torch.maximum(ri, rj)
    phi = (-COULOMB_LENGTH * z[i].to(r.dtype) * z[j] / (lo + hi)).clamp_min(
        -200.0
    )
    x = torch.stack(((lo * 1e9).log(), (hi / lo).log(), torch.asinh(phi)), -1)
    low = x.new_tensor(LOW)
    return (x - low) / (x.new_tensor(HIGH) - low), phi


def lattice_values(table, u):
    """Use trilinear interpolation of single-precision dimensionless labels."""
    # grid_sample expects width, height, depth, reversing the table axes.
    grid = (2 * u.float().flip((-1,)) - 1).reshape(1, 1, 1, -1, 3)
    return functional.grid_sample(
        table[None, None],
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).flatten()


def network(width):
    """Allocate a pointwise regressor of the transformed dimensional kernel."""
    return torch.nn.Sequential(
        torch.nn.Linear(3, width),
        torch.nn.SiLU(),
        torch.nn.Linear(width, width),
        torch.nn.SiLU(),
        torch.nn.Linear(width, 1),
    )


class FittedKernel:
    """Predict in-domain rates and refine selected sensitive cells."""

    def __init__(self, path, device, guard):
        """Keep the fit and its calibration mask on the forecast device."""
        cp = torch.load(path, map_location=device, weights_only=True)
        self.kind = cp["kind"]
        self.guard = guard
        self.mask = cp["mask"]
        self.shape = self.mask.shape
        self.cells = self.mask.new_tensor(self.shape, dtype=torch.int64)
        if self.kind == "lattice":
            self.table = cp["table"]
        else:
            self.net = network(cp["width"]).to(device)
            self.net.load_state_dict(cp["state"])
        self.fallback_pairs = 0
        self.pairs = 0

    def predict(self, r, m, z, i, j):
        """Return nonnegative rates, preserving the same-sign cutoff."""
        u, phi = coordinates(r, z, i, j)
        inside = ((u >= 0) & (u <= 1)).all(-1)
        uc = u.clamp(0, 1)
        pred = (
            lattice_values(self.table, uc)
            if self.kind == "lattice"
            else self.net(2 * uc.float() - 1)[:, 0]
        )
        rates = 1e-15 * torch.exp(pred.to(phi.dtype) + correction(phi))
        use = ~inside
        if self.guard:
            cell = torch.minimum((uc * self.cells).long(), self.cells - 1)
            use |= self.mask[cell[:, 0], cell[:, 1], cell[:, 2]]
        suppressed = phi * (PHYSICAL_LENGTH / COULOMB_LENGTH) < ZERO_PHI
        use &= ~suppressed
        rates.masked_fill_(suppressed, 0)
        if bool(use.any()):
            a, b = i[use], j[use]
            # Re-evaluate only fallback partners, avoiding unrelated nodes.
            rr = torch.stack((r[a], r[b]), 1)
            mm = torch.stack((m[a], m[b]), 1)
            zz = torch.stack((z[a], z[b]), 1)
            pair = torch.arange(2, device=r.device)
            rates[use] = charged_kernel(rr, mm, zz, pair[:1], pair[1:])[:, 0]
            self.fallback_pairs += len(a)
        self.pairs += len(i)
        return rates


def chunks(coords, fn):
    """Bound temporary storage when fitting the physical rate surface."""
    return torch.cat([fn(u) for u in coords.split(8192)])


def make_lattice(n):
    """Evaluate structured training knots on the normalized cube."""
    axis = torch.linspace(0, 1, n, dtype=torch.float64)
    pts = torch.stack(
        torch.meshgrid(axis, axis, axis, indexing="ij"), -1
    ).reshape(-1, 3)
    return chunks(pts, label).reshape(n, n, n).float()


@torch.no_grad()
def calibration(n, fn, tolerance):
    """Flag cells using independent interior quadrature probes."""
    axis = torch.arange(n - 1, dtype=torch.float64)
    base = torch.stack(
        torch.meshgrid(axis, axis, axis, indexing="ij"), -1
    ).reshape(-1, 3)
    flag = torch.zeros(len(base), dtype=torch.bool)
    for frac in [
        (0.5, 0.5, 0.5),
        (0.16, 0.67, 0.31),
        (0.79, 0.19, 0.83),
        (0.33, 0.87, 0.65),
    ]:
        u = (base + base.new_tensor(frac)) / (n - 1)
        truth = chunks(u, label)
        pred = chunks(u, fn).double()
        flag |= (pred - truth).abs() > math.log1p(tolerance)
    return flag.reshape(n - 1, n - 1, n - 1)


def fit(folder, kind, size, seed, updates, tolerance):
    """Fit on generated pairs and validate on a separate Sobol sequence."""
    tic = perf_counter()
    generator = torch.quasirandom.SobolEngine(3, scramble=True, seed=seed)
    validation = (
        torch.quasirandom.SobolEngine(3, scramble=True, seed=seed + 1000)
        .draw(8192)
        .double()
    )
    truth = label(validation)
    if kind == "lattice":
        table = make_lattice(size)
        cp = dict(kind=kind, table=table)

        def fn(u):
            return lattice_values(table, u)

        n = size
    else:
        torch.manual_seed(seed)
        model = network(size)
        opt = torch.optim.Adam(model.parameters(), lr=0.004)
        n = 34
        best = float("inf")
        for it in range(updates + 1):
            if it % 200 == 0 or it == updates:
                with torch.no_grad():
                    err = (
                        model(2 * validation.float() - 1)[:, 0] - truth
                    ).abs()
                    score = float(torch.quantile(err, 0.95))
                    if score < best:
                        best = score
                        state = {
                            k: w.detach().clone()
                            for k, w in model.state_dict().items()
                        }
                        best_it = it
            if it == updates:
                break
            u = generator.draw(1024).double()
            error = model(2 * u.float() - 1)[:, 0] - label(u)
            loss = (
                error.square().mean()
                + 0.1 * error.abs().topk(32).values.square().mean()
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            if it in (updates // 2, 3 * updates // 4):
                for g in opt.param_groups:
                    g["lr"] *= 0.4
        model.load_state_dict(state)
        cp = dict(kind=kind, width=size, state=state, best_update=best_it)

        def fn(u):
            return model(2 * u.float() - 1)[:, 0]

    with torch.no_grad():
        err = (fn(validation).double() - truth).expm1().abs()
        cp["mask"] = calibration(n, fn, tolerance)
        cp.update(
            seed=seed,
            updates=updates if kind == "mlp" else 0,
            tolerance=tolerance,
        )
        path = folder / f"{kind}-{size}-s{seed}.pt"
        torch.save(cp, path)
        return dict(
            kind=kind,
            size=size,
            seed=seed,
            updates=cp["updates"],
            checkpoint=path.name,
            bytes=path.stat().st_size,
            fit_and_calibration_s=perf_counter() - tic,
            cells_flagged_pct=float(cp["mask"].float().mean()) * 100,
            validation_median_pct=float(err.median()) * 100,
            validation_p99_pct=float(torch.quantile(err, 0.99)) * 100,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument("--kind", choices=["lattice", "mlp"], required=True)
    p.add_argument("--sizes", nargs="+", type=int, default=[32, 64])
    p.add_argument("--seed", type=int, default=401001)
    p.add_argument("--updates", type=int, default=3600)
    p.add_argument("--tolerance", type=float, default=0.003)
    a = p.parse_args()
    a.folder.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    for size in a.sizes:
        row = fit(a.folder, a.kind, size, a.seed, a.updates, a.tolerance)
        with (a.folder / (row["checkpoint"] + ".csv")).open("w") as file:
            w = csv.DictWriter(file, fieldnames=row)
            w.writeheader()
            w.writerow(row)
        print(row, flush=True)
