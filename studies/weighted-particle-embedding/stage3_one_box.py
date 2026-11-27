"""Linear-memory, explicit one-box benchmark for irreversible multivapor growth.

Particles have unit multiplicity. Gas is finite; the condensation coefficient
is diffusion-limited with a transition-regime correction. The randomized
matching estimates all-pair collisions with their correct inclusion probability.
"""

import argparse
import json
import math
from pathlib import Path
from time import perf_counter

import torch
from stage2_pair import charged_kernel, load_pair, physics_kernel
from stage3_charge_modes import initial_charge
from stage3_spectra import dry_radius


class Box:
    """Store one mass row per particle in a finite volume."""

    def __init__(
        self,
        n,
        species,
        density,
        charge,
        device,
        family,
        seed,
        kernel,
        *,
        compact=False,
    ):
        """Allocate unit-weight particles, charge and distinct vapor lanes."""
        self.device = torch.device(device)
        self.volume = n / density
        self.n0 = n
        self.charge_mode = charge
        self.species = species
        self.kernel = kernel
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.dtype = torch.float64
        # Deterministic functions of particle ID give comparable distinct
        # populations across devices and independent event streams.
        i = torch.arange(n, dtype=self.dtype, device=device)
        u = ((i + 1) * 0.618033988749895 + family * 0.0331).remainder(1)
        radius = dry_radius(u, family, torch.where)
        self.mass = torch.zeros(
            (n, 2 if compact else species + 1), dtype=self.dtype, device=device
        )
        self.mass[:, 0] = 4 * math.pi / 3 * 1400 * radius**3
        self.z = initial_charge(i, u, charge, torch.where).to(torch.int64)
        self.q0 = int(self.z.sum())
        j = torch.arange(species, dtype=self.dtype, device=device)
        self.diff = 5e-6 + 1.2e-5 * (1 + j) / (1 + species)
        self.alpha = 0.08 + 0.91 * (1 + ((j * 7) % species)) / (1 + species)
        self.path = 3 * self.diff / (150 + 150 * (j + 1) / species)
        # Every vapor has its own transport constants and reservoir (kg/m3).
        self.gas = (1 + j / species) * 1e-8 / species
        if compact:
            self.initial = torch.cat(
                (
                    self.mass[:, 0].sum().reshape(1) / self.volume,
                    self.gas.clone(),
                )
            )
        else:
            self.initial = self.mass.sum(0) / self.volume
            self.initial[1:] += self.gas
        self.collisions = self.pairs = self.colsteps = self.vsteps = (
            self.rejections
        ) = 0
        self.max_probability = 0.0
        self.nn = None
        if kernel == "learned":
            f = (
                "jump-w4-s901.pt"
                if charge == "neutral"
                else "charge8-weighted.pt"
            )
            self.nn = load_pair(
                Path(__file__).parent / "examples/stage-2" / f, device
            )

    def geometry(self):
        """Return current wet radii in metres and total particle masses."""
        total = self.mass.sum(-1)
        return (total * (3 / (4 * math.pi * 1400))).pow(1 / 3), total

    def conductance(self, radius, j):
        """Evaluate size-dependent uptake coefficients in cubic metres/s."""
        kn = self.path[j][None, :] / radius[:, None]
        a = self.alpha[j][None, :]
        beta = 0.75 * a * (1 + kn) / (kn * kn + (1 + 0.283 * a) * kn + 0.75 * a)
        return (4 * math.pi) * radius[:, None] * self.diff[j][None, :] * beta

    def vapor(self, duration, vmax, tile=4):
        """Limit radius change on the vapor clock."""
        left = duration
        while left > duration * 1e-12:
            radius, mass = self.geometry()
            dt = left
            if vmax > 0:
                maxrate = torch.zeros_like(radius)
                for start in range(0, self.species, tile):
                    j = slice(start, min(start + tile, self.species))
                    maxrate += (self.conductance(radius, j) * self.gas[j]).sum(
                        -1
                    )
                fastest = float((maxrate / mass).max())
                if fastest:
                    dt = min(left, ((1 + vmax) ** 3 - 1) / fastest)
            for start in range(0, self.species, tile):
                stop = min(start + tile, self.species)
                j = slice(start, stop)
                k = self.conductance(radius, j)
                sumk = k.sum(0)
                take = self.gas[j] * (-torch.expm1(-dt * sumk / self.volume))
                self.mass[:, start + 1 : stop + 1] += k * (
                    take * self.volume / sumk
                )
                self.gas[j] -= take
            self.vsteps += 1
            left -= dt
            if self.vsteps > 100000:
                raise RuntimeError("Vapor controller exhausted trial budget")

    def rates(self, r, m, z, ij):
        """Evaluate the collision kernel for the selected particle pairs."""
        i, j = ij
        r, m, z = r[None, :], m[None, :], z[None, :]
        if self.kernel == "constant":
            return torch.full(
                (i.numel(),), 2e-15, device=self.device, dtype=self.dtype
            )
        if self.nn is not None:
            active = torch.ones_like(r, dtype=torch.bool)
            return self.nn(r, m, z, active, i, j, self.nn.start(r))[0][0]
        if self.charge_mode == "neutral":
            return physics_kernel(r, m, i, j)[0]
        return charged_kernel(r, m, z, i, j)[0]

    def select_pairs(self, left, right, scale, exponential):
        """Return accepted partners while bounding neural-network workspace."""
        half = len(left)
        r, m = self.geometry()
        # Process independent pair blocks to bound charged-network workspace.
        chosen = []
        for start in range(0, half, 32768):
            i, j = left[start : start + 32768], right[start : start + 32768]
            ids = torch.cat((i, j))
            local = torch.arange(len(i), device=self.device)
            p = (
                self.rates(r[ids], m[ids], self.z[ids], (local, local + len(i)))
                * scale
            )
            if exponential:
                p = -torch.expm1(-p)
            peak = float(p.max())
            if not math.isfinite(peak) or peak > 1 or bool((p < 0).any()):
                raise ArithmeticError(
                    f"Invalid matched-pair probability {peak}"
                )
            self.max_probability = max(self.max_probability, peak)
            chosen.append(
                torch.rand(
                    len(i),
                    dtype=self.dtype,
                    device=self.device,
                    generator=self.generator,
                )
                < p
            )
        return torch.cat(chosen)

    def merger_pairs(self, dt, exponential, active=None):
        """Draw and accept disjoint partners among the live particle slots."""
        n = len(self.mass) if active is None else len(active)
        perm = torch.randperm(n, device=self.device, generator=self.generator)
        if active is not None:
            perm = active[perm]
        half = n // 2
        left, right = perm[:half], perm[half : 2 * half]
        scale = dt * ((n * (n - 1) / 2) / half / self.volume)
        keep = self.select_pairs(left, right, scale, exponential)
        return left[keep], right[keep]

    def collide(self, dt, exponential=False):
        """Commit disjoint sampled pairs using linear or exponential hazards."""
        n = len(self.mass)
        if self.kernel == "off" or n < 2:
            return
        i, j = self.merger_pairs(dt, exponential)
        if i.numel():
            self.mass[i] += self.mass[j]
            self.z[i] += self.z[j]
            live = torch.ones(n, dtype=torch.bool, device=self.device)
            live[j] = False
            self.mass, self.z = self.mass[live], self.z[live]
        self.collisions += len(i)
        self.colsteps += 1
        self.pairs += n // 2

    def measure(self):
        """Check invariants and extract population observables."""
        inventory = self.mass.sum(0) / self.volume
        total = inventory.clone()
        total[1:] += self.gas
        drift = float(((total - self.initial).abs() / self.initial).max())
        if (
            int(self.z.sum()) != self.q0
            or len(self.mass) != self.n0 - self.collisions
        ):
            raise ArithmeticError("Charge or count ledger failed")
        if (
            drift > 1e-9
            or not math.isfinite(drift)
            or bool((self.gas < 0).any())
        ):
            raise ArithmeticError(f"Inventory error {drift}")
        r, _ = self.geometry()
        idx = torch.bucketize(r, r.new_tensor([35e-9, 70e-9, 120e-9]))
        size = torch.bincount(idx, minlength=4)
        return dict(
            n=len(r),
            gas=self.gas.tolist(),
            particle=inventory.tolist(),
            bands=size.tolist(),
            charge_abs=float(self.z.abs().sum()),
            ledger=drift,
        )


@torch.inference_mode()
def simulate(
    n,
    species,
    density,
    charge,
    device,
    family,
    seed,
    kernel,
    horizon,
    steps,
    vmax,
    observations=4,
):
    """Measure one explicit coupled box on the selected device."""

    def sync():
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)

    sync()
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    tic = perf_counter()
    box = Box(n, species, density, charge, device, family, seed, kernel)
    history = [box.measure()]
    sync()
    setup = perf_counter() - tic
    timer = perf_counter()
    for _ in range(observations):
        h = horizon / (observations * steps)
        for _ in range(steps):
            box.vapor(h / 2, vmax)
            box.collide(h)
            box.vapor(h / 2, vmax)
        history.append(box.measure())
    sync()
    return dict(
        n=n,
        species=species,
        density=density,
        charge=charge,
        device=str(device),
        family=family,
        seed=seed,
        kernel=kernel,
        horizon=horizon,
        steps=steps,
        vmax=vmax,
        observations=observations,
        volume_m3=box.volume,
        setup_s=setup,
        advance_and_observe_s=perf_counter() - timer,
        total_s=perf_counter() - tic,
        cuda_peak_allocated=torch.cuda.max_memory_allocated(device)
        if str(device).startswith("cuda")
        else 0,
        cuda_peak_reserved=torch.cuda.max_memory_reserved(device)
        if str(device).startswith("cuda")
        else 0,
        final_n=len(box.mass),
        collisions=box.collisions,
        pairs=box.pairs,
        vapor_steps=box.vsteps,
        collision_steps=box.colsteps,
        max_probability=box.max_probability,
        history=history,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    for k, default in dict(
        n=256, species=2, family=101, seed=212001, steps=2
    ).items():
        p.add_argument("--" + k, type=int, default=default)
    for k, default in dict(density=1e12, horizon=1.0, vmax=0.02).items():
        p.add_argument("--" + k, type=float, default=default)
    p.add_argument("--charge", default="neutral")
    p.add_argument("--kernel", default="physical")
    p.add_argument("--device", default="cuda")
    args = vars(p.parse_args())
    path = args.pop("output")
    torch.set_num_threads(1)
    result = simulate(**args)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
