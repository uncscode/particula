"""Compare three collision operators inside the same native CUDA vapor run."""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from coupled_events import MFP_AIR, MU_AIR
from particula.util.constants import STANDARD_GRAVITY
from stage3_evidence import write
from stage3_native_sampled import ResidentPairs
from stage3_one_box import Box

MECHANISMS = {
    "Brownian": ("brownian",),
    "Charged": ("charged_hard_sphere",),
    "Settling": ("sedimentation_sp2016",),
    "Shear": ("turbulent_shear_st1956",),
    "Combined": ("brownian", "sedimentation_sp2016", "turbulent_shear_st1956"),
}
DISSIPATION = 1.0
AIR_DENSITY = 1.2
MATERIAL_DENSITY = 1400.0
METHODS = ("native", "sampled physical", "sampled neural")


def mechanical_rate(law, a, b):
    """Evaluate the native unit-efficiency settling or shear pair law, m³/s."""
    if law == "Shear":
        scale = math.sqrt(math.pi * DISSIPATION * AIR_DENSITY / (120 * MU_AIR))
        return scale * (2 * (a + b)) ** 3

    def velocity(r):
        kn = MFP_AIR / r
        slip = 1 + kn * (1.257 + 0.4 * torch.exp(-1.1 / kn))
        return (
            2 * STANDARD_GRAVITY * MATERIAL_DENSITY * r**2 * slip / (9 * MU_AIR)
        )

    return math.pi * (a + b) ** 2 * (velocity(a) - velocity(b)).abs()


class MechanicalFit(torch.nn.Module):
    """Learn a symmetric rate with an exact settling zero for equal radii."""

    def __init__(self, law):
        """Build the two-hidden-layer model for one law."""
        super().__init__()
        self.law = law
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(2, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, 1),
        ).double()

    def forward(self, a, b):
        """Return the fitted rate in cubic metres per second."""
        x, y = (a / 300e-9).log(), (b / 300e-9).log()
        features = torch.stack(((x + y) / 4, (x - y).abs() / 4), dim=-1)
        scale = 1e-15
        if self.law == "Settling":
            scale = scale * (a - b).abs() / (a + b)
        return scale * torch.exp(self.layers(features).squeeze(-1))


def fit_rates(folder, device="cuda"):
    """Fit on log-uniform radii, then evaluate a disjoint physical label set."""
    reports = []
    fits = {}
    for law in ("Shear", "Settling"):
        torch.manual_seed(77021)
        model = MechanicalFit(law)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.008)
        radii = (
            20e-9
            * (
                torch.rand((2, 32768), dtype=torch.float64) * math.log(200)
            ).exp()
        )
        target = mechanical_rate(law, *radii)
        for _ in range(1800):
            idx = torch.randint(32768, (256,))
            loss = ((model(*radii[:, idx]) / target[idx]).log() ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            holdout = (
                20e-9
                * (
                    torch.rand((2, 8192), dtype=torch.float64) * math.log(200)
                ).exp()
            )
            truth = mechanical_rate(law, *holdout)
            err = (model(*holdout) / truth - 1).abs() * 100
        reports.append(
            dict(
                kernel=law,
                samples=len(err),
                median_pct=float(err.median()),
                p99_pct=float(torch.quantile(err, 0.99)),
            )
        )
        fits[law] = model.eval().to(device)
    write(folder / "fits.csv", reports)
    return fits


class KernelBox(Box):
    """Reuse the existing random matching, replacing only the per-pair law."""

    def rates(self, r, mass, charge, ij):
        """Evaluate the selected law on the supplied pairs."""
        i, j = ij
        if self.law in ("Brownian", "Charged"):
            return super().rates(r, mass, charge, ij)
        a, b = r[i], r[j]
        if self.law == "Combined":
            return (
                super().rates(r, mass, charge, ij)
                + mechanical_rate("Settling", a, b)
                + mechanical_rate("Shear", a, b)
            )
        if self.fit is not None:
            return self.fit(a, b)
        return mechanical_rate(self.law, a, b)


class KernelPairs(ResidentPairs):
    """Bind the existing resident merger and sampler to the selected kernel."""

    def __init__(self, fixture, law, method, fits):
        """Attach the appropriate frozen law to the matching routine."""
        self.parts = 1
        self.kernel = method
        neural = method == "sampled neural"
        charged = "bipolar1" if law == "Charged" else "neutral"
        backbone = (
            "learned"
            if neural and law in ("Brownian", "Charged", "Combined")
            else "physical"
        )
        self.box = KernelBox(
            2,
            fixture.spec.species,
            2 / fixture.volume,
            charged,
            "cuda",
            307,
            1,
            backbone,
        )
        self.box.law = law
        self.box.fit = fits.get(law) if neural else None
        self.reset([1])


def build_solver(fixture, law, method, fits, steps, micro):
    """Use the kernel-specific full native baseline on the same vapor clock."""
    from particula.gpu.kernels.coagulation import CoagulationMechanismConfig
    from stage2_native_split import NativeSplit

    charged = law == "Charged"
    shear = law in ("Shear", "Combined")
    native = method == "native"
    return NativeSplit(
        [fixture],
        steps,
        micro,
        collisions=native,
        mechanism_config=CoagulationMechanismConfig(
            MECHANISMS[law],
            charged_model="gopalakrishnan2012" if charged else "hard_sphere",
        ),
        turbulent_dissipation=DISSIPATION if shear else None,
        fluid_density=AIR_DENSITY if shear else None,
        sampled_collision=None
        if native
        else KernelPairs(fixture, law, method, fits),
    )


def check(obs, fixture):
    """Verify species, charge and count budgets at each recorded time."""
    s = fixture.spec.species
    particulate = obs[:, 1 : s + 2]
    gas = obs[:, s + 2 : 2 * s + 2]
    inventory = np.column_stack((particulate[:, :s] + gas, particulate[:, s]))
    drift = np.max(np.abs(inventory / inventory[0] - 1)) * 100
    counts = np.rint(obs[:, 0] * fixture.volume).astype(int)
    if (
        not np.isfinite(obs).all()
        or drift >= 1e-7
        or np.any(np.diff(counts) > 0)
        or np.any(
            np.abs(obs[:, 2 * s + 3] * fixture.volume - fixture.charge.sum())
            > 1e-6
        )
        or np.any(particulate < 0)
        or np.any(gas < 0)
    ):
        raise ValueError("Invalid forecast or conservation budget")
    return drift, counts


def phase_fields(obs, species):
    """Return individual and aggregate particle and gas concentrations."""
    particle = obs[:, 1 : species + 2]
    gas = obs[:, species + 2 : 2 * species + 2]
    return np.column_stack((particle, gas, particle.sum(1), gas.sum(1)))


def radii_from_quantiles(u):
    """Assign 80% small particles and 20% larger collectors (radii in m)."""
    return np.where(
        u < 0.8,
        (40 + 50 * u / 0.8) * 1e-9,
        (0.7 + 0.9 * (u - 0.8) / 0.2) * 1e-6,
    )


def fixture(n, law):
    """Restore the same dry mixture and finite vapor reservoirs for each run."""
    from stage3_native_scale import box

    f = box(
        n, 32, 1e12, 307, 1.0, "bipolar1" if law == "Charged" else "neutral"
    )
    u = ((np.arange(n) + 1) * 0.618033988749895) % 1
    radius = radii_from_quantiles(u)
    f.mass[:, -1] = 4 * math.pi / 3 * MATERIAL_DENSITY * radius**3
    return f


def campaign(folder, sizes, repeats, steps, micro, laws):
    """Time matched forecasts after warming each operator on the device."""
    import warp as wp

    folder.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() or not wp.is_cuda_available():
        raise RuntimeError("CUDA required")
    torch.set_num_threads(1)
    fits = fit_rates(folder)
    rows = []
    for n in sizes:
        for law in laws:
            f = fixture(n, law)
            solvers = {}
            for method in METHODS:
                solvers[method] = build_solver(
                    f, law, method, fits, steps, micro
                )
                solvers[method].run([f], [77020])
            for k in range(repeats):
                seed = 78100 + k
                observations, times, budgets = {}, {}, {}
                for method in np.random.default_rng(seed).permutation(METHODS):
                    wp.synchronize_device("cuda:0")
                    start = perf_counter()
                    obs = solvers[method].run([f], [seed])[0]
                    wp.synchronize_device("cuda:0")
                    times[method] = perf_counter() - start
                    budgets[method] = check(obs, f)
                    observations[method] = obs
                reference = observations["native"]
                target = phase_fields(reference, f.spec.species)
                for method in METHODS:
                    drift, counts = budgets[method]
                    pred = observations[method]
                    phase_error = np.max(
                        np.abs(phase_fields(pred, f.spec.species) - target)
                        / np.maximum(target, 1e-11)
                    )
                    count_error = np.max(
                        np.abs(pred[:, 0] / reference[:, 0] - 1)
                    )
                    rows.append(
                        dict(
                            n=n,
                            kernel=law,
                            method=method,
                            seed=seed,
                            steps_per_observation=steps,
                            vapor_substeps_per_half=micro,
                            seconds=times[method],
                            mergers=n - counts[-1],
                            count_error_pct=100 * count_error,
                            phase_error_pct=100 * phase_error,
                            species_drift_pct=drift,
                        )
                    )
            write(folder / "trials.csv", rows)
            print(
                n,
                law,
                [
                    (m, round(times[m], 3), int(n - budgets[m][1][-1]))
                    for m in METHODS
                ],
                flush=True,
            )
            del solvers
    (folder / "settings.json").write_text(
        json.dumps(
            dict(
                device=torch.cuda.get_device_name(),
                torch=str(torch.__version__),
                warp=wp.__version__,
                volume_m3="n/1e12",
                temperature_K=293,
                pressure_Pa=101325,
                horizon_s=1,
                vapor_count=32,
                native_vapor_steps_per_half=micro,
                collision_steps=4 * steps,
                fit_seed=77021,
                first_event_seed=78100,
                repetitions=repeats,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--sizes", nargs="+", type=int, default=[4096, 32768])
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--micro", type=int, default=4)
    parser.add_argument(
        "--laws", nargs="+", choices=MECHANISMS, default=list(MECHANISMS)
    )
    args = parser.parse_args()
    if extra_site := os.getenv("EMBEDDING_WARP_SITE"):
        sys.path.append(extra_site)
    campaign(
        args.folder, args.sizes, args.repeats, args.steps, args.micro, args.laws
    )
