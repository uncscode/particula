"""Whole-spectrum Particula chamber replay. All radii compete for a shared vapor."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import particula as par
import warp as wp
from particula.dynamics.condensation.species_mapping import CondensationSpeciesMap

from workspace import WORK, measurement

HERE = WORK / "physical"
from observations import D, average, load

MW = 0.01801528
R = 8.314462618
RHO = 1000.0
MODE = "particle_resolved"
EDGES = np.arange(0.0, 152.0)
DIMS = (0.508, 0.508, 0.495)
RUN_EPS = {"forced": 4.0, "unforced": 0.004}
GAS = par.gas
DYN = par.dynamics


def condensation(temp, pressure):
    celsius = temp - 273.15
    es = 611.21 * np.exp((18.678 - celsius / 234.5) * celsius / (257.14 + celsius))
    diffusivity = 2.11e-5 * (temp / 273.15) ** 1.94 * 101325.0 / pressure
    return DYN.MassCondensation(
        DYN.CondensationLatentHeat(
            molar_mass=np.array([MW]),
            diffusion_coefficient=diffusivity,
            latent_heat=2.5e6,
            accommodation_coefficient=1.0,
            activity_strategy=par.particles.ActivityIdealMass(),
            surface_strategy=par.particles.SurfaceStrategyVolume(
                surface_tension=0.073, density=RHO
            ),
            vapor_pressure_strategy=GAS.ConstantVaporPressureStrategy(es),
            species_map=CondensationSpeciesMap(("water",), ((0, 0),)),
        )
    )


def coagulation(temp, pressure, epsilon):
    kernels = [DYN.BrownianCoagulationStrategy(MODE), DYN.SedimentationCoagulationStrategy(MODE)]
    if epsilon:
        rho_gas = pressure * 0.0280134 / (R * temp)
        kernels.append(DYN.TurbulentShearCoagulationStrategy(MODE, epsilon, rho_gas))
    return DYN.Coagulation(DYN.CombineCoagulationStrategy(kernels))


def initial(run, n, replicas):
    t, mat = load(run)
    pop = average(t, mat, 2.5, 3.0)
    vols = np.full(replicas, n / (1e6 * pop.sum()))
    cum = np.r_[0.0, np.cumsum(pop) / pop.sum()]
    samples = []
    for b in range(replicas):
        rng = np.random.default_rng(93017 + b)
        u = (np.arange(n) + rng.random(n)) / n
        j = np.searchsorted(cum, u, side="right") - 1
        diam = D[j] - 0.5 + (u - cum[j]) / (cum[j + 1] - cum[j])
        samples.append(rng.permutation(diam))
    mass = RHO * np.pi / 6 * (np.array(samples) * 1e-6) ** 3
    particles = par.particles.ParticleData(
        masses=mass[:, :, None],
        density=np.array([RHO]),
        concentration=np.ones_like(mass),
        charge=np.zeros_like(mass),
        volume=vols,
        distribution_type=MODE,
    )
    return particles


def inspect_state(state):
    p = state.particles
    mass = p.masses[:, :, 0]
    assert np.all(np.isfinite(mass)) and np.min(mass) >= 0
    active = (p.concentration > 0) & (mass > 0)
    assert np.all(p.concentration[active] == 1)
    number = active.sum(axis=1)
    liq = (mass * p.concentration).sum(axis=1) / p.volume
    gas = state.gas.concentration[:, 0]
    assert np.all(np.isfinite(gas)) and np.all(gas >= 0)
    return number, liq, gas


def spectrum(particles):
    d = 1e6 * np.cbrt(6 * particles.masses[:, :, 0] / (np.pi * RHO))
    active = (particles.concentration > 0) & (d > 0)
    if d.max() >= EDGES[-1]:
        raise ValueError("Increase reported histogram upper limit")
    return np.stack(
        [
            np.histogram(row[mask], EDGES)[0] / (vol * 1e6)
            for row, mask, vol in zip(d, active, particles.volume)
        ]
    )


def apply(proc, state, dt, steps, seed):
    with proc.prepare(state, device="cuda:0", time_step=dt / steps, root_seed=seed) as session:
        session.run(steps=steps)
        return session.snapshot()


def simulate(run, case, n, reps, split, output, pilot):
    start = time.monotonic()
    forcing = measurement("forcing/figure6.csv").query("run==@run and 2.75<time_s<7").copy()
    if case.startswith("sensor"):
        loc = int(case[-1])
        data = measurement("forcing/figureS2.csv")
        temperatures = (
            data.query("run==@run and sensor_from_center==@loc").set_index("time_s").temperature_K
        )
        forcing["temperature_K"] = temperatures.reindex(forcing.time_s).to_numpy()
    if pilot:
        forcing = forcing.iloc[:2]
    pres = 100 * (1015 - (347 if run == "forced" else 378))
    particles = initial(run, n, reps)
    eps = RUN_EPS[run]
    if case == "no_shear":
        eps = 0.0
    if case == "swap_shear":
        eps = RUN_EPS["forced" if run == "unforced" else "unforced"]
    eddy = {"low_wall": 0.1, "high_wall": 100.0}.get(case, 3.6)
    dims = (*DIMS[:2], 0.14 / (np.pi * 0.254**2)) if case == "viewport_height" else DIMS
    wall = DYN.WallLoss(DYN.RectangularWallLossStrategy(eddy, dims, MODE))
    use_wall = case not in ("cond", "cond_coag")
    use_coag = case not in ("cond", "cond_wall")
    all_spectra = []
    ledger = []
    gas_last = None
    water_error = 0.0
    coag_error = 0.0
    for i, row in enumerate(forcing.itertuples()):
        temp = row.temperature_K
        target = row.vapor_mol_mol * pres * MW / (R * temp)
        gas_mass = (
            gas_last if case == "closed_vapor" and gas_last is not None else np.full(reps, target)
        )
        external = np.zeros(reps) if gas_last is None else gas_mass - gas_last
        state = par.Aerosol(
            particles=particles,
            gas=GAS.GasData(
                name=["water"],
                molar_mass=np.array([MW]),
                concentration=gas_mass[:, None].copy(),
                partitioning=np.array([True]),
            ),
            environment=GAS.EnvironmentData(
                temperature=np.full(reps, temp),
                pressure=np.full(reps, pres),
                saturation_ratio=np.ones((reps, 1)),
            ),
        )
        cproc = condensation(temp, pres)
        kproc = coagulation(temp, pres, eps) if use_coag else None
        spec = np.zeros((reps, len(EDGES) - 1))
        counts = np.zeros((3, reps), int)
        wall_mass = np.zeros(reps)
        dt = 0.05 / split
        for j in range(split):
            before = spectrum(state.particles)
            n0, l0, g0 = inspect_state(state)
            seed = 170000 + i * 100 + j * 4
            state = apply(cproc, state, dt, 1, seed)
            n1, l1, g1 = inspect_state(state)
            water_error = max(
                water_error, np.max(abs(l0 + g0 - l1 - g1) / np.maximum(l0 + g0, 1e-30))
            )
            counts[0] += n0 - n1
            if use_coag:
                state = apply(kproc, state, dt, 1, seed + 1)
            n2, l2, g2 = inspect_state(state)
            coag_error = max(coag_error, np.max(abs(l2 - l1) / np.maximum(l1, 1e-30)))
            counts[1] += n1 - n2
            if use_wall:
                state = apply(wall, state, dt, 1, seed + 2)
            n3, l3, gas_last = inspect_state(state)
            counts[2] += n2 - n3
            wall_mass += l2 - l3
            assert np.all(n0 >= n1) and np.all(n1 >= n2) and np.all(n2 >= n3)
            assert np.array_equal(g1, g2) and np.array_equal(g2, gas_last)
            spec += (before + spectrum(state.particles)) / (2 * split)
        particles = state.particles
        for b in range(reps):
            ledger.append(
                [
                    row.time_s,
                    b,
                    *counts[:, b],
                    wall_mass[b],
                    external[b],
                    gas_last[b],
                    l3[b],
                    n3[b],
                    particles.volume[b],
                ]
            )
        all_spectra.append(spec)
    assert water_error < 1e-9 and coag_error < 1e-9
    meta = dict(
        run=run,
        case=case,
        n=n,
        replicas=reps,
        outer_step_s=0.05 / split,
        epsilon_m2_s3=eps,
        eddy_s_inv=eddy,
        dimensions_m=dims,
        coag=use_coag,
        wall=use_wall,
        device=wp.get_device("cuda:0").name,
        water_error=float(water_error),
        coag_error=float(coag_error),
        seconds=time.monotonic() - start,
    )
    folder = output / f"{run}-{case}-n{n}-s{split}"
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        folder / "spectrum.npz",
        D_um=(EDGES[:-1] + EDGES[1:]) / 2,
        t_s=forcing.time_s.to_numpy(),
        cm3=np.array(all_spectra),
    )
    pd.DataFrame(
        ledger,
        columns=[
            "t",
            "rep",
            "evaporated",
            "mergers",
            "deposited",
            "wall_kg_m3",
            "gas_reset_kg_m3",
            "gas_kg_m3",
            "liquid_kg_m3",
            "active",
            "volume_m3",
        ],
    ).to_csv(folder / "ledger.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    (folder / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(run, case, n, reps, round(meta["seconds"], 2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=["forced", "unforced"])
    p.add_argument("--cases", nargs="+", default=["full"])
    p.add_argument("--particles", type=int, default=256)
    p.add_argument("--replicas", type=int, default=12)
    p.add_argument("--split", type=int, default=1)
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--output", type=Path, default=HERE / "runs")
    args = p.parse_args()
    assert wp.get_device("cuda:0").is_cuda
    for run in args.runs:
        for case in args.cases:
            simulate(run, case, args.particles, args.replicas, args.split, args.output, args.pilot)
