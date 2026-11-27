"""Particle-resolved test of optical sampling and air transport on the 576-cell grid."""

import argparse, json, time
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from workspace import background
from chamber import HERE, HEIGHT0, temperature, MESHES
from mesh import Grid, air, gas_path, hologram
from geometry import inputs
from simulate import par, GAS, R, MW, RHO, condensation, apply, inspect_state, wp
from mixing import repack
from resolved import sampler


def simulate(
    run,
    flow,
    kernel,
    n=65536,
    reps=6,
    dt=0.05,
    mesh="576",
    scales=None,
    pair_schedule=None,
    output=None,
    thermal_schedule=None,
    matrix_schedule=None,
    hidden_population=None,
):
    if kernel != "standard":
        raise ValueError("Use a pair schedule to modify collisions")
    tick = time.monotonic()
    grid = Grid(*MESHES[mesh])
    nr = grid.nr
    b = grid.n
    d = np.arange(0.5, 351)
    times, p, base, chi, _, speed, oldloss, init, _, _ = inputs(run, "local", d, dt)
    x = background(run)
    temp = base[:, None] + x[0] + x[1] * temperature(grid, run, times)
    path = gas_path(grid, 0.230)
    if thermal_schedule is not None:
        radial = np.broadcast_to(grid.rc / 0.254, grid.shape).ravel().copy()
        radial -= path @ radial
        temp = temp + np.array([thermal_schedule(t) for t in times])[:, None] * radial / max(
            abs(radial)
        )
    M = splu(sparse.eye(b, format="csc") - dt * air(grid, run, flow, scales)).solve(np.eye(b))
    cdf = np.cumsum(M.T, axis=1)
    cdf[:, -1] = 1
    cdf = (cdf + 2.0 * np.arange(b)[:, None]).ravel()
    optics = np.stack(
        [hologram(grid, d, z=z)[0] for z in (HEIGHT0, HEIGHT0 - 0.055, HEIGHT0 + 0.055)]
    )
    optics = np.concatenate([optics, np.broadcast_to(grid.v[:, None], (1, b, len(d)))])
    profile = np.exp(x[4] * (grid.rc / 0.254 - 1 / 12) / (5 / 6))
    prof = np.broadcast_to(profile, grid.shape).ravel()[:, None]
    density = prof * init / (optics[0] * prof).sum(0)
    vols = n / (grid.v @ density.sum(1) * 1e6) * grid.v
    rng = [np.random.default_rng(70361 + i) for i in range(reps)]
    masses = []
    locations = []
    for gen in rng:
        chunks = []
        indices = []
        for j in range(b):
            count = int(np.floor(density[j].sum() * vols[j] * 1e6 + gen.random()))
            u = (np.arange(count) + gen.random(count)) / max(1, count)
            F = np.r_[0, density[j].cumsum() / density[j].sum()]
            idx = np.searchsorted(F, u, side="right") - 1
            diam = idx + (u - F[idx]) / (F[idx + 1] - F[idx])
            chunks.append(RHO * np.pi / 6 * (diam * 1e-6) ** 3)
            indices.append(np.full(count, j))
        masses.append(np.concatenate(chunks))
        locations.append(np.concatenate(indices))
    # Optional unresolved 4–10 µm inventory and appearance rate supplied by a fitted diagnostic.
    added = 0
    debit_kg = 0.0
    initial_extra = 0
    if hidden_population is not None:
        from scipy.special import ndtr, ndtri

        small_rng = [np.random.default_rng(89013 + i) for i in range(reps)]
        lo, hi = ndtr(np.log(np.array([4.0, 10.0]) / 7.0) / np.log(1.2))

        def add_small(concentration, gas=None):
            nonlocal added, debit_kg
            count = 0
            for k, gen in enumerate(small_rng):
                sites = np.repeat(np.arange(b), gen.poisson(concentration * vols * 1e6))
                diam = 7 * np.exp(np.log(1.2) * ndtri(gen.uniform(lo, hi, len(sites))))
                wet = RHO * np.pi / 6 * (diam * 1e-6) ** 3
                masses[k] = np.r_[masses[k], wet]
                locations[k] = np.r_[locations[k], sites]
                count += len(wet)
                if gas is not None:
                    delta = np.bincount(sites, weights=wet, minlength=b) / vols
                    gas[:, k] -= delta
                    debit_kg += wet.sum()
                    assert np.isclose(delta @ vols, wet.sum(), rtol=1e-12, atol=1e-24)
            if gas is not None:
                added += count
            return count

        initial_extra = add_small(hidden_population[0])

    def observation(ms, loc):
        j = (1e6 * np.cbrt(6 * ms / (RHO * np.pi))).astype(int)
        assert np.all(j < len(d))
        key = loc * len(d) + j
        hist = np.bincount(key, minlength=b * len(d)).reshape(b, -1) / (vols[:, None] * 1e6)
        return (optics * hist[None, :, :]).sum(1)

    prev = np.stack([observation(ms, idx) for ms, idx in zip(masses, locations)])
    carry = None
    rows = []
    eventlog = []
    eps_water = eps_coag = 0.0
    start_count = sum(map(len, masses))
    stepper = sampler(run, base.mean(), p)
    loss = np.zeros((*grid.shape, len(d)))
    loss[:, :, -1] += oldloss[11] * (11 / 36) / (1 - ((nr - 1) / nr) ** 2)
    loss[0] += grid.nz * oldloss[0] / 3
    loss[-1] += grid.nz * oldloss[0] / 3
    loss = loss.reshape(b, -1)
    for i, t in enumerate(times):
        factor = p * MW / (R * temp[i])
        target = (factor * chi[i])[:, None] * np.ones((1, reps))
        if carry is not None:
            target = carry + factor[:, None] * (chi[i] - (path @ carry) / (path @ factor))
        if hidden_population is not None:
            add_small(dt * hidden_population[1](t), target)
            assert target.min() > 0
        post = target.copy()
        groups = []
        mergers = evap = deposits = 0
        for ms, idx in zip(masses, locations):
            order = np.argsort(idx, kind="stable")
            groups.append(np.split(ms[order], np.cumsum(np.bincount(idx, minlength=b))[:-1]))
        mult = [1.0, 1.0]
        if pair_schedule is not None:
            mult = pair_schedule(t)
        if matrix_schedule is not None:
            stepper.set_multiplier(matrix_schedule(t, d))
        new_mass = [[] for _ in range(reps)]
        new_location = [[] for _ in range(reps)]
        for radial in range(nr):
            cells = np.arange(radial, b, nr)
            length = len(cells) * reps
            pack = repack(
                [groups[k][j] for j in cells for k in range(reps)], np.repeat(vols[cells], reps)
            )
            T = temp[i, radial]
            state = par.Aerosol(
                particles=pack,
                gas=GAS.GasData(
                    name=["water"],
                    molar_mass=np.array([MW]),
                    concentration=target[cells].reshape(-1, 1).copy(),
                    partitioning=np.array([True]),
                ),
                environment=GAS.EnvironmentData(
                    temperature=np.full(length, T),
                    pressure=np.full(length, p),
                    saturation_ratio=np.ones((length, 1)),
                ),
            )
            n0, l0, g0 = inspect_state(state)
            final = apply(condensation(T, p), state, dt, 1, 12400 + i * nr + radial)
            n1, l1, g1 = inspect_state(final)
            evap += int((n0 - n1).sum())
            eps_water = max(eps_water, float(np.max(abs(l1 + g1 - l0 - g0) / (l0 + g0))))
            post[cells] = g1.reshape(-1, reps)
            for ri, j in enumerate(cells):
                for k in range(reps):
                    q = ri * reps + k
                    mass = final.particles.masses[q, :, 0]
                    alive = mass[(mass > 0) & (final.particles.concentration[q] > 0)]
                    after = stepper(alive, vols[j], dt, *mult, 70361 + i * 10000000 + j * 100 + k)
                    eps_coag = max(
                        eps_coag, abs(after.sum() - alive.sum()) / max(alive.sum(), 1e-30)
                    )
                    mergers += len(alive) - len(after)
                    new_mass[k].append(after)
                    new_location[k].append(np.full(len(after), j))
        carry = (M @ (post * grid.v[:, None])) / grid.v[:, None]
        for k in range(reps):
            ms = np.concatenate(new_mass[k])
            idx = np.concatenate(new_location[k])
            gen = rng[k]
            diam = 1e6 * np.cbrt(6 * ms / (np.pi * RHO))
            jj = diam.astype(int)
            assert jj.max() < len(d)
            travel = dt * np.interp(diam, d, speed) / grid.dz
            down = np.floor(travel).astype(int) + (gen.random(len(ms)) < travel % 1)
            below = idx - down * nr * grid.na
            keep = (below >= 0) & (gen.random(len(ms)) < np.exp(-dt * loss[idx, jj]))
            deposits += int((~keep).sum())
            ms = ms[keep]
            below = below[keep]
            locations[k] = np.searchsorted(cdf, 2 * below + gen.random(len(ms)), side="right") % b
            masses[k] = ms
        current = np.stack([observation(ms, idx) for ms, idx in zip(masses, locations)])
        rows.append((prev + current) / 2)
        prev = current
        eventlog.append([evap, mergers, deposits])
    assert (
        start_count + added - sum(map(len, masses)) == np.sum(eventlog)
        and eps_water < 1e-10
        and eps_coag < 1e-10
    )
    folder = Path(output) if output is not None else HERE / "resolved"
    folder.mkdir(parents=True, exist_ok=True)
    key = f"{run}-{flow}-{kernel}"
    if mesh != "576" or n != 65536:
        key += f"-{mesh}cells-{n}particles"
    np.savez_compressed(
        folder / (key + ".npz"),
        t=times,
        d=d,
        spectra=np.array(rows).astype("float32"),
        events=eventlog,
    )
    meta = dict(
        run=run,
        flow=flow,
        kernel=kernel,
        n=n,
        reps=reps,
        cells=b,
        dt=dt,
        seed=70361,
        initial=start_count,
        water_error=eps_water,
        coag_error=eps_coag,
        seconds=time.monotonic() - tick,
        device=wp.get_device("cuda:0").name,
    )
    if hidden_population is not None:
        meta.update(initial_hidden=initial_extra, entries=added, vapor_debit_kg=float(debit_kg))
    (folder / (key + ".json")).write_text(json.dumps(meta, indent=2))
    print(meta, flush=True)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("run")
    a.add_argument("flow")
    a.add_argument("kernel")
    a.add_argument("--mesh", choices=MESHES, default="576")
    a.add_argument("--n", type=int, default=65536)
    q = a.parse_args()
    simulate(**vars(q))
