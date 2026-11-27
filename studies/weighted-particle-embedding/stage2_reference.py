"""Integrate particles, gas and collision hazard in continuous time."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.integrate import solve_ivp
from sparse_vapor_reference import ExplicitReservoir
from stage2_population import (
    collision_kernel,
    decoded,
    fixture,
    merge,
    observable_names,
    observables,
)


@dataclass
class Clock:
    """Carry the residual hazard and random stream."""

    rng: np.random.Generator
    remaining: float
    collisions: int = 0

    @classmethod
    def seeded(cls, seed):
        """Initialize the waiting hazard from an independent seed."""
        rng = np.random.default_rng(seed)
        return cls(rng, rng.exponential())

    def event(self, box, i, j, rates):
        """Choose a collision pair and draw the next hazard threshold."""
        cdf = np.cumsum(rates)
        index = np.searchsorted(cdf, self.rng.random() * cdf[-1], side="right")
        merge(box, i[index], j[index])
        self.remaining = self.rng.exponential()
        self.collisions += 1
        return int(i[index]), int(j[index])


def collisions(box, clock, interval, kernel):
    """Exact event sampling with fixed vapor loadings between mergers."""
    elapsed = 0.0
    while len(box.mass) > 1 and elapsed < interval:
        i, j = np.triu_indices(len(box.mass), 1)
        rates = kernel(box, i, j) / box.volume
        hazard = rates.sum()
        if hazard == 0.0 or (interval - elapsed) * hazard < clock.remaining:
            clock.remaining -= (interval - elapsed) * hazard
            break
        elapsed += clock.remaining / hazard
        clock.event(box, i, j, rates)


def collision_observations(box, clock, times, kind, observe, kernel):
    """Sample mergers at each requested time with fixed vapor masses."""
    history = []
    for dt in np.diff(times, prepend=0.0):
        if kind and len(box.mass):
            collisions(
                box,
                clock,
                dt,
                lambda b, i, j: kernel(b.mass, b.charge, b.spec, kind, i, j),
            )
        history.append(observe(box))
    return np.array(history), 0


def coupled_rates(
    t, y, model, use_collision, unpack, i, j, box, kind, s, kernel
):
    """Evaluate gas, particle loadings and integrated event hazard."""
    a = model.rates(t, y[:-1])
    if use_collision:
        m = unpack(y)
        # Tiny negative solver iterates can occur in depleted lanes.
        m[:, :s] = np.maximum(m[:, :s], 0)
        hazard = kernel(m, box.charge, box.spec, kind, i, j).sum() / box.volume
    else:
        hazard = 0.0
    return np.r_[a, hazard]


def update_clock(sol, box, clock, kind, i, j, use_collision, hazard, kernel):
    """Consume integrated hazard and commit any merger at the endpoint."""
    if sol.status == 1:
        rates = kernel(box.mass, box.charge, box.spec, kind, i, j) / box.volume
        clock.event(box, i, j, rates)
    elif use_collision:
        clock.remaining -= hazard


def advance(
    box,
    clock,
    times,
    tol,
    exchange=True,
    kind="brownian",
    method="LSODA",
    observe=observables,
    reservoir=ExplicitReservoir,
    kernel=collision_kernel,
):
    """Advance one closed interval and return its observations."""
    times = np.asarray(times)
    if times.ndim != 1 or np.any(np.diff(times) <= 0) or times[0] <= 0:
        raise ValueError("Observation times must increase and be positive")
    if not exchange or len(box.mass) == 0:
        return collision_observations(box, clock, times, kind, observe, kernel)
    out = []
    calls = 0
    s = box.spec.species
    elapsed, k = 0.0, 0
    while elapsed < times[-1]:
        n = len(box.mass)
        i, j = np.triu_indices(n, 1)
        model = reservoir(box.mass, np.ones(n), box.volume, box.spec)
        dry = model.local.dry[:, None]
        fixed = box.mass[:, s:].copy()
        use_collision = bool(kind) and n > 1

        def unpack(x, n=n, dry=dry, fixed=fixed):
            return np.concatenate(
                (x[: n * s].reshape(n, s) * dry, fixed), axis=1
            )

        def rhs(
            t, y, model=model, active=use_collision, unpack=unpack, i=i, j=j
        ):
            return coupled_rates(
                t, y, model, active, unpack, i, j, box, kind, s, kernel
            )

        rhs = getattr(model, "joint_rates", rhs)

        def crossing(_, y):
            return y[-1] - clock.remaining

        crossing.terminal = True
        crossing.direction = 1
        y0 = np.r_[(box.mass[:, :s] / dry).ravel(), box.gas / model.scale, 0.0]
        sol = solve_ivp(
            rhs,
            (elapsed, times[-1]),
            y0,
            rtol=tol,
            atol=tol * 1e-3,
            method=method,
            events=crossing if use_collision else None,
            dense_output=True,
        )
        if not sol.success:
            raise ArithmeticError(sol.message)
        calls += sol.nfev
        end = sol.t[-1]
        while k < len(times) and times[k] <= end:
            y = sol.sol(times[k])
            box.mass = unpack(y)
            box.gas = y[n * s : -1] * model.scale
            out.append(observe(box))
            k += 1
        y = sol.y[:, -1]
        box.mass = unpack(y)
        box.gas = y[n * s : -1] * model.scale
        if np.min(y[:-1], initial=0.0) < -tol * 0.03:
            raise FloatingPointError("Negative phase in coupled reference")
        update_clock(sol, box, clock, kind, i, j, use_collision, y[-1], kernel)
        elapsed = end
    return np.asarray(out), calls


def reference(
    box,
    seed,
    tol,
    exchange=True,
    kind="brownian",
    method="LSODA",
    observe=observables,
    reservoir=ExplicitReservoir,
    kernel=collision_kernel,
):
    """Integrate a complete forcing history with continuous collision times."""
    clock = Clock.seeded(seed)
    n0 = len(box.mass)
    q0 = box.charge.sum()
    expected = box.inventory().copy()
    original = expected.copy()
    starts = np.r_[0, np.flatnonzero(np.diff(box.t) == 0) + 1]
    reports = []
    drift = 0.0
    evaluations = 0
    for leg, (i, j) in enumerate(
        zip(starts, np.r_[starts[1:], len(box.t)], strict=True)
    ):
        old = box.gas.copy()
        box.gas = np.where(box.masks[leg], box.levels[leg], old)
        expected[: len(old)] += box.gas - old
        reports.append(observe(box)[None])
        rows, work = advance(
            box,
            clock,
            box.t[i + 1 : j] - box.t[i],
            tol,
            exchange,
            kind,
            method,
            observe,
            reservoir,
            kernel,
        )
        evaluations += work
        reports.append(rows)
        s = box.spec.species
        predicted = rows[:, 1 : 1 + s + 2].copy()
        predicted[:, :s] += rows[:, s + 3 : 2 * s + 3]
        drift = max(
            drift,
            float(
                np.max(
                    abs(predicted - expected) / np.maximum(abs(expected), 1e-11)
                )
            ),
        )
        if len(box.mass) != n0 - clock.collisions or box.charge.sum() != q0:
            raise ArithmeticError("Collision ledger did not close")
    if drift > 1e-8 or not np.isfinite(original).all():
        raise ArithmeticError(f"Coupled reference budget drift {drift}")
    return np.concatenate(reports), dict(
        evaluations=evaluations, events=clock.collisions, drift=drift
    )


def one(job):
    """Persist one reference trajectory and its numerical accounting."""
    (
        folder,
        family,
        mix,
        slots,
        density,
        process,
        z,
        seed,
        tol,
        rep,
        frames,
        legs,
        method,
    ) = job
    path = folder / (
        f"f{family}-m{mix}-n{slots}-d{density}-{process}-{z}-r{seed}"
        f"-{rep}-t{tol:.0e}-{method}-f{frames}-l{legs}.npz"
    )
    if path.exists():
        return
    box = decoded(
        fixture(family, mix, slots, density, process, z, frames, legs), rep
    )
    kind = (
        None
        if process in ("water", "organic", "both")
        else ("constant" if process == "constant" else "brownian")
    )
    if process in ("charge", "charge-only"):
        kind = "charged"
    start = perf_counter()
    obs, stats = reference(
        box,
        seed,
        tol,
        process not in ("coag", "constant", "charge-only"),
        kind,
        method,
    )
    np.savez_compressed(
        path,
        obs=obs,
        names=observable_names(box.spec),
        time=box.t,
        seconds=perf_counter() - start,
        **stats,
    )


def main(args):
    """Distribute independently identified reference cases across workers."""
    from itertools import product

    args.folder.mkdir(exist_ok=True, parents=True)
    jobs = (
        (args.folder, *x, args.frames, args.legs, args.method)
        for x in product(
            args.families,
            args.mix,
            args.slots,
            args.densities,
            args.process,
            args.charge,
            args.seeds,
            args.tol,
            args.rep,
        )
    )
    with ProcessPoolExecutor(args.workers) as pool:
        for k, _ in enumerate(pool.map(one, jobs, chunksize=1), 1):
            if k % 32 == 0:
                print(k, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    for name, default in (
        ("families", [101]),
        ("mix", [0, 1]),
        ("slots", [16]),
        ("densities", [1, 8]),
        ("seeds", [101]),
    ):
        p.add_argument("--" + name, nargs="+", type=int, default=default)
    p.add_argument("--tol", type=float, nargs="+", default=[1e-8, 1e-10])
    p.add_argument("--rep", nargs="+", default=["full"])
    p.add_argument("--charge", nargs="+", default=["zero"])
    p.add_argument(
        "--process",
        nargs="+",
        default=["water", "organic", "both", "coag", "coupled", "constant"],
    )
    p.add_argument("--method", default="LSODA")
    for k, v in (("workers", 4), ("frames", 8), ("legs", 4)):
        p.add_argument("--" + k, type=int, default=v)
    main(p.parse_args())
