"""Finite time, interacting-particle CLOUD box. Explicit Poisson births and mergers.

Each sampled particle represents one physical particle in an adjustable sampling
volume, the chamber remains well mixed. Gas concentrations are buffered by an
accounted external supply, and the sampled population has no initial background.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import numpy as np
import particula as par
from reference import ACID, AIR, J, NH3, OUT, T, dump_csv

AMU = 1.66053906660e-27
RHO = 1500.0
MA = 116.09428
MB = 17.03052


def diameter(a, b, mass_a=MA):
    return (6 * (a * mass_a + b * MB) * AMU / (np.pi * RHO)) ** (1 / 3) * 1e9


def source(line, label):
    if label in ("observed", "equal_total"):
        return J[line - 2]
    split = "248_low" if line in [334, 336, 338, 344] else "223_low"
    job = json.loads(
        (
            OUT / "fits" / f"{split}_{label}_linear_nominal_neutral_group.json"
        ).read_text()
    )
    r = min(job["runs"], key=lambda a: a["cost"])
    return r["prediction"][job["test_lines"].index(line)]


def transport(d, temp, molar, diff):
    speed = np.sqrt(8 * 8.314462618 * temp / (np.pi * molar))
    r = d * 0.5e-9
    return par.dynamics.get_first_order_mass_transport_k(
        r,
        par.particles.get_vapor_transition_correction(
            3 * diff / (speed * r), 1.0
        ),
        diff,
    )


def simulate(spec, conditions=None):
    """Replay one sampled box. Optional tables are provided on the native grid."""
    line, label, collision, step, resolution, seed = spec
    i = line - 2
    temp = T[i]
    rate = source(line, label) if conditions is None else conditions["J17"]
    # One hour of expected source supplies 'resolution' computational particles.
    vol_cm3 = resolution / (3600 * rate)
    rngs = [
        np.random.default_rng(x) for x in np.random.SeedSequence(seed).spawn(4)
    ]
    born, removed_wall, removed_dil, mergers = 0, 0, 0, 0
    supplied = np.zeros(2)
    outflow = np.zeros(2)
    uptake = np.zeros(2)
    a = np.empty(0)
    b = np.empty(0)
    dgrid = np.geomspace(1.39, 1000.0, 256)
    radius = dgrid * 0.5e-9
    K = (
        par.dynamics.get_brownian_kernel_via_system_state(
            radius, 4 * np.pi / 3 * radius**3 * RHO, temp, 1.005 * 101325
        )
        * 1e6
    )
    da_coeff = (
        transport(dgrid, temp, 0.098079, 8e-6 * (temp / 293.15) ** 1.75)
        * ACID[i]
        * 1.0e12
    )
    db_coeff = (
        transport(dgrid, temp, 0.01703052, 2.1e-5 * (temp / 293.15) ** 1.75)
        * NH3[i]
        * AIR[i]
        * 1.0e-6
    )
    mass_a = MA if conditions is None else conditions["acid_mass_amu"]
    dilution = 9.6e-5 if conditions is None else conditions["dilution"]
    if conditions is not None:
        da_coeff = conditions["acid_uptake_s"]
        db_coeff *= conditions["base_scale"]
    removed_bg = 0
    seed_a = (np.pi / 6 * (1.4e-9) ** 3 * RHO / AMU) / (mass_a + 2 * MB)
    edges = np.array([2.9, 9.7])
    flux = np.zeros((3, 2))
    mergeflux = np.zeros_like(flux)
    occ = np.zeros_like(flux)
    # Three 6 h windows allow detection of startup and residual drift.
    duration = 18 * 3600
    for epoch in range(0, duration, step):
        w = epoch // 21600
        nbirth = rngs[0].poisson(rate * vol_cm3 * step)
        if nbirth:
            a = np.r_[a, np.full(nbirth, seed_a)]
            b = np.r_[b, np.full(nbirth, 2 * seed_a)]
            supplied += nbirth * seed_a * np.array([1.0, 2.0])
            born += nbirth
        old = diameter(a, b, mass_a)
        if len(a) == 0:
            continue
        da = np.interp(old, dgrid, da_coeff) * step
        db = np.minimum(
            2 * (a + da) - b, np.interp(old, dgrid, db_coeff) * step
        )
        a += da
        b += db
        uptake += [da.sum(), db.sum()]
        new = diameter(a, b, mass_a)
        flux[w] += np.sum(
            (old[:, None] < edges) & (new[:, None] >= edges), axis=0
        )
        midpoint = (old + new) / 2
        if conditions is None:
            wall = 0.001 / midpoint
            bg = 0.0
        else:
            wall = np.interp(midpoint, dgrid, conditions["wall_hz"])
            bg = np.interp(midpoint, dgrid, conditions["background_hz"])
        dil = dilution
        u = rngs[1].random(len(a))
        hazard = wall + dil + bg
        tot = -np.expm1(-hazard * step)
        lost = u < tot
        nw = np.count_nonzero(u < tot * wall / hazard)
        nw_d = np.count_nonzero(u < tot * (wall + dil) / hazard)
        removed_wall += nw
        removed_dil += nw_d - nw
        removed_bg += np.sum(lost) - nw_d
        outflow += [a[lost].sum(), b[lost].sum()]
        a, b, new = a[~lost], b[~lost], new[~lost]
        if collision and len(a) > 1:
            # Exact finite-time pair selection for a frozen population, with rates
            # updated after each accepted collision; binary thinning in cm^3/s.
            # Index conservatively across the live radius interval.
            idx = np.searchsorted(dgrid, new).clip(1, len(dgrid) - 1)
            cap = K[
                idx.min() - 1 : idx.max() + 1, idx.min() - 1 : idx.max() + 1
            ].max()
            remaining = float(step)
            while len(a) > 1:
                hazard = len(a) * (len(a) - 1) * 0.5 * cap / vol_cm3
                remaining -= rngs[2].exponential() / hazard
                if remaining < 0:
                    break
                p = rngs[2].integers(len(a))
                q = rngs[2].integers(len(a) - 1)
                if q >= p:
                    q += 1
                dp, dq = new[p], new[q]
                ip, iq = idx[p], idx[q]
                wp = (dp - dgrid[ip - 1]) / (dgrid[ip] - dgrid[ip - 1])
                wq = (dq - dgrid[iq - 1]) / (dgrid[iq] - dgrid[iq - 1])
                kval = (1 - wp) * (
                    (1 - wq) * K[ip - 1, iq - 1] + wq * K[ip - 1, iq]
                ) + wp * ((1 - wq) * K[ip, iq - 1] + wq * K[ip, iq])
                if kval > cap * (1 + 1e-12):
                    raise ValueError("collision upper bound")
                if rngs[2].random() * cap >= kval:
                    continue
                a[p] += a[q]
                b[p] += b[q]
                dm = diameter(a[p], b[p], mass_a)
                crossed = (max(dp, dq) < edges) & (dm >= edges)
                flux[w] += crossed
                mergeflux[w] += crossed
                new[p] = dm
                a = np.delete(a, q)
                b = np.delete(b, q)
                new = np.delete(new, q)
                idx = np.searchsorted(dgrid, new).clip(1, len(dgrid) - 1)
                cap = K[
                    idx.min() - 1 : idx.max() + 1, idx.min() - 1 : idx.max() + 1
                ].max()
                mergers += 1
        occ[w] += np.sum(new[:, None] >= edges, axis=0) * step / vol_cm3
        if len(new) and new.max() > dgrid[-1]:
            raise ValueError("size range exceeded")
    budget = np.array([a.sum(), b.sum()]) + outflow - supplied - uptake
    err = float(np.max(abs(budget) / np.maximum(supplied + uptake, 1)))
    assert (
        err < 1e-10
        and len(a) == born - removed_wall - removed_dil - removed_bg - mergers
    )
    out = []
    for w in range(3):
        out.append(
            dict(
                line=line,
                source=label,
                J17=rate,
                coag=int(collision),
                dt_s=step,
                particles_per_source_hour=resolution,
                seed=seed,
                window_end_h=6 * (w + 1),
                sample_volume_cm3=vol_cm3,
                J32=flux[w, 0] / vol_cm3 / 21600,
                J10=flux[w, 1] / vol_cm3 / 21600,
                merger_J32=mergeflux[w, 0] / vol_cm3 / 21600,
                merger_J10=mergeflux[w, 1] / vol_cm3 / 21600,
                mean_N32_cm3=occ[w, 0] / 21600,
                mean_N10_cm3=occ[w, 1] / 21600,
                n_birth=born,
                n_merge=mergers,
                n_wall=removed_wall,
                n_dil=removed_dil,
                n_background=removed_bg,
                acid_molecules_supplied=supplied[0] + uptake[0],
                NH3_molecules_supplied=supplied[1] + uptake[1],
                relative_species_error=err,
            )
        )
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    cases = (
        [(336, "observed", True, 30, 256, 122)]
        if args.pilot
        else [
            (line, src, coal, dt, res, seed)
            for line, src in [
                (336, "observed"),
                (336, "full"),
                (336, "assisted"),
                (336, "equal_total"),
                (332, "observed"),
                (332, "full"),
                (332, "assisted"),
                (352, "observed"),
            ]
            for coal in [False, True]
            for dt, res in [(30, 512), (15, 512), (30, 1024)]
            for seed in [122, 123, 124, 125]
        ]
    )
    output = OUT / ("population_pilot.csv" if args.pilot else "population.csv")
    collected = []
    with ProcessPoolExecutor(8) as pool:
        for rows in pool.map(simulate, cases, chunksize=1):
            collected.extend(rows)
            if len(collected) % 36 == 0:
                dump_csv(output, collected)
    dump_csv(output, collected)
    print("Finished population simulations", len(cases))
