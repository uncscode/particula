"""Driven neutral monomer and dimer chemistry with absorbing growth to trimers.

A and AB share the measured monomer pool, ammonia is buffered. Retain all three
dimer compositions. Trimer return and cluster–cluster collisions above dimers
are omitted; this controlled truncation is scored against published dimers.
"""

import csv
import itertools
import numpy as np
from scipy.integrate import solve_ivp
import particula as par
from reference import AIR, ACID, NH3, RH, T, ROOT, OUT, dump_csv


def calculation(temp, rh, acid, base, chemical, hamaker, wall_coeff=0.001):
    masses = (
        np.array([98.079, 17.03052, 115.10952, 196.158, 213.18852, 230.21904])
        * 1.66053906660e-27
    )
    d = (6 * masses / (np.pi * 1615)) ** (1 / 3) * 1e9
    particles = par.particles.ParticleData(
        masses=masses[None, :, None],
        density=np.array([1615.0]),
        charge=np.zeros((1, 6)),
        volume=np.ones(1),
        concentration=np.ones((1, 6)),
        distribution_type="discrete",
    )
    kernel = par.dynamics.CoagulationFactory().get_strategy(
        "van_der_waals",
        dict(
            hamaker_constant=hamaker,
            distribution_type="discrete",
            relative_permittivity=1.0,
        ),
    )
    k = kernel.kernel(particles, temp, 1.005 * 101325) * 1e6
    thermal = 1.987204258 * temp
    std = 101325 / (1.380649e-23 * temp) / 1e6
    hs = [(16.7, 29.8), (27.8, 43.1), (19.3, 34.7)]
    if chemical == "posterior_median":
        hs = [(12.8, 30.0), (29.1, 42.9), (21.1, 34.2)]
    if chemical == "2015_adduct":
        hs[0] = (16.1, 26.4)
    ke = [
        pair * std * np.exp((-1000 * h + temp * s) / thermal)
        for (h, s), pair in zip(hs, [k[0, 1], k[0, 2], k[1, 4]])
    ]
    e0 = (
        0.5
        * k[0, 0]
        / (1.380649e-23 * temp * 1e6 * np.exp(10109 / temp - 35.03))
        * (20 / rh) ** 0.5
    )

    def reaction(_, y):
        ab, x, y1, y2 = y
        a = acid - ab
        aa = 0.5 * k[0, 0] * a * a
        aab = k[0, 2] * a * ab
        abab = 0.5 * k[2, 2] * ab * ab
        assoc = [k[3, 1] * base * x, k[4, 1] * base * y1]
        g = (k[3:, 0] * a + k[3:, 2] * ab) * y[1:]
        # Each AB used in a reaction lowers the AB inventory. Free A is
        # replenished externally to maintain the imposed total monomer.
        dab = (
            k[0, 1] * a * base
            - ke[0] * ab
            - aab
            - 2 * abab
            - ab * np.dot(k[3:, 2], y[1:])
            + ke[1] * y1
            - (wall_coeff / d[2] + 9.6e-5) * ab
        )
        dy = (
            np.array(
                [
                    aa - e0 * x - assoc[0],
                    aab + assoc[0] - assoc[1] - ke[1] * y1 + ke[2] * y2,
                    abab + assoc[1] - ke[2] * y2,
                ]
            )
            - (wall_coeff / d[3:] + 9.6e-5) * y[1:]
            - g
        )
        return np.r_[dab, dy]

    def integrate(tol):
        sol = solve_ivp(
            reaction,
            (0.0, 21600.0),
            np.zeros(4),
            method="BDF",
            rtol=tol,
            atol=tol * 0.01,
        )
        if not sol.success:
            raise RuntimeError(sol.message)
        last = sol.y[:, -1]
        if (
            min(last) < 0
            or last[0] > acid
            or np.max(abs(reaction(0, last)) / np.maximum(1, last)) > 1e-6
        ):
            raise ValueError("No steady state")
        return last

    y = integrate(1e-7)
    return dict(
        dimer_cm3=sum(y[1:]),
        adduct_fraction=y[0] / acid,
        bare_dimer_fraction=y[1] / sum(y[1:]),
        adduct_evap_s=ke[0],
        binary_evap_s=e0,
    )


if __name__ == "__main__":
    rows = []
    cases = []
    with (ROOT / "rate_sources/kurten_fig9_248K_dimers.csv").open() as f:
        for a in csv.reader(f):
            if a[0] == "temperature_K":
                continue
            temp, rh, nh, acid, obs = map(float, a[:5])
            cases.append(
                (
                    "248K_dimer",
                    temp,
                    rh,
                    acid,
                    nh * 1.005 * 101325 / (1.380649e-23 * temp) * 1e-18,
                    obs,
                )
            )
    cases.append(("265K_external_scenario", 265.0, 10.0, 1.9e9, 3.8e9, 1.1e7))
    for line in [332, 334, 336, 338, 344, 352, 366]:
        i = line - 2
        cases.append(
            (
                f"holdout_context_{line}",
                T[i],
                RH[i],
                ACID[i] * 1e6,
                NH3[i] * AIR[i] * 1e-12,
                None,
            )
        )
    for name, temp, rh, acid, base, obs in cases:
        for chemical, hamaker in itertools.product(
            ["optimized", "posterior_median", "2015_adduct"], [0.0, 6.4e-20]
        ):
            r = calculation(temp, rh, acid, base, chemical, hamaker)
            rows.append(
                dict(
                    case=name,
                    T_K=temp,
                    RH_pct=rh,
                    acid_cm3=acid,
                    ammonia_cm3=base,
                    observed_dimer_cm3=obs,
                    thermo=chemical,
                    hamaker_J=hamaker,
                    **r,
                    ratio=r["dimer_cm3"] / obs if obs else None,
                )
            )
    dump_csv(OUT / "dimer_scores.csv", rows)
