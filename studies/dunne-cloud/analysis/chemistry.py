"""Conditional fate of a tagged acid–ammonia pair using Table 1 evaporation energies.

This tagged-cluster experiment asks a narrower question than SANTIAGO. Follow a
single cluster while it contains base, with buffered monomer baths. Reaching a
pentamer is success, losing the last base (or dissociating to monomers) is exit.
"""

from functools import lru_cache
import numpy as np
import particula as par
from reference import AIR, ACID, NH3, Q, RH, T, OUT, WINDOW, dump_csv

# Published magnitudes of association enthalpy (kcal/mol) and entropy (cal/mol/K);
# Steihaug and posterior marginal medians from Kürten 2019, Table 1.
THERMO = {
    (1, 1): (16.7, 29.8, 12.8, 30.0),
    (2, 1): (27.8, 43.1, 29.1, 42.9),
    (2, 2): (19.3, 34.7, 21.1, 34.2),
    (3, 1): (18.3, 37.6, 20.0, 37.2),
    (3, 2): (28.1, 38.0, 30.6, 38.3),
    (3, 3): (25.7, 37.6, 27.1, 37.9),
    (4, 1): (21.8, 43.2, 20.7, 44.2),
    (4, 2): (22.9, 39.6, 24.1, 39.9),
    (4, 3): (27.9, 41.1, 30.8, 40.3),
    (4, 4): (19.2, 28.7, 20.3, 29.0),
}
STATES = list(THERMO)
SPECIES = [(1, 0), (0, 1), *STATES]
MASSES = (
    np.array([a * 98.079 + b * 17.03052 for a, b in SPECIES])
    * 1.66053906660e-27
)
DENS = 1615.0
DIAM = (6 * MASSES / (np.pi * DENS)) ** (1 / 3)


@lru_cache(None)
def kernel(temp, hamaker):
    particles = par.particles.ParticleData(
        masses=MASSES[None, :, None],
        density=np.array([DENS]),
        concentration=np.ones((1, len(MASSES))),
        charge=np.zeros((1, len(MASSES))),
        volume=np.ones(1),
        distribution_type="discrete",
    )
    strategy = par.dynamics.CoagulationFactory().get_strategy(
        "van_der_waals",
        dict(
            distribution_type="discrete",
            hamaker_constant=hamaker,
            relative_permittivity=1.0,
        ),
    )
    return strategy.kernel(particles, temp, 101325 * 1.005) * 1e6


def solve(
    temp, acid, base, thermo, hamaker, perturb=None, loss=True, evaporate=True
):
    k = kernel(temp, hamaker)
    G = np.zeros((len(STATES), len(STATES)))
    exits = np.zeros((len(STATES), 3))  # pentamer, monomer/base exit, chamber
    for i, (a, b) in enumerate(STATES):

        def jump(target, rate):
            if target in STATES:
                G[i, STATES.index(target)] += rate
            elif target[0] == 5:
                exits[i, 0] += rate
            else:
                exits[i, 1] += rate

        jump((a + 1, b), k[2 + i, 0] * acid)
        if b < a:
            jump((a, b + 1), k[2 + i, 1] * base)
        if evaporate:
            p = (a, b - 1) if a == b else (a - 1, b)
            released = 1 if a == b else 0
            h, s = THERMO[a, b][2 * thermo : 2 * thermo + 2]
            dg = -1000 * h + temp * s
            if perturb is not None and perturb[0] == (a, b):
                dg += perturb[1] * 1000
            standard = 101325 / (1.380649e-23 * temp) * 1e-6
            evaporation = (
                k[SPECIES.index(p), released]
                * standard
                * np.exp(dg / (1.987204258 * temp))
            )
            jump(p, evaporation)
        if loss:
            exits[i, 2] = 0.001 / (DIAM[2 + i] * 1e9) + 9.6e-5
        G[i, i] = -G[i].sum() - exits[i].sum()
    probabilities = np.linalg.solve(-G, exits)
    if (
        np.max(abs(probabilities.sum(axis=1) - 1)) > 2e-8
        or probabilities.min() < -1e-12
    ):
        raise ValueError("absorbing probability balance failed")
    return probabilities[0], np.linalg.solve(-G, np.ones(len(STATES)))[0]


if __name__ == "__main__":
    records = []
    sens = []
    for i in np.where(WINDOW & (Q == 0))[0]:
        args = T[i], ACID[i] * 1e6, NH3[i] * AIR[i] * 1e-12
        for label, model in [("optimized", 0), ("median", 1)]:
            for hamaker in [0.0, 6.4e-20]:
                p, tau = solve(*args, model, hamaker)
                records.append(
                    dict(
                        line=i + 2,
                        thermo=label,
                        hamaker_J=hamaker,
                        p_pentamer=p[0],
                        p_mon_exit=p[1],
                        p_chamber_loss=p[2],
                        mean_exit_s=tau,
                    )
                )
        if i + 2 in [332, 334, 336, 338, 344, 352, 366]:
            for cluster in STATES:
                pp, _ = solve(*args, 0, 6.4e-20, (cluster, 0.01))
                pm, _ = solve(*args, 0, 6.4e-20, (cluster, -0.01))
                sens.append(
                    dict(
                        line=i + 2,
                        cluster=str(cluster),
                        dlogP_per_kcal_mol=float(np.log(pp[0] / pm[0]) / 0.02),
                    )
                )
    oracle, _ = solve(
        248.3, 1.8e7, 1e7, 0, 6.4e-20, loss=False, evaporate=False
    )
    assert abs(oracle[0] - 1) < 1e-12
    dump_csv(OUT / "chemistry.csv", records)
    dump_csv(OUT / "cluster_sensitivity.csv", sens)
    print(
        "checked absorbing-state balances and irreversible limit", len(records)
    )
