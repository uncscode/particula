"""Check activity gradients, electroneutral salt inventories and phase-grid bounds.

Run against the pinned package with OPENBLAS_NUM_THREADS=1, as in the study.
The archived coarse-grid winner is pinned below for an independent replay.
"""

from itertools import product

import numpy as np
from scipy.optimize import brentq

from stability import grid, onset
from trial import MW_WATER, feed_ratio, prepare, states

CHARGE = {202: 1, 204: 1, 261: -2, 242: -1, 301: -1}


def finite_difference(salt, model, step=2e-5):
    # Coordinates are water mol/charge equivalent and Na and OS charge fractions.
    maximum = np.zeros(5)
    sodium_fractions = [0.13, 0.63] if salt == "AS" else [1.0]
    for m, u, v, temp in product(
        (0.09, 1.2, 6, 15, 50), sodium_fractions, (0.07, 0.41, 0.88), (275.15, 298.15)
    ):
        w = 1 / (MW_WATER * m)
        x = np.array([w, u, v])
        n, mu = states(model, salt, *x, temp)
        n, mu = n[0], mu[0]
        if salt == "AS":
            amounts = [u * v, (1 - u) * (1 - v) / 2, u * (1 - v) / 2, (1 - u) * v]
        else:
            amounts = [v, 1 - v]
        mass = np.r_[w, amounts] * np.r_[model.neutral_mw, model.salt_mw]
        native = model.evaluate(mass, temp)
        charge = abs(n[1:] @ [CHARGE[i] for i in model.ions])
        inventory = np.max(abs(np.array(amounts) @ model.stoich - n[1:]))
        molality = np.max(abs(native.ion_molalities - n[1:] * m))
        for i in [0, 1, 2] if salt == "AS" else [0, 2]:
            h = np.eye(3)[i] * step * (w if i == 0 else 1)
            a, ma = states(model, salt, *(x + h), temp)
            b, mb = states(model, salt, *(x - h), temp)
            # Parameter increments are relative in water, absolute in u and v.
            dmu = (ma[0] - mb[0]) / (2 * step)
            dn = (a[0] - b[0]) / (2 * step)
            dG = (a[0] @ ma[0] - b[0] @ mb[0]) / (2 * step)
            gradient = abs(dG - dn @ mu) / (1 + abs(dn @ mu))
            maximum = np.maximum(
                maximum, [charge, inventory, molality, abs(n @ dmu), gradient]
            )
    return maximum


def affine_check(tail, sodium, delta):
    models = [prepare("AS", tail, sodium, d) for d in (0.0, 1.0, delta)]
    # Include zero-component limits as used on the grid edges.
    w, u, v = np.meshgrid(
        [1.5, 12, 600], [0.0, 0.31, 0.65, 1.0], [0.0, 0.12, 0.77, 1.0]
    )
    potentials = [states(p, "AS", w, u, v, 275.15)[1] for p in models]
    a, b, direct = potentials
    return np.max(abs(direct - (a + delta * (b - a))))


def main():
    # Original coarse winner. Keep it fixed if another candidate wins after refinement.
    tail = 1.4500000000000002
    sodium = (0.0513126940332701, 1.028310161453074, 0.01)
    nh4_delta = 0.34
    print("Candidate tail, sodium parameters, NH4 offset:", tail, sodium, nh4_delta)
    print(
        "Finite differences: charge, salt inventory, molality, Gibbs–Duhem, relative G gradient"
    )
    for salt in ("NaCl", "AS"):
        model = prepare(salt, tail, sodium, nh4_delta)
        print(salt, model.ions, finite_difference(salt, model))
    print(
        "Maximum error of interpolated NH4-offset chemical potentials:",
        affine_check(tail, sodium, nh4_delta),
    )
    # For the archived coarse winner, verify an explicit continuous AS90 trial.
    # These compositions were found by minimizing TPD with independent u and v.
    model = prepare("AS", tail, sodium, nh4_delta)
    f = feed_ratio("AS", 90)
    for rh, m_trial, u, v in (
        (95.8, 2.784576, 0.629200, 0.461974),
        (95.5, 2.931888, 0.622274, 0.409059),
    ):

        def water_residual(m):
            _, mu = states(model, "AS", 1 / (MW_WATER * m), f, f, 275.15)
            return 100 * np.exp(mu[0, 0]) - rh

        m = brentq(water_residual, 0.04, 18.0)
        _, feed_mu = states(model, "AS", 1 / (MW_WATER * m), f, f, 275.15)
        n, mu = states(model, "AS", 1 / (MW_WATER * m_trial), u, v, 275.15)
        print(
            "AS90 explicit witness: RH, feed m, TPD", rh, m, n[0] @ (mu[0] - feed_mu[0])
        )
    print("salt ovf mesh maxmol -> onset_RH, witness_molality, first_feed_hit")
    for salt, ovf in (("AS", 50), ("AS", 70), ("AS", 90), ("NaCl", 50), ("NaCl", 70)):
        model = prepare(salt, tail, sodium, nh4_delta)
        for refine, cap in ((1, 80), (1, 160), (1, 320), (2, 160), (2, 320)):
            result = onset(grid(salt, ovf, 275.15, model, refine=refine, maxmol=cap))
            print(
                salt,
                ovf,
                refine,
                cap,
                *(round(float(v), 6) for v in result[:2]),
                bool(result[2]),
            )


if __name__ == "__main__":
    main()
