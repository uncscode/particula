"""Independent seeded A→V, condensed V→P benchmark for inverse chemistry."""

import numpy as np
from particula.dynamics.chemistry.discovery import compile_network
from particula.execution.inverse_chemistry import SeededChamber, VaporExchange
from particula.particles.properties.vapor_correction_module import (
    get_vapor_transition_correction,
)
from scipy.integrate import solve_ivp
from scipy.special import ndtr, ndtri


def chamber():
    """Build a synthetic gas-phase precursor and a reversible vapor pair."""
    names = ("gas:A", "gas:V", "particle:V", "particle:P", "particle:seed")
    mechanism = compile_network(
        ((-1, 1, 0, 0, 0), (0, 0, -1, 1, 0)),
        {
            n: {"molar_mass_kg_per_mol": 0.1, "elements": {"X": 1}}
            for n in names
        },
        [0.5, 0.2],
    )
    return SeededChamber(
        mechanism,
        {n: 1000.0 for n in names[2:]},
        {"particle:seed": 0.95, "particle:V": 0.05},
        (VaporExchange("gas:V", "particle:V", 1e-5, 6e-8),),
        ("particle:V", "particle:P"),
        ("gas:A", "gas:V"),
        (80e-9, 120e-9),
        0.12,
        3,
    )


def reference(times, *, dose, rates, saturation, seed):
    """Use independent per-particle amounts and suspended count coordinates."""
    n, median, gsd, wall = seed
    diam0 = median * gsd ** ndtri((np.arange(3) + 0.5) / 3)
    # Each mole occupies 1e-4 m³ for all synthetic materials.
    amount0 = np.pi / 6 * diam0**3 / 1e-4
    core = 0.95 * amount0
    # Particle coordinates are fractions of each cohort's original moles.
    initial = np.r_[
        np.array(dose) / 1e-7, np.full(3, 0.05), np.zeros(3), np.ones(3)
    ]

    def properties(y):
        fresh, aged, weight = y[2:5], y[5:8], y[8:]
        total = core / amount0 + fresh + aged
        diam = diam0 * np.cbrt(total)
        return fresh, aged, weight, total, diam

    def rhs(t, y):
        fresh, _, weight, total, diam = properties(y)
        radius = diam / 2
        beta = get_vapor_transition_correction(6e-8 / radius, 1.0)
        uptake = (
            4
            * np.pi
            * radius
            * 1e-5
            * beta
            * (1e-7 * y[1] - saturation[0] * fresh / total)
        )
        decay = rates[0] * y[0]
        return np.r_[
            -decay,
            decay - n / 3 * weight @ uptake / 1e-7,
            uptake / amount0 - rates[1] * fresh,
            rates[1] * fresh,
            -wall * 1e-7 / diam * weight,
        ]

    solution = solve_ivp(
        rhs,
        (0, max(times)),
        initial,
        t_eval=times,
        method="Radau",
        rtol=1e-10,
        atol=1e-12,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    predicted = []
    hard = []
    for state in solution.y.T:
        fresh, aged, weight, _, diam = properties(state)
        counts = n / 3 * weight
        cdf = ndtr(np.log(np.array([[80e-9], [120e-9]]) / diam) / 0.12)
        fractions = np.diff(np.vstack((np.zeros(3), cdf, np.ones(3))), axis=0)
        predicted.append(
            np.r_[
                state[:2] * 1e-8,
                counts @ ((fresh + aged) * amount0) * 0.1,
                fractions @ counts,
            ]
        )
        hard.append(
            np.histogram(diam, [0, 80e-9, 120e-9, np.inf], weights=counts)[0]
        )
    return np.array(predicted), np.array(hard)
