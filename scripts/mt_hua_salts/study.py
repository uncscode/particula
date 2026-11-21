"""Paired WSIM growth at 90% RH for the Mt. Hua filter extracts."""

import csv
import itertools
from pathlib import Path

import numpy as np
import warp as wp
from particula.gpu.dynamics.condensation_funcs import water_activity_kappa_wp
from particula.gpu.properties.particle_properties import (
    kelvin_radius_wp,
    kelvin_term_wp,
)
from scipy.optimize import brentq, linprog

# Rows NH4, Na, K, Ca, Mg; columns SO4, NO3, Cl. Liu et al. (2014), Table 2.
KAPPA = np.array(
    [
        [0.52, 0.68, 0.93],
        [0.76, 0.87, 1.24],
        [0.55, 0.53, 0.89],
        [0.0, 1.10, 0.78],
        [0.24, 0.63, 0.98],
    ]
)
# Nominal dry anhydrous crystal densities in kg/m3.
DENSITY = np.array(
    [
        [1770, 1725, 1530],
        [2664, 2260, 2165],
        [2660, 2110, 1984],
        [2960, 2500, 2150],
        [2660, 2320, 2320],
    ]
)
CATION_MASS = np.array([18.039, 22.990, 39.098, 40.078, 24.305])
ANION_MASS = np.array([96.06, 62.005, 35.45])
CATION_EQ = CATION_MASS / [1, 1, 1, 2, 2]
ANION_EQ = ANION_MASS / [2, 1, 1]
MASS_PER_EQ = CATION_EQ[:, None] + ANION_EQ
WATER = 997.0
DRY_RADIUS = 50e-9
TEMPERATURE = 298.15
RH = 0.90
A_RADIUS = 2 * 0.072 * 0.01801528 / (8.314462618 * TEMPERATURE * WATER)


def cpu_growth(kappa):
    """Solve the stable subsaturated root, including the insoluble limit."""
    if kappa == 0:
        return 1.0

    def residual(growth):
        water = growth**3 - 1
        return (
            water / (water + kappa) * np.exp(A_RADIUS / (DRY_RADIUS * growth))
            - RH
        )

    return brentq(residual, 1.0, 4.0, xtol=1e-14)


def pairings(cations, anions):
    """Return proportional, greedy, and optimized neutral salt allocations."""
    total = min(cations.sum(), anions.sum())
    if total <= 0:
        raise ValueError("No measured counterions for neutral reconstruction")
    pos = cations * (total / cations.sum())
    neg = anions * (total / anions.sum())
    cat_orders = list(itertools.permutations(range(5)))
    an_orders = list(itertools.permutations(range(3)))
    cat = np.repeat(cat_orders, len(an_orders), axis=0)
    an = np.tile(an_orders, (len(cat_orders), 1))
    case = np.arange(len(cat))
    left = np.tile(pos, (len(cat), 1))
    right = np.tile(neg, (len(cat), 1))
    allocation = np.zeros((len(cat), 5, 3))
    for i in range(5):
        for j in range(3):
            c, a = cat[:, i], an[:, j]
            used = np.minimum(left[case, c], right[case, a])
            allocation[case, c, a] = used
            left[case, c] -= used
            right[case, a] -= used
    proportional = np.outer(pos, neg) / total
    # Charnes-Cooper transform of min/max sum(kappa*V)/sum(V).
    # Unit dry volume fixes scale. The final variable restores ion budgets.
    weights = (MASS_PER_EQ / DENSITY).ravel()
    incidence = np.vstack(
        (np.repeat(np.eye(5), 3, axis=1), np.tile(np.eye(3), (1, 5)))
    )
    constraints = np.column_stack((incidence, -np.r_[pos, neg] / total))
    constraints = np.vstack((constraints, np.r_[weights, 0.0]))
    extremes = []
    for direction in (1.0, -1.0):
        fit = linprog(
            direction * np.r_[weights * KAPPA.ravel(), 0.0],
            A_eq=constraints,
            b_eq=np.r_[np.zeros(8), 1.0],
            bounds=(0, None),
        )
        if not fit.success:
            raise RuntimeError(fit.message)
        extremes.append(fit.x[:15].reshape(5, 3) * total / fit.x[-1])
    eq = np.concatenate(([proportional], allocation, extremes))
    np.testing.assert_allclose(
        eq.sum(axis=2), np.broadcast_to(pos, eq.shape[:2]), atol=1e-14
    )
    np.testing.assert_allclose(
        eq.sum(axis=1), np.broadcast_to(neg, (len(eq), 3)), atol=1e-14
    )
    volume = eq * MASS_PER_EQ / DENSITY
    return volume / volume.sum(axis=(1, 2), keepdims=True)


@wp.kernel
def equilibrate(
    masses: wp.array3d(dtype=wp.float64),
    rho: wp.array(dtype=wp.float64),
    kappas: wp.array(dtype=wp.float64),
    result: wp.array(dtype=wp.float64),
    surface: wp.float64,
    molar_mass: wp.float64,
    temperature: wp.float64,
    gas_constant: wp.float64,
    radius: wp.float64,
    dry_volume: wp.float64,
    humidity: wp.float64,
):
    """Solve one zero-flux diameter per thread."""
    i = wp.tid()
    low = wp.float64(1.0)
    high = wp.float64(4.0)
    kelvin = kelvin_radius_wp(
        surface, rho[15], molar_mass, temperature, gas_constant
    )
    for _ in range(48):
        g = (low + high) * wp.float64(0.5)
        masses[i, 0, 15] = (g * g * g - wp.float64(1.0)) * dry_volume * rho[15]
        saturation = water_activity_kappa_wp(
            masses, rho, kappas, i, 0, 15
        ) * kelvin_term_wp(radius * g, kelvin)
        if saturation > humidity:
            high = g
        else:
            low = g
    result[i] = (low + high) * wp.float64(0.5)


def cuda_growth(fractions):
    """Find equilibrium using native Particula condensation functions."""
    densities = np.r_[DENSITY.ravel(), WATER]
    dry = (
        fractions.reshape(-1, 15)
        * densities[:15]
        * (4 * np.pi / 3 * DRY_RADIUS**3)
    )
    masses = np.column_stack([dry, np.zeros(len(dry))])[:, None, :]
    device = wp.get_device("cuda:0")
    arrays = [
        wp.array(x, dtype=wp.float64, device=device)
        for x in (masses, densities, np.r_[KAPPA.ravel(), 0.0])
    ]
    out = wp.empty(len(dry), dtype=wp.float64, device=device)
    wp.launch(
        equilibrate,
        len(dry),
        inputs=[
            *arrays,
            out,
            0.072,
            0.01801528,
            TEMPERATURE,
            8.314462618,
            DRY_RADIUS,
            4 * np.pi / 3 * DRY_RADIUS**3,
            RH,
        ],
        device=device,
    )
    return out.numpy()


def simulate(path):
    """Reconstruct and qualify both salt sets for each observed sample."""
    with Path(path).open() as stream:
        samples = list(csv.DictReader(stream))
    pure = np.array([cpu_growth(k) for k in KAPPA.ravel()]).reshape(5, 3)
    np.testing.assert_allclose(
        cuda_growth(np.eye(15)), pure.ravel(), atol=1e-11, rtol=0
    )
    result = []
    for row in samples:
        cat_mass = np.array(
            [float(row[k]) for k in ("nh4", "na", "k", "ca", "mg")]
        )
        an_mass = np.array([float(row[k]) for k in ("so4", "no3", "cl")])
        sample = dict(
            site=row["site"],
            time=row["sample_start_local"],
            observed=float(row["growth_inorganic_90"]),
        )
        for include in (False, True):
            active = cat_mass.copy()
            if not include:
                active[2:] = 0
            cations, anions = active / CATION_EQ, an_mass / ANION_EQ
            name = "expanded" if include else "reduced"
            if min(cations.sum(), anions.sum()) <= 0:
                sample[name] = None
                continue
            fractions = pairings(cations, anions)
            gpu = cuda_growth(fractions)
            k_mix = (fractions * KAPPA).sum(axis=(1, 2))
            np.testing.assert_(
                (
                    k_mix.min() >= k_mix[-2] - 1e-10
                    and k_mix.max() <= k_mix[-1] + 1e-10
                )
            )
            # Independent forward equation checks every CUDA root.
            water = gpu**3 - 1
            soluble = k_mix > 0.0
            saturation = (
                water[soluble]
                / (water[soluble] + k_mix[soluble])
                * np.exp(A_RADIUS / (DRY_RADIUS * gpu[soluble]))
            )
            np.testing.assert_allclose(gpu[~soluble], 1.0, atol=1e-11, rtol=0)
            np.testing.assert_allclose(saturation, RH, atol=1e-11, rtol=0)
            np.testing.assert_allclose(
                gpu[0], cpu_growth(k_mix[0]), atol=1e-11, rtol=0
            )
            zsr = np.cbrt((fractions * pure**3).sum(axis=(1, 2)))
            paired = min(cations.sum(), anions.sum())
            retained = (
                active.sum() * paired / cations.sum()
                + an_mass.sum() * paired / anions.sum()
            ) / (cat_mass.sum() + an_mass.sum())
            sample["expanded" if include else "reduced"] = dict(
                growth=float(gpu[0]),
                low=float(gpu.min()),
                high=float(gpu.max()),
                zsr=float(zsr[0]),
                retained=float(retained),
                max_rh_error=float(np.max(abs(saturation - RH), initial=0.0)),
            )
        result.append(sample)
    np.testing.assert_(
        (
            len(result) == 80
            and len({(r["site"], r["time"]) for r in result}) == 80
        )
    )
    return dict(
        samples=result,
        allocations_per_case=723,
        gpu=wp.get_device("cuda:0").name,
    )
