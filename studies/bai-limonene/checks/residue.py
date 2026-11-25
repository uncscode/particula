"""Replay the paper's 100 nm dark-conditioning and labeled-product assays."""

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

# Each state is a fraction of the initial dry particle mass. The first 15
# components in each phase hold five slow parents, five fast parents and
# five first-generation products. Secondary reaction sends mass to vapor.
PARTICLE, GAS = slice(0, 15), slice(15, 30)
EXHAUST, PERMANENT = 30, 31
TAG_PARTICLE, TAG_GAS, TAG_EXHAUST, TAG_REACTED = range(32, 36)
CORE = 0.005
TEMP_K = 294.0
DRY_DIAMETER_NM = 100.0
RANKS = ["ranked", "independent", "inverse"]


class Assay:
    """Shared chemical and optical inputs for the 330 nm receivers."""

    def __init__(self, run_dir):
        """Load model inputs staged by the parent study replay."""
        # Use the same staged chemistry/optics inputs as the seasonal model.
        sunlight = run_dir.resolve() / "sunlight"
        sys.path.insert(0, str(sunlight))
        self.aging = importlib.import_module("aging")
        spectrum = pd.read_csv(sunlight / "solar/spectrum.csv").set_index(
            "wavelength_nm"
        )
        rates = pd.read_csv(sunlight / "solar/rates.csv").set_index("species")
        at330 = spectrum.loc[330]
        self.fast = 3600 * rates.loc["mvos", "standard_j_per_s"]
        self.slow = self.fast * at330.action_lvos_cm2 / at330.action_mvos_cm2
        # Absorbed cross section divided by dry volume times bulk absorption
        # coefficient gives the internal-light multiplier of the wet sphere.
        absorption_coefficient = (
            1400
            / 0.25
            * 6.02214076e23
            * 1.3e-23
            * (9.1e-21 / 1.3e-19) ** ((330 - 300) / 45)
        )
        with np.load(sunlight / "optics/cross_330nm.npz") as optical:
            diameters = optical["diameter_nm"]
            self.log_diameter = np.log(diameters)
            self.field = (
                optical["extinction_m2"] - optical["scattering_m2"]
            ) / (np.pi / 6 * (diameters * 1e-9) ** 3 * absorption_coefficient)
        if self.field.shape != diameters.shape or not (self.field > 0).all():
            raise ValueError("Invalid 330 nm Mie lookup table")
        # Micrograms per cubic meter at one particle per cubic centimeter.
        self.mass_per_number = (
            np.pi / 6 * self.aging.RHO * (DRY_DIAMETER_NM * 1e-9) ** 3 * 1e15
        )


class Box:
    """One ventilated population of initially monodisperse particles."""

    def __init__(self, assay, rank, gas_lifetime_h, number_cm3):
        """Partition the fresh organic material into volatility classes."""
        self.assay = assay
        self.number_cm3 = number_cm3
        self.gas_lifetime_h = gas_lifetime_h
        exposure = assay.aging.Exposure(
            inventory="wilson",
            common_fresh_k=True,
            spread_aged=True,
            rate_assignment=rank,
        )
        cstar, self.kappa, self.initial_parent, *_ = assay.aging.species(
            exposure
        )
        self.cstar = cstar / (assay.mass_per_number * number_cm3)
        self.thermal_speed = np.sqrt(
            8 * 8.314 * TEMP_K / (np.pi * assay.aging.MW)
        )

    def initial(self):
        """Return a fresh population with a 0.5% inert dry core."""
        state = np.zeros(36)
        state[:10] = (1 - CORE) * self.initial_parent
        return state

    def evolve(self, start, lamp, secondary, hours, target_mass=None):
        """Integrate to the requested time or first dry-mass crossing."""

        def rhs(t, y):
            particle, gas = y[PARTICLE], y[GAS]
            dry = CORE + particle.sum()
            diameter = DRY_DIAMETER_NM * np.cbrt(dry)
            kappa = (CORE * 0.0465 + particle @ self.kappa) / dry
            wet_radius = diameter * 5e-10 * (1 + kappa * 0.8 / 0.2) ** (1 / 3)
            kn = 3 * 8e-6 / (self.thermal_speed * wet_radius)
            transition = (1 + kn) / (1 + (4 / 3 + 0.377) * kn + 4 * kn**2 / 3)
            conduct = (
                4
                * np.pi
                * 8e-6
                * wet_radius
                * transition
                * self.number_cm3
                * 1e6
                * 3600
            )
            kelvin = np.exp(
                2
                * 0.03
                * self.assay.aging.MW
                / (8.314 * TEMP_K * self.assay.aging.RHO * wet_radius)
            )
            evaporation = conduct * kelvin * self.cstar / dry
            exchange = conduct * gas - evaporation * particle
            light = lamp * np.interp(
                np.log(diameter), self.assay.log_diameter, self.assay.field
            )
            reaction = (
                light
                * particle
                * np.repeat(
                    [
                        self.assay.slow,
                        self.assay.fast,
                        secondary * self.assay.slow,
                    ],
                    5,
                )
            )
            dy = np.zeros_like(y)
            dy[PARTICLE] = exchange - reaction
            dy[GAS] = -exchange - gas / self.gas_lifetime_h
            dy[EXHAUST] = gas.sum() / self.gas_lifetime_h
            slow, fast = reaction[:5].sum(), reaction[5:10].sum()
            dy[10] += 0.40 * slow
            dy[11:15] += (0.41 * slow + 0.68 * fast) / 4
            dy[PERMANENT] = 0.19 * slow + 0.32 * fast + reaction[10:].sum()
            tag_exchange = (
                conduct * y[TAG_GAS] - evaporation[10] * y[TAG_PARTICLE]
            )
            dy[TAG_REACTED] = (
                light * self.assay.slow * secondary * y[TAG_PARTICLE]
            )
            dy[TAG_PARTICLE] = tag_exchange - dy[TAG_REACTED]
            dy[TAG_EXHAUST] = y[TAG_GAS] / self.gas_lifetime_h
            dy[TAG_GAS] = -tag_exchange - dy[TAG_EXHAUST]
            return dy

        def mass_reached(t, state):
            return CORE + state[PARTICLE].sum() - target_mass

        mass_reached.direction = -1
        mass_reached.terminal = True
        result = solve_ivp(
            rhs,
            (0, hours),
            start,
            method="LSODA",
            rtol=3e-7,
            atol=3e-11,
            events=None if target_mass is None else mass_reached,
        )
        if not result.success or (
            target_mass is not None and result.status != 1
        ):
            raise RuntimeError(
                f"Integration failed or missed mass target: {result.message}"
            )
        end = result.y[:, -1]
        np.testing.assert_allclose(
            CORE + end[:TAG_PARTICLE].sum(), 1, atol=1e-7, rtol=0
        )
        np.testing.assert_allclose(
            end[TAG_PARTICLE:].sum(),
            start[TAG_PARTICLE:].sum(),
            atol=1e-7,
            rtol=0,
        )
        return end


def run(run_dir, output):
    """Compute both paired residue experiments and write compact tables."""
    assay = Assay(run_dir)
    fractions = []
    for rank in RANKS:
        for lifetime in [15 / 3600, 24]:
            box = Box(assay, rank, lifetime, 300)
            end = box.evolve(
                box.initial(), lamp=0, secondary=0, hours=60, target_mass=0.5
            )
            rate = (
                assay.slow * end[:5].sum() + assay.fast * end[5:10].sum()
            ) / end[PARTICLE].sum()
            ratio = rate / (0.55 * assay.slow + 0.45 * assay.fast)
            fractions.append(
                dict(
                    assignment=rank,
                    gas_lifetime_h=lifetime,
                    action_relative_to_fresh=ratio,
                )
            )

    marker = []
    # An identical prepared 2 µg/m³ sample enters each dark and light receiver.
    box = Box(assay, "ranked", 24, 2 / assay.mass_per_number)
    split = box.evolve(
        box.initial(), lamp=60, secondary=0.25, hours=60, target_mass=0.5
    )
    split[EXHAUST] += split[GAS].sum()
    split[GAS] = 0
    split[TAG_PARTICLE] = split[10]
    dark = box.evolve(split, lamp=0, secondary=0, hours=0.8)
    for secondary in [0, 0.25, 1]:
        lit = box.evolve(split, lamp=30, secondary=secondary, hours=0.8)
        marker.append(
            dict(
                secondary_relative_to_slow=secondary,
                reacted_fraction=lit[TAG_REACTED] / split[10],
                dark_intact_gas_and_exhaust_fraction=(
                    dark[TAG_GAS] + dark[TAG_EXHAUST]
                )
                / split[10],
            )
        )
    output.mkdir(parents=True, exist_ok=True)
    for name, records in [
        ("residual-action", fractions),
        ("product-turnover", marker),
    ]:
        frame = pd.DataFrame(records)
        frame.to_csv(output / f"{name}.csv", index=False, float_format="%.8g")
        print(frame.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Staging directory prepared with replay.py inputs",
    )
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.run_dir, args.output)
