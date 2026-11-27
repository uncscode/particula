"""Finite-precursor native nucleation with size-resolved newborn cohorts."""

import argparse
import csv
from pathlib import Path
from time import perf_counter

import numpy as np
from particula import Aerosol, dynamics, gas, particles
from particula.dynamics.condensation.species_mapping import (
    CondensationSpeciesMap,
)
from particula.dynamics.nucleation.species_mapping import NucleationSpeciesMap
from particula.particles.exhaustion import ExhaustionControls
from scipy.constants import Avogadro

RHO = 1500.0
MOLAR = 0.098
MOLECULES_PER_CLUSTER = 16
D0 = (6 * MOLECULES_PER_CLUSTER * MOLAR / (np.pi * RHO * Avogadro)) ** (1 / 3)


def build(dt, end, initial, coefficient, law):
    """Construct an empty resolved box and mapped native process owners."""
    capacity = int(np.ceil(end / dt)) + 8
    state = particles.ParticleData(
        masses=np.zeros((1, capacity, 1)),
        concentration=np.zeros((1, capacity)),
        charge=np.zeros((1, capacity)),
        density=np.array([RHO]),
        volume=np.array([1.0]),
        distribution_type="particle_resolved",
    )
    reservoir = gas.GasData(
        name=["precursor"],
        molar_mass=np.array([MOLAR]),
        partitioning=np.array([True]),
        concentration=np.array([[float(initial)]]),
    )
    env = gas.EnvironmentData(
        temperature=np.array([293.0]),
        pressure=np.array([101325.0]),
        saturation_ratio=np.ones((1, 1)),
    )
    builder = (
        dynamics.ActivationNucleationBuilder()
        if law == "activation"
        else dynamics.KineticNucleationBuilder()
    )
    strategy = builder.set_parameters(
        dict(
            coefficient=coefficient,
            coefficient_units="s^-1" if law == "activation" else "m^3/s",
            coefficient_provenance="Synthetic finite-reservoir study",
            precursor_number_concentration_lower=0.0,
            precursor_number_concentration_lower_units="1/m^3",
            precursor_number_concentration_upper=1e32,
            precursor_number_concentration_upper_units="1/m^3",
            temperature_lower=270.0,
            temperature_lower_units="K",
            temperature_upper=310.0,
            temperature_upper_units="K",
            injection_composition=[MOLECULES_PER_CLUSTER],
            formation_diameter=D0 * 1e9,
            formation_diameter_units="nm",
        )
    ).build()
    birth = dynamics.Nucleation(
        dynamics.NucleationSourceConfig(
            strategy=strategy,
            precursor_index=0,
            species_map=NucleationSpeciesMap(("precursor",), ((0, 0),)),
        ),
        dynamics.NucleationCommitConfig(
            maximum_slot_weight=1e25,
            source_charge=0.0,
            exhaustion_controls=ExhaustionControls(),
            requested_scale=np.ones(1),
            minimum_scale=np.ones(1),
            minimum_volume=np.array([1e-6]),
        ),
    )
    vp = gas.VaporPressureFactory().get_strategy(
        "constant", dict(vapor_pressure=1e-30, vapor_pressure_units="Pa")
    )
    growth = dynamics.MassCondensation(
        dynamics.CondensationIsothermal(
            molar_mass=MOLAR,
            accommodation_coefficient=1.0,
            diffusion_coefficient=1e-5,
            activity_strategy=particles.ActivityIdealMass(),
            surface_strategy=particles.SurfaceStrategyVolume(),
            vapor_pressure_strategy=vp,
            species_map=CondensationSpeciesMap(("precursor",), ((0, 0),)),
        )
    )
    return (
        Aerosol(particles=state, gas=reservoir, environment=env),
        birth,
        growth,
    )


def run(dt, end, initial, coefficient, law, substeps, order):
    """Advance births and growth and record the two diameter thresholds."""
    tic = perf_counter()
    b, birth, growth = build(dt, end, initial, coefficient, law)
    first = {5: None, 10: None}
    history = []
    for j in range(round(end / dt)):
        if order == "symmetric":
            growth.execute(b, time_step=dt / 2, sub_steps=substeps)
        birth.execute(b, time_step=dt, sub_steps=1)
        growth.execute(
            b,
            time_step=dt / (2 if order == "symmetric" else 1),
            sub_steps=substeps,
        )
        m = b.particles.masses[0, :, 0]
        w = b.particles.concentration[0]
        diameter = 1e9 * np.cbrt(6 * m / (np.pi * RHO))
        n5 = float(w[diameter >= 5].sum())
        n10 = float(w[diameter >= 10].sum())
        for threshold, n in [(5, n5), (10, n10)]:
            if n > 0 and first[threshold] is None:
                first[threshold] = (j + 1) * dt
        pm = float(w @ m)
        gm = float(b.gas.concentration[0, 0])
        drift = abs(gm + pm - initial) / max(1e-30, initial)
        if drift > 1e-10 or min(gm, pm, w.min()) < 0:
            raise ArithmeticError("Birth/growth budget failed")
        history.append(
            dict(
                time_s=(j + 1) * dt,
                number_m3=w.sum(),
                above_5nm_m3=n5,
                above_10nm_m3=n10,
                gas_kg_m3=gm,
                particle_kg_m3=pm,
                largest_diameter_nm=diameter.max(),
                ledger=drift,
                live_cohorts=int((w > 0).sum()),
            )
        )
    return dict(
        law=law,
        coefficient=coefficient,
        initial_kg_m3=initial,
        dt=dt,
        end_s=end,
        growth_substeps=substeps,
        order=order,
        first5_s=first[5],
        first10_s=first[10],
        runtime_s=perf_counter() - tic,
    ), history


def write(path, rows):
    """Replace a CSV atomically after a complete case."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument(
        "--initial", type=float, nargs="+", default=[0, 1e-10, 1e-9, 1e-8]
    )
    p.add_argument("--rate", type=float, nargs="+", default=[1e-7])
    p.add_argument("--dt", type=float, nargs="+", default=[8, 4, 2, 1])
    p.add_argument("--end", type=float, default=128)
    p.add_argument("--laws", nargs="+", default=["activation", "kinetic"])
    p.add_argument("--orders", nargs="+", default=["symmetric", "birth-first"])
    p.add_argument("--substeps", type=int, nargs="+", default=[1, 4])
    a = p.parse_args()
    from itertools import product

    rows = []
    for initial, rate, dt, sub, law, order in product(
        a.initial, a.rate, a.dt, a.substeps, a.laws, a.orders
    ):
        coeff = (
            rate if law == "activation" else rate / (1e-9 * Avogadro / MOLAR)
        )
        meta, history = run(dt, a.end, initial, coeff, law, sub, order)
        for row in history:
            rows.append({**meta, **row})
        write(a.out, rows)
        print(
            law,
            initial,
            dt,
            sub,
            order,
            meta["first5_s"],
            meta["first10_s"],
            flush=True,
        )
