"""Compare the differentiable chamber against native resolved-box processes."""

import json
import sys
from pathlib import Path

import numpy as np
import particula as par
from scipy.special import ndtri

from particula.dynamics.chemistry.discovery import compile_network
from particula.dynamics.chemistry.process import ChemistrySolver
from particula.dynamics.chemistry.resolved import ResolvedChemistry
from particula.dynamics.condensation.species_mapping import (
    CondensationSpeciesMap,
)
from particula.execution.inverse_chemistry import (
    SeededChamber,
    VaporExchange,
    simulate_chamber,
)
from particula.gas import EnvironmentData, GasData, get_molecule_mean_free_path
from particula.particles import ParticleData
from particula.util.constants import GAS_CONSTANT


def compare(step):
    """Match gas conversion, particle dimerization, and condensation."""
    masses = np.array([0.1, 0.1, 0.1, 0.2, 0.1])
    density = np.array([950.0, 1250.0, 1100.0])
    names = ["gas:A", "gas:V", "particle:V", "particle:D", "particle:seed"]
    mechanism = compile_network(
        ((-1, 1, 0, 0, 0), (0, 0, -2, 1, 0)),
        {
            n: {
                "molar_mass_kg_per_mol": float(m),
                "elements": {"X": round(m / 0.1)},
            }
            for n, m in zip(names, masses, strict=True)
        },
        [0.5, 0.002],
    )
    path = float(
        get_molecule_mean_free_path(
            molar_mass=0.1, temperature=298.0, pressure=101325.0
        )
    )
    box = SeededChamber(
        mechanism,
        dict(zip(names[2:], density, strict=True)),
        {"particle:seed": 0.95, "particle:V": 0.05},
        (VaporExchange(names[1], names[2], 1e-5, path),),
        tuple(names[2:4]),
        tuple(names[:2]),
        (80e-9, 120e-9),
        0.12,
        3,
    )
    doses = np.array([1e-7, 1e-9])
    seed = [1e10, 1e-7, 1.4, 0.0]
    number = np.full(3, seed[0] / 3.0)
    volume = (
        np.pi / 6 * (seed[1] * seed[2] ** ndtri((np.arange(3) + 0.5) / 3)) ** 3
    )
    state = par.Aerosol(
        particles=ParticleData(
            masses=(volume[:, None] * density * [0.05, 0, 0.95])[None, :, :],
            concentration=number[None, :],
            charge=np.zeros((1, 3)),
            density=density,
            volume=np.ones(1),
            distribution_type="particle_resolved",
        ),
        gas=GasData(
            ["A", "V"],
            masses[:2],
            (doses * masses[:2])[None, :],
            np.array([False, True]),
        ),
        environment=EnvironmentData(
            temperature=np.array([298.0]),
            pressure=np.array([101325.0]),
            saturation_ratio=np.zeros((1, 2)),
        ),
    )
    chem = ResolvedChemistry(
        mechanism.bind_resolved(state, {n: i for i, n in enumerate(names[2:])}),
        ChemistrySolver(rtol=1e-9, atol=1e-13),
    )
    cond = par.dynamics.MassCondensation(
        par.dynamics.CondensationIsothermal(
            molar_mass=masses[:2],
            diffusion_coefficient=1e-5,
            accommodation_coefficient=1.0,
            activity_strategy=par.particles.ActivityIdealMolar(masses[2:]),
            vapor_pressure_strategy=par.gas.ConstantVaporPressureStrategy(
                2e-8 * 298.0 * GAS_CONSTANT
            ),
            surface_strategy=par.particles.SurfaceStrategyVolume(
                surface_tension=1e-12, density=1000.0
            ),
            species_map=CondensationSpeciesMap(("A", "V"), ((1, 0),)),
        )
    )
    times = [0.5, 1.0, 1.5, 2.0]
    native = []
    for _ in times:
        for _ in range(round(0.5 / step)):
            chem.execute(state, step / 2)
            cond.execute(state, step)
            chem.execute(state, step / 2)
        native.append(
            np.r_[
                state.gas.concentration[0],
                number @ state.particles.masses[0, :, :2].sum(axis=1),
            ]
        )
    inverse = simulate_chamber(
        box,
        times,
        dose=doses,
        rates=[0.5, 0.002],
        saturation=[2e-8],
        seed=seed,
        max_step=step,
        device="cpu",
    )
    relative = np.max(
        abs(np.asarray(native) - inverse.predictions[:, :3])
        / np.max(native, axis=0),
        axis=0,
    )
    return dict(
        dt=step,
        native=np.array(native).tolist(),
        inverse=inverse.predictions[:, :3].tolist(),
        channel_peak_normalized_errors=relative.tolist(),
    )


if __name__ == "__main__":
    rows = [compare(step) for step in (0.1, 0.05, 0.025)]
    np.testing.assert_array_less(
        rows[-1]["channel_peak_normalized_errors"], 0.002
    )
    Path(sys.argv[1]).write_text(json.dumps(rows, indent=2) + "\n")
