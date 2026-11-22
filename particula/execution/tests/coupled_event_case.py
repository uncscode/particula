"""Fixed two-box regression case from the v0.3.3 coupled GPU tests."""

import numpy as np

from particula import Aerosol
from particula.dynamics import (
    BrownianCoagulationStrategy,
    Coagulation,
    Nucleation,
)
from particula.dynamics.condensation.condensation_strategies import (
    CondensationIsothermal,
)
from particula.dynamics.condensation.species_mapping import (
    CondensationSpeciesMap,
)
from particula.gas import EnvironmentData, GasData
from particula.gas.vapor_pressure_strategies import (
    ConstantVaporPressureStrategy,
    WaterBuckStrategy,
)
from particula.particles import ParticleData
from particula.particles.activity_strategies import ActivityIdealMolar
from particula.particles.surface_strategies import SurfaceStrategyVolume
from particula.util.constants import AVOGADRO_NUMBER, GAS_CONSTANT


def coupled_event_case():
    """Build two source events per box and water uptake for restart checks."""
    boxes, slots = 2, 64
    rng = np.random.default_rng(42)
    radii = np.clip(
        rng.lognormal(np.log(35e-9), np.log(1.6), (boxes, slots)), 5e-9, 200e-9
    )
    wet = 0.2 + 0.08 * np.tanh(np.log(radii / 35e-9))
    fractions = np.stack((wet, 1 - wet), axis=-1)
    masses = (4 * np.pi / 3 * 1000 * radii**3)[..., None] * fractions
    counts = np.ones((boxes, slots))
    inactive = rng.permutation(slots)[48:]
    counts[:, inactive] = 0
    masses[:, inactive] = 0
    charge = rng.integers(-2, 3, (boxes, slots)).astype(np.float64)
    charge[:, inactive] = 0

    # Eight precursor molecules in each representative volume give an exact
    # two-event pulse with the activation coefficient below.
    volume = np.full(boxes, 2.0**-30)
    temperature = np.full(boxes, 298.15)
    molar_mass = np.array([0.028, 0.2, 0.018])
    concentration = np.full((boxes, 3), 1e-8)
    concentration[:, 1] = 8 * molar_mass[1] / AVOGADRO_NUMBER / volume
    celsius = temperature - 273.15
    saturation = 611.21 * np.exp(
        (18.678 - celsius / 234.5) * celsius / (257.14 + celsius)
    )
    concentration[:, 2] = (
        1.3 * saturation * molar_mass[2] / (GAS_CONSTANT * temperature)
    )
    aerosol = Aerosol(
        particles=ParticleData(
            masses,
            counts,
            charge,
            np.full(2, 1000.0),
            volume,
            distribution_type="particle_resolved",
        ),
        gas=GasData(
            ["inert", "precursor", "water"],
            molar_mass,
            concentration,
            np.array([False, True, True]),
        ),
        environment=EnvironmentData(
            temperature, np.full(boxes, 101325.0), np.ones_like(concentration)
        ),
    )
    coagulation = Coagulation(BrownianCoagulationStrategy("particle_resolved"))
    condensation = CondensationIsothermal(
        molar_mass=molar_mass.copy(),
        diffusion_coefficient=np.full(3, 2e-5),
        accommodation_coefficient=np.full(3, 0.5),
        activity_strategy=ActivityIdealMolar(np.array([0.018, 0.2])),
        surface_strategy=SurfaceStrategyVolume(
            surface_tension=0.072, density=1000.0
        ),
        vapor_pressure_strategy=(
            ConstantVaporPressureStrategy(1.0),
            ConstantVaporPressureStrategy(1.0),
            WaterBuckStrategy(),
        ),
        species_map=CondensationSpeciesMap(tuple(aerosol.gas.name), ((2, 0),)),
    )
    return aerosol, _source(aerosol), coagulation, condensation


def _source(aerosol):
    """Two integral neutral precursor events in the initial one-second pulse."""
    from particula.dynamics.nucleation.nucleation_configuration import (
        NucleationSourceConfig,
    )
    from particula.dynamics.nucleation.nucleation_strategies import (
        ActivationNucleationStrategy,
        ClosedInterval,
        FormationMetadata,
        InjectionComposition,
        NucleationValidityDomain,
    )
    from particula.dynamics.nucleation.species_mapping import (
        NucleationSpeciesMap,
    )
    from particula.dynamics.particle_process import NucleationCommitConfig
    from particula.particles.exhaustion import ExhaustionControls

    boxes = aerosol.particles.n_boxes
    molecules = [0] * len(aerosol.gas.name)
    molecules[1] = 1
    strategy = ActivationNucleationStrategy(
        0.25,
        NucleationValidityDomain(
            ClosedInterval(0, 1e30), ClosedInterval(200, 400)
        ),
        InjectionComposition(tuple(molecules)),
        FormationMetadata(1e-9),
    )
    mapping = NucleationSpeciesMap(tuple(aerosol.gas.name), ((1, 1),))
    return Nucleation(
        NucleationSourceConfig(strategy, 1, mapping),
        NucleationCommitConfig(
            1.0,
            0.0,
            ExhaustionControls(False, False),
            np.ones(boxes),
            np.full(boxes, 0.001),
            np.full(boxes, 1e-30),
        ),
    )
