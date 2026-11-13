"""Deterministic, scientifically specified workloads for the RTX study campaign.

Only CPU construction happens here. Every GPU entry point explicitly requests
CUDA. Particle materials and gas species have independent widths and mappings.
"""

from dataclasses import dataclass
from functools import reduce
from operator import or_

import numpy as np
from particula import Aerosol
from particula.dynamics import (
    BrownianCoagulationStrategy,
    Coagulation,
    Dilution,
    DilutionStrategy,
    MassCondensation,
    Nucleation,
    WallLoss,
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


@dataclass(frozen=True)
class StudyCase:
    """Fixed layout, physics and reproducibility controls for one workload."""

    name: str
    family: str
    boxes: int
    slots: int
    materials: int = 2
    gases: int = 3
    occupancy: float = 1.0
    time_step: float = 0.125
    steps: int = 3
    seed: int = 42
    heterogeneous: bool = False
    experiment: str = "steady"


def water_pressure(temperature):
    """Independent Buck water-pressure expression [Pa], temperature in K."""
    celsius = np.asarray(temperature) - 273.15
    return 611.21 * np.exp(
        (18.678 - celsius / 234.5) * celsius / (257.14 + celsius)
    )


def build_case(case: StudyCase):
    """Build a polydisperse mixed-material aerosol and its ordered processes.

    Counts are unit multiplicities. Ordinary cases start at 5e13 particles/m³;
    heterogeneous volumes vary that density by a factor of sixteen. The full
    lifecycle instead uses a power-of-two volume and an exact two-event source
    pulse. The precursor must be explicitly removed after the first cycle so
    later dilution cannot make a new unit cohort incompatible with Brownian.
    """
    if case.materials < 2 or case.gases <= case.materials:
        raise ValueError(
            "study requires at least two materials and more gas lanes"
        )
    if not 0 < case.occupancy <= 1 or case.boxes < 1 or case.slots < 4:
        raise ValueError("invalid study layout")
    boxes, slots = case.boxes, case.slots
    rng = np.random.default_rng(case.seed)
    radii = np.clip(
        rng.lognormal(np.log(35e-9), np.log(1.6), (boxes, slots)), 5e-9, 200e-9
    )
    fractions = np.empty((boxes, slots, case.materials))
    wet = 0.2 + 0.08 * np.tanh(np.log(radii / 35e-9))
    if case.heterogeneous:
        wet += np.linspace(-0.1, 0.2, boxes)[:, None]
    fractions[..., 0] = wet
    fractions[..., 1:] = (1 - wet)[..., None] / (case.materials - 1)
    masses = (4 * np.pi / 3 * 1000 * radii**3)[..., None] * fractions
    count = int(slots * case.occupancy)
    counts = np.ones((boxes, slots))
    inactive = rng.permutation(slots)[count:]
    counts[:, inactive] = 0
    masses[:, inactive] = 0
    charge = rng.integers(-2, 3, (boxes, slots)).astype(np.float64)
    charge[:, inactive] = 0
    volume = np.full(boxes, count / 5e13)
    if case.heterogeneous:
        volume *= np.geomspace(0.25, 4.0, boxes)
    if case.family == "full":
        if slots - count < 2 or case.time_step != 1.0:
            raise ValueError("full lifecycle needs two free slots and dt=1 s")
        volume[:] = 2.0**-30
    temperature = np.full(boxes, 298.15)
    pressure = np.full(boxes, 101325.0)
    relative_humidity = np.full(boxes, 1.3)
    if case.heterogeneous:
        temperature[:] = np.linspace(275, 310, boxes)
        pressure[:] = np.linspace(85000, 105000, boxes)
        relative_humidity[:] = np.linspace(0.8, 1.8, boxes)
    molar_mass = np.array(
        [0.028, 0.2, 0.018] + [0.044 + 0.01 * i for i in range(case.gases - 3)]
    )
    concentration = np.full((boxes, case.gases), 1e-8)
    concentration[:, 2] = (
        relative_humidity
        * water_pressure(temperature)
        * molar_mass[2]
        / (GAS_CONSTANT * temperature)
    )
    if case.family == "full":
        concentration[:, 1] = 8 * molar_mass[1] / AVOGADRO_NUMBER / volume
    partitioning = np.zeros(case.gases, dtype=bool)
    partitioning[2] = True
    if case.family == "full":
        partitioning[1] = True
    aerosol = Aerosol(
        particles=ParticleData(
            masses,
            counts,
            charge,
            np.full(case.materials, 1000.0),
            volume,
            distribution_type="particle_resolved",
        ),
        gas=GasData(
            ["inert", "precursor", "water"]
            + [f"spectator_{i}" for i in range(case.gases - 3)],
            molar_mass,
            concentration,
            partitioning,
        ),
        environment=EnvironmentData(
            temperature, pressure, np.ones_like(concentration)
        ),
    )
    processes = _processes(case, aerosol)
    return aerosol, reduce(or_, processes), processes


def _processes(case, aerosol):
    """Configure only existing physics with explicit full-gas species maps."""
    molar_mass = aerosol.gas.molar_mass
    processes = []
    if case.family == "full":
        processes.append(_source(aerosol))
    if case.family in ("brownian", "full"):
        processes.append(
            Coagulation(BrownianCoagulationStrategy("particle_resolved"))
        )
    if case.family in ("condensation", "full"):
        vapor = [ConstantVaporPressureStrategy(1.0) for _ in range(case.gases)]
        vapor[2] = WaterBuckStrategy()
        processes.append(
            MassCondensation(
                CondensationIsothermal(
                    molar_mass=molar_mass.copy(),
                    diffusion_coefficient=np.full(case.gases, 2e-5),
                    accommodation_coefficient=np.full(case.gases, 0.5),
                    activity_strategy=ActivityIdealMolar(
                        np.array([0.018] + [0.2] * (case.materials - 1))
                    ),
                    surface_strategy=SurfaceStrategyVolume(
                        surface_tension=0.072, density=1000.0
                    ),
                    vapor_pressure_strategy=tuple(vapor),
                    species_map=CondensationSpeciesMap(
                        tuple(aerosol.gas.name), ((2, 0),)
                    ),
                )
            )
        )
    if case.family == "full":
        from particula.dynamics.wall_loss.wall_loss_strategies import (
            SphericalWallLossStrategy,
        )

        processes.append(
            WallLoss(SphericalWallLossStrategy(0.1, 0.02, "particle_resolved"))
        )
    if case.family in ("dilution", "condensation", "full"):
        processes.append(Dilution(DilutionStrategy(0.02)))
    if not processes:
        raise ValueError(f"unknown study family {case.family}")
    return tuple(processes)


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


def particle_inventory(aerosol):
    """Independent extensive mass [kg] and represented number per box."""
    p = aerosol.particles
    return np.sum(
        p.masses * p.concentration[..., None], axis=1
    ), p.concentration.sum(axis=1)


def radius_histogram(aerosol, edges=None):
    """Number concentration in logarithmic radius bins, resolved by box."""
    edges = np.geomspace(0.2e-9, 2e-6, 65) if edges is None else edges
    p = aerosol.particles
    radius = np.cbrt(np.sum(p.masses / p.density, axis=-1) * 3 / (4 * np.pi))
    rows = [
        np.histogram(r, bins=edges, weights=c / v)[0]
        for r, c, v in zip(radius, p.concentration, p.volume, strict=True)
    ]
    return edges, np.asarray(rows)
