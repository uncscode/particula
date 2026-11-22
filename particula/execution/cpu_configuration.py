"""Non-executable identity for supported native CPU configurations.

GPU plans intentionally impose GPU-specific support limits. CPU restart uses
an exact signature of known CPU physics objects instead. The signature is data
only: callers supply the actual strategy when resuming, and no type named in a
file is imported or instantiated. Unknown custom physics is rejected.
"""

import numpy as np

from particula.dynamics import (
    BrownianCoagulationStrategy,
    Coagulation,
    Dilution,
    MassCondensation,
    Nucleation,
    VanDerWaalsCoagulationStrategy,
    WallLoss,
)
from particula.dynamics.condensation import condensation_strategies as cs
from particula.dynamics.condensation.species_mapping import (
    CondensationSpeciesMap,
)
from particula.dynamics.dilution import DilutionStrategy
from particula.dynamics.nucleation import nucleation_strategies as ns
from particula.dynamics.nucleation.nucleation_configuration import (
    NucleationSourceConfig,
)
from particula.dynamics.nucleation.species_mapping import NucleationSpeciesMap
from particula.dynamics.particle_process import NucleationCommitConfig
from particula.dynamics.wall_loss import wall_loss_strategies as wl
from particula.execution.forcing import EnvironmentForcing
from particula.gas import latent_heat_strategies as lh
from particula.gas import vapor_pressure_strategies as vp
from particula.particles import activity_strategies as ac
from particula.particles import surface_strategies as su
from particula.particles.exhaustion import ExhaustionControls

NATIVE_PROCESSES = (
    MassCondensation,
    Coagulation,
    WallLoss,
    Dilution,
    Nucleation,
)

_CONFIGURATION_TYPES = (
    EnvironmentForcing,
    *NATIVE_PROCESSES,
    BrownianCoagulationStrategy,
    VanDerWaalsCoagulationStrategy,
    DilutionStrategy,
    wl.SphericalWallLossStrategy,
    wl.RectangularWallLossStrategy,
    wl.ChargedWallLossStrategy,
    NucleationSourceConfig,
    NucleationSpeciesMap,
    NucleationCommitConfig,
    ExhaustionControls,
    ns.ActivationNucleationStrategy,
    ns.KineticNucleationStrategy,
    ns.ClosedInterval,
    ns.NucleationValidityDomain,
    ns.InjectionComposition,
    ns.FormationMetadata,
    cs.CondensationIsothermal,
    cs.CondensationIsothermalStaggered,
    cs.CondensationLatentHeat,
    CondensationSpeciesMap,
    ac.ActivityIdealMass,
    ac.ActivityIdealMolar,
    ac.ActivityIdealVolume,
    ac.ActivityKappaParameter,
    ac.ActivityNonIdealBinary,
    ac.ActivityBAT,
    su.SurfaceStrategyMass,
    su.SurfaceStrategyMolar,
    su.SurfaceStrategyVolume,
    vp.ConstantVaporPressureStrategy,
    vp.AntoineVaporPressureStrategy,
    vp.ClausiusClapeyronStrategy,
    vp.WaterBuckStrategy,
    vp.TableVaporPressureStrategy,
    vp.ArblasterLiquidVaporPressureStrategy,
    vp.LiquidClausiusHybridStrategy,
    lh.ConstantLatentHeat,
    lh.LinearLatentHeat,
    lh.PowerLawLatentHeat,
)


def _freeze(value):
    """Represent configuration as data, without retaining live objects."""
    if value is None or type(value) in (str, bool, int, float):
        return value
    if isinstance(value, np.generic):
        return _freeze(value.item())
    if type(value) is np.ndarray:
        if value.dtype.kind not in "biuf" or not np.all(np.isfinite(value)):
            raise ValueError("configuration arrays must be finite numeric data")
        return ("array", value.dtype.str, value.shape, value.tobytes())
    if type(value) in (tuple, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, ac.ActivityStrategy):
        from particula.particles.prepared_activity import provider_identity

        identity = provider_identity(value)
        if identity is not None:
            return _freeze(identity)
    if type(value) in _CONFIGURATION_TYPES:
        # CPUExecutionContext owns stochastic draws, and latent energy is an
        # output diagnostic. Neither identifies the physical configuration.
        excluded = {
            "random_state",
            "random_generator",
            "last_latent_heat_energy",
        }
        if (
            type(value)
            in (
                su.SurfaceStrategyMass,
                su.SurfaceStrategyMolar,
                su.SurfaceStrategyVolume,
            )
            and value.surface_tension_table is not None
            and value.temperature_table is not None
        ):
            # Recomputed from the complete tables at every rate evaluation.
            excluded.add("surface_tension")
        attributes = tuple(
            (name, _freeze(item))
            for name, item in sorted(vars(value).items())
            if name not in excluded
        )
        return (type(value).__module__, type(value).__name__, attributes)
    raise TypeError(
        f"CPU restart has no configuration identity for {type(value).__name__}"
    )


def condensation_identity(strategy):
    """Retain all declared CPU transport, mapping and thermodynamic inputs."""
    return _freeze(strategy)


def cpu_process_identity(process):
    """Retain native configuration without imposing GPU capability limits."""
    from particula.dynamics.chemistry.process import GasChemistry
    from particula.dynamics.chemistry.resolved import ResolvedChemistry
    from particula.dynamics.optical_heating import OpticalHeating

    if type(process) in (GasChemistry, ResolvedChemistry, OpticalHeating):
        return (
            process.extension_id,
            process.version,
            process.name,
            process.configuration(),
        )
    if type(process) not in NATIVE_PROCESSES:
        raise TypeError("CPU restart requires a supported native runnable")
    return _freeze(process)
