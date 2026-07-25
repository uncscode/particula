"""Expose bounded P4 construction APIs for CPU nucleation potential rates.

This namespace exports immutable activation and kinetic potential-rate
strategies, their strict builders, ``NucleationFactory``, and source-selection
metadata. P2/P3 helpers in ``particle_source`` finalize source demand or mutate
particle and gas state; they intentionally remain concrete-module-only.
"""

from particula.dynamics.nucleation.nucleation_builders import (
    ActivationNucleationBuilder,
    KineticNucleationBuilder,
    NucleationSourceConfigBuilder,
)
from particula.dynamics.nucleation.nucleation_configuration import (
    NucleationSourceConfig,
)
from particula.dynamics.nucleation.nucleation_factories import NucleationFactory
from particula.dynamics.nucleation.nucleation_strategies import (
    ActivationNucleationStrategy,
    ClosedInterval,
    FormationMetadata,
    InjectionComposition,
    KineticNucleationStrategy,
    NucleationStrategy,
    NucleationValidityDomain,
)

__all__ = [
    "ActivationNucleationBuilder",
    "ActivationNucleationStrategy",
    "ClosedInterval",
    "FormationMetadata",
    "InjectionComposition",
    "KineticNucleationBuilder",
    "KineticNucleationStrategy",
    "NucleationFactory",
    "NucleationSourceConfig",
    "NucleationSourceConfigBuilder",
    "NucleationStrategy",
    "NucleationValidityDomain",
]
