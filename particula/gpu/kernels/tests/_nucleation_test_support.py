"""Explicit species-map setup shared by native source fixtures."""


def _binding(particles, gas):
    """Bind the fixture's explicit species order after replacement."""
    from particula.dynamics.nucleation.species_mapping import (
        NucleationSpeciesMap,
    )
    from particula.gpu.nucleation_species_mapping import (
        prepare_nucleation_species_map,
    )

    names = tuple(
        f"species_{index}" for index in range(gas.concentration.shape[1])
    )
    return prepare_nucleation_species_map(
        particles,
        gas,
        gas_names=names,
        species_map=NucleationSpeciesMap(
            names, tuple((index, index) for index in range(len(names)))
        ),
    )
