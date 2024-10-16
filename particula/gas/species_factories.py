"""Factory module to create a concrete Species object using builders."""

from typing import Union
from particula.abc_factory import StrategyFactory
from particula.gas.species_builders import (
    GasSpeciesBuilder,
    PresetGasSpeciesBuilder,
)
from particula.gas.species import GasSpecies


class GasSpeciesFactory(
    StrategyFactory[
        Union[
            GasSpeciesBuilder,
            PresetGasSpeciesBuilder,
        ],
        GasSpecies,
    ]
):
    """Factory class to create species builders

    Factory class to create species builders for creating gas species.

    Methods:
        get_builders: Returns the mapping of strategy types to builder
        instances.
        get_strategy: Gets the strategy instance
        for the specified strategy type.
            strategy_type: Type of species builder to use, can be
            'gas_species' or 'preset_gas_species'.
            parameters: Parameters required for the
            builder, dependent on the chosen strategy type.

    Returns:
        GasSpecies: An instance of the specified GasSpecies.

    Raises:
        ValueError: If an unknown strategy type is provided.
    """

    def get_builders(self):
        """Returns the mapping of strategy types to builder instances.

        Returns:
            A dictionary mapping strategy types to builder instances.
                - gas_species: GasSpeciesBuilder
                - preset_gas_species: PresetGasSpeciesBuilder
        """
        return {
            "gas_species": GasSpeciesBuilder(),
            "preset_gas_species": PresetGasSpeciesBuilder(),
        }
