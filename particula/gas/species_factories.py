"""Factory module to create a concrete Species object using builders."""

from typing import Union

from particula.abc_factory import StrategyFactoryABC
from particula.gas.species import GasSpecies
from particula.gas.species_builders import (
    GasSpeciesBuilder,
    PresetGasSpeciesBuilder,
)


class GasSpeciesFactory(
    StrategyFactoryABC[
        Union[
            GasSpeciesBuilder,
            PresetGasSpeciesBuilder,
        ],
        GasSpecies,
    ]
):
    """Factory for creating species builders that produce GasSpecies objects.

    This class provides methods to retrieve a builder (e.g., 'gas_species'
    or 'preset_gas_species') and instantiate a GasSpecies object from it
    using user-specified parameters.

    Methods:
        - get_builders : Return a dictionary of builder objects.
        - get_strategy : Construct and return a GasSpecies object with the
          chosen builder.

    Returns:
        - GasSpecies : A gas species instance from the specified builder.

    Raises:
        - ValueError : If an unknown strategy type is provided.

    Examples:
        ```py title="Create a preset gas species using the factory"
        import particula as par
        factory = par.gas.GasSpeciesFactory()
        gas_object = factory.get_strategy("preset_gas_species", parameters)
        ```

        ```py title="Create a gas species using the factory"
        import particula as par
        factory = par.gas.GasSpeciesFactory()
        parameters = {
            "name": "Oxygen",
            "molar_mass": 0.032,
            "vapor_pressure_strategy": ConstantVaporPressureStrategy(
                vapor_pressure=101325
            ),
            "condensable": False,
            "concentration": 1.2,
        }
        gas_object = factory.get_strategy("gas_species", parameters)
        ```

    """

    def get_builders(self):
        """Return a mapping of strategy types to builder instances.

        Returns:
            - dict[str, Union[GasSpeciesBuilder, PresetGasSpeciesBuilder]] :
              A dictionary where:
                * "gas_species" -> GasSpeciesBuilder
                * "preset_gas_species" -> PresetGasSpeciesBuilder

        Examples:
            ```py title="get_builders Example"
            import particula as par
            factory = par.gas.GasSpeciesFactory()
            builder_map = factory.get_builders()
            # builder_map["gas_species"] -> GasSpeciesBuilder()
            ```
        """
        return {
            "gas_species": GasSpeciesBuilder(),
            "preset_gas_species": PresetGasSpeciesBuilder(),
        }
