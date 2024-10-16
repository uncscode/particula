"""Activity strategy factories for calculating activity and partial pressure
of species in a mixture of liquids."""

from typing import Union
from particula.abc_factory import StrategyFactory
from particula.particles.activity_builders import (
    ActivityIdealMassBuilder,
    ActivityIdealMolarBuilder,
    ActivityKappaParameterBuilder,
)
from particula.particles.activity_strategies import (
    ActivityIdealMass,
    ActivityIdealMolar,
    ActivityKappaParameter,
)


# There is a bit of work in the constructor, but it is necessary to set the
# types of the builders and strategies correctly.
class ActivityFactory(
    StrategyFactory[
        Union[
            ActivityIdealMassBuilder,
            ActivityIdealMolarBuilder,
            ActivityKappaParameterBuilder,
        ],
        Union[ActivityIdealMass, ActivityIdealMolar, ActivityKappaParameter],
    ]
):
    """Factory class to create activity strategy builders

    Factory class to create activity strategy builders for calculating
    activity and partial pressure of species in a mixture of liquids.

    Methods:
        get_builders(): Returns the mapping of strategy types to builder
        instances.
        get_strategy(strategy_type, parameters): Gets the strategy instance
        for the specified strategy type.
            strategy_type: Type of activity strategy to use, can be
            'mass_ideal' (default), 'molar_ideal', or 'kappa_parameter'.
            parameters(Dict[str, Any], optional): Parameters required for the
            builder, dependent on the chosen strategy type.
                mass_ideal: No parameters are required.
                molar_ideal: molar_mass
                kappa | kappa_parameter: kappa, density, molar_mass,
                water_index

    Returns:
        ActivityStrategy: An instance of the specified ActivityStrategy.

    Raises:
        ValueError: If an unknown strategy type is provided.
        ValueError: If any required key is missing during check_keys or
            pre_build_check, or if trying to set an invalid parameter.

    Example:
    >>> strategy_is = ActivityFactory().get_strategy("mass_ideal")
    """

    def get_builders(self):
        """Returns the mapping of strategy types to builder instances.

        Returns:
            Dict[str, Any]: A dictionary mapping strategy types to builder
            instances.
                mass_ideal: IdealActivityMassBuilder
                molar_ideal: IdealActivityMolarBuilder
                kappa_parameter: KappaParameterActivityBuilder
        """
        return {
            "mass_ideal": ActivityIdealMassBuilder(),
            "molar_ideal": ActivityIdealMolarBuilder(),
            "kappa_parameter": ActivityKappaParameterBuilder(),
            "kappa": ActivityKappaParameterBuilder(),
        }
