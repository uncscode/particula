"""Activity strategy factories for calculating activity and partial pressure
of species in a mixture of liquids."""

from typing import Optional, Any, Dict
import logging
from particula.next.particles.activity_strategies import (
    ActivityStrategy)
from particula.next.particles.activity_builders import (
    IdealActivityMassBuilder, IdealActivityMolarBuilder,
    KappaParameterActivityBuilder
)

logger = logging.getLogger("particula")


def activity_factory(
    strategy_type: str = "mass_ideal",
    parameters: Optional[Dict[str, Any]] = None
) -> ActivityStrategy:
    """
    Factory function to create activity strategy builders for calculating
    activity and partial pressure of species in a mixture of liquids.

    Args:
    ----
    - strategy_type (str): Type of activity strategy to use, with options:
        'mass_ideal' (default), 'molar_ideal', or 'kappa_parameter'.
    - parameters (Dict[str, Any], optional): Parameters required for the
    builder, dependent on the chosen strategy type.
        - mass_ideal: No parameters are required.
        - molar_ideal: molar_mass
        - kappa|kappa_parameter: kappa, density, molar_mass, water_index

    Returns:
    - ActivityStrategy: An instance of the specified ActivityStrategy.

    Raises:
    - ValueError: If an unknown strategy type is provided.
    - ValueError: If any required key is missing during check_keys or
        pre_build_check, or if trying to set an invalid parameter.
    """
    builder_map = {
        "mass_ideal": IdealActivityMassBuilder(),
        "molar_ideal": IdealActivityMolarBuilder(),
        "kappa_parameter": KappaParameterActivityBuilder()
    }
    builder = builder_map.get(strategy_type.lower())
    if builder is None:
        message = f"Unknown strategy type: {strategy_type}"
        logger.error(message)
        raise ValueError(message)

    # Set the parameters for the builder
    if parameters and hasattr(builder, 'set_parameters'):
        builder.set_parameters(parameters)

    return builder.build()  # build the activity strategy
