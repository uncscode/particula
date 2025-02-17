"""
Builder class for coagulation strategies. This class is used to create
coagulation strategies based on the specified distribution type and kernel
strategy. This provides a validation layer to ensure that the correct values
are passed to the coagulation strategy.
"""

# pylint: disable=too-few-public-methods

from typing import Optional
import logging

logger = logging.getLogger("particula")


class BuilderDistributionTypeMixin:
    """Mixin class for distribution type in coagulation strategies.

    This mixin class is used to set the distribution type for coagulation
    strategies. It provides a validation layer to ensure that the correct
    distribution type is passed to the coagulation strategy.

    Methods:
        set_distribution_type(distribution_type): Set the distribution type.
    """

    def __init__(self):
        self.distribution_type = None

    def set_distribution_type(
        self,
        distribution_type: str,
        distribution_type_units: Optional[str] = None,
    ):
        """Set the distribution type.

        Args:
            distribution_type : The distribution type to be set.
                Options are "discrete", "continuous_pdf", "particle_resolved".
            distribution_type_units : Not used.

        Raises:
            ValueError: If the distribution type is not valid.
        """
        valid_distribution_types = [
            "discrete",
            "continuous_pdf",
            "particle_resolved",
        ]
        if distribution_type not in valid_distribution_types:
            message = (
                f"Invalid distribution type: {distribution_type}. "
                f"Valid types are: {valid_distribution_types}."
            )
            logger.error(message)
            raise ValueError(message)
        if distribution_type_units is not None:
            message = (
                f"Units for distribution type are not used. "
                f"Received: {distribution_type_units}."
            )
            logger.warning(message)
        self.distribution_type = distribution_type
        return self
class BuilderTurbulentShearMixin:
    """Mixin class for turbulent shear parameters.

    This mixin class is used to set the turbulent dissipation and fluid
    density for turbulent shear coagulation strategies. It provides a
    validation layer to ensure that the correct values are passed.
    """
    def __init__(self):
        BuilderTurbulentShearMixin.__init__(self)
