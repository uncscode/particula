"""Abstract Base Class for Builder classes.
"""

# pylint: disable=too-few-public-methods

from abc import ABC, abstractmethod
from typing import Any, Optional
import logging

logger = logging.getLogger("particula")


class BuilderABC(ABC):
    """Abstract base class for builders with common methods to check keys and
    set parameters from a dictionary.

    Args:
        - required_parameters: List of required parameters for the builder.

    Methods:
        - check_keys (parameters): Check if the keys you want to set are
        - present in the parameters dictionary.
        - set_parameters (parameters): Set parameters from a dictionary
            including optional suffix for units as '_units'.
        - pre_build_check(): Check if all required attribute parameters are set
            before building.
        - build (abstract): Build and return the strategy object.

    Raises:
        - ValueError : If any required key is missing during check_keys or
            pre_build_check, or if trying to set an invalid parameter.
        - Warning : If using default units for any parameter.

    References:
        - Builder Pattern : https://refactoring.guru/design-patterns/builder
    """

    def __init__(self, required_parameters: Optional[list[str]] = None):
        self.required_parameters = required_parameters or []

    def check_keys(self, parameters: dict[str, Any]):
        """Check if the keys are present and valid.

        Args:
            - parameters: The parameters dictionary to check.

        Raises:
            - ValueError: If any required key is missing or if trying to set an
                invalid parameter.
        """

        # Check if all required keys are present
        if missing := [
            p for p in self.required_parameters if p not in parameters
        ]:
            error_message = (
                f"Missing required parameter(s): {', '.join(missing)}"
            )
            logger.error(error_message)
            raise ValueError(error_message)

        # Generate a set of all valid keys
        valid_keys = set(
            self.required_parameters
            + [f"{key}_units" for key in self.required_parameters]
        )
        # Check for any invalid keys and handle them within the if condition
        if invalid_keys := [
            key for key in parameters if key not in valid_keys
        ]:
            error_message = (
                f"Trying to set an invalid parameter(s) '{invalid_keys}'. "
                f"The valid parameter(s) '{valid_keys}'."
            )
            logger.error(error_message)
            raise ValueError(error_message)

    def set_parameters(self, parameters: dict[str, Any]):
        """Set parameters from a dictionary including optional suffix for
        units as '_units'.

        Args:
            - parameters : The parameters dictionary to set.

        Returns:
            - The builder object with the set parameters.

        Raises:
            - ValueError : If any required key is missing.
            - Warning : If using default units for any parameter.
        """
        self.check_keys(parameters)
        for key in self.required_parameters:
            unit_key = f"{key}_units"
            if unit_key in parameters:
                # Call set method with units
                getattr(self, f"set_{key}")(
                    parameters[key], parameters[unit_key]
                )
            else:
                logger.warning("Using default units for parameter: '%s'.", key)
                # Call set method without units
                getattr(self, f"set_{key}")(parameters[key])
        return self

    def pre_build_check(self):
        """Check if all required attribute parameters are set before building.

            Raises:
                - ValueError : If any required parameter is missing.
        """
        if missing := [
            p for p in self.required_parameters if getattr(self, p) is None
        ]:
            error_message = (
                f"Required parameter(s) not set: {', '.join(missing)}"
            )
            logger.error(error_message)
            raise ValueError(error_message)

    @abstractmethod
    def build(self) -> Any:
        """Build and return the strategy object with the set parameters.

        Returns:
            - strategy : The built strategy object.
        """
