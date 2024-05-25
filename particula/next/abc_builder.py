"""Abstract Base Class for Builder classes"""

from abc import ABC, abstractmethod
from typing import Any, Optional
import logging

logger = logging.getLogger("particula")


class BuilderABC(ABC):
    """Abstract base class for builders with common methods to check keys and
    set parameters from a dictionary.

    Attributes:
    ----------
    - required_parameters (list): List of required parameters for the builder.

    Methods:
    -------
    - check_keys(parameters): Check if the keys you want to set are
    present in the parameters dictionary.
    - set_parameters(parameters): Set parameters from a dictionary including
        optional suffix for units as '_units'.
    - pre_build_check(): Check if all required attribute parameters are set
        before building.

    Abstract Methods:
    -----------------
    - build(): Build and return the strategy object with the set parameters.

    Raises:
    ------
    - ValueError: If any required key is missing during check_keys or
        pre_build_check, or if trying to set an invalid parameter.
    - Warning: If using default units for any parameter.
    """

    def __init__(self, required_parameters: Optional[list[str]] = None):
        self.required_parameters = required_parameters or []

    def check_keys(self, parameters: dict[str, Any]):
        """Check if the keys you want to set are present in the
        parameters dictionary and if all keys are valid.

        Args:
        ----
        - parameters (dict): The parameters dictionary to check.

        Returns:
        -------
        - None

        Raises:
        ------
        - ValueError: If any required key is missing or if trying to set an
        invalid parameter.
        """
        # Check if all required keys are present
        missing = [p for p in self.required_parameters if p not in parameters]
        if missing:
            logger.error(
                "Missing required parameter(s): %s",
                ', '.join(missing))
            raise ValueError(
                f"Missing required parameter(s): {', '.join(missing)}")

        # Check if all keys in parameters are valid, accounting for _units
        # suffix
        valid_keys = set(self.required_parameters +
                         [f"{key}_units" for key in self.required_parameters])
        invalid_keys = [key for key in parameters if key not in valid_keys]
        if invalid_keys:
            logger.error(
                "Trying to set an invalid parameter(s) '%s'. "
                "The valid parameter(s) '%s'.",
                invalid_keys,
                valid_keys)
            raise ValueError(
                f"Trying to set an invalid parameter(s) '{invalid_keys}'. "
                f"The valid parameter(s) '{valid_keys}'.")

    def set_parameters(self, parameters: dict[str, Any]):
        """Set parameters from a dictionary including optional suffix for
        units as '_units'.

        Args:
        ----
        - parameters (dict): The parameters dictionary to set.

        Returns:
        -------
        - self: The builder object with the set parameters.

        Raises:
        ------
        - ValueError: If any required key is missing.
        - Warning: If using default units for any parameter.
        """
        self.check_keys(parameters)
        for key in self.required_parameters:
            unit_key = f'{key}_units'
            if unit_key in parameters:
                # Call set method with units
                getattr(
                    self,
                    f'set_{key}')(
                    parameters[key],
                    parameters[unit_key])
            else:
                logger.warning("Using default units for parameter: '%s'.", key)
                # Call set method without units
                getattr(self, f'set_{key}')(parameters[key])
        return self

    def pre_build_check(self):
        """Check if all required attribute parameters are set before building.

        Returns:
        -------
        - None

        Raises:
        ------
        - ValueError: If any required parameter is missing.
        """
        missing = [
            p for p in self.required_parameters if getattr(
                self, p) is None]
        if missing:
            logger.error(
                "Required parameter(s) not set: %s",
                ', '.join(missing))
            raise ValueError(
                f"Required parameter(s) not set: {', '.join(missing)}")

    @abstractmethod
    def build(self) -> Any:
        """Build and return the strategy object with the set parameters.

        Returns:
        -------
        - strategy: The built strategy object.
        """
