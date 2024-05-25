"""Abstract Base Class for Builder classes"""

from abc import ABC, abstractmethod
from typing import Any, Optional
import logging

logger = logging.getLogger("particula")


class BuilderABC(ABC):
    """Abstract base class for builders with common methods to check keys and
    set parameters from dict.

    Attributes:
    ----------
    - required_parameters (list): List of required parameters for the builder.

    Methods:
    -------
    - check_keys(parameters): Check if the keys you want to set are present in
        the self.required_parameters dictionary.
    - set_parameters(parameters): Set parameters from a dictionary including
        optional suffix for units as '_units'.
    - pre_build_check(): Check if all required attribute parameters are set
        before building.

    Abstract Methods:
    -----------------
    - build(): Build and return the strategy object with the set parameters.

    Raises:
    ------
    - ValueError: If any required key is missing. During check_keys and
        pre_build_check. Or if trying to set an invalid parameter.
    - Warning: If using default units for any parameter.
    """

    def __init__(
        self,
        required_parameters: Optional[list[str]] = None
    ):
        self.required_parameters = required_parameters or []

    def check_keys(
        self,
        parameters: dict[str, Any],
    ):
        """Check if the keys you want to set are present in the
        self.required_parameters dictionary.

        Args:
        ----
        - parameters (dict): The parameters dictionary to check.
        - required_keys (list): List of required keys to be checked in the
        parameters.

        Returns:
        -------
        - None

        Raises:
        ------
        - ValueError: If you are trying to set an invalid parameter.
        """
        # check if all required keys are present
        missing = [p for p in self.required_parameters if p not in parameters]
        if missing:
            logger.error(
                "Missing required parameter(s): %s", ', '.join(missing))
            raise ValueError(
                f"Missing required parameter(s): {', '.join(missing)}")
        # check if all keys in parameters are valid, accounts _units suffix
        valid_keys = set(
            self.required_parameters
            + [f"{key}_units" for key in self.required_parameters]
        )
        key_to_set = [key for key in parameters
                      if key not in valid_keys]
        if key_to_set:
            logger.error(
                "Trying to set an invalid parameter(s) '%s'. "
                "The valid parameter(s) '%s'.",
                key_to_set, valid_keys
            )
            raise ValueError(
                f"Trying to set an invalid parameter(s) '{key_to_set}'."
                f" The valid parameter(s) '{valid_keys}'."
            )

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
        self.check_keys(parameters)  # check if all required keys are present
        for key in self.required_parameters:  # set the parameters
            unit_key = f'{key}_units'
            if unit_key in parameters:
                # build the set call set with units, from keys
                # e.g. self.set_a(params['a'], params['a_units'])
                getattr(self, f'set_{key}')(
                    parameters[key], parameters[unit_key]
                )
            else:
                logger.warning(
                    "Using default units for parameter: '%s'.", key)
                # build set call, e.g. self.set_a(params['a'])
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
        missing = [p for p in self.required_parameters
                   if getattr(self, p) is None]
        if missing:
            logger.error(
                "Required parameter(s) not set: %s", ', '.join(missing))
            raise ValueError(
                f"Required parameter(s) not set: {', '.join(missing)}")

    @abstractmethod
    def build(self) -> Any:
        """Build and return the strategy object with the set parameters.

        Returns:
        -------
        - strategy: The built strategy object.
        """
