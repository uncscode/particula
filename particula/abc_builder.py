"""Abstract Base Class for Builder classes.

References:
    - Builder Pattern : https://refactoring.guru/design-patterns/builder

"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

logger = logging.getLogger("particula")


class BuilderABC(ABC):
    """Abstract base class for builders to check keys and set parameters.

    Attributes:
        - required_parameters: List of required parameters for the builder.

    Raises:
        - ValueError: If any required key is missing during check_keys or
          pre_build_check, or if trying to set an invalid parameter.
        - Warning: If using default units for any parameter.

    Examples:
        ```py
        class MyBuilder(BuilderABC):
            def set_parameter1(self, value, units=None):
                ...
            def set_parameter2(self, value, units=None):
                ...
            def build(self):
                return SomeStrategy()

        strategy = (
            MyBuilder()
            .set_parameters1(10, 'm')
            .set_parameters2(20, 's')
            .build()
        )
        ```

    References:
        - "Builder Pattern,"
        [Refactoring Guru](https://refactoring.guru/design-patterns/builder)
    """

    def __init__(self, required_parameters: Optional[list[str]] = None):
        """Initialize builder with required parameters."""
        self.required_parameters = required_parameters or []

    def check_keys(self, parameters: dict[str, Any]):
        """Check if the keys are present and valid.

        Arguments:
            - parameters: The parameters dictionary to check.

        Raises:
            - ValueError: If any required key is missing or if trying to set
              an invalid parameter.

        Examples:
            ```py
            builder = Builder()
            builder.check_keys({
                "parameter1": 1,
                "parameter2": 2,
            })
            ```
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
        if invalid_keys := [key for key in parameters if key not in valid_keys]:
            error_message = (
                f"Trying to set an invalid parameter(s) '{invalid_keys}'. "
                f"The valid parameter(s) '{valid_keys}'."
            )
            logger.error(error_message)
            raise ValueError(error_message)

    def set_parameters(self, parameters: dict[str, Any]):
        """Set parameters from a dictionary, handling any '_units' suffix.

        Arguments:
            - parameters: The parameters dictionary to set.

        Returns:
            BuilderABC: This builder object with the set parameters.

        Raises:
            - ValueError: If any required key is missing.
            - Warning: If using default units for any parameter.

        Examples:
            ```py
            builder = Builder().set_parameters({
                "parameter1": 1,
                "parameter2": 2,
                "parameter2_units": "K",
            })
            ```
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
            - ValueError: If any required parameter is missing.

        Examples:
            ```py
            builder = Builder()
            builder.pre_build_check()
            ```
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
            Any: The built strategy object.

        Examples:
            ```py
            builder = Builder()
            strategy = builder.build()
            ```
        """
