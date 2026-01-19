"""Builder class for Activity objects with validation and error handling.

Change to MixinMolar classes, after PR integration.
"""

import logging
import sys
import warnings
from typing import Any, List, Optional, Union

import numpy as np
from numpy.typing import NDArray

# Self was added to typing in Python 3.11
if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from particula.abc_builder import (
    BuilderABC,
)
from particula.builder_mixin import (
    BuilderDensityMixin,
    BuilderMolarMassMixin,
)
from particula.particles.activity_strategies import (
    ActivityIdealMass,
    ActivityIdealMolar,
    ActivityKappaParameter,
    ActivityNonIdealBinary,
    ActivityStrategy,
)

logger = logging.getLogger("particula")


class ActivityIdealMassBuilder(BuilderABC):
    """Builds an ActivityIdealMass object for calculating activity based on
    ideal mass fractions.

    A concise builder for ActivityIdealMass. This class requires no extra
    parameters beyond the defaults. It ensures the returned strategy follows
    Raoult's Law for mass-based activities.

    Methods:
        - build: Validates any required parameters and returns the strategy.

    Examples:
        ```py title="Example Usage"
        import particula as par
        builder = par.particles.ActivityIdealMassBuilder()
        strategy = builder.build()
        result = strategy.activity([1.0, 2.0, 3.0])
        # result -> ...
        ```

    References:
    - "Raoult's Law,"
        [Wikipedia](https://en.wikipedia.org/wiki/Raoult%27s_law).
    """

    def __init__(self):
        """Initialize the ActivityIdealMass builder.

        Sets up the builder with no required parameters for creating
        an ActivityIdealMass strategy.
        """
        required_parameters = None
        BuilderABC.__init__(self, required_parameters)

    def build(self) -> ActivityStrategy:
        """Validate and return an ActivityIdealMass strategy instance.

        Returns:
            - ActivityIdealMass : The validated strategy for
              ideal mass-based activity calculations.

        Examples:
            ```py title="Build Method Example"
            builder = par.particles.ActivityIdealMassBuilder()
            mass_activity_strategy = builder.build()
            # Use mass_activity_strategy.activity(...)
            ```
        """
        return ActivityIdealMass()


class ActivityIdealMolarBuilder(BuilderABC, BuilderMolarMassMixin):
    """Builds an ActivityIdealMolar object for calculating activity from
    ideal mole fractions.

    This builder sets up any required parameters (e.g., molar mass) and
    creates an ActivityIdealMolar strategy. Uses Raoult's Law in terms
    of mole fraction.

    Attributes:
        - molar_mass : Molar mass for each species, in kilograms per mole.

    Methods:
        - set_molar_mass: Assigns the molar masses (with unit
            handling).
        - set_parameters: Batch-assign parameters from a dictionary.
        - build: Finalizes the builder and returns the strategy.

    Examples:
        ```py title="Example Usage"
        import particula as par
        builder = (
            par.particles.ActivityIdealMolarBuilder()
            .set_molar_mass(0.01815, "kg/mol")
        )
        strategy = builder.build()
        result = strategy.activity([1.0, 2.0, 3.0])
        # result -> ...
        ```

    References:
        - "Raoult's Law,"
        [Wikipedia](https://en.wikipedia.org/wiki/Raoult%27s_law).
    """

    def __init__(self):
        """Initialize the ActivityIdealMolar builder.

        Sets up the builder with required molar_mass parameter for
        creating an ActivityIdealMolar strategy.
        """
        required_parameters = ["molar_mass"]
        BuilderABC.__init__(self, required_parameters)
        BuilderMolarMassMixin.__init__(self)

    def build(self) -> ActivityStrategy:
        """Validate parameters and create an ActivityIdealMolar strategy.

        Ensures molar_mass is properly configured before building.

        Returns:
            - ActivityIdealMolar : An ideal strategy based on mole fractions.

        Examples:
            ```py title="Build Method Example"
            builder = (
                par.particles.ActivityIdealMolarBuilder()
                .set_molar_mass(0.028, "kg/mol")
            )
            molar_activity_strategy = builder.build()
            # molar_activity_strategy.activity(...)
            ```
        """
        self.pre_build_check()
        molar_mass_value = self.molar_mass
        if molar_mass_value is None:
            error_message = (
                "Required parameter 'molar_mass' not set before building."
            )
            logger.error(error_message)
            raise ValueError(error_message)
        return ActivityIdealMolar(molar_mass=molar_mass_value)


class ActivityKappaParameterBuilder(
    BuilderABC, BuilderDensityMixin, BuilderMolarMassMixin
):
    """Builds an ActivityKappaParameter object for non-ideal activity
    calculations.

    This builder requires kappa, density, molar_mass, and water_index.
    Kappa is the hygroscopicity parameter, used to capture non-ideal
    behavior. The optional water_index identifies which species is water.

    Attributes:
        - kappa : NDArray of kappa parameters for each species.
        - density : NDArray of densities, in kilograms per cubic meter.
        - molar_mass : NDArray of molar masses, in kilograms per mole.
        - water_index : Integer index of the water species.

    Methods:
        - set_kappa: Assigns kappa values (must be nonnegative).
        - set_water_index: Sets the index of the water species.
        - set_density: Assigns density values (with unit handling).
        - set_molar_mass: Assigns molar mass values (with unit
            handling).
        - set_parameters: Batch-assign parameters from a dictionary.
        - build: Finalizes checks and returns the strategy.

    Examples:
        ```py title="Example Usage"
        import particula as par
        import numpy as np

        builder = (
            par.particles.ActivityKappaParameterBuilder()
            .set_kappa(np.array([0.1, 0.0]))
            .set_density(np.array([1000, 1200]), "kg/m^3"))
            .set_molar_mass(np.array([0.018, 0.058]), "kg/mol")
            .set_water_index(0)
        )
        strategy = builder.build()
        result = strategy.activity(np.array([1.0, 2.0]))
        # result -> ...
        ```

    References:
        - Petters, M. D., and Kreidenweis, S. M. (2007).
          "A single parameter representation of hygroscopic growth and
           cloud condensation nucleus activity," Atmospheric Chemistry
           and Physics, 7(8), 1961–1971.
           [DOI](https://doi.org/10.5194/acp-7-1961-2007)
    """

    def __init__(self):
        """Initialize the ActivityKappaParameter builder.

        Sets up the builder with required parameters for creating an
        ActivityKappaParameter strategy, including kappa, density,
        molar mass, and water index.
        """
        required_parameters = ["kappa", "density", "molar_mass", "water_index"]
        BuilderABC.__init__(self, required_parameters)
        BuilderDensityMixin.__init__(self)
        BuilderMolarMassMixin.__init__(self)
        self.kappa = None
        self.water_index = None

    def set_kappa(
        self,
        kappa: Union[float, NDArray[np.float64]],
        kappa_units: Optional[str] = None,
    ):
        """Set the kappa parameter for the activity calculation.

        Args:
            kappa: The kappa parameter for the activity calculation.
            kappa_units: Not used. (for interface consistency)
        """
        if np.any(kappa < 0):
            error_message = "Kappa parameter must be a positive value."
            logger.error(error_message)
            raise ValueError(error_message)
        if kappa_units is not None:
            logger.warning("Ignoring units for kappa parameter.")
        self.kappa = kappa
        return self

    def set_water_index(
        self, water_index: int, water_index_units: Optional[str] = None
    ):
        """Set the array index of the species.

        Args:
            water_index: The array index of the species.
            water_index_units: Not used. (for interface consistency)
        """
        if not isinstance(water_index, int):  # type: ignore
            error_message = "Water index must be an integer."
            logger.error(error_message)
            raise TypeError(error_message)
        if water_index_units is not None:
            logger.warning("Ignoring units for water index.")
        self.water_index = water_index
        return self

    def build(self) -> ActivityStrategy:
        """Validate parameters and build ActivityKappaParameter strategy.

        Returns:
            - ActivityKappaParameter : The non-ideal activity strategy
              utilizing the kappa hygroscopic parameter.

        Examples:
            ```py title="Build Method Example"
            kappa_activity_strategy = (
                par.particles.ActivityKappaParameterBuilder()
                .set_kappa([0.1, 0.2])
                .set_density([1000, 1200], "kg/m^3")
                .set_molar_mass([0.018, 0.046], "kg/mol")
                .set_water_index(0)
                .build()
            )
            # kappa_activity_strategy ...
            ```
        """
        self.pre_build_check()
        return ActivityKappaParameter(
            kappa=self.kappa,  # type: ignore
            density=self.density,  # type: ignore
            molar_mass=self.molar_mass,  # type: ignore
            water_index=self.water_index,  # type: ignore
        )


class ActivityNonIdealBinaryBuilder(
    BuilderABC, BuilderMolarMassMixin, BuilderDensityMixin
):
    """Builder for ActivityNonIdealBinary strategy using BAT model.

    Provides a fluent interface to configure the binary non-ideal
    activity strategy with validation of required parameters and optional
    functional group metadata.

    Required parameters (via setters or ``set_parameters``):
        - molar_mass: Organic molar mass in kg/mol.
        - oxygen2carbon: Oxygen to carbon atomic ratio (dimensionless).
        - density: Organic density in kg/m^3.

    Optional parameters:
        - functional_group: Functional group identifier string or list.

    Examples:
        >>> import particula as par
        >>> builder = (
        ...     par.particles.ActivityNonIdealBinaryBuilder()
        ...     .set_molar_mass(0.200, "kg/mol")
        ...     .set_oxygen2carbon(0.4)
        ...     .set_density(1400.0, "kg/m^3")
        ...     .set_functional_group("carboxylic_acid")
        ... )
        >>> strategy = builder.build()
        >>> strategy.get_name()
        'ActivityNonIdealBinary'
    """

    def __init__(self) -> None:
        """Initialize the builder with required parameters."""
        required_parameters = ["molar_mass", "oxygen2carbon", "density"]
        BuilderABC.__init__(self, required_parameters)
        BuilderMolarMassMixin.__init__(self)
        BuilderDensityMixin.__init__(self)
        self.molar_mass: Optional[Union[float, NDArray[np.float64]]] = None
        self.density: Optional[Union[float, NDArray[np.float64]]] = None
        self.oxygen2carbon: Optional[float] = None
        self.functional_group: Optional[Union[str, List[str]]] = None

    @staticmethod
    def _to_scalar(value: Union[float, NDArray[np.float64]]) -> float:
        """Convert scalar-like input to float, rejecting multi-value arrays."""
        array_value = np.asarray(value, dtype=np.float64)
        if array_value.size != 1:
            error_message = "Expected a scalar value for parameter assignment."
            logger.error(error_message)
            raise ValueError(error_message)
        return float(array_value.item())

    def set_oxygen2carbon(
        self,
        oxygen2carbon: Union[float, NDArray[np.float64]],
        oxygen2carbon_units: Optional[str] = None,
    ) -> Self:
        """Set oxygen-to-carbon ratio ensuring nonnegativity.

        Args:
            oxygen2carbon: Oxygen to carbon atomic ratio (>=0).
            oxygen2carbon_units: Ignored (dimensionless); warns if provided.

        Returns:
            Self for fluent chaining.

        Raises:
            ValueError: If ``oxygen2carbon`` contains negative values.
        """
        array_value = np.asarray(oxygen2carbon, dtype=np.float64)
        if np.any(array_value < 0):
            error_message = "Oxygen to carbon ratio must be nonnegative."
            logger.error(error_message)
            raise ValueError(error_message)
        if oxygen2carbon_units is not None:
            warnings.warn(
                "Ignoring units for oxygen2carbon (dimensionless).",
                UserWarning,
                stacklevel=2,
            )
        self.oxygen2carbon = self._to_scalar(array_value)
        return self

    def set_functional_group(
        self,
        functional_group: Optional[Union[str, List[str]]],
        functional_group_units: Optional[str] = None,
    ) -> Self:
        """Set optional functional group identifier(s).

        Args:
            functional_group: Functional group value; accepts None, str,
                or list.
            functional_group_units: Ignored; warns if provided.


        Returns:
            Self for fluent chaining.
        """
        if functional_group_units is not None:
            warnings.warn(
                "Ignoring units for functional_group.",
                UserWarning,
                stacklevel=2,
            )
        self.functional_group = functional_group
        return self

    def set_parameters(
        self, parameters: dict[str, Any]
    ) -> "ActivityNonIdealBinaryBuilder":
        """Batch assign parameters with optional units and validation."""
        missing = [p for p in self.required_parameters if p not in parameters]
        if missing:
            error_message = (
                f"Missing required parameter(s): {', '.join(missing)}"
            )
            logger.error(error_message)
            raise ValueError(error_message)

        valid_keys = set(
            self.required_parameters
            + [f"{key}_units" for key in self.required_parameters]
            + ["functional_group", "functional_group_units"]
        )
        invalid_keys = [key for key in parameters if key not in valid_keys]
        if invalid_keys:
            error_message = (
                f"Trying to set an invalid parameter(s) '{invalid_keys}'. "
                f"The valid parameter(s) '{valid_keys}'."
            )
            logger.error(error_message)
            raise ValueError(error_message)

        if "molar_mass_units" in parameters:
            self.set_molar_mass(
                parameters["molar_mass"], parameters["molar_mass_units"]
            )
        else:
            logger.warning("Using default units for parameter: 'molar_mass'.")
            self.set_molar_mass(parameters["molar_mass"], "kg/mol")

        if "oxygen2carbon_units" in parameters:
            self.set_oxygen2carbon(
                parameters["oxygen2carbon"],
                parameters["oxygen2carbon_units"],
            )
        else:
            logger.warning(
                "Using default units for parameter: 'oxygen2carbon'."
            )
            self.set_oxygen2carbon(parameters["oxygen2carbon"])

        if "density_units" in parameters:
            self.set_density(parameters["density"], parameters["density_units"])
        else:
            logger.warning("Using default units for parameter: 'density'.")
            self.set_density(parameters["density"], "kg/m^3")

        if "functional_group" in parameters:
            self.set_functional_group(
                parameters["functional_group"],
                parameters.get("functional_group_units"),
            )

        return self

    def build(self) -> ActivityNonIdealBinary:
        """Validate required inputs then build the strategy."""
        self.pre_build_check()
        if self.molar_mass is None:
            error_message = (
                "Required parameter 'molar_mass' not set before building."
            )
            logger.error(error_message)
            raise ValueError(error_message)
        if self.oxygen2carbon is None:
            error_message = (
                "Required parameter 'oxygen2carbon' not set before building."
            )
            logger.error(error_message)
            raise ValueError(error_message)
        if self.density is None:
            error_message = (
                "Required parameter 'density' not set before building."
            )
            logger.error(error_message)
            raise ValueError(error_message)
        molar_mass_value = self._to_scalar(self.molar_mass)
        oxygen2carbon_value = self._to_scalar(self.oxygen2carbon)
        density_value = self._to_scalar(self.density)
        return ActivityNonIdealBinary(
            molar_mass=molar_mass_value,
            oxygen2carbon=oxygen2carbon_value,
            density=density_value,
            functional_group=self.functional_group,
        )
