"""Builder class for Activity objects with validation and error handling.

Change to MixinMolar classes, after PR integration.
"""

import logging
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray

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
        return ActivityIdealMolar(molar_mass=self.molar_mass)


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
