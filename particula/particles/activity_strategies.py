"""Common interface for activity calculations.

Class strategies for activities and vapor pressure over mixture of liquids
surface Using Raoult's Law, and strategies ideal, non-ideal, kappa hygroscopic
parameterizations.
"""

# pyright: reportArgumentType=false

from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from numpy.typing import NDArray

from particula.particles.properties.activity_module import (
    get_ideal_activity_mass,
    get_ideal_activity_molar,
    get_ideal_activity_volume,
    get_kappa_activity,
    get_surface_partial_pressure,
)


class ActivityStrategy(ABC):
    """Abstract base class for vapor pressure and activity calculations.

    This interface is used by subclasses for computing particle activity
    and partial pressures. General methods include activity() and
    partial_pressure().

    Attributes:
        - None

    Methods:
    - get_name : Return the type of the activity strategy.
    - activity : Calculate the activity of a species. (abstract method)
    - partial_pressure : Calculate the partial pressure of a species
        using its pure vapor pressure and computed activity.

    Examples:
        ```py title="Example Subclass"
        class CustomActivity(ActivityStrategy):
            def activity(self, mass_concentration):
                return 1.0

        my_activity = CustomActivity()
        pvap = my_activity.partial_pressure(101325.0, 1.0)
        # pvap -> 101325.0
        ```

    References:
    - "Vapor Pressure,"
        [Wikipedia](https://en.wikipedia.org/wiki/Vapor_pressure).
    """

    @abstractmethod
    def activity(
        self, mass_concentration: Union[float, NDArray[np.float64]]
    ) -> Union[float, NDArray[np.float64]]:
        """Calculate the activity of a species based on its mass concentration.

        Arguments:
            - mass_concentration : Concentration of the species in kg/m^3.

        Returns:
            - Activity of the species, unitless.
        """

    def get_name(self) -> str:
        """Return the type of the activity strategy."""
        return self.__class__.__name__

    def partial_pressure(
        self,
        pure_vapor_pressure: Union[float, NDArray[np.float64]],
        mass_concentration: Union[float, NDArray[np.float64]],
    ) -> Union[float, NDArray[np.float64]]:
        """Calculate the vapor pressure of species in the particle phase.

        Arguments:
            - pure_vapor_pressure : Pure vapor pressure of the species in Pa.
            - mass_concentration : Concentration of the species in kg/m^3.

        Returns:
            - Vapor pressure of the particle in Pa.
        """
        return get_surface_partial_pressure(
            pure_vapor_pressure=pure_vapor_pressure,
            activity=self.activity(mass_concentration),
        )


class ActivityIdealMolar(ActivityStrategy):
    """Calculate ideal activity based on mole fractions (Raoult's Law).

    Attributes:
        - molar_mass : Molar mass of the species in kg/mol.

    Methods:
        - activity : Computes ideal activity from mass concentration
          and molar mass.

    Examples:
        ```py title="Example Usage"
        import particula as par
        strategy = par.particles.ActivityIdealMolar(molar_mass=0.018)
        # mass_concentration in kg/m^3
        a = strategy.activity(np.array([1.2, 2.5, 3.0]))
        # a -> ...
        ```

    References:
        - "Raoult's Law,"
        [Wikipedia](https://en.wikipedia.org/wiki/Raoult%27s_law).
    """

    def __init__(self, molar_mass: Union[float, NDArray[np.float64]] = 0.0):
        """Initialize the ActivityIdealMolar strategy.

        Arguments:
            - molar_mass : Molar mass in kg/mol.
        """
        self.molar_mass = molar_mass

    def activity(
        self, mass_concentration: Union[float, NDArray[np.float64]]
    ) -> Union[float, NDArray[np.float64]]:
        """Calculate the activity of a species based on mass concentration.

        Arguments:
            - mass_concentration : Concentration of the species in kg/m^3.

        Returns:
            - Activity of the species, unitless.
        """
        return get_ideal_activity_molar(
            mass_concentration=mass_concentration, molar_mass=self.molar_mass
        )


class ActivityIdealMass(ActivityStrategy):
    """Calculate ideal activity based on mass fractions (Raoult's Law).

    Attributes:
        - None

    Methods:
    - activity : Computes activity from mass concentration,
        treating mass fractions as ideal.

    Examples:
        ```py title="Example Usage"
        import particula as par
        strategy = par.particles.ActivityIdealMass()
        a = strategy.activity([0.5, 1.0, 1.5])
        # a -> ...
        ```

    References:
    - "Raoult's Law,"
        [Wikipedia](https://en.wikipedia.org/wiki/Raoult%27s_law).
    """

    def activity(
        self, mass_concentration: Union[float, NDArray[np.float64]]
    ) -> Union[float, NDArray[np.float64]]:
        """Calculate the activity of a species based on mass concentration.

        Arguments:
            - mass_concentration : Concentration of the species in kg/m^3.

        Returns:
            - Activity of the species, unitless.
        """
        return get_ideal_activity_mass(mass_concentration=mass_concentration)


class ActivityIdealVolume(ActivityStrategy):
    """Calculate ideal activity based on volume fractions (Raoult's Law).

    Attributes:
        - density : The density of the species in kg/m^3, used to
          derive volume fractions from mass concentrations.

    Methods:
    - activity : Computes activity from mass concentration and density.

    Examples:
        ```py title="Example Usage"
        strategy = ActivityIdealVolume(density=1000.0)
        a = strategy.activity(2.5)
        # a -> ...
        ```

    References:
    - "Raoult's Law,"
        [Wikipedia](https://en.wikipedia.org/wiki/Raoult%27s_law).
    """

    def __init__(self, density: Union[float, NDArray[np.float64]] = 0.0):
        """Initialize the ActivityIdealVolume strategy.

        Arguments:
            - density : Density of the species in kg/m^3.
        """
        self.density = density

    def activity(
        self, mass_concentration: Union[float, NDArray[np.float64]]
    ) -> Union[float, NDArray[np.float64]]:
        """Calculate the activity of a species based on mass concentration.

        Arguments:
            - mass_concentration : Concentration of the species in kg/m^3.
            - density : Density of the species in kg/m^3.

        Returns:
            - Activity of the species, unitless.
        """
        return get_ideal_activity_volume(
            mass_concentration=mass_concentration, density=self.density
        )


# Non-ideal activity strategies
class ActivityKappaParameter(ActivityStrategy):
    """Non-ideal activity strategy using the kappa hygroscopic parameter.

    Attributes:
        - kappa : Kappa hygroscopic parameters (array or scalar).
        - density : Densities (array or scalar) in kg/m^3.
        - molar_mass : Molar masses (array or scalar) in kg/mol.
        - water_index : Index identifying the water species in arrays.

    Methods:
    - activity : Computes non-ideal activity using kappa
      hygroscopicity approach.

    Examples:
        ```py title="Example Usage"
        import particula as par
        import numpy as np
        strategy = par.particles.ActivityKappaParameter(
            kappa=np.array([0.1, 0.0]),
            density=np.array([1000.0, 1200.0]),
            molar_mass=np.array([0.018, 0.058]),
            water_index=0,
        )
        a = strategy.activity(np.array([1.0, 2.0]))
        # a -> ...
        ```

    References:
        - Petters, M. D., & Kreidenweis, S. M. (2007). A single parameter
          representation of hygroscopic growth and cloud condensation
          nucleus activity. Atmospheric Chemistry and Physics, 7(8),
          1961-1971. [DOI](https://doi.org/10.5194/acp-7-1961-2007).
    """

    def __init__(
        self,
        kappa: NDArray[np.float64] | None = None,
        density: NDArray[np.float64] | None = None,
        molar_mass: NDArray[np.float64] | None = None,
        water_index: int = 0,
    ):
        """Initialize the ActivityKappaParameter strategy.

        Arguments:
            - kappa : Kappa hygroscopic parameters (array or scalar).
            - density : Densities in kg/m^3 (array or scalar).
            - molar_mass : Molar masses in kg/mol (array or scalar).
            - water_index : Index of the water species.
        """
        self.kappa = (
            kappa if kappa is not None else np.array([0.0], dtype=np.float64)
        )
        self.density = (
            density
            if density is not None
            else np.array([0.0], dtype=np.float64)
        )
        self.molar_mass = (
            molar_mass
            if molar_mass is not None
            else np.array([0.0], dtype=np.float64)
        )
        self.water_index = water_index

    def activity(
        self, mass_concentration: Union[float, NDArray[np.float64]]
    ) -> Union[float, NDArray[np.float64]]:
        """Calculate the activity of a species based on mass concentration.

        Arguments:
            - mass_concentration : Concentration of the species in kg/m^3.

        Returns:
            - Activity of the species, unitless.

        References:
            - Petters, M. D., & Kreidenweis, S. M. (2007). A single parameter
              representation of hygroscopic growth and cloud condensation
              nucleus activity. Atmospheric Chemistry and Physics, 7(8),
              1961-1971. [DOI](https://doi.org/10.5194/acp-7-1961-2007).
        """
        return get_kappa_activity(
            mass_concentration=mass_concentration,
            kappa=self.kappa,
            density=self.density,
            molar_mass=self.molar_mass,
            water_index=self.water_index,
        )
