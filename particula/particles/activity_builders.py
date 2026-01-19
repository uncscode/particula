"""Builders for activity strategies with parameter validation.

This module defines builders for mass-, mole-, kappa-, and non-ideal
binary-based activity strategies. Each builder validates the parameters
required by its corresponding strategy before instantiating it.
"""

import logging
from typing import Any, Optional, Union

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
    ActivityNonIdealBinary,
    ActivityStrategy,
)

logger = logging.getLogger("particula")


class ActivityIdealMassBuilder(BuilderABC):
    """Builder for ActivityIdealMass strategies based on ideal mass fractions.

    This builder requires no additional parameters and returns a strategy that
    applies Raoult's Law using mass-based ratios.
    """

    def __init__(self):
        """Initialize the ActivityIdealMass builder with default requirements.

        The builder has no required parameters beyond the mixin defaults, so
        it simply initializes the base BuilderABC without additional state.
        """
        required_parameters = None
        BuilderABC.__init__(self, required_parameters)

    def build(self) -> ActivityStrategy:
        """Return an ActivityIdealMass strategy.

        Returns:
            ActivityIdealMass: Strategy that computes activity using
                ideal mass fractions.
        """
        return ActivityIdealMass()


class ActivityIdealMolarBuilder(BuilderABC, BuilderMolarMassMixin):
    """Builder for ActivityIdealMolar strategies that require molar mass.

    This builder collects the molar mass data necessary to apply Raoult's Law
    for mole fraction–based activity calculations.

    Attributes:
        molar_mass: Molar mass for each species, in kilograms per mole.
    """

    def __init__(self):
        """Initialize base builders for molar mass handling.

        Registers molar_mass as a required parameter for building the
        ActivityIdealMolar strategy and initializes the molar mass mixin.
        """
        required_parameters = ["molar_mass"]
        BuilderABC.__init__(self, required_parameters)
        BuilderMolarMassMixin.__init__(self)

    def build(self) -> ActivityStrategy:
        """Validate molar mass configuration and create the strategy.

        Returns:
            ActivityIdealMolar: Strategy that applies Raoult's Law using mole
                fractions.
        """
        self.pre_build_check()
        if self.molar_mass is None:
            raise ValueError("Required parameter 'molar_mass' is not set.")
        return ActivityIdealMolar(molar_mass=self.molar_mass)


class ActivityKappaParameterBuilder(
    BuilderABC, BuilderDensityMixin, BuilderMolarMassMixin
):
    """Builder for ActivityKappaParameter strategies using hygroscopic kappa.

    This builder validates kappa, density, molar mass, and water index inputs
    before instantiating ActivityKappaParameter.

    Attributes:
        kappa: NDArray of kappa parameters for each species.
        density: NDArray of densities, in kilograms per cubic meter.
        molar_mass: NDArray of molar masses, in kilograms per mole.
        water_index: Integer index of the water species.
    """

    def __init__(self):
        """Initialize the kappa parameter builder with required inputs.

        Registers kappa, density, molar mass, and water index as required
        parameters and initializes the density and molar mass mixins.
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
    """Builder for the ActivityNonIdealBinary strategy.

    Attributes:
        - molar_mass : Organic molar mass in kg/mol.
        - density : Organic density in kg/m^3.
        - oxygen2carbon : Required oxygen-to-carbon ratio.
        - functional_group : Optional BAT functional group identifier.
    """

    OPTIONAL_PARAMETERS = {"functional_group"}

    def __init__(self):
        """Initialize builder and register required parameters."""
        required_parameters = ["molar_mass", "density", "oxygen2carbon"]
        BuilderABC.__init__(self, required_parameters)
        BuilderMolarMassMixin.__init__(self)
        BuilderDensityMixin.__init__(self)
        self.oxygen2carbon: Optional[float] = None
        self.functional_group: Optional[str] = None

    def check_keys(self, parameters: dict[str, Any]):
        """Validate parameter keys while honoring optional fields.

        Args:
            parameters: Raw parameter dictionary received from the factory
                interface.
        """
        filtered = {
            key: value
            for key, value in parameters.items()
            if key not in self.OPTIONAL_PARAMETERS
        }
        super().check_keys(filtered)

    def set_parameters(
        self, parameters: dict[str, Any]
    ) -> "ActivityNonIdealBinaryBuilder":
        """Set builder inputs while handling optional bridges.

        Args:
            parameters: Dictionary containing molar_mass, density,
                oxygen2carbon, and optional functional_group entries.
        """
        optional_functional_group = parameters.get("functional_group")
        if optional_functional_group is not None:
            self.set_functional_group(optional_functional_group)
        filtered = {
            key: value
            for key, value in parameters.items()
            if key not in self.OPTIONAL_PARAMETERS
        }
        if "molar_mass" in filtered and "molar_mass_units" not in filtered:
            filtered["molar_mass_units"] = "kg/mol"
        if "density" in filtered and "density_units" not in filtered:
            filtered["density_units"] = "kg/m^3"
        super().set_parameters(filtered)
        return self

    def set_oxygen2carbon(
        self, oxygen2carbon: float
    ) -> "ActivityNonIdealBinaryBuilder":
        """Assign the oxygen-to-carbon ratio with validation.

        Args:
            oxygen2carbon: Ratio of oxygen to carbon for the organic compound.
        """
        oxygen_value = float(oxygen2carbon)
        if oxygen_value < 0:
            error_message = "oxygen2carbon must be non-negative."
            logger.error(error_message)
            raise ValueError(error_message)
        self.oxygen2carbon = oxygen_value
        return self

    def set_functional_group(
        self, functional_group: Optional[str]
    ) -> "ActivityNonIdealBinaryBuilder":
        """Assign optional functional group tags for BAT helpers.

        Args:
            functional_group: Optional BAT functional group identifier.
        """
        if functional_group is not None and not isinstance(
            functional_group, str
        ):
            raise TypeError("functional_group must be a string when provided.")
        self.functional_group = functional_group
        return self

    def build(self) -> ActivityStrategy:
        """Validate builder state and construct the BAT strategy.

        Returns:
            ActivityNonIdealBinary: BAT strategy configured with current
                parameters.
        """
        self.pre_build_check()
        if self.oxygen2carbon is None:
            raise ValueError("Required parameter 'oxygen2carbon' is not set.")
        return ActivityNonIdealBinary(
            molar_mass=self.molar_mass,  # type: ignore
            density=self.density,  # type: ignore
            oxygen2carbon=self.oxygen2carbon,  # type: ignore
            functional_group=self.functional_group,
        )
