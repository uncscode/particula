"""Builder class for Surface strategies, for how to account for
surface tension in the calculation of the Kelvin effect.

This builds the strategy and checks that the required parameters are set,
and converts the units of the parameters if necessary.
"""
# pylint: disable=too-many-ancestors

import logging

from particula.abc_builder import (
    BuilderABC,
)
from particula.builder_mixin import (
    BuilderDensityMixin,
    BuilderMolarMassMixin,
    BuilderPhaseIndexMixin,
    BuilderSurfaceTensionMixin,
    BuilderSurfaceTensionTableMixin,
    BuilderTemperatureTableMixin,
)
from particula.particles.surface_strategies import (
    SurfaceStrategyMass,
    SurfaceStrategyMolar,
    SurfaceStrategyVolume,
)

logger = logging.getLogger("particula")


class SurfaceStrategyMolarBuilder(
    BuilderABC,
    BuilderDensityMixin,
    BuilderSurfaceTensionMixin,
    BuilderSurfaceTensionTableMixin,
    BuilderTemperatureTableMixin,
    BuilderMolarMassMixin,
    BuilderPhaseIndexMixin,
):
    """Builder class for SurfaceStrategyMolar objects.

    For calculating the Kelvin effect based on molar mass. Needed for
    the effective surface tension calculation.

    Methods:
    - set_surface_tension : Set the surface tension in N/m.
    - set_density : Set the density in kg/m^3.
    - set_molar_mass : Set the molar mass in kg/mol.
    - set_parameters : Configure multiple parameters at once.
    - set_phase_index : Optionally assign species to phases.
    - build : Validate parameters and return the strategy.
    """

    def __init__(self):
        """Initialize the SurfaceStrategyMolarBuilder.

        Sets up the builder with required parameters for creating a
        SurfaceStrategyMolar instance including surface tension, density,
        and molar mass.
        """
        required_parameters = ["surface_tension", "density", "molar_mass"]
        BuilderABC.__init__(self, required_parameters)
        BuilderSurfaceTensionMixin.__init__(self)
        BuilderSurfaceTensionTableMixin.__init__(self)
        BuilderTemperatureTableMixin.__init__(self)
        BuilderDensityMixin.__init__(self)
        BuilderMolarMassMixin.__init__(self)
        BuilderPhaseIndexMixin.__init__(self)

    def build(self) -> SurfaceStrategyMolar:
        """Validate and return the SurfaceStrategyMolar object.

        Returns:
            SurfaceStrategyMolar: Instance of the SurfaceStrategyMolar object.
        """
        self.pre_build_check()
        return SurfaceStrategyMolar(
            surface_tension=self.surface_tension,  # type: ignore
            density=self.density,  # type: ignore
            molar_mass=self.molar_mass,  # type: ignore
            phase_index=self.phase_index,  # type: ignore
            surface_tension_table=self.surface_tension_table,  # type: ignore
            temperature_table=self.temperature_table,  # type: ignore
        )


class SurfaceStrategyMassBuilder(
    BuilderABC,
    BuilderSurfaceTensionMixin,
    BuilderSurfaceTensionTableMixin,
    BuilderTemperatureTableMixin,
    BuilderDensityMixin,
    BuilderPhaseIndexMixin,
):
    """Builder class for SurfaceStrategyMass objects.

    For calculating the Kelvin effect based on mass mixing. Needed for
    the effective surface tension calculation.

    Methods:
    - set_surface_tension : Set the surface tension in N/m.
    - set_density : Set the density in kg/m^3.
    - set_parameters : Configure multiple parameters at once.
    - set_phase_index : Optionally assign species to phases.
    - build : Validate parameters and return the strategy.
    """

    def __init__(self):
        """Initialize the SurfaceStrategyMassBuilder.

        Sets up the builder with required parameters for creating a
        SurfaceStrategyMass instance including surface tension and density.
        """
        required_parameters = ["surface_tension", "density"]
        BuilderABC.__init__(self, required_parameters)
        BuilderSurfaceTensionMixin.__init__(self)
        BuilderSurfaceTensionTableMixin.__init__(self)
        BuilderTemperatureTableMixin.__init__(self)
        BuilderDensityMixin.__init__(self)
        BuilderPhaseIndexMixin.__init__(self)

    def build(self) -> SurfaceStrategyMass:
        """Validate and return the SurfaceStrategyMass object.

        Returns:
            SurfaceStrategyMass: Instance of the SurfaceStrategyMass object.
        """
        self.pre_build_check()
        return SurfaceStrategyMass(
            surface_tension=self.surface_tension,  # type: ignore
            density=self.density,  # type: ignore
            phase_index=self.phase_index,  # type: ignore
            surface_tension_table=self.surface_tension_table,  # type: ignore
            temperature_table=self.temperature_table,  # type: ignore
        )


class SurfaceStrategyVolumeBuilder(
    BuilderABC,
    BuilderSurfaceTensionMixin,
    BuilderSurfaceTensionTableMixin,
    BuilderTemperatureTableMixin,
    BuilderDensityMixin,
    BuilderPhaseIndexMixin,
):
    """Builder class for SurfaceStrategyVolume objects.

    For calculating the Kelvin effect based on volume mixing. Needed for
    the effective surface tension calculation.

    Methods:
    - set_surface_tension : Set the surface tension in N/m.
    - set_density : Set the density in kg/m^3.
    - set_parameters : Configure multiple parameters at once.
    - set_phase_index : Optionally assign species to phases.
    - build : Validate parameters and return the strategy.
    """

    def __init__(self):
        """Initialize the SurfaceStrategyVolumeBuilder.

        Sets up the builder with required parameters for creating a
        SurfaceStrategyVolume instance including surface tension and density.
        """
        required_parameters = ["surface_tension", "density"]
        BuilderABC.__init__(self, required_parameters)
        BuilderSurfaceTensionMixin.__init__(self)
        BuilderSurfaceTensionTableMixin.__init__(self)
        BuilderTemperatureTableMixin.__init__(self)
        BuilderDensityMixin.__init__(self)
        BuilderPhaseIndexMixin.__init__(self)

    def build(self) -> SurfaceStrategyVolume:
        """Validate and return the SurfaceStrategyVolume object.

        Returns:
            SurfaceStrategyVolume: Instance of the SurfaceStrategyVolume
                object.
        """
        self.pre_build_check()
        return SurfaceStrategyVolume(
            surface_tension=self.surface_tension,  # type: ignore
            density=self.density,  # type: ignore
            phase_index=self.phase_index,  # type: ignore
            surface_tension_table=self.surface_tension_table,  # type: ignore
            temperature_table=self.temperature_table,  # type: ignore
        )
