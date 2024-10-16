"""Builder class for Surface strategies, for how to account for
surface tension in the calculation of the Kelvin effect.

This builds the strategy and checks that the required parameters are set,
and converts the units of the parameters if necessary.
"""

import logging
from particula.abc_builder import (
    BuilderABC,
)
from particula.builder_mixin import (
    BuilderDensityMixin,
    BuilderSurfaceTensionMixin,
    BuilderMolarMassMixin,
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
    BuilderMolarMassMixin,
):
    """Builder class for SurfaceStrategyMolar objects.

    Methods:
        set_surface_tension(surface_tension, surface_tension_units): Set the
            surface tension of the particle in N/m. Default units are 'N/m'.
        set_density(density, density_units): Set the density of the particle in
            kg/m^3. Default units are 'kg/m^3'.
        set_molar_mass(molar_mass, molar_mass_units): Set the molar mass of the
            particle in kg/mol. Default units are 'kg/mol'.
        set_parameters(params): Set the parameters of the SurfaceStrategyMolar
            object from a dictionary including optional units.
        build(): Validate and return the SurfaceStrategyMolar object.
    """

    def __init__(self):
        required_parameters = ["surface_tension", "density", "molar_mass"]
        BuilderABC.__init__(self, required_parameters)
        BuilderSurfaceTensionMixin.__init__(self)
        BuilderDensityMixin.__init__(self)
        BuilderMolarMassMixin.__init__(self)

    def build(self) -> SurfaceStrategyMolar:
        """Validate and return the SurfaceStrategyMass object.

        Returns:
            SurfaceStrategyMolar: Instance of the SurfaceStrategyMolar object.
        """
        self.pre_build_check()
        return SurfaceStrategyMolar(
            surface_tension=self.surface_tension,  # type: ignore
            density=self.density,  # type: ignore
            molar_mass=self.molar_mass,  # type: ignore
        )


class SurfaceStrategyMassBuilder(
    BuilderABC, BuilderSurfaceTensionMixin, BuilderDensityMixin
):
    """Builder class for SurfaceStrategyMass objects.

    Methods:
        set_surface_tension(surface_tension, surface_tension_units): Set the
            surface tension of the particle in N/m. Default units are 'N/m'.
        set_density(density, density_units): Set the density of the particle in
            kg/m^3. Default units are 'kg/m^3'.
        set_parameters(params): Set the parameters of the SurfaceStrategyMass
            object from a dictionary including optional units.
        build(): Validate and return the SurfaceStrategyMass object.
    """

    def __init__(self):
        required_parameters = ["surface_tension", "density"]
        BuilderABC.__init__(self, required_parameters)
        BuilderSurfaceTensionMixin.__init__(self)
        BuilderDensityMixin.__init__(self)

    def build(self) -> SurfaceStrategyMass:
        """Validate and return the SurfaceStrategyMass object.

        Returns:
            SurfaceStrategyMass: Instance of the SurfaceStrategyMass object.
        """
        self.pre_build_check()
        return SurfaceStrategyMass(
            surface_tension=self.surface_tension,  # type: ignore
            density=self.density,  # type: ignore
        )


class SurfaceStrategyVolumeBuilder(
    BuilderABC, BuilderSurfaceTensionMixin, BuilderDensityMixin
):
    """Builder class for SurfaceStrategyVolume objects.

    Methods:
        set_surface_tension(surface_tension, surface_tension_units): Set the
            surface tension of the particle in N/m. Default units are 'N/m'.
        set_density(density, density_units): Set the density of the particle in
            kg/m^3. Default units are 'kg/m^3'.
        set_parameters(params): Set the parameters of the SurfaceStrategyVolume
            object from a dictionary including optional units.
        build(): Validate and return the SurfaceStrategyVolume object.
    """

    def __init__(self):
        required_parameters = ["surface_tension", "density"]
        BuilderABC.__init__(self, required_parameters)
        BuilderSurfaceTensionMixin.__init__(self)
        BuilderDensityMixin.__init__(self)

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
        )
