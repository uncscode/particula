"""Builder for a compact-sphere dispersion and polarization strategy."""

from particula.abc_builder import BuilderABC
from particula.util.convert_units import get_unit_conversion

from ..coagulation_strategy.van_der_waals_coagulation_strategy import (
    VanDerWaalsCoagulationStrategy,
)
from .coagulation_builder_mixin import BuilderDistributionTypeMixin


class VanDerWaalsCoagulationBuilder(BuilderABC, BuilderDistributionTypeMixin):
    """Build a thermal encounter strategy for a specified material."""

    def __init__(self):
        """Require both material constants and a population representation."""
        BuilderABC.__init__(
            self,
            ["distribution_type", "hamaker_constant", "relative_permittivity"],
        )
        BuilderDistributionTypeMixin.__init__(self)
        self.hamaker_constant = None
        self.relative_permittivity = None

    def set_hamaker_constant(
        self, hamaker_constant, hamaker_constant_units="J"
    ):
        """Set the effective Hamaker constant in joules."""
        if hamaker_constant_units != "J":
            hamaker_constant *= get_unit_conversion(hamaker_constant_units, "J")
        self.hamaker_constant = hamaker_constant
        return self

    def set_relative_permittivity(
        self,
        relative_permittivity,
        relative_permittivity_units="dimensionless",
    ):
        """Set the dimensionless static relative permittivity."""
        if relative_permittivity_units not in ("dimensionless", "1"):
            raise ValueError("relative_permittivity is dimensionless")
        self.relative_permittivity = relative_permittivity
        return self

    def build(self):
        """Return the configured coagulation strategy."""
        self.pre_build_check()
        return VanDerWaalsCoagulationStrategy(
            self.distribution_type,
            self.hamaker_constant,
            self.relative_permittivity,
        )
