"""Builder for the CondensationIsothermal strategy."""

from particula.abc_builder import BuilderABC
from particula.builder_mixin import BuilderMolarMassMixin
from particula.dynamics.condensation.condensation_strategies import (
    CondensationIsothermal,
    CondensationStrategy,
)

from .condensation_builder_mixin import (
    BuilderAccommodationCoefficientMixin,
    BuilderDiffusionCoefficientMixin,
    BuilderUpdateGasesMixin,
)


class CondensationIsothermalBuilder(
    BuilderABC,
    BuilderMolarMassMixin,
    BuilderDiffusionCoefficientMixin,
    BuilderAccommodationCoefficientMixin,
    BuilderUpdateGasesMixin,
):
    """Fluent builder for :class:`CondensationIsothermal`."""

    def __init__(self) -> None:
        """Initialize the Condensation Isothermal builder.

        Sets up the builder with required parameters for creating a
        CondensationIsothermal strategy, including molar mass,
        diffusion coefficient, and accommodation coefficient.
        """
        required_parameters = [
            "molar_mass",
            "diffusion_coefficient",
            "accommodation_coefficient",
        ]
        BuilderABC.__init__(self, required_parameters)
        BuilderMolarMassMixin.__init__(self)
        BuilderDiffusionCoefficientMixin.__init__(self)
        BuilderAccommodationCoefficientMixin.__init__(self)
        BuilderUpdateGasesMixin.__init__(self)

    def build(self) -> CondensationStrategy:
        """Validate parameters and create a condensation strategy."""
        self.pre_build_check()

        # Type guards: pre_build_check ensures these are not None
        if self.diffusion_coefficient is None:
            raise ValueError("diffusion_coefficient must be set")
        if self.accommodation_coefficient is None:
            raise ValueError("accommodation_coefficient must be set")

        return CondensationIsothermal(
            molar_mass=self.molar_mass,
            diffusion_coefficient=self.diffusion_coefficient,
            accommodation_coefficient=self.accommodation_coefficient,
            update_gases=self.update_gases,
        )
