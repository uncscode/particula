"""Test cases for CondensationIsothermalBuilder in particula.dynamics."""

import pytest

from particula.dynamics.condensation.condensation_builder.condensation_builder_mixin import (  # noqa: E501
    BuilderAccommodationCoefficientMixin,
    BuilderDiffusionCoefficientMixin,
    BuilderUpdateGasesMixin,
)


class MixinTester(
    BuilderDiffusionCoefficientMixin,
    BuilderAccommodationCoefficientMixin,
    BuilderUpdateGasesMixin,
):
    """Simple class to test condensation builder mixins."""


def test_set_diffusion_coefficient_units():
    """Test setting diffusion coefficient with units."""
    tester = MixinTester()
    tester.set_diffusion_coefficient(1.0, "m^2/s")
    assert pytest.approx(tester.diffusion_coefficient) == 1.0


def test_set_accommodation_coefficient():
    """Test setting accommodation coefficient."""
    tester = MixinTester()
    tester.set_accommodation_coefficient(0.5)
    assert tester.accommodation_coefficient == pytest.approx(0.5)


def test_set_update_gases():
    """Test setting update gases."""
    tester = MixinTester()
    tester.set_update_gases(False)
    assert tester.update_gases is False
