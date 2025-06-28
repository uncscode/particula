"""Tests for condensation builder mixins."""

import pytest

from particula.dynamics.condensation.condensation_builder.condensation_builder_mixin import (
    BuilderDiffusionCoefficientMixin,
    BuilderAccommodationCoefficientMixin,
    BuilderUpdateGasesMixin,
)


class MixinTester(
    BuilderDiffusionCoefficientMixin,
    BuilderAccommodationCoefficientMixin,
    BuilderUpdateGasesMixin,
):
    """Simple class to test condensation builder mixins."""


def test_set_diffusion_coefficient_units():
    """Diffusion coefficient units are converted to SI."""
    tester = MixinTester()
    tester.set_diffusion_coefficient(1.0, "cm^2/s")
    assert pytest.approx(tester.diffusion_coefficient) == 1.0e-4


def test_set_accommodation_coefficient():
    """Accommodation coefficient is stored correctly."""
    tester = MixinTester()
    tester.set_accommodation_coefficient(0.5)
    assert tester.accommodation_coefficient == pytest.approx(0.5)


def test_set_update_gases():
    """Update-gases flag toggles state."""
    tester = MixinTester()
    tester.set_update_gases(False)
    assert tester.update_gases is False
