import numpy as np
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
    tester = MixinTester()
    tester.set_diffusion_coefficient(1.0, "cm^2/s")
    assert pytest.approx(tester.diffusion_coefficient) == 1.0e-4


def test_set_accommodation_coefficient():
    tester = MixinTester()
    tester.set_accommodation_coefficient(0.5)
    assert tester.accommodation_coefficient == pytest.approx(0.5)


def test_set_update_gases():
    tester = MixinTester()
    tester.set_update_gases(False)
    assert tester.update_gases is False
