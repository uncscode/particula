"""
Tests for Converting Units
"""

import unittest
from particula.util.converting.units import convert_units

# flake8: noqa
try:
    import pint  # pylint: disable=unused-import
except ImportError:
    IS_PINT_AVAILABE = False
else:
    IS_PINT_AVAILABE = True


class TestUnitConversion(unittest.TestCase):
    """
    Test for Unit Conversion
    """

    def setUp(self) -> None:
        """
        Setup the test environment
        """

    def test_import_warning(self) -> None:
        """
        Test for import warning if pint is not installed
        """
        if not IS_PINT_AVAILABE:
            with self.assertRaises(ImportError):
                convert_units("degC", "degF")
        else:
            self.skipTest("Pint installed. Skipping import warning test.")

    def test_example_conversion(self) -> None:
        """
        Test for example conversion when pint is installed
        """
        if IS_PINT_AVAILABE:
            result = convert_units("ug/m^3", "kg/m^3")
            self.assertAlmostEqual(result, 1e-9)
        else:
            self.skipTest("Pint not installed. Skipping conversion test.")
