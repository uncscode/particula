"""
Tests for Converting Units
"""
import unittest
from particula.util.converting.units import convert_units


class TestUnitConversion(unittest.TestCase):
    """
    Test for Unit Conversion
    """

    def setUp(self) -> None:
        """
        Setup the test environment
        """
        try:
            import pint  # noqa
        except ImportError:
            self.pint_installed = False
        else:
            self.pint_installed = True

    def test_import_warning(self) -> None:
        """
        Test for import warning if pint is not installed
        """
        if not self.pint_installed:
            with self.assertRaises(ImportError):
                convert_units("degC", "degF")
        else:
            self.skipTest("Pint installed. Skipping import warning test.")

    def test_example_conversion(self) -> None:
        """
        Test for example conversion when pint is installed
        """
        if self.pint_installed:
            result = convert_units("ug/m^3", "kg/m^3")
            self.assertAlmostEqual(result, 1e-9)
        else:
            self.skipTest("Pint not installed. Skipping conversion test.")
