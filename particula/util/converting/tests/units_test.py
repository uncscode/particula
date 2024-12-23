"""
Tests for Converting Units
"""
import unittest
from particula.util.converting.units import (
    UnitConverter,
)  # Assuming the class is saved in unit_converter.py


class TestUnitConverter(unittest.TestCase):
    def setUp(self):
        """Set up a UnitConverter instance for testing."""
        self.converter = UnitConverter()

    def test_si_prefix_conversion(self):
        """Test conversions using SI prefixes."""
        self.assertAlmostEqual(
            self.converter.convert("kiloPascal", "Pa", 1), 1000
        )
        self.assertAlmostEqual(self.converter.convert("megaWatt", "W", 1), 1e6)
        self.assertAlmostEqual(
            self.converter.convert("milliAmpere", "A", 1), 1e-3
        )

    def test_base_unit_conversion(self):
        """Test conversions between base units."""
        self.assertAlmostEqual(self.converter.convert("meter", "meter", 1), 1)
        self.assertAlmostEqual(self.converter.convert("liter", "liter", 1), 1)

    def test_complex_conversion(self):
        """Test complex conversions registered in the class."""
        self.assertAlmostEqual(self.converter.convert("degC", "K", 25), 298.15)
        self.assertAlmostEqual(self.converter.convert("K", "degC", 298.15), 25)
        self.assertAlmostEqual(self.converter.convert("feet", "m", 1), 0.3048)
        self.assertAlmostEqual(
            self.converter.convert("m", "feet", 1), 1 / 0.3048
        )

    def test_register_new_conversion(self):
        """Test registering a new complex conversion."""
        self.converter.register_complex_conversion(
            "mile", "km", lambda x: x * 1.60934
        )
        self.assertAlmostEqual(
            self.converter.convert("mile", "km", 10), 16.0934
        )

    def test_invalid_unit(self):
        """Test handling of invalid unit strings."""
        with self.assertRaises(ValueError):
            self.converter.convert("unknownUnit", "Pa", 1)

        with self.assertRaises(ValueError):
            self.converter.convert("kiloPascal", "unknownUnit", 1)

    def test_edge_case_no_prefix(self):
        """Test edge cases with no prefix."""
        self.assertAlmostEqual(self.converter.convert("Pa", "Pa", 1), 1)
        self.assertAlmostEqual(self.converter.convert("g", "g", 1), 1)
