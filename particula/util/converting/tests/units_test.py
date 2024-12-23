"""
Tests for Converting Units
"""

import unittest

from particula.util.converting import units


class TestUnitConversions(unittest.TestCase):

    def setUp(self) -> None:
        self.conversions = [
            ("Pa", "bar", 1e-5),
            ("bar", "Pa", 1e5),
            ("Pa", "Atm", 1 / 101325),
            ("Atm", "Pa", 101325),
            ("degC", "K", 273.15),
            ("K", "degC", -273.15),
            ("degF", "K", (32 - 32) * 5/9 + 273.15),
            ("K", "degF", (273.15 - 273.15) * 9/5 + 32),
            ("feet", "m", 0.3048),
            ("m", "feet", 1 / 0.3048),
        ]

    def test_complex_conversions(self) -> None:
        for from_unit, to_unit, expected in self.conversions:
            with self.subTest(from_unit=from_unit, to_unit=to_unit):
                result = units.convert(from_unit, to_unit, 1)
                self.assertAlmostEqual(result, expected, places=5)

    def test_parse_to_base(self) -> None:
        self.assertEqual(units.parse_to_base("Kilogram"), 1e3)
        self.assertEqual(units.parse_to_base("kg"), 1e3)
        self.assertEqual(units.parse_to_base("Meter"), 1)
        self.assertEqual(units.parse_to_base("m"), 1)
        self.assertEqual(units.parse_to_base("Centimeter"), 1e-2)
        self.assertEqual(units.parse_to_base("cm"), 1e-2)

