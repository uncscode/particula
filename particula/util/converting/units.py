"""
Convert Common Units
"""
import re
from typing import Callable, Dict, Tuple


class UnitConverter:
    """A class to convert units."""

    def __init__(self):
        self.si_prefixes = {
            **{
                "yotta": 1e24,
                "zetta": 1e21,
                "exa": 1e18,
                "peta": 1e15,
                "tera": 1e12,
                "giga": 1e9,
                "mega": 1e6,
                "kilo": 1e3,
                "hecto": 1e2,
                "deka": 1e1,
                "deci": 1e-1,
                "centi": 1e-2,
                "milli": 1e-3,
                "micro": 1e-6,
                "nano": 1e-9,
                "pico": 1e-12,
                "femto": 1e-15,
                "atto": 1e-18,
                "zepto": 1e-21,
                "yocto": 1e-24,
            },
            **{
                "Y": 1e24,
                "Z": 1e21,
                "E": 1e18,
                "P": 1e15,
                "T": 1e12,
                "G": 1e9,
                "M": 1e6,
                "k": 1e3,
                "h": 1e2,
                "da": 1e1,
                "d": 1e-1,
                "c": 1e-2,
                "m": 1e-3,
                "u": 1e-6,
                "n": 1e-9,
                "p": 1e-12,
                "f": 1e-15,
                "a": 1e-18,
                "z": 1e-21,
                "y": 1e-24,
            },
        }
        self.base_units = {
            "meter": 1,
            "Pascal": 1,
            "gram": 1,
            "second": 1,
            "Ampere": 1,
            "Kelvin": 1,
            "Newton": 1,
            "Joule": 1,
            "Watt": 1,
            "Coulomb": 1,
            "Hertz": 1,
            "liter": 1,
            "m": 1,
            "Pa": 1,
            "g": 1,
            "s": 1,
            "A": 1,
            "K": 1,
            "N": 1,
            "J": 1,
            "W": 1,
            "C": 1,
            "Hz": 1,
            "L": 1,
        }
        self.complex_conversions: Dict[
            Tuple[str, str], Callable[[float], float]
        ] = {
            ("Pa", "bar"): lambda x: x * 1e-5,
            ("bar", "Pa"): lambda x: x * 1e5,
            ("Pa", "Atm"): lambda x: x / 101325,
            ("Atm", "Pa"): lambda x: x * 101325,
            ("degC", "K"): lambda x: x + 273.15,
            ("K", "degC"): lambda x: x - 273.15,
            ("degF", "K"): lambda x: (x - 32) * 5 / 9 + 273.15,
            ("K", "degF"): lambda x: (x - 273.15) * 9 / 5 + 32,
            ("feet", "m"): lambda x: x * 0.3048,
            ("m", "feet"): lambda x: x / 0.3048,
        }

    def register_complex_conversion(
        self, from_unit: str, to_unit: str, func: Callable[[float], float]
    ):
        """Register a new complex conversion."""
        self.complex_conversions[(from_unit, to_unit)] = func

    def parse_unit(self, unit: str) -> float:
        """Parse a unit string to its base unit scale factor."""
        match = re.match(r"([a-zA-Z]+)?(.+)", unit)
        if not match:
            raise ValueError(f"Invalid unit format: {unit}")
        prefix, base_unit = match.groups()
        prefix_factor = self.si_prefixes.get(prefix, 1)
        base_factor = self.base_units.get(base_unit)
        if base_factor is None:
            raise ValueError(f"Unknown unit: {base_unit}")
        return prefix_factor * base_factor

    def convert(
        self, from_unit: str, to_unit: str, value: float = 1.0
    ) -> float:
        """Convert a value from one unit to another."""
        if (from_unit, to_unit) in self.complex_conversions:
            return self.complex_conversions[(from_unit, to_unit)](value)
        from_factor = self.parse_unit(from_unit)
        to_factor = self.parse_unit(to_unit)
        return value * from_factor / to_factor
