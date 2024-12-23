"""
Convert Common Units
"""

si_prefixes_long = {
    "yotta": 1e24,  # Y
    "zetta": 1e21,  # Z
    "exa": 1e18,  # E
    "peta": 1e15,  # P
    "tera": 1e12,  # T
    "giga": 1e9,  # G
    "mega": 1e6,  # M
    "kilo": 1e3,  # k
    "hecto": 1e2,  # h
    "deka": 1e1,  # da
    "deci": 1e-1,  # d
    "centi": 1e-2,  # c
    "milli": 1e-3,  # m
    "micro": 1e-6,  # μ
    "nano": 1e-9,  # n
    "pico": 1e-12,  # p
    "femto": 1e-15,  # f
    "atto": 1e-18,  # a
    "zepto": 1e-21,  # z
    "yocto": 1e-24,  # y
}
si_prefixes_short = {
    "Y": 1e24,  # Y
    "Z": 1e21,  # Z
    "E": 1e18,  # E
    "P": 1e15,  # P
    "T": 1e12,  # T
    "G": 1e9,  # G
    "M": 1e6,  # M
    "k": 1e3,  # k
    "h": 1e2,  # h
    "da": 1e1,  # da
    "d": 1e-1,  # d
    "c": 1e-2,  # c
    "m": 1e-3,  # m
    "u": 1e-6,  # u
    "n": 1e-9,  # n
    "p": 1e-12,  # p
    "f": 1e-15,  # f
    "a": 1e-18,  # a
    "z": 1e-21,  # z
    "y": 1e-24,  # y
}

base_si_suffix = {
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
}

base_si_suffix_short = {
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

# Conversion Functions
complex_conversions = {
    ("Pa", "bar"): lambda x: x * 1e-5,
    ("bar", "Pa"): lambda x: x * 1e5,
    ("Pa", "Atm"): lambda x: x / 101325,
    ("Atm", "Pa"): lambda x: x * 101325,
    ("degC", "K"): lambda x: x + 273.15,
    ("K", "degC"): lambda x: x - 273.15,
    ("degF", "K"): lambda x: (x - 32) * 5/9 + 273.15,
    ("K", "degF"): lambda x: (x - 273.15) * 9/5 + 32,
    ("feet", "m"): lambda x: x * 0.3048,
    ("m", "feet"): lambda x: x / 0.3048,
}


def parse_to_base(unit: str) -> float:
    """
    Convert a unit string to its base unit scale factor.

    Args:
        - unit : The unit string to be converted.

    Examples:
        ``` py title="Kilogram to gram"
            parse_to_base("Kilogram") #1e3
            # or
            parse_to_base("kg") # 1e3
        ```
    """

    for prefix, factor in si_prefixes_long.items():
        if unit.startswith(prefix):
            base_unit = unit[len(prefix):]
            if base_unit in base_si_suffix:
                return factor * base_si_suffix[base_unit]
    for prefix, factor in si_prefixes_short.items():
        if unit.startswith(prefix):
            base_unit = unit[len(prefix):]
            if base_unit in base_si_suffix_short:
                return factor * base_si_suffix_short[base_unit]
    raise ValueError(f"Unknown unit: {unit}")


def convert(
    from_unit: str,
    to_unit: str,
    value: float = 1.0
) -> float:
    """
    Convert a value from one unit to another using base unit scale factors.

    Args:
        - from_unit : The unit of the input value.
        - to_unit : The unit to convert the value to.
        - value : The numerical value to be converted.

    Returns:
        - The scale factor or the converted value.
    """
    if (from_unit, to_unit) in complex_conversions:
        return complex_conversions[(from_unit, to_unit)](value)
    from_base_factor = parse_to_base(from_unit)
    to_base_factor = parse_to_base(to_unit)
    base_value = value * from_base_factor
    return base_value / to_base_factor
