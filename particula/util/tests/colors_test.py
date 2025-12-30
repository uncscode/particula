"""Integrity checks for Tailwind palette values."""

import re

from particula.util.colors import TAILWIND


def test_tailwind_includes_expected_families():
    """Selected key families should always be present."""
    expected = {"slate", "gray", "blue", "orange", "emerald", "rose"}
    assert expected.issubset(TAILWIND)


def test_tailwind_values_are_hex_codes():
    """Every palette entry matches a #RRGGBB pattern."""
    pattern = re.compile(r"^#[0-9a-fA-F]{6}$")
    for shades in TAILWIND.values():
        for hex_value in shades.values():
            assert pattern.match(hex_value)


def test_tailwind_canonical_values_are_stable():
    """Pin a small subset of canonical shades to protect regressions."""
    canonical = {
        ("blue", "500"): "#3b82f6",
        ("emerald", "500"): "#10b981",
        ("gray", "900"): "#111827",
        ("orange", "400"): "#fb923c",
    }
    for (family, shade), expected in canonical.items():
        assert TAILWIND[family][shade] == expected


def test_tailwind_family_values_are_unique():
    """Within each family, hex values should not repeat."""
    for shades in TAILWIND.values():
        values = list(shades.values())
        assert len(set(values)) == len(values)
