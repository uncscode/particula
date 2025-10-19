"""Unit tests for particula.util.materials.chemical_search.

These tests check:
    • normal behaviour when the optional ``thermo`` package is present;
    • expected ImportError when ``thermo`` is absent or deliberately disabled.
"""

import pytest

from particula.util.chemical.chemical_search import (
    CHEMICALS_AVAILABLE,
    get_chemical_search,
)


@pytest.mark.skipif(not CHEMICALS_AVAILABLE, reason="thermo not installed")
def test_get_chemical_search_exact_when_thermo_present():
    """Exact match should be returned when thermo is available."""
    assert get_chemical_search("water") == "water"


@pytest.mark.skipif(not CHEMICALS_AVAILABLE, reason="thermo not installed")
def test_get_chemical_search_fuzzy_when_thermo_present():
    """Fuzzy query resolves to closest name when thermo is available."""
    suggestion = get_chemical_search("watr")
    assert suggestion and suggestion.lower() == "water"


@pytest.mark.skipif(CHEMICALS_AVAILABLE, reason="thermo installed")
def test_get_chemical_search_raises_without_thermo():
    """ImportError is expected when thermo is not installed."""
    with pytest.raises(ImportError):
        get_chemical_search("water")
