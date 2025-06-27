"""Unit tests for particula.util.materials.chemical_search.

These tests check:
    • normal behaviour when the optional ``thermo`` package is present;
    • expected ImportError when ``thermo`` is absent or deliberately disabled.
"""

import pytest
import particula.util.chemical.chemical_search as chemical_search

from particula.util.chemical.chemical_search import CHEMICALS_AVAILABLE


@pytest.mark.skipif(not CHEMICALS_AVAILABLE, reason="thermo not installed")
def test_get_chemical_search_exact_when_thermo_present():
    """Exact match should be returned when thermo is available."""
    assert chemical_search.get_chemical_search("water") == "water"


@pytest.mark.skipif(not CHEMICALS_AVAILABLE, reason="thermo not installed")
def test_get_chemical_search_fuzzy_when_thermo_present():
    """Fuzzy query should resolve to the closest name when thermo is available."""
    suggestion = chemical_search.get_chemical_search("watr")
    assert suggestion and suggestion.lower() == "water"


@pytest.mark.skipif(CHEMICALS_AVAILABLE, reason="thermo installed")
def test_get_chemical_search_raises_without_thermo():
    """ImportError is expected when thermo is not installed."""
    with pytest.raises(ImportError):
        chemical_search.get_chemical_search("water")


def test_forced_flag_disables_search(monkeypatch):
    """Forcing THERMO_AVAILABLE=False must trigger ImportError even if thermo exists."""
    if not chemical_search.CHEMICALS_AVAILABLE:
        pytest.skip("thermo not installed – behaviour already tested elsewhere")
    # Override the module flag
    monkeypatch.setattr(chemical_search, "CHEMICALS_AVAILABLE", False)
    with pytest.raises(ImportError):
        chemical_search.get_chemical_search("water")
