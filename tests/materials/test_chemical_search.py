import pytest

import particula.util.materials.chemical_search as chemical_search

THERMO_AVAILABLE = chemical_search.Chemical is not None


def test_get_chemical_search_exact():
    if not THERMO_AVAILABLE:
        pytest.skip("thermo is not installed")
    assert chemical_search.get_chemical_search("water") is not None


def test_get_chemical_search_fuzzy():
    if not THERMO_AVAILABLE:
        pytest.skip("thermo is not installed")
    suggestion = chemical_search.get_chemical_search("watr")
    assert suggestion is not None
