"""Smoke tests for public util exports."""

import particula.util as util

EXPECTED_EXPORTS = {
    "get_unit_conversion",
    "TAILWIND",
    "get_effective_refractive_index",
    "get_arbitrary_round",
    "get_safe_exp",
    "get_safe_log",
    "get_safe_log10",
    "get_reduced_value",
    "get_reduced_self_broadcast",
    "get_coerced_type",
    "get_dict_from_list",
    "get_shape_check",
    "get_values_of_dict",
    "constants",
    "get_chemical_search",
    "get_chemical_surface_tension",
    "get_chemical_vapor_pressure",
    "get_chemical_stp_properties",
}


def test_expected_exports_are_present():
    """All documented util exports should be importable."""
    missing = {name for name in EXPECTED_EXPORTS if not hasattr(util, name)}
    assert not missing


def test_all_matches_attributes_when_defined():
    """If __all__ is defined, ensure it aligns with documented exports."""
    exported = set(getattr(util, "__all__", ()))
    if exported:
        assert EXPECTED_EXPORTS.issubset(exported)
