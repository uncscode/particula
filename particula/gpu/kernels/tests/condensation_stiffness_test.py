"""Discoverable GPU condensation stiffness and candidate-evidence tests."""

from __future__ import annotations

import pytest

from particula.gpu.kernels.tests import _condensation_test_support as support

pytestmark = pytest.mark.warp

device = support.device


def _export_stiffness_tests() -> tuple[str, ...]:
    """Expose stiffness and candidate support tests through discovery."""
    exported = tuple(
        sorted(
            name
            for name in dir(support)
            if name.startswith("test_")
            and (
                "stiffness" in name
                or "candidate" in name
                or name.startswith("test_fractional_mass_change")
                or name.startswith("test_zero_mass_entries")
            )
        )
    )
    globals().update({name: getattr(support, name) for name in exported})
    return exported


EXPORTED_STIFFNESS_TESTS = _export_stiffness_tests()


def test_condensation_stiffness_wrapper_exports_support_evidence() -> None:
    """The stiffness wrapper stays discoverable and non-empty."""
    assert len(EXPORTED_STIFFNESS_TESTS) >= 10
    assert "test_condensation_stiffness_case_builds_named_regimes" in (
        EXPORTED_STIFFNESS_TESTS
    )
    assert (
        "test_candidate_matches_cpu_reference_with_documented_tolerance"
        in EXPORTED_STIFFNESS_TESTS
    )
