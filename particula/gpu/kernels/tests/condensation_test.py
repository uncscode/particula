"""Discoverable GPU condensation kernel contract tests."""

from __future__ import annotations

from particula.gpu.kernels.tests import _condensation_test_support as support

device = support.device


def _export_condensation_tests() -> tuple[str, ...]:
    """Expose non-stiffness support tests through a discoverable module."""
    exported = tuple(
        sorted(
            name
            for name in dir(support)
            if name.startswith("test_")
            and "stiffness" not in name
            and "candidate" not in name
            and not name.startswith("test_fractional_mass_change")
            and not name.startswith("test_zero_mass_entries")
        )
    )
    globals().update({name: getattr(support, name) for name in exported})
    return exported


EXPORTED_CONDENSATION_TESTS = _export_condensation_tests()


def test_condensation_wrapper_exports_support_contract_tests() -> None:
    """The discoverable wrapper continues to expose real support-backed tests."""
    assert len(EXPORTED_CONDENSATION_TESTS) >= 20
    assert (
        "test_condensation_step_gpu_signature_keeps_environment_keyword_only"
        in EXPORTED_CONDENSATION_TESTS
    )
    assert "test_condensation_step_gpu_scalar_positional_call_remains_valid" in (
        EXPORTED_CONDENSATION_TESTS
    )
    assert "test_condensation_step_gpu_matches_cpu_single_box" in (
        EXPORTED_CONDENSATION_TESTS
    )
