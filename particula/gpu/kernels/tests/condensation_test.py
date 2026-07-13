"""Discoverable GPU condensation kernel contract tests."""

from __future__ import annotations

import pytest

from particula.gpu.kernels.tests import _condensation_test_support as support

pytestmark = pytest.mark.warp

device = support.device

EXCLUDED_CONDENSATION_TEST_PREFIXES = (
    "test_fractional_mass_change",
    "test_zero_mass_entries",
)
EXCLUDED_CONDENSATION_TEST_SUBSTRINGS = ("stiffness", "candidate")


def _is_condensation_contract_test(name: str) -> bool:
    """Return whether a support-backed test belongs in this wrapper."""
    return (
        name.startswith("test_")
        and not any(
            excluded in name
            for excluded in EXCLUDED_CONDENSATION_TEST_SUBSTRINGS
        )
        and not any(
            name.startswith(prefix)
            for prefix in EXCLUDED_CONDENSATION_TEST_PREFIXES
        )
    )


EXPORTED_CONDENSATION_TESTS = tuple(
    sorted(
        name for name in dir(support) if _is_condensation_contract_test(name)
    )
)
globals().update(
    {name: getattr(support, name) for name in EXPORTED_CONDENSATION_TESTS}
)


def test_condensation_wrapper_exports_support_contract_tests() -> None:
    """The discoverable wrapper continues to expose real support-backed tests."""
    assert len(EXPORTED_CONDENSATION_TESTS) >= 20
    assert (
        "test_condensation_step_gpu_signature_keeps_keyword_only_inputs"
        in EXPORTED_CONDENSATION_TESTS
    )
    assert (
        "test_condensation_step_gpu_scalar_positional_call_remains_valid"
        in (EXPORTED_CONDENSATION_TESTS)
    )
    assert "test_condensation_step_gpu_matches_cpu_single_box" in (
        EXPORTED_CONDENSATION_TESTS
    )
    assert "test_condensation_step_gpu_rejects_mixed_environment_inputs" in (
        EXPORTED_CONDENSATION_TESTS
    )
    assert (
        "test_condensation_step_gpu_environment_device_mismatch_raises_value_error"
        in EXPORTED_CONDENSATION_TESTS
    )
    assert (
        "test_condensation_step_gpu_direct_pressure_device_mismatch_raises"
        in EXPORTED_CONDENSATION_TESTS
    )
