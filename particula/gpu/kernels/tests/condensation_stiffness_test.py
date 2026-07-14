"""Discoverable GPU condensation stiffness and candidate-evidence tests."""

from __future__ import annotations

import pytest

from particula.gpu.kernels.tests import _condensation_test_support as support

pytestmark = pytest.mark.warp

device = support.device
warp_cpu_device = support.warp_cpu_device
cuda_device = support.cuda_device


@pytest.fixture(autouse=True)
def _selected_warp_test_runtime(request: pytest.FixtureRequest) -> None:
    """Load Warp only while executing a selected Warp-backed export."""
    if request.node.get_closest_marker("warp") is not None:
        support._load_warp_runtime()


def _export_stiffness_tests() -> tuple[str, ...]:
    """Expose only production stiffness support tests through discovery."""
    exported = tuple(
        sorted(
            name
            for name in dir(support)
            if name.startswith("test_condensation_production_stiffness_")
        )
    )
    globals().update({name: getattr(support, name) for name in exported})
    return exported


EXPORTED_STIFFNESS_TESTS = _export_stiffness_tests()


def test_condensation_stiffness_wrapper_exports_support_evidence() -> None:
    """The stiffness wrapper stays discoverable and non-empty."""
    assert EXPORTED_STIFFNESS_TESTS == (
        "test_condensation_production_stiffness_cuda_slice",
        "test_condensation_production_stiffness_recorded_contract",
    )
    assert not any("candidate" in name for name in EXPORTED_STIFFNESS_TESTS)
