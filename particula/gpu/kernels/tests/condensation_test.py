"""Discoverable GPU condensation kernel contract tests."""

from __future__ import annotations

import pytest

from particula.gpu.kernels.tests import _condensation_test_support as support

device = support.device
warp_cpu_device = support.warp_cpu_device
cuda_device = support.cuda_device


@pytest.fixture(autouse=True)
def _selected_warp_test_runtime(request: pytest.FixtureRequest) -> None:
    """Load Warp only while executing a selected Warp-backed export."""
    if request.node.get_closest_marker("warp") is not None:
        support._load_warp_runtime()


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
    {
        name: pytest.mark.warp(getattr(support, name))
        for name in EXPORTED_CONDENSATION_TESTS
    }
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
    assert (
        "test_condensation_activity_surface_matches_independent_reference"
        in EXPORTED_CONDENSATION_TESTS
    )
    assert (
        "test_condensation_activity_surface_invalid_sidecar_is_atomic"
        in EXPORTED_CONDENSATION_TESTS
    )
    assert (
        "test_condensation_activity_surface_warp_cpu_matches_numpy_reference"
        in EXPORTED_CONDENSATION_TESTS
    )
    assert (
        "test_condensation_activity_surface_cuda_matches_numpy_reference"
        in EXPORTED_CONDENSATION_TESTS
    )
    assert (
        "test_condensation_public_inventory_warp_cpu_matches_oracle"
        in EXPORTED_CONDENSATION_TESTS
    )
    assert (
        "test_condensation_public_inventory_cuda_matches_oracle"
        in EXPORTED_CONDENSATION_TESTS
    )
    assert (
        "test_condensation_step_gpu_energy_transfer_reuses_and_overwrites_output"
        in EXPORTED_CONDENSATION_TESTS
    )
    assert (
        "test_condensation_step_gpu_energy_transfer_aggregates_by_box_and_species"
        in EXPORTED_CONDENSATION_TESTS
    )
    assert (
        "test_condensation_composed_scalar_route_matches_four_substep_oracle"
        in EXPORTED_CONDENSATION_TESTS
    )
    assert (
        "test_condensation_composed_environment_route_matches_four_substep_oracle"
        in EXPORTED_CONDENSATION_TESTS
    )
    assert "test_condensation_composed_cuda_matches_four_substep_oracle" in (
        EXPORTED_CONDENSATION_TESTS
    )
    assert (
        "test_condensation_step_gpu_without_energy_transfer_skips_energy_kernels"
        in EXPORTED_CONDENSATION_TESTS
    )
    assert "test_condensation_energy_transfer_preflight_is_atomic" in (
        EXPORTED_CONDENSATION_TESTS
    )
    assert "test_condensation_energy_transfer_alias_preflight_is_atomic" in (
        EXPORTED_CONDENSATION_TESTS
    )
    assert "test_condensation_cuda_invalid_species_sidecar_is_atomic" in (
        EXPORTED_CONDENSATION_TESTS
    )
    assert (
        "test_condensation_energy_transfer_rejects_thermodynamic_parameters_alias"
        in EXPORTED_CONDENSATION_TESTS
    )
    assert (
        "test_condensation_energy_transfer_cuda_matches_box_species_oracle"
        in EXPORTED_CONDENSATION_TESTS
    )
    assert (
        "test_finalize_inventory_limited_transfer_matches_numpy_oracle"
        in EXPORTED_CONDENSATION_TESTS
    )
    assert (
        "test_finalize_inventory_rejects_invalid_physical_inputs_atomically"
        in EXPORTED_CONDENSATION_TESTS
    )
    assert (
        "test_finalize_inventory_rejects_p2_sidecar_aliases_atomically"
        in EXPORTED_CONDENSATION_TESTS
    )
    assert (
        "test_condensation_step_gpu_couples_four_substeps_to_numpy_oracle"
        in EXPORTED_CONDENSATION_TESTS
    )
    assert (
        "test_condensation_step_gpu_rejects_p2_vapor_pressure_alias_atomically"
        in EXPORTED_CONDENSATION_TESTS
    )
    assert (
        "test_condensation_scratch_ownership_aliases_are_atomic"
        in EXPORTED_CONDENSATION_TESTS
    )
    assert (
        "test_condensation_energy_retains_committed_substeps_after_proposal_failure"
        in EXPORTED_CONDENSATION_TESTS
    )
    assert (
        "test_condensation_rejects_nonfinite_extreme_gas_delta_before_coupling"
        in EXPORTED_CONDENSATION_TESTS
    )
    assert (
        "test_condensation_public_insufficient_inventory_scales_uptake_and_conserves"
        in EXPORTED_CONDENSATION_TESTS
    )
