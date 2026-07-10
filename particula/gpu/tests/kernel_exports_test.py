"""Regression tests for the public GPU kernel import surface."""

from __future__ import annotations

import pytest

SUPPORTED_STEP_SYMBOLS = (
    "coagulation_step_gpu",
    "condensation_step_gpu",
)

INTERNAL_HELPER_SYMBOLS = (
    "apply_coagulation_kernel",
    "apply_mass_transfer_kernel",
    "condensation_mass_transfer_kernel",
    "initialize_coagulation_rng_states",
)


def test_gpu_top_level_does_not_reexport_kernel_steps() -> None:
    """Top-level particula.gpu stays focused on transfer and context helpers."""
    import particula.gpu as gpu

    for symbol_name in SUPPORTED_STEP_SYMBOLS:
        assert not hasattr(gpu, symbol_name)
        assert symbol_name not in gpu.__all__


@pytest.mark.parametrize("symbol_name", SUPPORTED_STEP_SYMBOLS)
def test_public_kernels_package_exports_supported_step_function(
    symbol_name: str,
) -> None:
    """Direct kernel step functions resolve from particula.gpu.kernels."""
    pytest.importorskip("warp")

    import particula.gpu.kernels as kernels
    from particula.gpu.kernels.coagulation import coagulation_step_gpu
    from particula.gpu.kernels.condensation import condensation_step_gpu

    concrete_symbol_map = {
        "coagulation_step_gpu": coagulation_step_gpu,
        "condensation_step_gpu": condensation_step_gpu,
    }

    assert getattr(kernels, symbol_name) is concrete_symbol_map[symbol_name]


def test_kernels_package_all_is_exact_supported_surface() -> None:
    """The package-level public surface is limited to the two step functions."""
    pytest.importorskip("warp")

    import particula.gpu.kernels as kernels

    assert kernels.__all__ == ["coagulation_step_gpu", "condensation_step_gpu"]


@pytest.mark.parametrize("helper_name", INTERNAL_HELPER_SYMBOLS)
def test_kernels_package_does_not_export_internal_helpers(
    helper_name: str,
) -> None:
    """Representative helpers stay internal to the concrete kernel modules."""
    pytest.importorskip("warp")

    import particula.gpu.kernels as kernels

    assert not hasattr(kernels, helper_name)
    assert helper_name not in kernels.__all__


def test_concrete_kernel_modules_still_expose_supported_and_internal_symbols() -> (
    None
):
    """Concrete modules keep both public steps and representative internals."""
    pytest.importorskip("warp")

    import particula.gpu.kernels as kernels
    from particula.gpu.kernels.coagulation import (
        coagulation_step_gpu,
        initialize_coagulation_rng_states,
    )
    from particula.gpu.kernels.condensation import condensation_step_gpu

    assert kernels.coagulation_step_gpu is coagulation_step_gpu
    assert kernels.condensation_step_gpu is condensation_step_gpu
    assert initialize_coagulation_rng_states is not None
