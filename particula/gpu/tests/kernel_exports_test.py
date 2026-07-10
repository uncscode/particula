"""Regression tests for the public GPU kernel import surface."""

from __future__ import annotations

import pytest


def test_gpu_top_level_does_not_reexport_kernel_steps() -> None:
    """Top-level particula.gpu stays focused on transfer and context helpers."""
    import particula.gpu as gpu

    assert not hasattr(gpu, "coagulation_step_gpu")
    assert not hasattr(gpu, "condensation_step_gpu")
    assert "coagulation_step_gpu" not in gpu.__all__
    assert "condensation_step_gpu" not in gpu.__all__


def test_kernel_step_functions_import_from_public_kernels_package() -> None:
    """Direct kernel step functions resolve from particula.gpu.kernels."""
    pytest.importorskip("warp")

    from particula.gpu.kernels import (
        coagulation_step_gpu,
        condensation_step_gpu,
    )
    from particula.gpu.kernels.coagulation import (
        coagulation_step_gpu as coagulation_step_gpu_impl,
    )
    from particula.gpu.kernels.condensation import (
        condensation_step_gpu as condensation_step_gpu_impl,
    )

    assert coagulation_step_gpu is coagulation_step_gpu_impl
    assert condensation_step_gpu is condensation_step_gpu_impl


def test_kernels_package_exports_only_supported_step_functions() -> None:
    """The package-level public surface is limited to the two step functions."""
    pytest.importorskip("warp")

    import particula.gpu.kernels as kernels

    assert kernels.__all__ == ["coagulation_step_gpu", "condensation_step_gpu"]
    assert hasattr(kernels, "coagulation_step_gpu")
    assert hasattr(kernels, "condensation_step_gpu")
    assert not hasattr(kernels, "apply_coagulation_kernel")
    assert not hasattr(kernels, "apply_mass_transfer_kernel")
    assert not hasattr(kernels, "brownian_coagulation_kernel")
    assert not hasattr(kernels, "condensation_mass_transfer_kernel")
    assert not hasattr(kernels, "initialize_coagulation_rng_states")


def test_kernels_steps_import_from_concrete_modules() -> None:
    """The supported step functions remain importable from concrete modules."""
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
