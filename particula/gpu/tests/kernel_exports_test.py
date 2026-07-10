"""Regression tests for the public GPU kernel import surface."""

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


def test_kernels_all_contains_only_supported_step_functions() -> None:
    """Kernel package __all__ matches the supported public surface."""
    pytest.importorskip("warp")

    import particula.gpu.kernels as kernels

    assert kernels.__all__ == [
        "coagulation_step_gpu",
        "condensation_step_gpu",
    ]
