"""Regression tests for the public GPU kernel import surface."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.warp

ROOT = Path(__file__).resolve().parents[3]

SUPPORTED_STEP_SYMBOLS = (
    "coagulation_step_gpu",
    "activate_slots_gpu",
    "condensation_step_gpu",
    "dilution_step_gpu",
    "wall_loss_step_gpu",
)

INTERNAL_HELPER_SYMBOLS = (
    "brownian_coagulation_kernel",
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
    from particula.gpu.kernels.dilution import dilution_step_gpu
    from particula.gpu.kernels.slot_management import activate_slots_gpu
    from particula.gpu.kernels.wall_loss import wall_loss_step_gpu

    concrete_symbol_map = {
        "coagulation_step_gpu": coagulation_step_gpu,
        "activate_slots_gpu": activate_slots_gpu,
        "condensation_step_gpu": condensation_step_gpu,
        "dilution_step_gpu": dilution_step_gpu,
        "wall_loss_step_gpu": wall_loss_step_gpu,
    }

    assert getattr(kernels, symbol_name) is concrete_symbol_map[symbol_name]


def test_kernels_package_all_is_exact_supported_surface() -> None:
    """The package-level public surface is limited to supported step functions."""
    pytest.importorskip("warp")

    import particula.gpu.kernels as kernels

    assert kernels.__all__ == [
        "coagulation_step_gpu",
        "activate_slots_gpu",
        "condensation_step_gpu",
        "dilution_step_gpu",
        "wall_loss_step_gpu",
    ]


def test_kernels_package_keeps_dilution_module_lazy_in_fresh_process() -> None:
    """Package import publishes dilution without importing its implementation."""
    script = "\n".join(
        (
            "import sys",
            "import particula.gpu.kernels as kernels",
            "assert 'dilution_step_gpu' in kernels.__all__",
            "assert 'particula.gpu.kernels.dilution' not in sys.modules",
        )
    )
    environment = os.environ | {
        "PYTHONPATH": os.pathsep.join(
            filter(None, (str(ROOT), os.environ.get("PYTHONPATH")))
        )
    }

    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


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
        brownian_coagulation_kernel,
        coagulation_step_gpu,
        initialize_coagulation_rng_states,
    )
    from particula.gpu.kernels.condensation import (
        apply_mass_transfer_kernel,
        condensation_step_gpu,
    )

    assert kernels.coagulation_step_gpu is coagulation_step_gpu
    assert kernels.condensation_step_gpu is condensation_step_gpu
    assert brownian_coagulation_kernel is not None
    assert initialize_coagulation_rng_states is not None
    assert apply_mass_transfer_kernel is not None


def test_kernels_package_lazily_imports_only_requested_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accessing one public symbol should not import the unrelated module."""
    import particula.gpu.kernels as kernels

    imported: list[str] = []
    fake_symbol = object()

    def _fake_import_module(name: str) -> object:
        imported.append(name)
        if name == "particula.gpu.kernels.coagulation":
            return SimpleNamespace(coagulation_step_gpu=fake_symbol)
        raise AssertionError(f"Unexpected import: {name}")

    monkeypatch.delattr(kernels, "coagulation_step_gpu", raising=False)
    monkeypatch.delattr(kernels, "condensation_step_gpu", raising=False)
    monkeypatch.setattr(kernels, "import_module", _fake_import_module)

    assert kernels.coagulation_step_gpu is fake_symbol
    assert imported == ["particula.gpu.kernels.coagulation"]


def test_kernels_package_keeps_unrelated_broken_module_from_blocking_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken unrelated kernel module should not block selected symbol access."""
    import particula.gpu.kernels as kernels

    fake_symbol = object()

    def _fake_import_module(name: str) -> object:
        if name == "particula.gpu.kernels.coagulation":
            return SimpleNamespace(coagulation_step_gpu=fake_symbol)
        if name == "particula.gpu.kernels.condensation":
            raise RuntimeError("broken condensation import")
        raise AssertionError(f"Unexpected import: {name}")

    monkeypatch.delattr(kernels, "coagulation_step_gpu", raising=False)
    monkeypatch.delattr(kernels, "condensation_step_gpu", raising=False)
    monkeypatch.setattr(kernels, "import_module", _fake_import_module)

    assert kernels.coagulation_step_gpu is fake_symbol
