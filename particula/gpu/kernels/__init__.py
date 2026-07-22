"""Public GPU kernel entry points.

Import direct low-level step functions from this package:

    from particula.gpu.kernels import (
        coagulation_step_gpu,
        condensation_step_gpu,
        dilution_step_gpu,
    )

Lower-level helper kernels remain importable from their concrete modules.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "coagulation_step_gpu",
    "condensation_step_gpu",
    "dilution_step_gpu",
]

_SYMBOL_TO_MODULE = {
    "coagulation_step_gpu": "particula.gpu.kernels.coagulation",
    "condensation_step_gpu": "particula.gpu.kernels.condensation",
    "dilution_step_gpu": "particula.gpu.kernels.dilution",
}


def __getattr__(name: str) -> Any:
    """Lazily resolve the supported public direct-kernel entry points."""
    module_name = _SYMBOL_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
