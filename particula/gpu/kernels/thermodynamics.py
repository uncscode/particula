"""Validate caller-owned thermodynamic sidecars for GPU kernels.

This Python-side sidecar supplies a fixed device-local schema for future
vapor-pressure models. Validation does not evaluate a model, launch a Warp
kernel, calculate vapor pressure, or refresh gas container state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import warp as wp
except ImportError as exc:  # pragma: no cover - handled via import guards
    raise ImportError(
        "Warp is required for GPU thermodynamics helpers. "
        "Install with: pip install warp-lang"
    ) from exc

from particula.gpu.kernels.environment import _is_warp_array_like

__all__ = [
    "THERMODYNAMICS_MODE_BUCK",
    "THERMODYNAMICS_MODE_CONSTANT",
    "ThermodynamicsConfig",
]


THERMODYNAMICS_MODE_CONSTANT = wp.int32(0)
"""Constant vapor-pressure mode; parameter zero is pressure in Pa."""

THERMODYNAMICS_MODE_BUCK = wp.int32(1)
"""Buck mode: reference pressure plus three coefficients."""


@dataclass(frozen=True)
class ThermodynamicsConfig:
    """Store caller-owned, device-local thermodynamic model inputs.

    The frozen dataclass preserves field bindings, but its caller-owned Warp
    buffers remain mutable. It is a validation-only sidecar and does not
    calculate vapor pressure or update ``gas.vapor_pressure``.

    Attributes:
        modes: Per-species ``wp.int32`` model codes with shape ``(n_species,)``.
        parameters: Per-species ``wp.float64`` model parameters with shape
            ``(n_species, 4)``.
        molar_mass_reference: Ordered ``wp.float64`` compatibility fingerprint
            with shape ``(n_species,)``.
    """

    modes: Any
    parameters: Any
    molar_mass_reference: Any


def _validate_array_metadata(
    name: str,
    values: Any,
    expected_shape: tuple[int, ...],
    expected_dtype: Any,
    device: Any,
    caller_name: str,
) -> None:
    """Validate one sidecar field's metadata without reading its contents.

    Args:
        name: Field name used in validation errors.
        values: Candidate caller-owned Warp array.
        expected_shape: Required array shape.
        expected_dtype: Required Warp scalar dtype.
        device: Required active Warp device.
        caller_name: Public entry point used in error messages.

    Raises:
        ValueError: If the field is not a Warp array or has incompatible dtype,
            shape, or device.
    """
    if not _is_warp_array_like(values):
        raise ValueError(
            f"thermodynamics.{name} must be a Warp array in {caller_name}."
        )
    if values.dtype != expected_dtype:
        raise ValueError(
            f"thermodynamics.{name} must use dtype {expected_dtype} in "
            f"{caller_name}."
        )
    if values.shape != expected_shape:
        raise ValueError(
            f"thermodynamics.{name} shape {values.shape} does not match "
            f"expected {expected_shape} in {caller_name}."
        )
    if str(values.device) != str(device):
        raise ValueError(
            f"thermodynamics.{name} device does not match expected device in "
            f"{caller_name}."
        )


def _read_array(values: Any) -> np.ndarray:
    """Read one metadata-validated Warp array once into host memory.

    Args:
        values: Metadata-validated caller-owned Warp array.

    Returns:
        Host array containing the field values.
    """
    return np.asarray(values.numpy())


def validate_thermodynamics_config(
    thermodynamics: ThermodynamicsConfig | Any | None,
    n_species: int,
    device: Any,
    gas_molar_mass: Any,
    caller_name: str,
) -> ThermodynamicsConfig:
    """Validate a thermodynamic sidecar and return its identical object.

    Structural metadata is validated before any device-to-host readback. Valid
    caller buffers are never allocated, replaced, or mutated. This
    validation-only function does not launch kernels, evaluate thermodynamic
    parameters, calculate vapor pressure, or refresh ``gas.vapor_pressure``.

    Args:
        thermodynamics: Candidate caller-owned configuration.
        n_species: Expected species count.
        device: Active particle and gas Warp device.
        gas_molar_mass: Device-local gas molar masses used as the ordered
            compatibility fingerprint.
        caller_name: Public entry point used in error messages.

    Returns:
        The exact ``ThermodynamicsConfig`` instance supplied by the caller.

    Raises:
        ValueError: If the configuration schema, device locality, values, or
            ordered molar-mass fingerprint is invalid.
    """
    if not isinstance(thermodynamics, ThermodynamicsConfig):
        raise ValueError(
            f"thermodynamics must be a ThermodynamicsConfig in {caller_name}."
        )
    _validate_array_metadata(
        "modes",
        thermodynamics.modes,
        (n_species,),
        wp.int32,
        device,
        caller_name,
    )
    _validate_array_metadata(
        "parameters",
        thermodynamics.parameters,
        (n_species, 4),
        wp.float64,
        device,
        caller_name,
    )
    _validate_array_metadata(
        "molar_mass_reference",
        thermodynamics.molar_mass_reference,
        (n_species,),
        wp.float64,
        device,
        caller_name,
    )
    _validate_array_metadata(
        "gas.molar_mass",
        gas_molar_mass,
        (n_species,),
        wp.float64,
        device,
        caller_name,
    )

    modes = _read_array(thermodynamics.modes)
    parameters = _read_array(thermodynamics.parameters)
    references = _read_array(thermodynamics.molar_mass_reference)
    gas_masses = _read_array(gas_molar_mass)

    supported_modes = (
        int(THERMODYNAMICS_MODE_CONSTANT),
        int(THERMODYNAMICS_MODE_BUCK),
    )
    if not np.all(np.isin(modes, supported_modes)):
        raise ValueError(
            "thermodynamics.modes contains an unsupported mode in "
            f"{caller_name}."
        )
    if not np.all(np.isfinite(parameters)) or np.any(parameters < 0.0):
        raise ValueError(
            "thermodynamics.parameters must be finite and non-negative in "
            f"{caller_name}."
        )
    if not np.all(np.isfinite(references)) or np.any(references < 0.0):
        raise ValueError(
            "thermodynamics.molar_mass_reference must be finite and "
            f"non-negative in {caller_name}."
        )
    if not np.array_equal(references, gas_masses):
        raise ValueError(
            "thermodynamics.molar_mass_reference must exactly match "
            f"gas.molar_mass in {caller_name}."
        )
    return thermodynamics
