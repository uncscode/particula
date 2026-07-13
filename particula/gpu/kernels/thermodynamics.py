"""Validate thermodynamic sidecars and refresh GPU vapor pressures.

This Python-side sidecar supplies a fixed device-local schema for future
vapor-pressure models. The concrete-module refresh API validates these
caller-owned buffers before launching the device-resident evaluator.
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

from particula.gpu.kernels.environment import (
    _is_warp_array_like,
    _validate_positive_finite_array,
)

__all__ = [
    "THERMODYNAMICS_MODE_BUCK",
    "THERMODYNAMICS_MODE_CONSTANT",
    "ThermodynamicsConfig",
    "refresh_vapor_pressure_gpu",
    "validate_thermodynamics_config",
]


THERMODYNAMICS_MODE_CONSTANT = wp.int32(0)
"""Constant vapor-pressure mode; parameter zero is pressure in Pa."""

THERMODYNAMICS_MODE_BUCK = wp.int32(1)
"""Canonical Buck water/ice mode; all four parameter slots are unused."""


@dataclass(frozen=True)
class ThermodynamicsConfig:
    """Store caller-owned, device-local thermodynamic model inputs.

    The frozen dataclass preserves field bindings, but its caller-owned Warp
    buffers remain mutable. It is a validation-only sidecar and does not
    calculate vapor pressure or update ``gas.vapor_pressure``. Constant-mode
    parameter zero is vapor pressure in Pa. Buck-mode parameter slots are
    reserved by the fixed schema and are ignored by the canonical evaluator.

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
    field_prefix: str = "thermodynamics",
) -> None:
    """Validate one sidecar field's metadata without reading its contents.

    Args:
        name: Field name used in validation errors.
        values: Candidate caller-owned Warp array.
        expected_shape: Required array shape.
        expected_dtype: Required Warp scalar dtype.
        device: Required active Warp device.
        caller_name: Public entry point used in error messages.
        field_prefix: Namespace prepended to ``name`` in validation errors.

    Raises:
        ValueError: If the field is not a Warp array or has incompatible dtype,
            shape, or device.
    """
    field_name = f"{field_prefix}.{name}" if field_prefix else name
    if not _is_warp_array_like(values):
        raise ValueError(f"{field_name} must be a Warp array in {caller_name}.")
    if values.dtype != expected_dtype:
        raise ValueError(
            f"{field_name} must use dtype {expected_dtype} in {caller_name}."
        )
    if values.shape != expected_shape:
        raise ValueError(
            f"{field_name} shape {values.shape} does not match "
            f"expected {expected_shape} in {caller_name}."
        )
    if str(values.device) != str(device):
        raise ValueError(
            f"{field_name} device does not match expected device in "
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


@wp.func
def _constant_vapor_pressure(
    parameters: Any,
    species_idx: Any,
) -> wp.float64:
    """Return the constant-mode vapor pressure in Pa."""
    return parameters[species_idx, 0]


@wp.func
def _buck_vapor_pressure(temperature: wp.float64) -> wp.float64:
    """Return canonical Buck water/ice vapor pressure in Pa."""
    temperature_celsius = temperature - wp.float64(273.15)
    if temperature_celsius < wp.float64(0.0):
        return (
            wp.float64(6.1115)
            * wp.exp(
                (wp.float64(23.036) - temperature_celsius / wp.float64(333.7))
                * temperature_celsius
                / (wp.float64(279.82) + temperature_celsius)
            )
            * wp.float64(100.0)
        )
    return (
        wp.float64(6.1121)
        * wp.exp(
            (wp.float64(18.678) - temperature_celsius / wp.float64(234.5))
            * temperature_celsius
            / (wp.float64(257.14) + temperature_celsius)
        )
        * wp.float64(100.0)
    )


@wp.func
def _evaluate_vapor_pressure(
    mode: wp.int32,
    parameters: Any,
    species_idx: Any,
    temperature: wp.float64,
) -> wp.float64:
    """Evaluate one validated per-species vapor-pressure model."""
    if mode == THERMODYNAMICS_MODE_CONSTANT:
        return _constant_vapor_pressure(parameters, species_idx)
    return _buck_vapor_pressure(temperature)


@wp.kernel
def _refresh_vapor_pressure_kernel(
    modes: Any,
    parameters: Any,
    temperature: Any,
    vapor_pressure: Any,
) -> None:
    """Refresh all per-box, per-species vapor-pressure values."""
    box_idx, species_idx = wp.tid()  # type: ignore[misc]
    vapor_pressure[box_idx, species_idx] = _evaluate_vapor_pressure(
        modes[species_idx],
        parameters,
        species_idx,
        temperature[box_idx],
    )


def _validate_refresh_gas(gas: Any, caller_name: str) -> tuple[int, int, Any]:
    """Validate gas refresh buffers and return their shape and device."""
    vapor_pressure = getattr(gas, "vapor_pressure", None)
    if vapor_pressure is None or not _is_warp_array_like(vapor_pressure):
        raise ValueError(
            f"gas.vapor_pressure must be a Warp array in {caller_name}."
        )
    if vapor_pressure.dtype != wp.float64:
        raise ValueError(
            f"gas.vapor_pressure must use dtype {wp.float64} in {caller_name}."
        )
    if len(vapor_pressure.shape) != 2:
        raise ValueError(
            "gas.vapor_pressure must have shape (n_boxes, n_species) in "
            f"{caller_name}."
        )
    n_boxes, n_species = vapor_pressure.shape
    device = vapor_pressure.device
    _validate_array_metadata(
        "molar_mass",
        getattr(gas, "molar_mass", None),
        (n_species,),
        wp.float64,
        device,
        caller_name,
        "gas",
    )
    return n_boxes, n_species, device


def refresh_vapor_pressure_gpu(
    thermodynamics: ThermodynamicsConfig,
    gas: Any,
    temperature: Any,
) -> None:
    """Refresh vapor pressure from device-local thermodynamic models.

    Import this concrete-module API from
    ``particula.gpu.kernels.thermodynamics``; it is intentionally not exported
    from ``particula.gpu.kernels``.

    Args:
        thermodynamics: Validated device-local per-species model sidecar.
        gas: Caller-owned ``WarpGasData`` whose vapor pressure is overwritten.
        temperature: Device-local ``wp.float64`` temperatures in K with shape
            ``(n_boxes,)``.

    Raises:
        ValueError: If any input violates the device-array contract.
    """
    caller_name = "refresh_vapor_pressure_gpu"
    n_boxes, n_species, device = _validate_refresh_gas(gas, caller_name)
    _validate_array_metadata(
        "temperature",
        temperature,
        (n_boxes,),
        wp.float64,
        device,
        caller_name,
        "",
    )
    _validate_positive_finite_array("temperature", temperature, caller_name)
    validate_thermodynamics_config(
        thermodynamics,
        n_species,
        device,
        gas.molar_mass,
        caller_name,
    )
    wp.launch(
        _refresh_vapor_pressure_kernel,
        dim=(n_boxes, n_species),
        inputs=[
            thermodynamics.modes,
            thermodynamics.parameters,
            temperature,
            gas.vapor_pressure,
        ],
        device=device,
    )
