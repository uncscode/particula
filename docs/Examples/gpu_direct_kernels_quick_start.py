"""Run direct condensation with explicit CPU↔Warp transfers.

This example demonstrates only the supported low-level, gas-coupled
condensation path. It lazily imports Warp-only APIs, reuses fixed-shape,
caller-owned sidecars across two direct calls, and explicitly restores final
particle and gas checkpoints. Without Warp, or when the force-no-Warp flag is
set, it constructs CPU fixtures and reports that no kernel ran.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from particula.gas import GasData
from particula.gpu import (
    WARP_AVAILABLE,
    from_warp_gas_data,
    from_warp_particle_data,
    to_warp_gas_data,
    to_warp_particle_data,
)
from particula.particles import ParticleData

_FORCE_NO_WARP_ENV = "PARTICULA_EXAMPLE_FORCE_NO_WARP"


@dataclass
class ExampleRun:
    """Store output and optional Warp-path checkpoints from the example.

    Warp-backed fields are ``None`` when Warp is unavailable or disabled. This
    keeps importing and executing the CPU-only branch independent of Warp.

    Attributes:
        output: Deterministic, human-readable execution metadata.
        particle_data: Final particle checkpoint restored to CPU data.
        gas_data: Final gas checkpoint restored to CPU data.
        total_mass_transfer: Caller-owned final transfer accumulator.
        scratch_buffers: Caller-owned condensation scratch sidecar.
        latent_heat: Caller-owned per-species latent-heat sidecar.
        energy_transfer: Caller-owned per-box energy diagnostic sidecar.
    """

    output: list[str]
    particle_data: ParticleData | None = None
    gas_data: GasData | None = None
    total_mass_transfer: Any | None = None
    scratch_buffers: Any | None = None
    latent_heat: Any | None = None
    energy_transfer: Any | None = None


def _build_particle_data() -> ParticleData:
    """Create one-box, two-particle, one-species CPU particle data.

    The sole mass column corresponds to the ``Water`` species in
    :func:`_build_gas_data`.

    Returns:
        Particle data with ``float64`` state arrays and masses shaped
        ``(1, 2, 1)``.
    """
    return ParticleData(
        masses=np.array([[[1.0e-18], [1.2e-18]]], dtype=np.float64),
        concentration=np.array([[1.0, 1.0]], dtype=np.float64),
        charge=np.array([[0.0, 0.0]], dtype=np.float64),
        density=np.array([1000.0], dtype=np.float64),
        volume=np.array([1.0e-6], dtype=np.float64),
    )


def _build_gas_data() -> GasData:
    """Create one-box gas data ordered to match the particle mass column.

    Returns:
        Gas data for ``Water`` with ``float64`` molar mass and concentration,
        plus a Boolean partitioning mask.
    """
    return GasData(
        name=["Water"],
        molar_mass=np.array([0.018], dtype=np.float64),
        concentration=np.array([[1.0e-6]], dtype=np.float64),
        partitioning=np.array([True]),
    )


def _build_vapor_pressure() -> np.ndarray:
    """Create initial derived GPU vapor-pressure storage in Pa.

    Condensation overwrites this GPU helper storage from the thermodynamic
    sidecar before transfer, so it is not a caller-supplied physics source.

    Returns:
        Initial ``float64`` storage shaped ``(1, 1)`` with pressure in Pa.
    """
    return np.array([[2330.0]], dtype=np.float64)


def _warp_enabled() -> bool:
    """Return whether optional Warp execution is available and enabled.

    The ``PARTICULA_EXAMPLE_FORCE_NO_WARP=1`` environment variable disables
    the direct path even when Warp is installed.

    Returns:
        ``True`` only when Warp is available and the force-no-Warp flag is not
        set.
    """
    return WARP_AVAILABLE and os.getenv(_FORCE_NO_WARP_ENV) != "1"


def _load_gpu_runtime() -> tuple[Any, Any, Any, Any]:
    """Lazily load Warp and the direct condensation runtime contract.

    This helper runs only after the Warp guard succeeds. It deliberately loads
    the public step entry point and the concrete sidecar classes rather than a
    high-level runnable API.

    Returns:
        A tuple containing Warp, public ``condensation_step_gpu``, concrete
        ``CondensationScratchBuffers``, and concrete
        ``ThermodynamicsConfig``.
    """
    wp = importlib.import_module("warp")
    kernels = importlib.import_module("particula.gpu.kernels")
    condensation = importlib.import_module("particula.gpu.kernels.condensation")
    thermodynamics = importlib.import_module(
        "particula.gpu.kernels.thermodynamics"
    )
    return (
        wp,
        kernels.condensation_step_gpu,
        condensation.CondensationScratchBuffers,
        thermodynamics.ThermodynamicsConfig,
    )


def _output_prefix(particle_data: ParticleData, gas_data: GasData) -> list[str]:
    """Build deterministic metadata describing the CPU fixtures.

    Args:
        particle_data: CPU particle fixture to describe.
        gas_data: CPU gas fixture to describe.

    Returns:
        Human-readable fixture metadata in its printed order.
    """
    return [
        "Canonical path: docs/Examples/gpu_direct_kernels_quick_start.py",
        (
            "ParticleData constructed: "
            f"masses={particle_data.masses.shape}, "
            f"concentration={particle_data.concentration.shape}, "
            f"charge={particle_data.charge.shape}, "
            f"density={particle_data.density.shape}, "
            f"volume={particle_data.volume.shape}"
        ),
        (
            "GasData constructed: "
            f"concentration={gas_data.concentration.shape}, "
            f"molar_mass={gas_data.molar_mass.shape}, "
            f"partitioning={gas_data.partitioning.shape}, "
            f"names={gas_data.name}"
        ),
    ]


def run_example(device: str = "cpu") -> ExampleRun:
    """Run two direct-condensation calls with reused caller-owned sidecars.

    The enabled path explicitly converts CPU fixtures to Warp data, executes
    two sequential low-level calls with the same scratch and physical-property
    sidecars, then restores final particle and gas checkpoints. The transfer
    and energy outputs are cleared per successful call, so returned diagnostics
    describe the final call only. It does not provide a fallback kernel path:
    when Warp is unavailable or disabled, it returns CPU-fixture metadata
    stating that no kernel ran.

    Args:
        device: Warp device for the optional kernel path. Defaults to Warp CPU.

    Returns:
        An :class:`ExampleRun` containing CPU-only metadata and ``None``
        Warp fields when disabled, or restored final checkpoints and the exact
        caller-owned sidecars after both direct calls.

    Raises:
        Exception: Propagates conversion, allocation, and direct-kernel errors
            without claiming a successful checkpoint or rolling back device
            state.
    """
    particle_data = _build_particle_data()
    gas_data = _build_gas_data()
    output = _output_prefix(particle_data, gas_data)

    if not _warp_enabled():
        output.append("Warp is unavailable or disabled; no kernel ran.")
        return ExampleRun(output=output)

    (
        wp,
        condensation_step_gpu,
        condensation_scratch_buffers,
        thermodynamics_config,
    ) = _load_gpu_runtime()
    saved_names = list(gas_data.name)
    gpu_particle_data = to_warp_particle_data(particle_data, device=device)
    gpu_gas_data = to_warp_gas_data(
        gas_data,
        device=device,
        vapor_pressure=_build_vapor_pressure(),
    )

    transfer_shape = particle_data.masses.shape
    box_shape = (particle_data.n_boxes,)
    species_shape = gas_data.concentration.shape
    scratch_buffers = condensation_scratch_buffers(
        work_mass_transfer=wp.zeros(
            transfer_shape, dtype=wp.float64, device=device
        ),
        total_mass_transfer=wp.zeros(
            transfer_shape, dtype=wp.float64, device=device
        ),
        dynamic_viscosity=wp.zeros(box_shape, dtype=wp.float64, device=device),
        mean_free_path=wp.zeros(box_shape, dtype=wp.float64, device=device),
        positive_mass_transfer_demand=wp.zeros(
            species_shape, dtype=wp.float64, device=device
        ),
        negative_mass_transfer_release=wp.zeros(
            species_shape, dtype=wp.float64, device=device
        ),
        positive_mass_transfer_scale=wp.zeros(
            species_shape, dtype=wp.float64, device=device
        ),
    )
    latent_heat = wp.array(
        np.array([2.26e6], dtype=np.float64), dtype=wp.float64, device=device
    )
    energy_transfer = wp.zeros(species_shape, dtype=wp.float64, device=device)
    surface_tension = wp.array(
        np.array([0.072], dtype=np.float64), dtype=wp.float64, device=device
    )
    mass_accommodation = wp.array(
        np.array([1.0], dtype=np.float64), dtype=wp.float64, device=device
    )
    diffusion_coefficient_vapor = wp.array(
        np.array([2.0e-5], dtype=np.float64), dtype=wp.float64, device=device
    )
    thermodynamics = thermodynamics_config(
        modes=wp.array(
            np.array([0], dtype=np.int32), dtype=wp.int32, device=device
        ),
        parameters=wp.array(
            np.array([[1.0e-6, 0.0, 0.0, 0.0]], dtype=np.float64),
            dtype=wp.float64,
            device=device,
        ),
        molar_mass_reference=wp.array(
            gas_data.molar_mass, dtype=wp.float64, device=device
        ),
    )

    _, total_mass_transfer = condensation_step_gpu(
        gpu_particle_data,
        gpu_gas_data,
        temperature=298.15,
        pressure=101325.0,
        time_step=0.1,
        surface_tension=surface_tension,
        mass_accommodation=mass_accommodation,
        diffusion_coefficient_vapor=diffusion_coefficient_vapor,
        thermodynamics=thermodynamics,
        scratch_buffers=scratch_buffers,
        latent_heat=latent_heat,
        energy_transfer=energy_transfer,
    )
    _, total_mass_transfer = condensation_step_gpu(
        gpu_particle_data,
        gpu_gas_data,
        temperature=298.15,
        pressure=101325.0,
        time_step=0.1,
        surface_tension=surface_tension,
        mass_accommodation=mass_accommodation,
        diffusion_coefficient_vapor=diffusion_coefficient_vapor,
        thermodynamics=thermodynamics,
        scratch_buffers=scratch_buffers,
        latent_heat=latent_heat,
        energy_transfer=energy_transfer,
    )
    restored_particle_data = from_warp_particle_data(gpu_particle_data)
    restored_gas_data = from_warp_gas_data(gpu_gas_data, name=saved_names)
    output.extend(
        [
            "Explicit helpers: CPU→Warp conversion -> direct condensation -> CPU checkpoints",
            (
                "Direct condensation complete: "
                f"device={device}, calls=2, final_call_transfer_shape="
                f"{total_mass_transfer.shape}"
            ),
            (
                "Final checkpoints restored: "
                f"particle_masses={restored_particle_data.masses.shape}, "
                f"gas_concentration={restored_gas_data.concentration.shape}, "
                f"names={restored_gas_data.name}"
            ),
            "Two-item kernel return; energy remains a caller-owned sidecar.",
            (
                "Fixed-shape fp64 scratch, physical-property, latent-heat, "
                "and energy sidecars reused."
            ),
            "Transfer and energy diagnostics are reset per call and report the final call.",
        ]
    )
    return ExampleRun(
        output=output,
        particle_data=restored_particle_data,
        gas_data=restored_gas_data,
        total_mass_transfer=total_mass_transfer,
        scratch_buffers=scratch_buffers,
        latent_heat=latent_heat,
        energy_transfer=energy_transfer,
    )


def main() -> None:
    """Run the example and print its deterministic execution metadata.

    The metadata identifies the no-Warp outcome or the explicit conversion,
    direct-call, sidecar-reuse, and final-checkpoint restoration route.
    """
    for line in run_example().output:
        print(line)


if __name__ == "__main__":
    main()
