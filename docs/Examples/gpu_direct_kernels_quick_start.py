"""Canonical direct GPU kernel quick-start with explicit CPU↔Warp transfers.

This runnable example lives at ``docs/Examples/`` as the single canonical
quick-start for the low-level Warp kernel path. It keeps CPU-owned
``ParticleData`` and ``GasData`` fixtures explicit, transfers them with the
published ``to_warp_*`` / ``from_warp_*`` helpers, and imports direct step
functions only from ``particula.gpu.kernels`` after the optional
``WARP_AVAILABLE`` guard passes. The condensation call supplies a
caller-owned thermodynamic sidecar and an initial GPU vapor-pressure buffer.
After validation, the call refreshes that derived buffer on-device from the
current temperature before mass transfer.
"""

from __future__ import annotations

import importlib
import os
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


def _build_particle_data() -> ParticleData:
    """Create deterministic particle data for the quick-start example.

    Returns:
        Single-box particle data with two particles and one species.
    """
    return ParticleData(
        masses=np.array([[[1.0e-18], [1.2e-18]]], dtype=np.float64),
        concentration=np.array([[1.0, 1.0]], dtype=np.float64),
        charge=np.array([[0.0, 0.0]], dtype=np.float64),
        density=np.array([1000.0], dtype=np.float64),
        volume=np.array([1.0e-6], dtype=np.float64),
    )


def _build_gas_data() -> GasData:
    """Create deterministic gas data for the quick-start example.

    Returns:
        Single-box gas data for one condensable species.
    """
    return GasData(
        name=["Water"],
        molar_mass=np.array([0.018], dtype=np.float64),
        concentration=np.array([[1.0e-6]], dtype=np.float64),
        partitioning=np.array([True]),
    )


def _build_vapor_pressure() -> np.ndarray:
    """Create deterministic vapor pressure input for condensation.

    Returns:
        Vapor pressure array shaped ``(n_boxes, n_species)``.
    """
    return np.array([[2330.0]], dtype=np.float64)


def _warp_enabled() -> bool:
    """Return whether the optional Warp execution branch should run.

    Returns:
        ``True`` when Warp is available and the example is not forced into the
        CPU-only path.
    """
    return WARP_AVAILABLE and os.getenv(_FORCE_NO_WARP_ENV) != "1"


def _load_gpu_runtime() -> tuple[Any, Any, Any]:
    """Load Warp and the supported direct kernel entry points lazily.

    Returns:
        Tuple of the imported ``warp`` module plus the supported condensation
        and coagulation direct-kernel callables.
    """
    wp = importlib.import_module("warp")
    kernels = importlib.import_module("particula.gpu.kernels")
    return wp, kernels.condensation_step_gpu, kernels.coagulation_step_gpu


def run_example(device: str = "cpu") -> list[str]:
    """Run the canonical direct-kernel quick-start.

    The condensation branch builds caller-owned thermodynamic buffers and
    refreshes the derived vapor-pressure buffer on-device before mass transfer.

    Args:
        device: Warp device for the optional kernel path. Defaults to the Warp
            CPU backend so the example remains portable.

    Returns:
        Human-readable output lines describing the example execution.

    Raises:
        RuntimeError: Propagated if a direct kernel call fails.
    """
    particle_data = _build_particle_data()
    gas_data = _build_gas_data()

    output = [
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
            f"partitioning={gas_data.partitioning.shape}"
        ),
    ]

    if not _warp_enabled():
        output.append(
            "Warp is optional; direct particula.gpu.kernels imports stay "
            "deferred until WARP_AVAILABLE passes, so this CPU-default "
            "quick-start finished without Warp."
        )
        return output

    wp, condensation_step_gpu, coagulation_step_gpu = _load_gpu_runtime()
    thermodynamics_module = importlib.import_module(
        "particula.gpu.kernels.thermodynamics"
    )

    gpu_particle_data = to_warp_particle_data(particle_data, device=device)
    gpu_gas_data = to_warp_gas_data(
        gas_data,
        device=device,
        vapor_pressure=_build_vapor_pressure(),
    )
    thermodynamics = thermodynamics_module.ThermodynamicsConfig(
        modes=wp.zeros((1,), dtype=wp.int32, device=device),
        parameters=wp.zeros((1, 4), dtype=wp.float64, device=device),
        molar_mass_reference=wp.array(
            gas_data.molar_mass, dtype=wp.float64, device=device
        ),
    )
    _, mass_transfer = condensation_step_gpu(
        gpu_particle_data,
        gpu_gas_data,
        temperature=298.15,
        pressure=101325.0,
        time_step=0.1,
        thermodynamics=thermodynamics,
    )
    restored_gas_data = from_warp_gas_data(gpu_gas_data, name=gas_data.name)

    rng_states = wp.zeros(
        (particle_data.n_boxes,), dtype=wp.uint32, device=device
    )
    _, _, collision_counts = coagulation_step_gpu(
        gpu_particle_data,
        temperature=298.15,
        pressure=101325.0,
        time_step=0.1,
        rng_states=rng_states,
        initialize_rng=True,
        rng_seed=41,
    )
    restored_particle_data = from_warp_particle_data(gpu_particle_data)

    output.extend(
        [
            (
                "Explicit helpers: to_warp_particle_data/to_warp_gas_data -> "
                "particula.gpu.kernels -> "
                "from_warp_particle_data/from_warp_gas_data"
            ),
            (
                "Condensation kernel complete: "
                f"device={device}, temperature=298.15 K, pressure=101325.0 Pa, "
                f"mass_transfer_shape={mass_transfer.shape}"
            ),
            (
                "Gas round trip: "
                f"restored_concentration={restored_gas_data.concentration.shape}, "
                f"restored_names={restored_gas_data.name}"
            ),
            (
                "Coagulation kernel complete: "
                f"device={device}, rng_states=caller-owned, "
                "initialize_rng=True, rng_seed=41, "
                f"collision_counts_shape={collision_counts.shape}"
            ),
            (
                "Particle round trip: "
                f"restored_masses={restored_particle_data.masses.shape}, "
                "restored_concentration="
                f"{restored_particle_data.concentration.shape}"
            ),
            (
                "Direct GPU kernel quick-start complete on the Warp "
                f"{device} path."
            ),
        ]
    )
    return output


def main() -> None:
    """Print the example output lines for manual validation.

    Raises:
        RuntimeError: Propagated if ``run_example()`` fails in the Warp path.
    """
    for line in run_example():
        print(line)


if __name__ == "__main__":
    main()
