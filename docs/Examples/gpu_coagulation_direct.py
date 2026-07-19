"""Run direct GPU coagulation with explicit CPU-to-Warp transfers.

This standalone example demonstrates the bounded, low-level,
particle-resolved Brownian coagulation path. It defaults to Warp ``device="cpu"``
and reuses caller-owned collision and RNG sidecars for two direct calls before
explicitly restoring a CPU checkpoint. It has no hidden CPU fallback, Runnable
API, CUDA requirement, or performance claim.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from particula.particles import ParticleData

_FORCE_NO_WARP_ENV = "PARTICULA_EXAMPLE_FORCE_NO_WARP"


@dataclass
class ExampleRun:
    """Store metadata and optional outputs from the direct coagulation path.

    All optional fields are ``None`` when Warp is unavailable or disabled.

    Attributes:
        output: Deterministic, human-readable execution metadata.
        particle_data: CPU checkpoint restored after both direct calls.
        collision_pairs: Caller-owned collision-pair sidecar.
        n_collisions: Caller-owned per-box collision-count sidecar.
        rng_states: Caller-owned persistent RNG sidecar.
    """

    output: list[str]
    particle_data: ParticleData | None = None
    collision_pairs: Any | None = None
    n_collisions: Any | None = None
    rng_states: Any | None = None


def _build_particle_data() -> ParticleData:
    """Create a deterministic one-box particle-resolved CPU fixture.

    Returns:
        Eight particle slots with six active particles and two inactive slots.
    """
    return ParticleData(
        masses=np.array(
            [
                [
                    [1.0e-21],
                    [1.3e-21],
                    [1.7e-21],
                    [2.2e-21],
                    [2.8e-21],
                    [3.5e-21],
                    [0.0],
                    [4.0e-21],
                ]
            ],
            dtype=np.float64,
        ),
        concentration=np.array(
            [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]],
            dtype=np.float64,
        ),
        charge=np.array(
            [[1.0, -1.0, 2.0, -2.0, 0.0, 3.0, -3.0, 4.0]],
            dtype=np.float64,
        ),
        density=np.array([1000.0], dtype=np.float64),
        volume=np.array([1.0e-18], dtype=np.float64),
    )


def _warp_enabled() -> bool:
    """Return whether optional Warp execution is available and enabled.

    Returns:
        ``True`` unless Warp is unavailable or the force-no-Warp flag is set.
    """
    if os.getenv(_FORCE_NO_WARP_ENV) == "1":
        return False
    warp_available, _, _ = _load_gpu_helpers()
    return warp_available


def _load_gpu_helpers() -> tuple[bool, Any, Any]:
    """Lazily load GPU availability and explicit particle-transfer helpers.

    Returns:
        Warp availability, CPU-to-Warp conversion, and Warp-to-CPU restoration
        helpers in that order.
    """
    gpu = importlib.import_module("particula.gpu")
    return (
        gpu.WARP_AVAILABLE,
        gpu.to_warp_particle_data,
        gpu.from_warp_particle_data,
    )


def _load_gpu_runtime() -> tuple[Any, Any, Any]:
    """Lazily load Warp plus the public step and concrete configuration.

    Returns:
        Warp, public ``coagulation_step_gpu``, and concrete
        ``CoagulationMechanismConfig`` in that order.
    """
    wp = importlib.import_module("warp")
    kernels = importlib.import_module("particula.gpu.kernels")
    coagulation = importlib.import_module("particula.gpu.kernels.coagulation")
    return (
        wp,
        kernels.coagulation_step_gpu,
        coagulation.CoagulationMechanismConfig,
    )


def _output_prefix(particle_data: ParticleData) -> list[str]:
    """Build deterministic metadata describing the CPU fixture."""
    return [
        "Canonical path: docs/Examples/gpu_coagulation_direct.py",
        (
            "ParticleData constructed: "
            f"masses={particle_data.masses.shape}, "
            f"concentration={particle_data.concentration.shape}, "
            f"charge={particle_data.charge.shape}, "
            f"density={particle_data.density.shape}, "
            f"volume={particle_data.volume.shape}"
        ),
    ]


def run_example(device: str = "cpu") -> ExampleRun:
    """Run two direct Brownian calls with persistent caller-owned sidecars.

    The enabled route transfers the CPU fixture explicitly, executes the public
    low-level step twice, and restores a CPU checkpoint only after both calls
    succeed. Failures from runtime loading, conversion, allocation, execution,
    or restoration propagate without a success result.

    Args:
        device: Warp device for the optional direct path. Defaults to Warp CPU.

    Returns:
        Metadata only when disabled, otherwise the restored checkpoint and
        caller-owned sidecars.
    """
    particle_data = _build_particle_data()
    output = _output_prefix(particle_data)
    if not _warp_enabled():
        output.append("Warp is unavailable or disabled; no kernel ran.")
        return ExampleRun(output=output)

    _, to_warp_particle_data, from_warp_particle_data = _load_gpu_helpers()
    wp, coagulation_step_gpu, mechanism_config_type = _load_gpu_runtime()
    gpu_particle_data = to_warp_particle_data(particle_data, device=device)
    n_boxes = particle_data.n_boxes
    collision_pairs = wp.zeros((n_boxes, 1, 2), dtype=wp.int32, device=device)
    n_collisions = wp.zeros((n_boxes,), dtype=wp.int32, device=device)
    rng_states = wp.zeros((n_boxes,), dtype=wp.uint32, device=device)
    mechanism_config = mechanism_config_type(
        mechanisms=("brownian",),
        distribution_type="particle_resolved",
    )
    common_arguments = {
        "temperature": 298.15,
        "pressure": 101325.0,
        "time_step": 1.0,
        "volume": gpu_particle_data.volume,
        "max_collisions": 1,
        "rng_seed": 41,
        "collision_pairs": collision_pairs,
        "n_collisions": n_collisions,
        "rng_states": rng_states,
        "mechanism_config": mechanism_config,
    }
    _, collision_pairs, n_collisions = coagulation_step_gpu(
        gpu_particle_data,
        initialize_rng=True,
        **common_arguments,
    )
    _, collision_pairs, n_collisions = coagulation_step_gpu(
        gpu_particle_data,
        initialize_rng=False,
        **common_arguments,
    )
    restored_particle_data = from_warp_particle_data(gpu_particle_data)
    output.extend(
        [
            "Explicit helpers: CPU→Warp conversion -> direct coagulation -> CPU checkpoint",
            (
                "Direct Brownian coagulation complete: "
                f"device={device}, calls=2, collision_pairs="
                f"{collision_pairs.shape}, n_collisions={n_collisions.shape}"
            ),
            (
                "Final checkpoint restored: "
                f"particle_masses={restored_particle_data.masses.shape}"
            ),
            "Three-item direct return; collision and RNG sidecars remain caller-owned.",
            "Persistent RNG state is initialized once and reused by the second call.",
        ]
    )
    return ExampleRun(
        output=output,
        particle_data=restored_particle_data,
        collision_pairs=collision_pairs,
        n_collisions=n_collisions,
        rng_states=rng_states,
    )


def main() -> None:
    """Run the example and print metadata only after successful execution."""
    for line in run_example().output:
        print(line)


if __name__ == "__main__":
    main()
