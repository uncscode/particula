# %% [markdown]
# # Direct Brownian coagulation on CUDA
#
# Six active particles and two inactive slots make the bookkeeping visible:
# collisions reduce particle number while preserving represented mass and
# charge. The same random stream and collision storage survive both calls.
# Run with `--cuda` to reproduce the published figures. The unflagged command
# retains the small Warp CPU regression; large stochastic ensembles are shown
# in the prepared simulation studies.

# %%
"""Run selected-adapter GPU coagulation with explicit CPU-to-Warp transfers.

This standalone example demonstrates the bounded, low-level,
particle-resolved Brownian coagulation path. It defaults to Warp ``device="cpu"``
and reuses caller-owned collision and RNG sidecars for two concrete
selected-adapter dispatches before explicitly restoring CPU particle data. It
has no hidden CPU fallback, Runnable API, CUDA requirement, restart/checkpoint
support, or performance claim.

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
    """Store metadata and optional outputs from selected-adapter dispatch.

    All optional fields are ``None`` when Warp is unavailable or disabled.

    Attributes:
        output: Deterministic, human-readable execution metadata.
        particle_data: CPU particle data restored after both selected dispatches.
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
    """Create six unit-count particles in 1e-18 m³ and two inactive slots.

    Inactive material/charge is deliberately retained to demonstrate that
    coagulation ignores zero-count slots. These are not canonical free slots
    suitable for nucleation or activation.

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
        distribution_type="particle_resolved",
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


def _load_gpu_runtime() -> tuple[Any, Any, Any, Any, Any]:
    """Lazily load Warp and concrete selected-Brownian adapter classes.

    Returns:
        Warp, the Brownian marker, Warp state, execution state, and execution
        adapter classes in that order.
    """
    wp = importlib.import_module("warp")
    coagulation = importlib.import_module(
        "particula.execution.adapters.coagulation"
    )
    return (
        wp,
        coagulation.BrownianCoagulationConfig,
        coagulation.WarpBrownianCoagulationState,
        coagulation.WarpBrownianCoagulationExecutionState,
        coagulation.WarpBrownianCoagulationExecutionAdapter,
    )


def _output_prefix(particle_data: ParticleData) -> list[str]:
    """Build deterministic metadata describing the CPU fixture."""
    return [
        "Canonical path: docs/examples/gpu/direct/gpu_coagulation_direct.py",
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
    """Run two selected-Brownian calls with persistent caller-owned sidecars.

    The enabled route transfers the CPU fixture explicitly, executes the
    concrete-only selected adapter twice, synchronizes once, and restores a CPU
    particle data only after both calls succeed. Failures from runtime loading,
    conversion, allocation, execution, synchronization, or restoration
    propagate without a success result.

    Args:
        device: Warp device for the optional selected path. Defaults to Warp CPU.

    Returns:
        Metadata only when disabled, otherwise the restored particle data and
        caller-owned sidecars.
    """
    particle_data = _build_particle_data()
    output = _output_prefix(particle_data)
    if not _warp_enabled():
        output.append("Warp is unavailable or disabled; no kernel ran.")
        return ExampleRun(output=output)

    _, to_warp_particle_data, from_warp_particle_data = _load_gpu_helpers()
    (
        wp,
        config_type,
        state_type,
        execution_state_type,
        adapter_type,
    ) = _load_gpu_runtime()
    gpu_particle_data = to_warp_particle_data(particle_data, device=device)
    n_boxes = particle_data.n_boxes
    collision_capacity = max(1, particle_data.n_particles // 2)
    collision_pairs = wp.zeros(
        (n_boxes, collision_capacity, 2), dtype=wp.int32, device=device
    )
    n_collisions = wp.zeros((n_boxes,), dtype=wp.int32, device=device)
    rng_states = wp.zeros((n_boxes,), dtype=wp.uint32, device=device)
    adapter = adapter_type()
    for initialize_rng in (True, False):
        state = state_type(
            config_type(),
            gpu_particle_data,
            298.15,
            101325.0,
            1.0,
            volume=gpu_particle_data.volume,
            collision_pairs=collision_pairs,
            n_collisions=n_collisions,
            rng_states=rng_states,
            rng_seed=41,
            initialize_rng=initialize_rng,
        )
        result = adapter.execute(execution_state_type(state))
        backend_result = result.backend_result.value
        collision_pairs = backend_result.collision_pairs
        n_collisions = backend_result.n_collisions
    wp.synchronize()
    restored_particle_data = from_warp_particle_data(
        gpu_particle_data, sync=False
    )
    # Resolved concentration is a count, so extensive species mass has no
    # extra volume factor. Do not count material in zero-count inactive slots.
    np.testing.assert_allclose(
        np.sum(
            restored_particle_data.masses
            * restored_particle_data.concentration[:, :, None],
            axis=1,
        ),
        np.sum(
            particle_data.masses * particle_data.concentration[:, :, None],
            axis=1,
        ),
        rtol=1e-12,
        atol=1e-30,
    )
    output.extend(
        [
            "Explicit helpers: CPU→Warp conversion -> selected-adapter dispatch -> CPU restoration",
            (
                "Selected Brownian coagulation complete: "
                f"device={device}, calls=2, collision_pairs="
                f"{collision_pairs.shape}, n_collisions={n_collisions.shape}"
            ),
            (
                "Final particle data restored: "
                f"particle_masses={restored_particle_data.masses.shape}"
            ),
            "Adapter result retains supplied collision and RNG sidecars by identity.",
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


# %% [markdown]
# ## Inspect the changed size distribution
#
# Histogram weights are represented counts divided by physical box volume.
# Inactive slots contribute nothing, even if they retain old mass values.
# A single six-particle realization illustrates merging; it is not an estimate
# of ensemble variability or a throughput benchmark.


# %%
def run_publication():
    """Require CUDA and show particle-number change alongside conserved mass."""
    import matplotlib.pyplot as plt
    import warp as wp

    if not _warp_enabled() or not wp.is_cuda_available():
        raise RuntimeError("This publication run requires CUDA")
    initial = _build_particle_data()
    result = run_example(device="cuda:0")
    final = result.particle_data
    if final is None:
        raise RuntimeError("CUDA produced no particle snapshot")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    edges = np.geomspace(1, 30, 16)
    for data, label in ((initial, "Initial"), (final, "After two calls")):
        volume = np.sum(data.masses / data.density, axis=2)[0]
        radius = np.cbrt(3 * volume / (4 * np.pi)) * 1e9
        histogram, _ = np.histogram(
            radius, bins=edges, weights=data.concentration[0] / data.volume[0]
        )
        axes[0].stairs(histogram, edges, label=label, linewidth=2)
    axes[0].set(
        xscale="log",
        xlabel="Radius (nm)",
        ylabel="Number density per bin (m$^{-3}$)",
    )
    axes[0].legend()
    initial_mass = np.sum(initial.masses * initial.concentration[..., None])
    final_mass = np.sum(final.masses * final.concentration[..., None])
    fraction = [
        final.concentration.sum() / initial.concentration.sum(),
        final_mass / initial_mass,
    ]
    axes[1].bar(
        ["Represented number", "Represented mass"],
        fraction,
        color=["#be6c39", "#287a90"],
    )
    axes[1].axhline(1, color="0.5", linestyle="--")
    axes[1].set(ylabel="Final / initial", ylim=(0, 1.1))
    fig.suptitle("Brownian collisions · " + wp.get_device("cuda:0").name)
    print("\n".join(result.output))
    plt.show()
    return result


def main() -> None:
    """Run the example and print metadata only after successful execution."""
    for line in run_example().output:
        print(line)


if __name__ == "__main__":
    import sys

    if "--cuda" in sys.argv[1:]:
        run_publication()
    else:
        main()
