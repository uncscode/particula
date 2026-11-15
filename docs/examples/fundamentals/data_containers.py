# %% [markdown]
# # What survives a data transfer?
#
# Moving an aerosol between CPU storage and a device requires preserving physical
# meaning as well as array shape. This example constructs two weighted particles
# and two gas species, then optionally transfers them to Warp on the CPU and back.
# It is a transfer demonstration; no aerosol process or CUDA workload runs.
#
# The particle axes are box, slot, material. Counts `[1, 0.25]` in 10⁻⁶ m³ imply
# slot number densities of 10⁶ and 2.5 × 10⁵ m⁻³. Material masses are kilograms per
# particle, whereas gas concentrations are kg m⁻³. These different measures must
# not be converted with the same volume factor.
#
# Read the construction helpers first, then `run_example()`. The printed shapes
# make the axes visible. On restoration, the caller supplies gas names explicitly;
# vapor pressure is device helper data rather than restored CPU gas storage.
# See [distribution measures](particles/aerosol_distributions.py) for the count and
# density distinction.

# %%
"""Demonstrate data-container construction and optional Warp CPU round trips.

This published entrypoint is the canonical runnable example. It builds public
``ParticleData`` and ``GasData`` containers with the documented single-box
shapes, then optionally exercises Warp-backed transfer helpers on the Warp CPU
backend when Warp is available.
"""

from __future__ import annotations

import os

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
    """Create a single-box particle container for the example.

    Returns:
        ParticleData: Example particle data with documented public shapes for
            masses, concentration, charge, density, and volume.
    """
    return ParticleData(
        masses=np.array([[[1.0e-18, 2.5e-18], [3.5e-18, 0.5e-18]]]),
        concentration=np.array([[1.0, 0.25]]),  # represented counts
        charge=np.array([[0.0, 1.0]]),
        density=np.array([1000.0, 1200.0]),
        volume=np.array([1.0e-6]),
        distribution_type="particle_resolved",
    )


def _build_gas_data() -> GasData:
    """Create a single-box gas container for the example.

    Returns:
        GasData: Example gas data with documented public shapes for species
            names, molar mass, concentration, and partitioning.
    """
    return GasData(
        name=["Water", "H2SO4"],
        molar_mass=np.array([0.018, 0.098]),
        concentration=np.array([[1.0e-6, 2.0e-10]]),
        partitioning=np.array([True, True]),
    )


def _warp_enabled() -> bool:
    """Return whether optional Warp-backed transfers should run.

    Returns:
        bool: ``True`` when Warp is available and the opt-out environment
            variable is not set.
    """
    return WARP_AVAILABLE and os.getenv(_FORCE_NO_WARP_ENV) != "1"


def run_example() -> list[str]:
    """Run the documented CPU example and optional Warp round trips.

    Returns:
        list[str]: Human-readable output lines describing the constructed data
            containers and any optional Warp round-trip results.
    """
    particle_data = _build_particle_data()
    particle_data.validate_representation()
    gas_data = _build_gas_data()
    vapor_pressure = np.array([[2330.0, 120.0]], dtype=np.float64)

    output = [
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
            "Warp-backed transfers are optional; CPU container example "
            "completed without Warp."
        )
        return output

    gpu_particle_data = to_warp_particle_data(particle_data, device="cpu")
    restored_particle_data = from_warp_particle_data(gpu_particle_data)
    output.append(
        "Warp particle round trip: "
        f"restored_masses={restored_particle_data.masses.shape}, "
        f"restored_concentration={restored_particle_data.concentration.shape}"
    )

    gpu_gas_data = to_warp_gas_data(
        gas_data,
        device="cpu",
        vapor_pressure=vapor_pressure,
    )
    restored_gas_data = from_warp_gas_data(
        gpu_gas_data,
        name=gas_data.name,
    )
    output.append(
        "Warp gas round trip: "
        f"restored_concentration={restored_gas_data.concentration.shape}, "
        f"restored_names={restored_gas_data.name}"
    )
    output.append(
        "Gas restore note: names are caller-supplied on restore and "
        "vapor_pressure remains GPU-only helper state."
    )
    return output


def main() -> None:
    """Print the example output lines for manual validation."""
    for line in run_example():
        print(line)


if __name__ == "__main__":
    main()

# %% [markdown]
# ## Read the output as a transfer contract
#
# A round-trip message shows which transfers actually ran and the restored shapes
# and gas names. A CPU-only message means the optional transfer path was skipped;
# it is not evidence of a successful device round trip. Shape agreement alone is
# also weaker than numerical equality of every field.
#
# Before adapting this to measured populations, compare pre/post-transfer masses,
# weights, charge, volume, and ordered species explicitly. Then check number and
# material inventories using the declared representation. Only after these checks
# should a difference be attributed to the process solver. Continue to the
# [direct condensation example](../gpu/direct/gpu_direct_kernels_quick_start.py)
# to see transfers combined with a physical update.
