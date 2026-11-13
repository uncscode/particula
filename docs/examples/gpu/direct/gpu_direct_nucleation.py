"""Run the bounded direct-Warp nucleation step with explicit transfers.

The package-exported ``nucleation_step_gpu`` is used for execution, while
configuration and sidecar records are imported from their concrete modules.
Warp is required; this example intentionally has no CPU-physics fallback.
"""

import numpy as np
import numpy.testing as npt
import warp as wp
from particula.dynamics.nucleation.species_mapping import NucleationSpeciesMap
from particula.gas import EnvironmentData, GasData
from particula.gpu import (
    from_warp_environment_data,
    from_warp_gas_data,
    from_warp_particle_data,
    to_warp_environment_data,
    to_warp_gas_data,
    to_warp_particle_data,
)
from particula.gpu.kernels import nucleation_step_gpu
from particula.gpu.kernels.exhaustion import ResamplingBuffers
from particula.gpu.kernels.nucleation import (
    NucleationConfig,
    NucleationDiagnosticBuffers,
    NucleationExhaustionBuffers,
    NucleationExhaustionControls,
    NucleationFinalizedDemandBuffers,
    NucleationScratchBuffers,
)
from particula.gpu.nucleation_species_mapping import (
    prepare_nucleation_species_map,
)
from particula.particles import ParticleData
from particula.util.constants import AVOGADRO_NUMBER


def _buffers(
    device: str,
) -> tuple[
    NucleationScratchBuffers,
    NucleationFinalizedDemandBuffers,
    NucleationDiagnosticBuffers,
    NucleationExhaustionBuffers,
]:
    """Allocate caller-owned sidecars for the B=1, N=2, S=1 fixture.

    Args:
        device: Warp device that owns every allocated sidecar.

    Returns:
        Scratch, finalized-demand, diagnostic, and exhaustion sidecars on
        ``device``.
    """

    def f64(shape: tuple[int, ...]) -> object:
        """Allocate a float64 sidecar on the selected Warp device.

        Args:
            shape: Dimensions of the sidecar.

        Returns:
            A zero-initialized Warp float64 array.
        """
        return wp.zeros(shape, dtype=wp.float64, device=device)

    def ones(shape: tuple[int, ...]) -> object:
        """Allocate a float64 sidecar initialized to one.

        Args:
            shape: Dimensions of the sidecar.

        Returns:
            A one-initialized Warp float64 array.
        """
        return wp.ones(shape, dtype=wp.float64, device=device)

    def i32(shape: tuple[int, ...]) -> object:
        """Allocate an int32 sidecar on the selected Warp device.

        Args:
            shape: Dimensions of the sidecar.

        Returns:
            A zero-initialized Warp int32 array.
        """
        return wp.zeros(shape, dtype=wp.int32, device=device)

    resampling = ResamplingBuffers(
        i32((1,)),
        i32((1,)),
        i32((1, 2)),
        i32((1, 2)),
        i32((1, 2)),
        f64((1, 2, 1)),
        f64((1, 2)),
        f64((1, 2)),
        f64((1, 2)),
        f64((1,)),
        f64((1,)),
        f64((1,)),
        f64((1,)),
        i32((1,)),
    )
    return (
        NucleationScratchBuffers(f64((1,)), f64((1,)), f64((1,))),
        NucleationFinalizedDemandBuffers(i32((1,)), f64((1,)), f64((1, 1))),
        NucleationDiagnosticBuffers(
            i32((1,)), i32((1, 2)), i32((1, 2)), i32((1,)), i32((1,))
        ),
        NucleationExhaustionBuffers(
            resampling,
            f64((1,)),
            f64((1,)),
            ones((1,)),
            ones((1,)),
            ones((1,)),
            f64((1,)),
            i32((1,)),
            i32((1,)),
            i32((1,)),
            i32((1,)),
            i32((1, 2)),
        ),
    )


def run_example(
    volume: float = 1.0,
) -> tuple[ParticleData, GasData, EnvironmentData]:
    """Run one deterministic direct-Warp nucleation event on Warp CPU.

    Args:
        volume: Represented volume in m³. Particle concentration stores counts;
            gas concentration stores kg/m³. The two slots accommodate the
            default one-event demonstration and its two-m³ regression case.

    Returns:
        Restored CPU particle, gas, and environment data after synchronization.
    """
    device = "cpu"
    particles = ParticleData(
        masses=np.zeros((1, 2, 1), dtype=np.float64),
        concentration=np.zeros((1, 2), dtype=np.float64),
        charge=np.zeros((1, 2), dtype=np.float64),
        density=np.array([1000.0], dtype=np.float64),
        volume=np.array([volume], dtype=np.float64),
        distribution_type="particle_resolved",
    )
    gas = GasData(
        name=["precursor"],
        molar_mass=np.array([0.1]),
        concentration=np.array([[1.0]], dtype=np.float64),
        partitioning=np.array([True]),
    )
    environment = EnvironmentData(
        temperature=np.array([300.0]),
        pressure=np.array([101325.0]),
        saturation_ratio=np.array([[1.0]]),
    )
    initial_inventory = gas.concentration * particles.volume[:, None]
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(gas, device=device)
    gpu_environment = to_warp_environment_data(environment, device=device)
    # Names remain host-owned; upload explicit configuration for these exact
    # carriers after transfer. Equal species widths do not define a mapping.
    species_binding = prepare_nucleation_species_map(
        gpu_particles,
        gpu_gas,
        gas_names=gas.name,
        species_map=NucleationSpeciesMap(tuple(gas.name), ((0, 0),)),
    )
    scratch, finalized, diagnostics, exhaustion = _buffers(device)
    config = NucleationConfig(
        rate_law="activation",
        coefficient=0.1 / AVOGADRO_NUMBER,
        survival_factor=1.0,
        precursor_index=0,
        molecule_counts=(1,),
        formation_diameter=1.0e-9,
        precursor_number_concentration_lower=0.0,
        precursor_number_concentration_upper=1.0e30,
        temperature_lower=200.0,
        temperature_upper=400.0,
    )
    returned_particles, returned_gas = nucleation_step_gpu(
        gpu_particles,
        gpu_gas,
        config,
        1.0,
        environment=gpu_environment,
        species_binding=species_binding,
        scratch=scratch,
        finalized_demand=finalized,
        diagnostics=diagnostics,
        exhaustion_controls=NucleationExhaustionControls(False, False),
        exhaustion_buffers=exhaustion,
    )
    assert returned_particles is gpu_particles and returned_gas is gpu_gas
    wp.synchronize()
    restored_particles = from_warp_particle_data(gpu_particles, sync=False)
    restored_gas = from_warp_gas_data(gpu_gas, name=gas.name, sync=False)
    restored_environment = from_warp_environment_data(
        gpu_environment,
        sync=False,
    )
    assert restored_particles.masses.shape == (1, 2, 1)
    assert restored_gas.concentration.shape == (1, 1)
    assert int(np.count_nonzero(restored_particles.concentration)) > 0
    assert restored_gas.concentration[0, 0] <= gas.concentration[0, 0]
    particle_inventory = np.sum(
        restored_particles.masses
        * restored_particles.concentration[:, :, None],
        axis=1,
    )
    npt.assert_allclose(
        particle_inventory
        + restored_gas.concentration * restored_particles.volume[:, None],
        initial_inventory,
        rtol=1e-12,
        atol=1e-30,
    )
    return restored_particles, restored_gas, restored_environment


if __name__ == "__main__":
    run_example()
    print("Direct Warp nucleation example completed on device=cpu.")
