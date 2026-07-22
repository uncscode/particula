"""Run a deterministic CPU dilution process through the public API."""

from __future__ import annotations

import numpy as np
import particula as par


class ExampleResult:
    """Immutable snapshots and metadata from the CPU dilution example."""

    __slots__ = (
        "particle_initial",
        "particle_final",
        "partitioning_initial",
        "partitioning_final",
        "gas_only_initial",
        "gas_only_final",
        "coefficient",
        "time_step",
        "sub_steps",
    )

    particle_initial: np.ndarray
    particle_final: np.ndarray
    partitioning_initial: np.ndarray
    partitioning_final: np.ndarray
    gas_only_initial: np.ndarray
    gas_only_final: np.ndarray
    coefficient: float
    time_step: float
    sub_steps: int

    def __init__(
        self,
        particle_initial: np.ndarray,
        particle_final: np.ndarray,
        partitioning_initial: np.ndarray,
        partitioning_final: np.ndarray,
        gas_only_initial: np.ndarray,
        gas_only_final: np.ndarray,
        coefficient: float,
        time_step: float,
        sub_steps: int,
    ) -> None:
        """Store detached snapshots and execution metadata."""
        object.__setattr__(self, "particle_initial", particle_initial)
        object.__setattr__(self, "particle_final", particle_final)
        object.__setattr__(self, "partitioning_initial", partitioning_initial)
        object.__setattr__(self, "partitioning_final", partitioning_final)
        object.__setattr__(self, "gas_only_initial", gas_only_initial)
        object.__setattr__(self, "gas_only_final", gas_only_final)
        object.__setattr__(self, "coefficient", coefficient)
        object.__setattr__(self, "time_step", time_step)
        object.__setattr__(self, "sub_steps", sub_steps)

    def __setattr__(self, name: str, value: object) -> None:
        """Reject reassignment after initialization."""
        raise AttributeError("ExampleResult is immutable")


def _build_aerosol() -> par.Aerosol:
    """Build a small aerosol with every CPU dilution concentration domain."""
    particles = par.particles.ParticleRepresentation(
        strategy=par.particles.MassBasedMovingBin(),
        activity=par.particles.ActivityIdealMass(),
        surface=par.particles.SurfaceStrategyVolume(),
        distribution=np.array([1e-18, 2e-18], dtype=np.float64),
        density=np.array([1000.0], dtype=np.float64),
        concentration=np.array([4.0, 8.0], dtype=np.float64),
        charge=np.array([1.0, -1.0], dtype=np.float64),
        volume=2.0,
    )
    partitioning = par.gas.GasSpecies(
        name="partitioning",
        molar_mass=0.1,
        concentration=3.0,
        partitioning=True,
    )
    gas_only = par.gas.GasSpecies(
        name=np.array(["gas_a", "gas_b"]),
        molar_mass=np.array([0.02, 0.03], dtype=np.float64),
        concentration=np.array([5.0, 7.0], dtype=np.float64),
        partitioning=False,
    )
    atmosphere = par.gas.Atmosphere(
        temperature=298.15,
        total_pressure=101325.0,
        partitioning_species=partitioning,
        gas_only_species=gas_only,
    )
    return par.Aerosol(atmosphere=atmosphere, particles=particles)


def run_example() -> ExampleResult:
    """Execute exact CPU dilution and return detached concentration snapshots."""
    coefficient = 0.25
    time_step = 4.0
    sub_steps = 2
    aerosol = _build_aerosol()
    particle_initial = aerosol.particles.get_concentration().copy()
    partitioning_initial = np.asarray(
        aerosol.atmosphere.partitioning_species.get_concentration()
    ).copy()
    gas_only_initial = (
        aerosol.atmosphere.gas_only_species.get_concentration().copy()
    )

    strategy = par.dynamics.DilutionStrategy(coefficient)
    dilution = par.dynamics.Dilution(strategy)
    assert dilution.execute(aerosol, time_step, sub_steps) is aerosol

    particle_final = aerosol.particles.get_concentration().copy()
    partitioning_final = np.asarray(
        aerosol.atmosphere.partitioning_species.get_concentration()
    ).copy()
    gas_only_final = (
        aerosol.atmosphere.gas_only_species.get_concentration().copy()
    )
    decay_factor = np.exp(-coefficient * time_step)
    for initial, final in (
        (particle_initial, particle_final),
        (partitioning_initial, partitioning_final),
        (gas_only_initial, gas_only_final),
    ):
        np.testing.assert_allclose(
            final,
            initial * decay_factor,
            rtol=1e-12,
            atol=0.0,
        )

    return ExampleResult(
        particle_initial,
        particle_final,
        partitioning_initial,
        partitioning_final,
        gas_only_initial,
        gas_only_final,
        coefficient,
        time_step,
        sub_steps,
    )


def main() -> None:
    """Run the example and report its deterministic concentration decay."""
    result = run_example()
    decay_factor = np.exp(-result.coefficient * result.time_step)
    print(f"coefficient: {result.coefficient} 1/s")
    print(f"duration: {result.time_step} s ({result.sub_steps} substeps)")
    print(f"decay factor: {decay_factor}")
    print(f"particle before: {result.particle_initial}")
    print(f"particle after: {result.particle_final}")
    print(f"partitioning before: {result.partitioning_initial}")
    print(f"partitioning after: {result.partitioning_final}")
    print(f"gas-only before: {result.gas_only_initial}")
    print(f"gas-only after: {result.gas_only_final}")


if __name__ == "__main__":
    main()
