"""Run deterministic CPU dilution with the public Particula API.

Run from the repository root:
    python docs/Examples/cpu_dilution.py
"""

from __future__ import annotations

import numpy as np
import particula as par


class ExampleResult:
    """Detached concentration snapshots from one dilution execution."""

    __slots__ = (
        "coefficient",
        "time_step",
        "sub_steps",
        "particle_initial",
        "particle_final",
        "partitioning_initial",
        "partitioning_final",
        "gas_only_initial",
        "gas_only_final",
        "_initialized",
    )

    coefficient: float
    time_step: float
    sub_steps: int
    particle_initial: np.ndarray
    particle_final: np.ndarray
    partitioning_initial: np.ndarray
    partitioning_final: np.ndarray
    gas_only_initial: np.ndarray
    gas_only_final: np.ndarray
    _initialized: bool

    def __init__(
        self,
        coefficient: float,
        time_step: float,
        sub_steps: int,
        particle_initial: np.ndarray,
        particle_final: np.ndarray,
        partitioning_initial: np.ndarray,
        partitioning_final: np.ndarray,
        gas_only_initial: np.ndarray,
        gas_only_final: np.ndarray,
    ) -> None:
        """Store immutable execution metadata and detached array snapshots."""
        object.__setattr__(self, "coefficient", coefficient)
        object.__setattr__(self, "time_step", time_step)
        object.__setattr__(self, "sub_steps", sub_steps)
        object.__setattr__(self, "particle_initial", particle_initial)
        object.__setattr__(self, "particle_final", particle_final)
        object.__setattr__(self, "partitioning_initial", partitioning_initial)
        object.__setattr__(self, "partitioning_final", partitioning_final)
        object.__setattr__(self, "gas_only_initial", gas_only_initial)
        object.__setattr__(self, "gas_only_final", gas_only_final)
        object.__setattr__(self, "_initialized", True)

    def __setattr__(self, _name: str, _value: object) -> None:
        """Prevent reassignment after construction."""
        raise AttributeError("ExampleResult is immutable")


def _build_aerosol() -> par.Aerosol:
    """Build a deterministic CPU aerosol with three concentration domains."""
    particles = par.particles.ParticleRepresentation(
        strategy=par.particles.MassBasedMovingBin(),
        activity=par.particles.ActivityIdealMass(),
        surface=par.particles.SurfaceStrategyVolume(),
        distribution=np.array([1.0e-18, 2.0e-18]),
        density=np.array([1000.0]),
        concentration=np.array([4.0, 8.0]),
        charge=np.array([1.0, -1.0]),
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
        molar_mass=np.array([0.02, 0.03]),
        concentration=np.array([5.0, 7.0]),
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
    """Execute two equal public-runnable dilution substeps and verify decay."""
    coefficient = 0.25  # 1/s
    time_step = 4.0  # s
    sub_steps = 2
    aerosol = _build_aerosol()

    particle_initial = aerosol.particles.get_concentration().copy()
    partitioning_initial = np.asarray(
        aerosol.atmosphere.partitioning_species.get_concentration()
    ).copy()
    gas_only_initial = (
        aerosol.atmosphere.gas_only_species.get_concentration().copy()
    )

    strategy = par.dynamics.DilutionStrategy(coefficient=coefficient)
    dilution = par.dynamics.Dilution(strategy)
    result = dilution.execute(
        aerosol,
        time_step=time_step,
        sub_steps=sub_steps,
    )
    assert result is aerosol

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
        coefficient=coefficient,
        time_step=time_step,
        sub_steps=sub_steps,
        particle_initial=particle_initial,
        particle_final=particle_final,
        partitioning_initial=partitioning_initial,
        partitioning_final=partitioning_final,
        gas_only_initial=gas_only_initial,
        gas_only_final=gas_only_final,
    )


def main() -> None:
    """Run the example and print deterministic concentration snapshots."""
    result = run_example()
    decay_factor = np.exp(-result.coefficient * result.time_step)

    print(f"Coefficient: {result.coefficient:.3f} 1/s")
    print(f"Duration: {result.time_step:.1f} s ({result.sub_steps} substeps)")
    print(f"Decay factor: {decay_factor:.12f}")
    print(f"Particle before: {result.particle_initial}")
    print(f"Particle after:  {result.particle_final}")
    print(f"Partitioning before: {result.partitioning_initial}")
    print(f"Partitioning after:  {result.partitioning_final}")
    print(f"Gas-only before: {result.gas_only_initial}")
    print(f"Gas-only after:  {result.gas_only_final}")


if __name__ == "__main__":
    main()
