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
    # Raw represented counts [4, 8] in 2 m³ give densities [2, 4] m^-3.
    particles = par.particles.create_resolved_particle_data(
        masses=np.array([1.0e-18, 2.0e-18]),
        density=np.array([1000.0]),
        concentration=np.array([4.0, 8.0]),
        charge=np.array([1.0, -1.0]),
        volume=2.0,
    )
    gas = par.gas.GasData(
        name=["partitioning", "gas_a", "gas_b"],
        molar_mass=np.array([0.1, 0.02, 0.03]),
        concentration=np.array([[3.0, 5.0, 7.0]]),
        partitioning=np.array([True, False, False]),
    )
    environment = par.gas.EnvironmentData(
        temperature=np.array([298.15]),
        pressure=np.array([101325.0]),
        saturation_ratio=np.zeros((1, 3)),
    )
    return par.Aerosol(particles=particles, gas=gas, environment=environment)


def run_example() -> ExampleResult:
    """Execute two equal public-runnable dilution substeps and verify decay."""
    coefficient = 0.25  # 1/s
    time_step = 4.0  # s
    sub_steps = 2
    aerosol = _build_aerosol()

    particles, gas, environment = (
        aerosol.particles,
        aerosol.gas,
        aerosol.environment,
    )
    initial_masses = particles.masses.copy()
    initial_charge = particles.charge.copy()
    initial_volume = particles.volume.copy()
    particle_initial = aerosol.particles.slot_concentration_density[0].copy()
    partitioning_initial = np.asarray(aerosol.gas.concentration[0, :1]).copy()
    gas_only_initial = aerosol.gas.concentration[0, 1:].copy()

    strategy = par.dynamics.DilutionStrategy(coefficient=coefficient)
    dilution = par.dynamics.Dilution(strategy)
    result = dilution.execute(
        aerosol,
        time_step=time_step,
        sub_steps=sub_steps,
    )
    assert result is aerosol
    assert aerosol.particles is particles
    assert aerosol.gas is gas
    assert aerosol.environment is environment
    np.testing.assert_array_equal(particles.masses, initial_masses)
    np.testing.assert_array_equal(particles.charge, initial_charge)
    np.testing.assert_array_equal(particles.volume, initial_volume)

    particle_final = aerosol.particles.slot_concentration_density[0].copy()
    partitioning_final = np.asarray(aerosol.gas.concentration[0, :1]).copy()
    gas_only_final = aerosol.gas.concentration[0, 1:].copy()
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
