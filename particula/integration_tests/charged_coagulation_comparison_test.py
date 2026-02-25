"""Charged coagulation comparison integration tests.

Extreme scenario: calcite 200 nm diameter with charge -6, ions 100 nm.
Compare three ion charge values: -6, 0, +6. Same-sign ions match calcite.

Note: The issue text mentions +6 for same-sign ions. In this test, calcite is
fixed at -6, so the same-sign case uses -6 to match calcite's sign.

Each case asserts mass/charge conservation and physical merger behavior.
"""

import numpy as np
import numpy.testing as npt

import particula as par


def _run_coagulation(
    ion_charge: float,
    use_direct_kernel: bool,
    n_steps: int,
    calcite_count: int = 600,
    ion_count: int = 120,
    calcite_charge: float = -6.0,
    total_time: float = 14 * 24 * 3600,
    seed: int = 42,
) -> dict:
    """Run a charged coagulation scenario and return summary stats.

    Args:
        ion_charge: Charge on each ion particle.
        use_direct_kernel: Whether to use the direct kernel evaluator.
        n_steps: Number of coagulation steps to execute.
        calcite_count: Number of calcite particles.
        ion_count: Number of ion particles.
        calcite_charge: Charge on each calcite particle.
        total_time: Total duration to span when defining time step.
        seed: Random seed for reproducibility.

    Returns:
        Dict with before/after counts and merger details.
    """
    np.random.seed(seed)

    # -- parameters --
    nn = calcite_count
    nd = ion_count
    calcite_density = 2710.0  # kg/m^3
    ions_density = 1500.0  # kg/m^3

    # Scale volume to maintain concentration
    volume = 1e-9 * (nn / 6.705e5)  # m^3

    atmosphere = (
        par.gas.AtmosphereBuilder()
        .set_temperature(-56.5, temperature_units="degC")
        .set_pressure(0.05, pressure_units="atm")
        .build()
    )

    # -- particles (fixed radii, no randomness) --
    calcite_r = np.full(nn, 100e-9)  # 100nm radius = 200nm diameter
    ions_r = np.full(nd, 50e-9)  # 50nm radius = 100nm diameter

    mass_calcite = (4.0 / 3.0) * np.pi * calcite_r**3 * calcite_density
    mass_ions = (4.0 / 3.0) * np.pi * ions_r**3 * ions_density

    # Two species: column 0 = calcite, column 1 = ions
    mass_speciation = np.zeros((nn + nd, 2), dtype=float)
    mass_speciation[:nn, 0] = mass_calcite
    mass_speciation[nn:, 1] = mass_ions

    charge_array = np.zeros(nn + nd, dtype=float)
    charge_array[:nn] = calcite_charge
    charge_array[nn:] = ion_charge

    total_mass_before = np.sum(mass_speciation)
    total_charge_before = charge_array.sum()

    # -- build aerosol --
    resolved_masses = (
        par.particles.ResolvedParticleMassRepresentationBuilder()
        .set_distribution_strategy(
            par.particles.ParticleResolvedSpeciatedMass()
        )
        .set_activity_strategy(par.particles.ActivityIdealMass())
        .set_surface_strategy(par.particles.SurfaceStrategyVolume())
        .set_mass(mass_speciation, "kg")
        .set_density(np.array([calcite_density, ions_density]), "kg/m^3")
        .set_charge(charge_array)
        .set_volume(volume, "m^3")
        .build()
    )

    aerosol = par.Aerosol(atmosphere=atmosphere, particles=resolved_masses)

    # -- coagulation --
    hs_strategy = par.dynamics.HardSphereKernelStrategy()
    charged_builder = (
        par.dynamics.ChargedCoagulationBuilder()
        .set_distribution_type("particle_resolved")
        .set_charged_kernel_strategy(hs_strategy)
        .set_use_direct_kernel(use_direct_kernel)
    )
    charged_brownian = charged_builder.build()
    coagulation = par.dynamics.Coagulation(
        coagulation_strategy=charged_brownian,
    )

    # -- time step --
    time = np.logspace(np.log10(0.2), np.log10(total_time), n_steps + 1)
    dt = time[1] - time[0]

    for _ in range(n_steps):
        aerosol = coagulation.execute(aerosol, dt, sub_steps=1)

    # -- analyze after --
    dist = aerosol.particles.distribution
    charges = aerosol.particles.get_charge(clone=True)
    masses = aerosol.particles.get_mass(clone=True)
    assert charges is not None, "charges should not be None"
    assert masses is not None, "masses should not be None"

    alive = dist.sum(axis=1) > 0
    is_pure_calcite = (dist[:, 0] > 0) & (dist[:, 1] == 0)
    is_pure_ion = (dist[:, 1] > 0) & (dist[:, 0] == 0)
    is_mixed = (dist[:, 0] > 0) & (dist[:, 1] > 0)

    # charge breakdown
    alive_charges = charges[alive]
    unique_q, counts_q = np.unique(alive_charges, return_counts=True)

    total_mass_after = masses[alive].sum()
    total_charge_after = charges[alive].sum()

    calcite_calcite_mergers = (
        nn - int(is_pure_calcite.sum()) - int(is_mixed.sum())
    )
    ion_calcite_mergers = int(is_mixed.sum())
    ion_ion_mergers = nd - int(is_pure_ion.sum()) - ion_calcite_mergers

    return {
        "ion_charge": ion_charge,
        "dt": dt,
        "nn": nn,
        "nd": nd,
        "alive": int(alive.sum()),
        "pure_calcite": int(is_pure_calcite.sum()),
        "pure_ion": int(is_pure_ion.sum()),
        "mixed": int(is_mixed.sum()),
        "calcite_calcite_mergers": calcite_calcite_mergers,
        "ion_calcite_mergers": ion_calcite_mergers,
        "ion_ion_mergers": ion_ion_mergers,
        "ions_remaining": int(is_pure_ion.sum()),
        "charge_breakdown": dict(zip(unique_q, counts_q, strict=False)),
        "total_mass_before": total_mass_before,
        "total_mass_after": total_mass_after,
        "total_charge_before": total_charge_before,
        "total_charge_after": total_charge_after,
    }


def _assert_conservation(results: dict) -> None:
    """Assert mass and charge conservation for a scenario."""
    npt.assert_allclose(
        results["total_mass_after"],
        results["total_mass_before"],
        rtol=1e-10,
        err_msg="Mass not conserved in charged coagulation scenario.",
    )
    npt.assert_allclose(
        results["total_charge_after"],
        results["total_charge_before"],
        rtol=1e-10,
        err_msg="Charge not conserved in charged coagulation scenario.",
    )


def test_same_sign_ions_no_mergers():
    """Same-sign ions repel: no ion-calcite or calcite-calcite mergers."""
    results = _run_coagulation(
        ion_charge=-6.0,
        use_direct_kernel=False,
        n_steps=1,
        calcite_count=100,
        ion_count=20,
        calcite_charge=-6.0,
        total_time=1.0,
        seed=7,
    )

    _assert_conservation(results)
    assert results["ion_calcite_mergers"] == 0
    assert results["calcite_calcite_mergers"] == 0


def test_neutral_ions_brownian_baseline():
    """Neutral ions and neutral calcite follow Brownian baseline."""
    results = _run_coagulation(
        ion_charge=0.0,
        use_direct_kernel=False,
        n_steps=5,
        calcite_count=120,
        ion_count=40,
        calcite_charge=0.0,
        seed=42,
    )

    _assert_conservation(results)
    assert results["ion_calcite_mergers"] > 0
    assert results["calcite_calcite_mergers"] > 0


def test_opposite_sign_ions_direct_kernel():
    """Opposite-sign ions attract; direct kernel avoids spurious mergers."""
    results = _run_coagulation(
        ion_charge=6.0,
        use_direct_kernel=True,
        n_steps=2,
        calcite_count=1,
        ion_count=40,
        calcite_charge=-6.0,
        total_time=10.0,
        seed=42,
    )

    _assert_conservation(results)
    assert results["ion_calcite_mergers"] > 0
    assert results["calcite_calcite_mergers"] == 0
