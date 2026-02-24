"""Charge coagulation comparison integration test.

Extreme scenario: calcite 200nm diameter with charge -6, ions 100nm.
Compare three ion charge values: -6, 0, +6.

For each case, run one coagulation step and report:
  - calcite-calcite mergers
  - ion-calcite mergers
  - total alive particles

This isolates how ion charge affects coagulation outcomes.

Known bugs documented by this test (M6):
  - Charge is NOT conserved due to numpy duplicate-index += bug
    in collide_pairs() (M6-P1 will fix)
  - Calcite-calcite mergers in q=+6 case are spurious due to
    charge-blind kernel interpolation (M6-P2 will fix)
"""

import logging
import sys

import numpy as np
import pytest

import particula as par


def _run_coagulation(
    ion_charge: float,
    seed: int = 42,
) -> dict:
    """Run a single coagulation step and return summary stats.

    Args:
        ion_charge: Charge on each ion particle.
        seed: Random seed for reproducibility.

    Returns:
        Dict with before/after counts and merger details.
    """
    np.random.seed(seed)

    # -- parameters --
    Nn = int(1e4)  # calcite particles
    Nd = 100  # ions (more to make effect visible)
    calcite_charge = -6.0
    calcite_density = 2710.0  # kg/m^3
    ions_density = 1500.0  # kg/m^3

    # Scale volume to maintain concentration
    volume = 1e-9 * (Nn / 6.705e5)  # m^3

    atmosphere = (
        par.gas.AtmosphereBuilder()
        .set_temperature(-56.5, temperature_units="degC")
        .set_pressure(0.05, pressure_units="atm")
        .build()
    )

    # -- particles (fixed radii, no randomness) --
    calcite_r = np.full(Nn, 100e-9)  # 100nm radius = 200nm diameter
    ions_r = np.full(Nd, 50e-9)  # 50nm radius = 100nm diameter

    mass_calcite = (4.0 / 3.0) * np.pi * calcite_r**3 * calcite_density
    mass_ions = (4.0 / 3.0) * np.pi * ions_r**3 * ions_density

    # Two species: column 0 = calcite, column 1 = ions
    mass_speciation = np.zeros((Nn + Nd, 2), dtype=float)
    mass_speciation[:Nn, 0] = mass_calcite
    mass_speciation[Nn:, 1] = mass_ions

    charge_array = np.zeros(Nn + Nd, dtype=float)
    charge_array[:Nn] = calcite_charge
    charge_array[Nn:] = ion_charge

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
    sedimentation = (
        par.dynamics.SedimentationCoagulationBuilder()
        .set_distribution_type("particle_resolved")
        .build()
    )
    hs_strategy = par.dynamics.HardSphereKernelStrategy()
    charged_brownian = (
        par.dynamics.ChargedCoagulationBuilder()
        .set_distribution_type("particle_resolved")
        .set_charged_kernel_strategy(hs_strategy)
        .build()
    )
    coagulation = par.dynamics.Coagulation(
        coagulation_strategy=par.dynamics.CombineCoagulationStrategy(
            strategies=[charged_brownian, sedimentation]
        )
    )

    # -- time step --
    total_time = 14 * 24 * 3600
    time = np.logspace(np.log10(0.2), np.log10(total_time), 100)
    dt = time[1] - time[0]

    aerosol = coagulation.execute(aerosol, dt, sub_steps=1)

    # -- analyze after --
    dist = aerosol.particles.distribution
    charges = aerosol.particles.get_charge(clone=True)
    masses = aerosol.particles.get_mass(clone=True)

    alive = dist.sum(axis=1) > 0
    is_pure_calcite = (dist[:, 0] > 0) & (dist[:, 1] == 0)
    is_pure_ion = (dist[:, 1] > 0) & (dist[:, 0] == 0)
    is_mixed = (dist[:, 0] > 0) & (dist[:, 1] > 0)

    # charge breakdown
    alive_charges = charges[alive]
    unique_q, counts_q = np.unique(alive_charges, return_counts=True)

    total_mass_after = masses[alive].sum()
    total_charge_after = charges[alive].sum()

    return {
        "ion_charge": ion_charge,
        "dt": dt,
        "Nn": Nn,
        "Nd": Nd,
        "alive": int(alive.sum()),
        "pure_calcite": int(is_pure_calcite.sum()),
        "pure_ion": int(is_pure_ion.sum()),
        "mixed": int(is_mixed.sum()),
        "calcite_calcite_mergers": Nn
        - int(is_pure_calcite.sum())
        - int(is_mixed.sum()),
        "ion_calcite_mergers": int(is_mixed.sum()),
        "ions_remaining": int(is_pure_ion.sum()),
        "charge_breakdown": dict(zip(unique_q, counts_q)),
        "mass_delta": total_mass_after - total_mass_before,
        "charge_delta": total_charge_after - total_charge_before,
    }


@pytest.mark.slow
def test_charge_coagulation_comparison():
    """Compare coagulation with ion charges -6, 0, +6.

    Documents current (broken) behavior. After M6 fixes:
    - Charge delta should be 0 for all cases
    - Calcite-calcite mergers in q=+6 should match q=0 baseline
    """
    logger = logging.getLogger("charge_coag_compare")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
        logger.addHandler(handler)

    ion_charges = [-6.0, 0.0, 6.0]
    results = []

    for q in ion_charges:
        logger.info(f"Running ion_charge = {q:+.0f} ...")
        r = _run_coagulation(ion_charge=q)
        results.append(r)
        logger.info(f"  Done: {r['alive']} alive")

    # -- print comparison table --
    sep = "-" * 80
    print("\n" + "=" * 80)
    print("  CHARGE COAGULATION COMPARISON")
    print("  Calcite: 10000 x 200nm diameter, charge -6")
    print("  Ions:    100 x 100nm diameter")
    print(f"  dt = {results[0]['dt']:.4f} s")
    print("=" * 80)

    header = (
        f"  {'':>25s}  {'ion q=-6':>12s}  {'ion q=0':>12s}  {'ion q=+6':>12s}"
    )
    print(header)
    print("  " + sep)

    rows = [
        ("Alive particles", "alive"),
        ("Pure calcite", "pure_calcite"),
        ("Pure ions remaining", "ions_remaining"),
        ("Mixed (ion+calcite)", "mixed"),
        ("Calcite-calcite mergers", "calcite_calcite_mergers"),
        ("Ion-calcite mergers", "ion_calcite_mergers"),
    ]
    for label, key in rows:
        vals = [r[key] for r in results]
        print(f"  {label:>25s}  {vals[0]:>12d}  {vals[1]:>12d}  {vals[2]:>12d}")

    print("  " + sep)
    for label, key, fmt in [
        ("Mass delta (kg)", "mass_delta", ".2e"),
        ("Charge delta (e)", "charge_delta", ".2f"),
    ]:
        vals = [r[key] for r in results]
        print(
            f"  {label:>25s}"
            f"  {vals[0]:>12{fmt}}"
            f"  {vals[1]:>12{fmt}}"
            f"  {vals[2]:>12{fmt}}"
        )

    for r in results:
        print(f"\n  Charge breakdown (ion q={r['ion_charge']:+.0f}):")
        for q in sorted(r["charge_breakdown"].keys()):
            cnt = r["charge_breakdown"][q]
            print(f"    charge {q:+8.1f}: {cnt:>8d}")

    print("\n" + "=" * 80)
    print("  EXPECTED BEHAVIOR:")
    print("  ion q=-6 (same sign as calcite): minimal ion-calcite mergers")
    print("           (repulsive Coulomb), calcite-calcite same as no ions")
    print("  ion q= 0 (neutral): moderate ion-calcite mergers")
    print("           (no Coulomb, just Brownian)")
    print("  ion q=+6 (opposite sign): many ion-calcite mergers")
    print("           (attractive Coulomb enhancement)")
    print("  Calcite-calcite mergers should be similar across all three")
    print("  (ions shouldn't affect calcite-calcite rate)")
    print("=" * 80 + "\n")

    r_neg, r_zero, r_pos = results

    # =============================================================
    # Assertions documenting CURRENT (broken) behavior.
    #
    # Two known bugs are captured here — see maintenance plan M6:
    #   adw-docs/dev-plans/maintenance/
    #       M6-particle-resolved-coagulation-fixes.md
    #
    # When the bugs are fixed, flip the assertions marked "WRONG"
    # to the "CORRECT" versions shown in the comments beside them.
    # =============================================================

    # ---------------------------------------------------------
    # q=-6 (same sign as calcite): strong Coulomb repulsion
    # everywhere.  All particles carry charge -6, so both
    # ion-calcite AND calcite-calcite pairs are repulsive.
    # We expect essentially zero mergers of any kind.
    # This case is CORRECT today because no mergers means
    # neither bug is triggered.
    # ---------------------------------------------------------
    assert r_neg["ion_calcite_mergers"] == 0
    assert r_neg["calcite_calcite_mergers"] <= 5

    # ---------------------------------------------------------
    # q=0 (neutral ions): ions have no charge, calcite has -6.
    # Ion-calcite pairs see only Brownian diffusion (no Coulomb).
    # Calcite-calcite pairs are still repulsive (-6 vs -6).
    # We expect moderate ion-calcite mergers and some
    # calcite-calcite mergers from the Brownian baseline.
    # ---------------------------------------------------------
    assert r_zero["ion_calcite_mergers"] > 0
    assert r_zero["calcite_calcite_mergers"] > 0

    # ---------------------------------------------------------
    # q=+6 (opposite sign to calcite): strong Coulomb attraction
    # between ions(+6) and calcite(-6).  We expect MANY
    # ion-calcite mergers — more than the neutral case.
    # ---------------------------------------------------------
    assert r_pos["ion_calcite_mergers"] > r_zero["ion_calcite_mergers"]

    # ---------------------------------------------------------
    # BUG (M6-P1): Charge is NOT conserved.
    #
    # Root cause: collide_pairs() uses numpy fancy-index +=,
    # which is buffered.  When a large particle absorbs multiple
    # smalls in one step (duplicate large_index), only the LAST
    # charge addition takes effect.  The other charges vanish.
    #
    # The q=-6 case is trivially conserved because almost
    # nothing merges.  The q=0 and q=+6 cases lose charge.
    #
    # WRONG (current broken behavior) — passes today:
    assert r_neg["charge_delta"] == 0.0  # trivially OK
    assert r_zero["charge_delta"] != 0.0  # BUG: charge lost
    assert r_pos["charge_delta"] != 0.0  # BUG: charge lost
    # CORRECT (after M6-P1 fix) — flip to these:
    #   assert r_neg["charge_delta"] == 0.0
    #   assert r_zero["charge_delta"] == 0.0
    #   assert r_pos["charge_delta"] == 0.0
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # BUG (M6-P2): Spurious calcite-calcite mergers in q=+6.
    #
    # Root cause: the kernel interpolator is radius-only.  It
    # maps (r_small, r_large) -> kernel_value, but the kernel
    # matrix was built with charge baked in.  The attractive
    # ion-calcite kernel (~1e-13) bleeds via linear
    # interpolation into the calcite-calcite radius range,
    # giving calcite(-6)+calcite(-6) pairs an artificially high
    # kernel value.  This causes thousands of spurious
    # calcite-calcite mergers.
    #
    # WRONG (current broken behavior) — passes today:
    #   q=+6 has >10x the calcite-calcite mergers of q=0, which
    #   is physically impossible (ions should NOT boost the
    #   calcite-calcite coagulation rate).
    assert (
        r_pos["calcite_calcite_mergers"]
        > 10 * r_zero["calcite_calcite_mergers"]
    )
    # CORRECT (after M6-P2 fix, use_direct_kernel=True):
    #   Calcite-calcite mergers should be similar across all
    #   three ion charge scenarios because ions don't affect
    #   the calcite-calcite coagulation rate.
    #   assert (
    #       r_pos["calcite_calcite_mergers"]
    #       < 2 * r_zero["calcite_calcite_mergers"]
    #   )
    # ---------------------------------------------------------
