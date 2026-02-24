"""Diagnostic: compare direct kernel calculation (with charge)
against the interpolated kernel (radius-only) for the same radius
pairs but different charge combinations.

Original scenarios on a 4-bin radius grid [5, 30, 50, 70] nm:
  1) Wide opposite:  bin0 = +1, bins1-3 = -6  (ion + calcite)
  2) Same sign:      bin0 = +1, bins1-3 = +1  (all same charge)
  3) Narrow opposite: bin0 = +1, bins1-3 = -1  (small attraction)

Coag-comparison scenarios on a 4-bin grid [50, 75, 100, 150] nm:
  4) Calcite(-6)+Ion(-6): same sign, repulsive
  5) Calcite(-6)+Ion(0):  neutral ions, Brownian only
  6) Calcite(-6)+Ion(+6): opposite sign, attractive

Known issue documented by scenarios 5 and 6 (M6-P2):
  The kernel interpolator is radius-only.  When the kernel matrix
  has charge-dependent structure (e.g., attractive ion-calcite vs
  repulsive calcite-calcite), the interpolator linearly blends
  them based on radius alone.  This causes wrong kernel values
  in the transition zone between ion and calcite radii.

  See maintenance plan M6:
    adw-docs/dev-plans/maintenance/
        M6-particle-resolved-coagulation-fixes.md
"""

import numpy as np
from particula.dynamics.coagulation.charged_kernel_strategy import (
    HardSphereKernelStrategy,
)
from particula.dynamics.coagulation.particle_resolved_step.particle_resolved_method import (  # noqa: E501
    _interpolate_kernel,
)
from particula.gas.properties.dynamic_viscosity import (
    get_dynamic_viscosity,
)
from particula.gas.properties.mean_free_path import (
    get_molecule_mean_free_path,
)
from particula.particles.properties.coulomb_enhancement import (
    get_coulomb_enhancement_ratio,
)
from particula.particles.properties.diffusive_knudsen_module import (
    get_diffusive_knudsen_number,
)
from particula.particles.properties.friction_factor_module import (
    get_friction_factor,
)
from particula.particles.properties.knudsen_number_module import (
    get_knudsen_number,
)
from particula.particles.properties.slip_correction_module import (
    get_cunningham_slip_correction,
)
from particula.util.reduced_quantity import get_reduced_self_broadcast

# -- shared constants --
TEMPERATURE = 216.65  # K
PRESSURE = 5065.0  # Pa
DENSITY = 2710.0  # kg/m^3
BIN_RADII = np.array([5e-9, 30e-9, 50e-9, 70e-9])
SCAN_RADII = np.array([5, 10, 15, 20, 25, 30, 35, 40, 50]) * 1e-9


def _compute_kernel_matrix(
    radii: np.ndarray,
    charges: np.ndarray,
    temperature: float = TEMPERATURE,
    pressure: float = PRESSURE,
    density: float = DENSITY,
) -> np.ndarray:
    """Compute pairwise kernel matrix from radii + charges."""
    masses = (4.0 / 3.0) * np.pi * radii**3 * density

    dynamic_viscosity = get_dynamic_viscosity(temperature=temperature)
    mfp = get_molecule_mean_free_path(
        temperature=temperature,
        pressure=pressure,
        dynamic_viscosity=dynamic_viscosity,
    )
    kn = get_knudsen_number(mean_free_path=mfp, particle_radius=radii)
    slip = get_cunningham_slip_correction(knudsen_number=kn)
    friction = get_friction_factor(
        particle_radius=radii,
        dynamic_viscosity=dynamic_viscosity,
        slip_correction=slip,
    )

    coulomb_phi = get_coulomb_enhancement_ratio(
        particle_radius=radii,
        charge=charges,
        temperature=temperature,
    )
    diff_kn = get_diffusive_knudsen_number(
        particle_radius=radii,
        particle_mass=masses,
        friction_factor=friction,
        coulomb_potential_ratio=coulomb_phi,
        temperature=temperature,
    )

    hs = HardSphereKernelStrategy()
    h_kernel = hs.dimensionless(
        diffusive_knudsen=diff_kn,
        coulomb_potential_ratio=coulomb_phi,
    )

    sum_of_radii = radii[:, np.newaxis] + radii
    reduced_mass = get_reduced_self_broadcast(masses)
    reduced_friction = get_reduced_self_broadcast(friction)

    return hs.kernel(
        dimensionless_kernel=h_kernel,
        coulomb_potential_ratio=coulomb_phi,
        sum_of_radii=sum_of_radii,
        reduced_mass=reduced_mass,
        reduced_friction_factor=reduced_friction,
    )


def _run_scenario(
    label: str,
    bin_charges: np.ndarray,
    query_charge_pairs: list[tuple[str, np.ndarray]],
):
    """Run a single scenario: build interpolator, compare direct
    vs interp.

    Args:
        label: Scenario name for printing.
        bin_charges: Charges for the 4-bin grid [5, 30, 50, 70] nm.
        query_charge_pairs: List of (name, charges_array) for direct
            kernel at (5nm, 50nm).
    """
    kernel_matrix = _compute_kernel_matrix(BIN_RADII, bin_charges)
    interp = _interpolate_kernel(kernel_matrix, BIN_RADII)

    k_interp_5_50 = interp(np.array([[5e-9, 50e-9]])).item()

    sep = "-" * 76
    print("\n" + "=" * 76)
    print(f"  {label}")
    print(f"  Bin charges: {bin_charges}")
    print("=" * 76)

    # -- kernel matrix --
    print("\n  Kernel matrix:")
    print(f"  {'':>8s}", end="")
    for r in BIN_RADII:
        print(f"  {r * 1e9:8.0f}nm", end="")
    print()
    for i in range(4):
        print(f"  {BIN_RADII[i] * 1e9:6.0f}nm", end="")
        for j in range(4):
            print(f"  {kernel_matrix[i, j]:10.2e}", end="")
        print()

    # -- direct vs interpolated at (5nm, 50nm) --
    print("\n  Direct kernel at (5 nm, 50 nm) with specific charges:")
    print(f"  {'case':>35s}  {'charges':>10s}  {'kernel':>14s}")
    print("  " + sep)
    for name, q_pair in query_charge_pairs:
        r_pair = np.array([5e-9, 50e-9])
        kd = _compute_kernel_matrix(r_pair, q_pair)
        print(f"  {name:>35s}  {str(q_pair):>10s}  {kd[0, 1]:14.6e}")
    print("  " + sep)
    print(
        f"  {'interpolated (radius only)':>35s}"
        f"  {'???':>10s}  {k_interp_5_50:14.6e}"
    )

    # -- bleed zone scan --
    ion_charge = bin_charges[0]
    bulk_charge = bin_charges[1]
    print("\n  Bleed zone at (r, 50nm):")
    print(
        f"  {'r (nm)':>8s}  {'K interp':>12s}"
        f"  {'K({ion_charge:+.0f},{bulk_charge:+.0f})':>12s}"
        f"  {'K({bulk_charge:+.0f},{bulk_charge:+.0f})':>12s}"
        f"  {'interp matches':>16s}"
    )
    print("  " + sep)

    for r in SCAN_RADII:
        ki = interp(np.array([[r, 50e-9]])).item()
        r_pair = np.array([r, 50e-9])
        kd_cross = _compute_kernel_matrix(
            r_pair, np.array([ion_charge, bulk_charge])
        )
        kd_same = _compute_kernel_matrix(
            r_pair, np.array([bulk_charge, bulk_charge])
        )
        diff_cross = abs(ki - kd_cross[0, 1])
        diff_same = abs(ki - kd_same[0, 1])
        if diff_cross < diff_same:
            match_label = f"({ion_charge:+.0f},{bulk_charge:+.0f})"
        elif diff_same < diff_cross:
            match_label = f"({bulk_charge:+.0f},{bulk_charge:+.0f})"
        else:
            match_label = "both equal"
        print(
            f"  {r * 1e9:8.1f}"
            f"  {ki:12.4e}"
            f"  {kd_cross[0, 1]:12.4e}"
            f"  {kd_same[0, 1]:12.4e}"
            f"  {match_label:>16s}"
        )

    print("=" * 76)


# =============================================================
# Test 1: Wide opposite charge (+1 vs -6)
# =============================================================
def test_wide_opposite_charge():
    """Bin 0 = +1 (ion), bins 1-3 = -6 (calcite).
    Large attraction.
    """
    _run_scenario(
        label="SCENARIO 1: Wide opposite charge (+1 vs -6)",
        bin_charges=np.array([1.0, -6.0, -6.0, -6.0]),
        query_charge_pairs=[
            (
                "ion+calcite (attractive)",
                np.array([1.0, -6.0]),
            ),
            (
                "calcite+calcite (repulsive)",
                np.array([-6.0, -6.0]),
            ),
            (
                "ion+ion (repulsive)",
                np.array([1.0, 1.0]),
            ),
        ],
    )


# =============================================================
# Test 2: Same sign charge (+1 vs +1)
# =============================================================
def test_same_sign_charge():
    """All bins = +1. No cross-charge attraction, all
    repulsive.
    """
    _run_scenario(
        label="SCENARIO 2: Same sign charge (+1 vs +1)",
        bin_charges=np.array([1.0, 1.0, 1.0, 1.0]),
        query_charge_pairs=[
            ("(+1, +1) same sign", np.array([1.0, 1.0])),
            ("(0, 0) neutral", np.array([0.0, 0.0])),
        ],
    )


# =============================================================
# Test 3: Narrow opposite charge (+1 vs -1)
# =============================================================
def test_narrow_opposite_charge():
    """Bin 0 = +1, bins 1-3 = -1. Small attraction."""
    _run_scenario(
        label="SCENARIO 3: Narrow opposite charge (+1 vs -1)",
        bin_charges=np.array([1.0, -1.0, -1.0, -1.0]),
        query_charge_pairs=[
            ("(+1, -1) attractive", np.array([1.0, -1.0])),
            ("(-1, -1) repulsive", np.array([-1.0, -1.0])),
            ("(+1, +1) repulsive", np.array([1.0, 1.0])),
        ],
    )


# =============================================================
# Coag-comparison matched scenarios
# Grid: [50, 75, 100, 150] nm  (ions=50nm, calcite=100nm)
# =============================================================
COAG_BIN_RADII = np.array([50e-9, 75e-9, 100e-9, 150e-9])
COAG_SCAN_RADII = np.array([50, 60, 70, 80, 90, 100, 120, 150]) * 1e-9


def _run_coag_scenario(
    label: str,
    bin_charges: np.ndarray,
    query_charge_pairs: list[tuple[str, np.ndarray]],
    bin_radii: np.ndarray = COAG_BIN_RADII,
    scan_radii: np.ndarray = COAG_SCAN_RADII,
):
    """Run scenario matched to the coag comparison test setup.

    Same as _run_scenario but uses the coag-comparison grid and
    prints an extra section comparing ion-calcite vs
    calcite-calcite kernel values at exact particle sizes
    (50nm, 100nm).
    """
    kernel_matrix = _compute_kernel_matrix(bin_radii, bin_charges)
    interp = _interpolate_kernel(kernel_matrix, bin_radii)

    # Interpolated value at ion(50nm) vs calcite(100nm)
    k_interp_50_100 = interp(np.array([[50e-9, 100e-9]])).item()

    sep = "-" * 76
    print("\n" + "=" * 76)
    print(f"  {label}")
    print(f"  Bin radii (nm): {bin_radii * 1e9}")
    print(f"  Bin charges: {bin_charges}")
    print("=" * 76)

    # -- kernel matrix --
    print("\n  Kernel matrix:")
    print(f"  {'':>8s}", end="")
    for r in bin_radii:
        print(f"  {r * 1e9:8.0f}nm", end="")
    print()
    for i in range(len(bin_radii)):
        print(f"  {bin_radii[i] * 1e9:6.0f}nm", end="")
        for j in range(len(bin_radii)):
            print(
                f"  {kernel_matrix[i, j]:10.2e}",
                end="",
            )
        print()

    # -- direct vs interpolated at (50nm, 100nm) --
    print("\n  Direct kernel at (50 nm, 100 nm) with specific charges:")
    print(f"  {'case':>40s}  {'charges':>12s}  {'kernel':>14s}")
    print("  " + sep)
    for name, q_pair in query_charge_pairs:
        r_pair = np.array([50e-9, 100e-9])
        kd = _compute_kernel_matrix(r_pair, q_pair)
        print(f"  {name:>40s}  {str(q_pair):>12s}  {kd[0, 1]:14.6e}")
    print("  " + sep)
    print(
        f"  {'interpolated (radius only)':>40s}"
        f"  {'???':>12s}  {k_interp_50_100:14.6e}"
    )

    # -- ratio analysis --
    print("\n  Kernel ratios (vs neutral Brownian baseline):")
    r_pair = np.array([50e-9, 100e-9])
    k_neutral = _compute_kernel_matrix(r_pair, np.array([0.0, 0.0]))[0, 1]
    print(f"  {'neutral (0,0) baseline':>40s}  {k_neutral:14.6e}")
    for name, q_pair in query_charge_pairs:
        kd = _compute_kernel_matrix(r_pair, q_pair)[0, 1]
        ratio = kd / k_neutral if k_neutral > 0 else float("inf")
        print(f"  {name:>40s}  {kd:14.6e}  ratio={ratio:.4f}")
    ratio_interp = (
        k_interp_50_100 / k_neutral if k_neutral > 0 else float("inf")
    )
    print(
        f"  {'interpolated':>40s}"
        f"  {k_interp_50_100:14.6e}"
        f"  ratio={ratio_interp:.4f}"
    )

    # -- bleed zone scan --
    ion_charge = bin_charges[0]
    bulk_charge = bin_charges[2]  # calcite charge at 100nm bin
    print("\n  Bleed zone at (r, 100nm):")
    print(
        f"  {'r (nm)':>8s}  {'K interp':>12s}"
        f"  {'K({ion_charge:+.0f},{bulk_charge:+.0f})':>12s}"
        f"  {'K({bulk_charge:+.0f},{bulk_charge:+.0f})':>12s}"
        f"  {'interp matches':>16s}"
    )
    print("  " + sep)

    for r in scan_radii:
        ki = interp(np.array([[r, 100e-9]])).item()
        r_pair = np.array([r, 100e-9])
        kd_cross = _compute_kernel_matrix(
            r_pair, np.array([ion_charge, bulk_charge])
        )
        kd_same = _compute_kernel_matrix(
            r_pair, np.array([bulk_charge, bulk_charge])
        )
        diff_cross = abs(ki - kd_cross[0, 1])
        diff_same = abs(ki - kd_same[0, 1])
        if diff_cross < diff_same:
            match_label = f"({ion_charge:+.0f},{bulk_charge:+.0f})"
        elif diff_same < diff_cross:
            match_label = f"({bulk_charge:+.0f},{bulk_charge:+.0f})"
        else:
            match_label = "both equal"
        print(
            f"  {r * 1e9:8.1f}"
            f"  {ki:12.4e}"
            f"  {kd_cross[0, 1]:12.4e}"
            f"  {kd_same[0, 1]:12.4e}"
            f"  {match_label:>16s}"
        )

    print("=" * 76)


# =============================================================
# Test 4: Calcite(-6) + Ion(-6) -- same sign, repulsive
# =============================================================
def test_coag_same_sign_repulsive():
    """Ion charge = -6, calcite charge = -6. All repulsive.

    All charges identical, so the interpolator is consistent
    with direct — no bleed problem.  Kernel values are
    ~11 orders of magnitude below Brownian baseline.
    """
    _run_coag_scenario(
        label=("SCENARIO 4: Calcite(-6) + Ion(-6) -- all repulsive"),
        bin_charges=np.array([-6.0, -6.0, -6.0, -6.0]),
        query_charge_pairs=[
            (
                "ion+calcite (-6,-6) repulsive",
                np.array([-6.0, -6.0]),
            ),
            (
                "neutral (0,0) Brownian",
                np.array([0.0, 0.0]),
            ),
        ],
    )


# =============================================================
# Test 5: Calcite(-6) + Ion(0) -- neutral ions
#
# KNOWN ISSUE (M6-P2): The interpolator blends the neutral
# ion row (K ~ 4.6e-15) with the repulsive calcite rows
# (K ~ 2.8e-24).  At 80nm the interpolated value drops from
# ~1e-15 to ~8e-25 — a 10-order-of-magnitude cliff.
# Particles near the bin boundary get wrong kernel values.
# =============================================================
def test_coag_neutral_ions():
    """Ion charge = 0, calcite charge = -6. Mixed charge
    grid.
    """
    _run_coag_scenario(
        label=("SCENARIO 5: Calcite(-6) + Ion(0) -- neutral ions"),
        bin_charges=np.array([0.0, -6.0, -6.0, -6.0]),
        query_charge_pairs=[
            (
                "ion+calcite (0,-6) neutral-charged",
                np.array([0.0, -6.0]),
            ),
            (
                "calcite+calcite (-6,-6) repulsive",
                np.array([-6.0, -6.0]),
            ),
            (
                "neutral (0,0) Brownian",
                np.array([0.0, 0.0]),
            ),
        ],
    )


# =============================================================
# Test 6: Calcite(-6) + Ion(+6) -- opposite sign, attractive
#
# KNOWN ISSUE (M6-P2): The interpolator gives the cross-charge
# attractive kernel (~1e-13, 22x Brownian) at 50nm, but by
# 80nm it crashes to ~8e-25 (the repulsive calcite-calcite
# value).  Any calcite particle that falls near the 60-75nm
# boundary gets kernel values 10+ orders of magnitude too
# high, causing thousands of spurious calcite-calcite mergers
# in the full coagulation simulation.
# =============================================================
def test_coag_opposite_sign_attractive():
    """Ion charge = +6, calcite charge = -6. Strong
    attraction.
    """
    _run_coag_scenario(
        label=("SCENARIO 6: Calcite(-6) + Ion(+6) -- attractive"),
        bin_charges=np.array([6.0, -6.0, -6.0, -6.0]),
        query_charge_pairs=[
            (
                "ion+calcite (+6,-6) attractive",
                np.array([6.0, -6.0]),
            ),
            (
                "calcite+calcite (-6,-6) repulsive",
                np.array([-6.0, -6.0]),
            ),
            (
                "neutral (0,0) Brownian",
                np.array([0.0, 0.0]),
            ),
        ],
    )
