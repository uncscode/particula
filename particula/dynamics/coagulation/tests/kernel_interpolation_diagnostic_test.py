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
import numpy.testing as npt
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
    """Run a single scenario and return interpolated/direct kernels.

    Args:
        label: Scenario name for identification.
        bin_charges: Charges for the 4-bin grid [5, 30, 50, 70] nm.
        query_charge_pairs: List of (name, charges_array) for direct
            kernel at (5nm, 50nm).
    """
    kernel_matrix = _compute_kernel_matrix(BIN_RADII, bin_charges)
    interp = _interpolate_kernel(kernel_matrix, BIN_RADII)

    k_interp_5_50 = interp(np.array([[5e-9, 50e-9]])).item()
    direct_kernels = {}
    for name, q_pair in query_charge_pairs:
        r_pair = np.array([5e-9, 50e-9])
        kd = _compute_kernel_matrix(r_pair, q_pair)[0, 1]
        direct_kernels[name] = kd

    return {
        "label": label,
        "k_interp_5_50": k_interp_5_50,
        "direct_kernels": direct_kernels,
    }


# =============================================================
# Test 1: Wide opposite charge (+1 vs -6)
# =============================================================
def test_wide_opposite_charge():
    """Bin 0 = +1 (ion), bins 1-3 = -6 (calcite).
    Large attraction.
    """
    results = _run_scenario(
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
    assert np.isfinite(results["k_interp_5_50"])
    assert results["k_interp_5_50"] >= 0
    for kernel_value in results["direct_kernels"].values():
        assert np.isfinite(kernel_value)
        assert kernel_value >= 0


# =============================================================
# Test 2: Same sign charge (+1 vs +1)
# =============================================================
def test_same_sign_charge():
    """All bins = +1. No cross-charge attraction, all
    repulsive.
    """
    results = _run_scenario(
        label="SCENARIO 2: Same sign charge (+1 vs +1)",
        bin_charges=np.array([1.0, 1.0, 1.0, 1.0]),
        query_charge_pairs=[
            ("(+1, +1) same sign", np.array([1.0, 1.0])),
            ("(0, 0) neutral", np.array([0.0, 0.0])),
        ],
    )
    assert np.isfinite(results["k_interp_5_50"])
    assert results["k_interp_5_50"] >= 0
    for kernel_value in results["direct_kernels"].values():
        assert np.isfinite(kernel_value)
        assert kernel_value >= 0


# =============================================================
# Test 3: Narrow opposite charge (+1 vs -1)
# =============================================================
def test_narrow_opposite_charge():
    """Bin 0 = +1, bins 1-3 = -1. Small attraction."""
    results = _run_scenario(
        label="SCENARIO 3: Narrow opposite charge (+1 vs -1)",
        bin_charges=np.array([1.0, -1.0, -1.0, -1.0]),
        query_charge_pairs=[
            ("(+1, -1) attractive", np.array([1.0, -1.0])),
            ("(-1, -1) repulsive", np.array([-1.0, -1.0])),
            ("(+1, +1) repulsive", np.array([1.0, 1.0])),
        ],
    )
    assert np.isfinite(results["k_interp_5_50"])
    assert results["k_interp_5_50"] >= 0
    for kernel_value in results["direct_kernels"].values():
        assert np.isfinite(kernel_value)
        assert kernel_value >= 0


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
    """Run scenario matched to the coag comparison test setup."""
    kernel_matrix = _compute_kernel_matrix(bin_radii, bin_charges)
    interp = _interpolate_kernel(kernel_matrix, bin_radii)

    # Interpolated value at ion(50nm) vs calcite(100nm)
    k_interp_50_100 = interp(np.array([[50e-9, 100e-9]])).item()

    r_pair = np.array([50e-9, 100e-9])
    k_neutral = _compute_kernel_matrix(r_pair, np.array([0.0, 0.0]))[0, 1]

    direct_kernels = {}
    for name, q_pair in query_charge_pairs:
        kd = _compute_kernel_matrix(r_pair, q_pair)[0, 1]
        direct_kernels[name] = kd

    return {
        "label": label,
        "k_interp_50_100": k_interp_50_100,
        "k_neutral": k_neutral,
        "direct_kernels": direct_kernels,
        "bin_radii": bin_radii,
        "scan_radii": scan_radii,
    }


# =============================================================
# Test 4: Calcite(-6) + Ion(-6) -- same sign, repulsive
# =============================================================
def test_coag_same_sign_repulsive():
    """Ion charge = -6, calcite charge = -6. All repulsive.

    All charges identical, so the interpolator is consistent
    with direct — no bleed problem.  Kernel values are
    ~11 orders of magnitude below Brownian baseline. Small
    rtol/atol is used because kernels are near machine precision.
    """
    results = _run_coag_scenario(
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
    k_repulsive = results["direct_kernels"]["ion+calcite (-6,-6) repulsive"]
    k_neutral = results["direct_kernels"]["neutral (0,0) Brownian"]
    k_interp = results["k_interp_50_100"]

    assert k_repulsive < k_neutral * 1e-8
    # Allow a small tolerance because kernels are ~1e-24 to 1e-23 in this case.
    npt.assert_allclose(
        k_interp,
        k_repulsive,
        rtol=1e-6,
        atol=k_neutral * 1e-12,
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

    Neutral direct kernel should match the Brownian baseline and the
    interpolation should align within a small numeric tolerance.
    """
    results = _run_coag_scenario(
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
    k_neutral = results["direct_kernels"]["neutral (0,0) Brownian"]
    k_neutral_charged = results["direct_kernels"][
        "ion+calcite (0,-6) neutral-charged"
    ]
    k_interp = results["k_interp_50_100"]

    # Small tolerance used for floating-point roundoff at ~1e-15 scale.
    npt.assert_allclose(k_neutral_charged, k_neutral, rtol=1e-6)
    npt.assert_allclose(k_interp, k_neutral_charged, rtol=1e-6)


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

    Attractive direct kernel should exceed neutral and interpolation
    should match within a small numeric tolerance.
    """
    results = _run_coag_scenario(
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
    k_attractive = results["direct_kernels"]["ion+calcite (+6,-6) attractive"]
    k_neutral = results["direct_kernels"]["neutral (0,0) Brownian"]
    k_interp = results["k_interp_50_100"]

    assert k_attractive > k_neutral * 10.0
    # Allow a small tolerance due to scale differences across charge pairs.
    npt.assert_allclose(k_interp, k_attractive, rtol=1e-6)
