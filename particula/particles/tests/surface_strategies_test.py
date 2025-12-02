"""Test the surface module."""

import numpy as np
import pytest

from particula.particles.surface_strategies import (
    SurfaceStrategyMass,
    SurfaceStrategyMolar,
    SurfaceStrategyVolume,
    _as_2d,
    _broadcast_weights,
    _interp_surface_tension,  # NEW
    _weighted_average_by_phase,
)


# pylint: disable=W0612
def test_molar_surface_strategy():
    """Test MolarSurfaceStrategy class."""
    surface_tension = np.array([0.072, 0.05])  # water, oil
    density = np.array([1000, 800])  # water, oil
    molar_mass = np.array([0.01815, 0.03])  # water, oil

    SurfaceStrategyMolar(surface_tension, density, molar_mass)

    # Test effective surface tension
    np.array([100, 200])  # water, oil
    expected_st_scalar = surface_tension
    np.full_like(surface_tension, expected_st_scalar)


def test_molar_strategy_update_surface_tension():
    """Test updating surface tension via temperature tables for Molar."""
    st_table = np.array(
        [
            [0.072, 0.07],
            [0.071, 0.069],
            [0.070, 0.065],
            [0.069, 0.060],
            [0.068, 0.058],
        ]
    )
    t_table_in = np.array([273, 283, 293, 303, 313])

    strategy = SurfaceStrategyMolar(
        surface_tension=0.072,  # default fallback
        density=np.array([1000, 800]),
        molar_mass=np.array([0.01815, 0.03]),
        surface_tension_table=st_table,
        temperature_table=t_table_in,
    )

    mass_concentration = np.array([150.0, 150.0])  # water-like, dual-species

    expected_surface_tension = np.array([0.070, 0.065])  # Expected at 293 K

    # Check at 293 K
    st_293 = strategy.effective_surface_tension(
        mass_concentration, temperature=293
    )
    # Expect interpolation → row 2 of st_table: [0.070, 0.065]
    np.testing.assert_allclose(st_293, [0.070, 0.065], rtol=1e-5)

    molar_mass_water = np.array([0.01815, 0.03])[0]  # Molar mass of water

    # Check at 313 K
    st_313 = strategy.effective_surface_tension(
        mass_concentration, temperature=313
    )
    # Expect last row: [0.068, 0.058]
    np.testing.assert_allclose(st_313, [0.068, 0.058], rtol=1e-5)

    expected_density = np.array([1000, 800])  # Expected density

    expected_kelvin_radius = (
        2 * expected_surface_tension * molar_mass_water
    ) / (8.314 * 298 * expected_density)
    radius = 1e-7
    kelvin_val = strategy.kelvin_term(radius, 0.01815, mass_concentration, 293)
    assert kelvin_val.squeeze().shape == (2,), (
        f"Expected shape (2,) got {kelvin_val.shape}"
    )
    np.testing.assert_allclose(
        strategy.effective_surface_tension(mass_concentration),
        expected_surface_tension,
    )

    # Test effective density
    expected_density = np.array([1000, 800])  # Expected density
    np.testing.assert_allclose(strategy.get_density(), expected_density)

    # Test kelvin_radius
    molar_mass_water = np.array([0.01815, 0.03])[0]  # Molar mass of water
    expected_kelvin_radius = (
        2 * expected_surface_tension * molar_mass_water
    ) / (8.314 * 298 * expected_density)
    np.testing.assert_allclose(
        strategy.kelvin_radius(molar_mass_water, mass_concentration, 298),
        expected_kelvin_radius,
        rtol=5e-2,
    )

    # Test kelvin_term
    radius = 1e-6
    expected_kelvin_term = np.exp(expected_kelvin_radius / radius)
    np.testing.assert_allclose(
        strategy.kelvin_term(
            radius, molar_mass_water, mass_concentration, 298
        ).squeeze(),
        expected_kelvin_term,
        rtol=1e-4,
    )


# Test MassSurfaceStrategy
def test_mass_surface_strategy():
    """Test MassSurfaceStrategy class."""
    surface_tension = np.array([0.072, 0.05])  # water, oil
    density = np.array([1000, 800])  # water, oil

    strategy = SurfaceStrategyMass(surface_tension, density)

    # Test effective surface tension
    mass_concentration = np.array([100, 200])  # water, oil
    expected_surface_tension = surface_tension  # no mixing
    np.testing.assert_allclose(
        strategy.effective_surface_tension(mass_concentration),
        expected_surface_tension,
    )

    # Test effective density
    expected_density = density
    np.testing.assert_allclose(strategy.get_density(), expected_density)

    # Test kelvin_radius
    expected_kelvin_radius = (2 * expected_surface_tension * 0.01815) / (
        8.314 * 298 * expected_density
    )
    np.testing.assert_allclose(
        strategy.kelvin_radius(0.01815, mass_concentration, 298),
        expected_kelvin_radius,
        rtol=1e-4,
    )

    # Test kelvin_term
    radius = 1e-6
    expected_kelvin_term = np.exp(expected_kelvin_radius / radius)
    np.testing.assert_allclose(
        strategy.kelvin_term(
            radius, 0.01815, mass_concentration, 298
        ).squeeze(),
        expected_kelvin_term,
        rtol=1e-4,
    )


# Test VolumeSurfaceStrategy
def test_volume_surface_strategy():
    """Test VolumeSurfaceStrategy class."""
    surface_tension = np.array([0.072, 0.05])  # water, oil
    density = np.array([1000, 800])  # water, oil

    strategy = SurfaceStrategyVolume(surface_tension, density)

    # Test effective surface tension
    mass_concentration = np.array([100, 200])  # water, oil
    expected_surface_tension = surface_tension  # no mixing
    np.testing.assert_allclose(
        strategy.effective_surface_tension(mass_concentration),
        expected_surface_tension,
    )

    # Test effective density
    expected_density = density
    np.testing.assert_allclose(strategy.get_density(), expected_density)

    # Test kelvin_radius
    expected_kelvin_radius = (2 * expected_surface_tension * 0.01815) / (
        8.314 * 298 * expected_density
    )
    np.testing.assert_allclose(
        strategy.kelvin_radius(0.01815, mass_concentration, 298),
        expected_kelvin_radius,
        rtol=1e-4,
    )

    # Test kelvin_term
    radius = 1e-6
    expected_kelvin_term = np.exp(expected_kelvin_radius / radius)
    np.testing.assert_allclose(
        strategy.kelvin_term(
            radius, 0.01815, mass_concentration, 298
        ).squeeze(),
        expected_kelvin_term,
        rtol=1e-4,
    )


def test_molar_surface_strategy_scalar_input():
    """Test MolarSurfaceStrategy with scalar inputs."""
    surface_tension = 0.072  # scalar
    density = 1000  # scalar
    molar_mass = 0.01815  # scalar
    mass_concentration = 150  # any positive value
    strat = SurfaceStrategyMolar(surface_tension, density, molar_mass)

    # Assert get_density for scalar input
    assert strat.get_density() == pytest.approx(density)
    assert strat.effective_surface_tension(mass_concentration) == pytest.approx(
        surface_tension
    )


def test_surface_strategy_phase_index():
    """Test phase-index mixing option."""
    surface_tension = np.array([0.072, 0.05])
    density = np.array([1000, 800])
    molar_mass = np.array([0.01815, 0.03])
    phase_index = np.array([0, 1])
    mass_concentration = np.array([100, 200])

    strat = SurfaceStrategyMolar(
        surface_tension, density, molar_mass, phase_index=phase_index
    )
    expected_st = surface_tension
    np.testing.assert_allclose(
        strat.effective_surface_tension(mass_concentration), expected_st
    )

    strat_mass = SurfaceStrategyMass(
        surface_tension, density, phase_index=phase_index
    )
    np.testing.assert_allclose(
        strat_mass.effective_surface_tension(mass_concentration),
        surface_tension,
    )

    strat_vol = SurfaceStrategyVolume(
        surface_tension, density, phase_index=phase_index
    )
    np.testing.assert_allclose(
        strat_vol.effective_surface_tension(mass_concentration),
        surface_tension,
    )


def test_surface_strategy_multi_phase_mixing():
    """Validate mixing when phases contain more than one species."""
    # ---- input data --------------------------------------------------------
    surface_tension = np.array([0.072, 0.050, 0.060, 0.055])
    density = np.array([1000, 800, 850, 900])
    molar_mass = np.array([0.01815, 0.030, 0.040, 0.050])
    mass_conc = np.array([100.0, 200.0, 100.0, 150.0])
    phase_index = [0, 1, 1, 2]  # three phases

    # helper that reproduces the weighting rule -----------------------------
    def _expected_st(st, weights, p_index):
        """Return array with phase-wise weighted surface tension."""
        exp = np.zeros_like(st, dtype=np.float64)
        for ph in np.unique(p_index):
            m = p_index == ph
            w = weights[m] / weights[m].sum()
            exp[m] = (st[m] * w).sum()
        return exp

    # -------- expected values for each strategy ----------------------------
    # molar weighting
    mole_counts = mass_conc / molar_mass
    exp_molar = _expected_st(surface_tension, mole_counts, phase_index)
    # mass weighting
    exp_mass = _expected_st(surface_tension, mass_conc, phase_index)
    # volume weighting  (volume = mass / density)
    volumes = mass_conc / density
    exp_volume = _expected_st(surface_tension, volumes, phase_index)
    # ------------- instantiate strategies ----------------------------------
    strat_molar = SurfaceStrategyMolar(
        surface_tension, density, molar_mass, phase_index=phase_index
    )
    strat_mass = SurfaceStrategyMass(
        surface_tension, density, phase_index=phase_index
    )
    strat_volume = SurfaceStrategyVolume(
        surface_tension, density, phase_index=phase_index
    )
    # --------------------- assertions --------------------------------------
    np.testing.assert_allclose(
        strat_molar.effective_surface_tension(mass_conc), exp_molar
    )
    np.testing.assert_allclose(
        strat_mass.effective_surface_tension(mass_conc), exp_mass
    )
    np.testing.assert_allclose(
        strat_volume.effective_surface_tension(mass_conc), exp_volume
    )


def test_weighted_average_by_phase_helper():
    """Ensure helper returns correct phase-weighted averages."""
    vals = np.array([0.072, 0.050, 0.060])
    wts = np.array([1.0, 1.0, 2.0])
    phases = np.array([0, 1, 1])

    expected = np.array(
        [
            0.072,  # phase 0: single member
            (0.050 * 1 + 0.060 * 2) / 3,  # phase 1 average
            (0.050 * 1 + 0.060 * 2) / 3,
        ]
    )

    np.testing.assert_allclose(
        _weighted_average_by_phase(vals, wts, phases),
        expected,
    )


def test_helper_shape_management():
    """Validate _as_2d and _broadcast_weights helper behaviour."""
    # ----- _as_2d ----------------------------------------------------------
    vec = np.array([1.0, 2.0, 3.0])
    mat, flag = _as_2d(vec)
    assert flag is True
    assert mat.shape == (1, 3)
    mat2, flag2 = _as_2d(mat)
    assert flag2 is False
    assert mat2 is mat  # returns the same object

    # ----- _broadcast_weights ---------------------------------------------
    target_shape = (4, 3)
    # (a) 1-D weights  -> broadcast
    w1 = np.array([1.0, 1.0, 1.0])
    w1b = _broadcast_weights(w1, target_shape)
    assert w1b.shape == target_shape
    assert np.all(w1b == 1.0)

    # (b) single-row 2-D weights -> broadcast
    w2 = np.array([[2.0, 2.0, 2.0]])  # (1,3)
    w2b = _broadcast_weights(w2, target_shape)
    assert np.all(w2b == 2.0)

    # (c) already-matching 2-D weights returned unchanged
    w3 = np.arange(12, dtype=float).reshape(4, 3)
    w3b = _broadcast_weights(w3, target_shape)
    assert np.array_equal(w3b, w3)


def test_weighted_average_by_phase_multi_bin():
    """Check _weighted_average_by_phase for n_bins=4, n_species=3."""
    # ----------------------- set up input ---------------------------------
    values = np.array(
        [
            [1.0, 10.0, 2.0],
            [3.0, 20.0, 6.0],
            [5.0, 30.0, 10.0],
            [7.0, 40.0, 14.0],
        ]
    )  # (4,3)
    weights_1d = np.array([1.0, 1.0, 1.0])  # broadcast case
    phase_idx = np.array([1, 0, 1])  # species 0&2 phase-1, species 1 phase-0

    # ---------------------- expected result -------------------------------
    # phase-0 average = value of species-1
    # phase-1 average = mean(values[:,0], values[:,2])  (equal weights)
    expected = np.array(
        [
            [1.5, 10.0, 1.5],
            [4.5, 20.0, 4.5],
            [7.5, 30.0, 7.5],
            [10.5, 40.0, 10.5],
        ]
    )

    # -------------------- execute & assertions ----------------------------
    out1 = _weighted_average_by_phase(values, weights_1d, phase_idx)
    np.testing.assert_allclose(out1, expected)

    # also test when weights are provided already 2-D ----------------------
    weights_2d = np.tile(weights_1d, (4, 1))  # (4,3)
    out2 = _weighted_average_by_phase(values, weights_2d, phase_idx)
    np.testing.assert_allclose(out2, expected)


def test_weighted_average_by_phase_zero_weight_fallback():
    """When all weights in a phase are zero, fall back to un-weighted mean."""
    # single bin, two species, same phase
    vals = np.array([2.0, 6.0])  # (n_species,)
    wts = np.array([0.0, 0.0])  # all zeros
    phase = np.array([0, 0])  # single phase

    # expected: mean of values for every species
    exp = np.array([4.0, 4.0])  # arithmetic mean of 2 & 6

    np.testing.assert_allclose(
        _weighted_average_by_phase(vals, wts, phase),
        exp,
    )


def test_update_surface_tension_1d_table():
    """SurfaceStrategy must cope with a 1-D lookup‐table."""
    st_table = np.array([0.072, 0.071, 0.070, 0.069])  # N/m
    t_table = np.array([273, 283, 293, 303])  # K

    strat = SurfaceStrategyMass(
        surface_tension=0.072,
        density=1000,
        surface_tension_table=st_table,
        temperature_table=t_table,
    )

    mass_conc = 100.0  # arbitrary, scalar
    # at 293 K the table gives 0.070 N/m
    out = strat.effective_surface_tension(mass_conc, temperature=293)
    assert np.allclose(out, 0.070, rtol=1e-6)


def test_interp_surface_tension_helper():
    """Validate _interp_surface_tension for 1-D and 2-D tables."""
    # --- 1-D table --------------------------------------------------------
    t_tab = np.array([270.0, 280.0, 290.0])  # K
    st_1d = np.array([0.072, 0.071, 0.070])  # N / m
    val_1d = _interp_surface_tension(275.0, st_1d, t_tab)
    assert np.isclose(val_1d, np.interp(275.0, t_tab, st_1d))

    # --- 2-D table --------------------------------------------------------
    st_2d = np.array(
        [
            [0.072, 0.050],
            [0.071, 0.049],
            [0.070, 0.048],
        ]
    )  # shape (3 temps, 2 species)
    val_2d = _interp_surface_tension(285.0, st_2d, t_tab)
    expected = np.array(
        [
            np.interp(285.0, t_tab, st_2d[:, 0]),
            np.interp(285.0, t_tab, st_2d[:, 1]),
        ]
    )
    np.testing.assert_allclose(val_2d, expected)

    # --- unsupported dimensionality should raise -------------------------
    with pytest.raises(ValueError):
        bad_table = st_2d[:, :, np.newaxis]  # 3-D input
        _ = _interp_surface_tension(285.0, bad_table, t_tab)
