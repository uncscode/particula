"""Tests for partitioning helpers, validation, and regression safeguards."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from particula.equilibria import partitioning
from particula.particles.properties.organic_density_module import (
    get_organic_density_array,
)
from scipy.optimize import OptimizeResult


def _build_partitioning_inputs(
    species_count: int = 5, water_activity: float = 0.7
):
    """Create representative inputs for partitioning helpers."""
    species_count = max(1, species_count)
    c_star_j_dry = np.linspace(1e-3, 1.0, species_count, dtype=float)
    concentration_organic_matter = np.linspace(
        0.2, 1.0, species_count, dtype=float
    )
    molar_mass = np.full(species_count, 200.0, dtype=float)
    oxygen2carbon = np.linspace(0.25, 0.45, species_count, dtype=float)
    density = get_organic_density_array(
        molar_mass=molar_mass,
        oxygen2carbon=oxygen2carbon,
        hydrogen2carbon=None,
        nitrogen2carbon=None,
    )
    water_activity_array = np.full(species_count, water_activity, dtype=float)
    gamma_organic_ab, mass_fraction_water_ab, q_ab = (
        partitioning.get_properties_for_liquid_vapor_partitioning(
            water_activity_desired=water_activity_array,
            molar_mass=molar_mass,
            oxygen2carbon=oxygen2carbon,
            density=density,
        )
    )
    return (
        c_star_j_dry,
        concentration_organic_matter,
        molar_mass,
        gamma_organic_ab,
        mass_fraction_water_ab,
        q_ab,
    )


def test_calculate_alpha_phase_basic_and_zero_water():
    """Validate alpha-phase computation for finite and unity water cases."""
    c_j_liquid = np.array([1.0, 2.0], dtype=float)
    q_ab = np.array([[0.5, 0.2], [1.0, 0.0]], dtype=float)
    mass_fraction_water_ab = np.array([[0.2, 0.0], [1.0, 0.0]], dtype=float)

    (
        c_j_alpha,
        c_j_aq_alpha,
        c_alpha_total,
        c_aq_alpha,
        denominator_alpha,
    ) = partitioning._calculate_alpha_phase(
        c_j_liquid=c_j_liquid,
        q_ab=q_ab,
        mass_fraction_water_ab=mass_fraction_water_ab,
    )

    expected_c_j_alpha = c_j_liquid * q_ab[:, 0]
    expected_denominator = np.where(
        mass_fraction_water_ab[:, 0] >= 1.0,
        0.0,
        1.0 / (1.0 - mass_fraction_water_ab[:, 0] + partitioning.EPSILON),
    )
    expected_c_j_aq_alpha = (
        expected_c_j_alpha
        * mass_fraction_water_ab[:, 0]
        * (expected_denominator)
    )
    expected_c_alpha_total = float(
        np.sum(expected_c_j_alpha) + np.sum(expected_c_j_aq_alpha)
    )
    expected_c_aq_alpha = float(np.sum(expected_c_j_aq_alpha))

    assert_allclose(c_j_alpha, expected_c_j_alpha)
    assert_allclose(denominator_alpha, expected_denominator)
    assert_allclose(c_j_aq_alpha, expected_c_j_aq_alpha)
    assert c_alpha_total == pytest.approx(expected_c_alpha_total)
    assert c_aq_alpha == pytest.approx(expected_c_aq_alpha)


def test_calculate_beta_phase_basic_and_zero_water():
    """Validate beta-phase computation for finite and unity water cases."""
    c_j_liquid = np.array([1.0, 3.0], dtype=float)
    q_ab = np.array([[0.1, 0.9], [0.0, 1.0]], dtype=float)
    mass_fraction_water_ab = np.array([[0.3, 0.7], [0.0, 1.0]], dtype=float)

    (
        c_j_beta,
        c_j_aq_beta,
        c_beta_total,
        c_aq_beta,
        denominator_beta,
    ) = partitioning._calculate_beta_phase(
        c_j_liquid=c_j_liquid,
        q_ab=q_ab,
        mass_fraction_water_ab=mass_fraction_water_ab,
    )

    expected_c_j_beta = c_j_liquid * q_ab[:, 1]
    expected_denominator = np.where(
        mass_fraction_water_ab[:, 1] >= 1.0,
        0.0,
        1.0 / (1.0 - mass_fraction_water_ab[:, 1] + partitioning.EPSILON),
    )
    expected_c_j_aq_beta = (
        expected_c_j_beta
        * mass_fraction_water_ab[:, 1]
        * (expected_denominator)
    )
    expected_c_beta_total = float(
        np.sum(expected_c_j_beta) + np.sum(expected_c_j_aq_beta)
    )
    expected_c_aq_beta = float(np.sum(expected_c_j_aq_beta))

    assert_allclose(c_j_beta, expected_c_j_beta)
    assert_allclose(denominator_beta, expected_denominator)
    assert_allclose(c_j_aq_beta, expected_c_j_aq_beta)
    assert c_beta_total == pytest.approx(expected_c_beta_total)
    assert c_aq_beta == pytest.approx(expected_c_aq_beta)


def test_calculate_cstar_zero_mass_weighted_returns_zero():
    """Return zeros for C* when mass-weighted molar mass is zero."""
    c_star_j_dry = np.array([1e-3, 5e-3], dtype=float)

    gamma_phase = np.ones_like(c_star_j_dry)
    q_phase = np.full_like(c_star_j_dry, 0.5)
    molar_mass = np.full_like(c_star_j_dry, 200.0)

    c_star_output = partitioning._calculate_cstar(
        c_star_j_dry=c_star_j_dry,
        gamma_organic_phase=gamma_phase,
        q_phase=q_phase,
        c_liquid_total=1.0,
        molar_mass=molar_mass,
        mass_weighted_molar_mass=0.0,
    )

    assert_allclose(c_star_output, np.zeros_like(c_star_j_dry))


def test_get_properties_beta_none_zero_fill(monkeypatch):
    """Zero-fill beta arrays when water activity yields no beta phase."""

    def _fake_fixed_water_activity(*args, **kwargs):
        alpha = (
            np.array([0.5], dtype=float),
            np.array([0.5], dtype=float),
            np.array([0.1], dtype=float),
            np.array([0.9], dtype=float),
            np.array([1.0], dtype=float),
            np.array([2.0], dtype=float),
        )
        beta = None
        q_alpha = np.array([1.0], dtype=float)
        return alpha, beta, q_alpha

    monkeypatch.setattr(
        partitioning, "fixed_water_activity", _fake_fixed_water_activity
    )

    gamma_organic_ab, mass_fraction_water_ab, q_ab = (
        partitioning.get_properties_for_liquid_vapor_partitioning(
            water_activity_desired=0.5,
            molar_mass=np.array([150.0], dtype=float),
            oxygen2carbon=np.array([0.4], dtype=float),
            density=np.array([1100.0], dtype=float),
        )
    )

    assert gamma_organic_ab[0, 1] == 0.0
    assert mass_fraction_water_ab[0, 1] == 0.0
    assert q_ab[0, 1] == 0.0
    assert q_ab[0, 0] == 1.0


def test_liquid_vapor_partitioning_regression_shapes_and_finiteness():
    """Safeguard shapes and finiteness of partitioning regression outputs."""
    (
        c_star_j_dry,
        concentration_organic_matter,
        molar_mass,
        gamma_organic_ab,
        mass_fraction_water_ab,
        q_ab,
    ) = _build_partitioning_inputs(species_count=5)

    alpha_phase, beta_phase, system_output, fit_result = (
        partitioning.liquid_vapor_partitioning(
            c_star_j_dry=c_star_j_dry,
            concentration_organic_matter=concentration_organic_matter,
            molar_mass=molar_mass,
            gamma_organic_ab=gamma_organic_ab,
            mass_fraction_water_ab=mass_fraction_water_ab,
            q_ab=q_ab,
            partition_coefficient_guess=None,
        )
    )

    assert alpha_phase[0].shape == c_star_j_dry.shape
    assert beta_phase[0].shape == c_star_j_dry.shape
    assert np.isfinite(system_output[0])
    assert np.isfinite(system_output[1])
    assert system_output[2].shape == c_star_j_dry.shape
    assert fit_result.x.shape == c_star_j_dry.shape


def test_liquid_vapor_partitioning_single_species_and_zero_concentration():
    """Handle single-species and zero-concentration edge cases without fail."""
    (
        c_star_j_dry,
        _,
        molar_mass,
        gamma_organic_ab,
        mass_fraction_water_ab,
        q_ab,
    ) = _build_partitioning_inputs(species_count=1)
    concentration_organic_matter = np.zeros_like(c_star_j_dry)

    alpha_phase, beta_phase, system_output, _ = (
        partitioning.liquid_vapor_partitioning(
            c_star_j_dry=c_star_j_dry,
            concentration_organic_matter=concentration_organic_matter,
            molar_mass=molar_mass,
            gamma_organic_ab=gamma_organic_ab,
            mass_fraction_water_ab=mass_fraction_water_ab,
            q_ab=q_ab,
            partition_coefficient_guess=None,
        )
    )

    assert alpha_phase[0].shape == (1,)
    assert beta_phase[0].shape == (1,)
    assert np.isfinite(system_output[0])
    assert np.isfinite(system_output[2]).all()


def test_partition_coefficient_guess_matches_length(monkeypatch):
    """Partition guess defaults to correct length and passes into optimizer."""
    (
        c_star_j_dry,
        concentration_organic_matter,
        molar_mass,
        gamma_organic_ab,
        mass_fraction_water_ab,
        q_ab,
    ) = _build_partitioning_inputs(species_count=3)

    captured: dict[str, np.ndarray] = {}

    def _fake_minimize(**kwargs):
        captured["x0"] = np.asarray(kwargs["x0"])
        return OptimizeResult(x=np.zeros_like(captured["x0"]))

    monkeypatch.setattr(partitioning, "minimize", _fake_minimize)

    partitioning.liquid_vapor_partitioning(
        c_star_j_dry=c_star_j_dry,
        concentration_organic_matter=concentration_organic_matter,
        molar_mass=molar_mass,
        gamma_organic_ab=gamma_organic_ab,
        mass_fraction_water_ab=mass_fraction_water_ab,
        q_ab=q_ab,
        partition_coefficient_guess=None,
    )

    assert "x0" in captured
    assert captured["x0"].shape == c_star_j_dry.shape


def test_partition_coefficient_guess_length_mismatch_raises():
    """Raise when partition guess length mismatches the species count."""
    (
        c_star_j_dry,
        concentration_organic_matter,
        molar_mass,
        gamma_organic_ab,
        mass_fraction_water_ab,
        q_ab,
    ) = _build_partitioning_inputs(species_count=2)
    bad_guess = np.ones(c_star_j_dry.size + 1, dtype=float)

    with pytest.raises(ValueError):
        partitioning.liquid_vapor_partitioning(
            c_star_j_dry=c_star_j_dry,
            concentration_organic_matter=concentration_organic_matter,
            molar_mass=molar_mass,
            gamma_organic_ab=gamma_organic_ab,
            mass_fraction_water_ab=mass_fraction_water_ab,
            q_ab=q_ab,
            partition_coefficient_guess=bad_guess,
        )


def test_liquid_vapor_partitioning_rejects_negative_or_nonfinite_inputs():
    """Validation decorator rejects negative or non-finite inputs."""
    (
        c_star_j_dry,
        concentration_organic_matter,
        molar_mass,
        gamma_organic_ab,
        mass_fraction_water_ab,
        q_ab,
    ) = _build_partitioning_inputs(species_count=1)

    with pytest.raises(ValueError):
        partitioning.liquid_vapor_partitioning(
            c_star_j_dry=-np.abs(c_star_j_dry),
            concentration_organic_matter=concentration_organic_matter,
            molar_mass=molar_mass,
            gamma_organic_ab=gamma_organic_ab,
            mass_fraction_water_ab=mass_fraction_water_ab,
            q_ab=q_ab,
            partition_coefficient_guess=None,
        )

    with pytest.raises(ValueError):
        partitioning.liquid_vapor_partitioning(
            c_star_j_dry=c_star_j_dry,
            concentration_organic_matter=np.array([np.nan], dtype=float),
            molar_mass=molar_mass,
            gamma_organic_ab=gamma_organic_ab,
            mass_fraction_water_ab=mass_fraction_water_ab,
            q_ab=q_ab,
            partition_coefficient_guess=None,
        )


def test_get_properties_length_mismatch_raises():
    """Input length mismatch in property helper raises informative errors."""
    with pytest.raises(ValueError):
        partitioning.get_properties_for_liquid_vapor_partitioning(
            water_activity_desired=np.array([0.4, 0.4, 0.4], dtype=float),
            molar_mass=np.array([150.0, 160.0], dtype=float),
            oxygen2carbon=np.array([0.3, 0.4], dtype=float),
            density=np.array([1100.0, 1150.0], dtype=float),
        )


def test_liquid_vapor_partitioning_rejects_1d_arrays():
    """Validation rejects 1D arrays for gamma, mass_fraction, q."""
    (
        c_star_j_dry,
        concentration_organic_matter,
        molar_mass,
        _,
        _,
        _,
    ) = _build_partitioning_inputs(species_count=2)

    # Create 1D arrays instead of 2D arrays
    gamma_1d = np.array([1.0, 1.0], dtype=float)
    mass_fraction_1d = np.array([0.5, 0.5], dtype=float)
    q_1d = np.array([0.5, 0.5], dtype=float)

    with pytest.raises(ValueError, match="must be 2D arrays"):
        partitioning.liquid_vapor_partitioning(
            c_star_j_dry=c_star_j_dry,
            concentration_organic_matter=concentration_organic_matter,
            molar_mass=molar_mass,
            gamma_organic_ab=gamma_1d,
            mass_fraction_water_ab=mass_fraction_1d,
            q_ab=q_1d,
        )


def test_liquid_vapor_partitioning_rejects_wrong_column_count():
    """Validation rejects 2D arrays with wrong number of columns."""
    (
        c_star_j_dry,
        concentration_organic_matter,
        molar_mass,
        _,
        _,
        _,
    ) = _build_partitioning_inputs(species_count=2)

    # Create 2D arrays with wrong number of columns (3 instead of 2)
    gamma_3cols = np.ones((2, 3), dtype=float)
    mass_fraction_3cols = np.ones((2, 3), dtype=float)
    q_3cols = np.ones((2, 3), dtype=float)

    with pytest.raises(ValueError, match="must have two columns"):
        partitioning.liquid_vapor_partitioning(
            c_star_j_dry=c_star_j_dry,
            concentration_organic_matter=concentration_organic_matter,
            molar_mass=molar_mass,
            gamma_organic_ab=gamma_3cols,
            mass_fraction_water_ab=mass_fraction_3cols,
            q_ab=q_3cols,
        )
