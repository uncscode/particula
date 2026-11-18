"""Test for activity_coefficients function."""

import numpy as np
from particula.activity.activity_coefficients import bat_activity_coefficients


def test_activity_coefficients():
    """Test for activity_coefficents function."""
    molar_mass_ratio = 18.016 / 250
    organic_mole_fraction = np.linspace(0.1, 1, 10)
    oxygen2carbon = 0.3
    density = 2500

    activity_coefficients = bat_activity_coefficients(
        molar_mass_ratio=molar_mass_ratio,
        organic_mole_fraction=organic_mole_fraction,
        oxygen2carbon=oxygen2carbon,
        density=density,
        functional_group=None,
    )
    assert np.all(activity_coefficients[0] >= 0)
    assert np.all(activity_coefficients[1] >= 0)
    assert np.all(activity_coefficients[2] >= 0)
    assert np.all(activity_coefficients[3] >= 0)


def test_activity_coefficients_with_functional_group():
    """Test for activity_coefficents function with functional group."""
    molar_mass_ratio = 18.016 / 250
    organic_mole_fraction = np.linspace(0.1, 1, 10)
    oxygen2carbon = 0.3
    density = 2500
    functional_group = "alcohol"

    activity_coefficients = bat_activity_coefficients(
        molar_mass_ratio=molar_mass_ratio,
        organic_mole_fraction=organic_mole_fraction,
        oxygen2carbon=oxygen2carbon,
        density=density,
        functional_group=functional_group,
    )
    assert np.all(activity_coefficients[0] >= 0)
    assert np.all(activity_coefficients[1] >= 0)
    assert np.all(activity_coefficients[2] >= 0)
    assert np.all(activity_coefficients[3] >= 0)


def test_activity_coefficients_edge_cases():
    """Test for activity_coefficents function with edge cases."""
    molar_mass_ratio = 18.016 / 250
    organic_mole_fraction = np.array([1e-12, 1.0])
    oxygen2carbon = 0.3
    density = 2500

    activity_coefficients = bat_activity_coefficients(
        molar_mass_ratio=molar_mass_ratio,
        organic_mole_fraction=organic_mole_fraction,
        oxygen2carbon=oxygen2carbon,
        density=density,
        functional_group=None,
    )
    assert np.all(activity_coefficients[0] >= 0)
    assert np.all(activity_coefficients[1] >= 0)
    assert np.all(activity_coefficients[2] >= 0)
    assert np.all(activity_coefficients[3] >= 0)


def test_activity_coefficients_with_arrays():
    """Test for activity_coefficents function with np.array inputs."""
    molar_mass_ratio = np.array([18.016 / 250, 18.016 / 200])
    organic_mole_fraction = np.array([0.2, 0.8])
    oxygen2carbon = np.array([0.3, 0.4])
    density = np.array([2500, 2400])

    activity_coefficients = bat_activity_coefficients(
        molar_mass_ratio=molar_mass_ratio,
        organic_mole_fraction=organic_mole_fraction,
        oxygen2carbon=oxygen2carbon,
        density=density,
        functional_group=None,
    )
    assert np.all(activity_coefficients[0] >= 0)
    assert np.all(activity_coefficients[1] >= 0)
    assert np.all(activity_coefficients[2] >= 0)
    assert np.all(activity_coefficients[3] >= 0)


def test_activity_coefficients_with_mixed_inputs():
    """Test for activity_coefficents function with mixed inputs."""
    molar_mass_ratio = 18.016 / 250
    organic_mole_fraction = np.array([0.1, 0.9])
    oxygen2carbon = 0.3
    density = np.array([2500, 2400])

    activity_coefficients = bat_activity_coefficients(
        molar_mass_ratio=molar_mass_ratio,
        organic_mole_fraction=organic_mole_fraction,
        oxygen2carbon=oxygen2carbon,
        density=density,
        functional_group=None,
    )
    assert np.all(activity_coefficients[0] >= 0)
    assert np.all(activity_coefficients[1] >= 0)
    assert np.all(activity_coefficients[2] >= 0)
    assert np.all(activity_coefficients[3] >= 0)


def test_citric_acid_figure_2b_reference():
    """Reproduce Fig2!I:K at water mole fractions 0.1, 0.5 and 0.9.

    Published workbook data_in_paper_figures.xlsx, BAT MATLAB revision
    6e0d5a4d1a9af4571358f0539682cbc0cc1e0257. Figure script uses entry
    5 of binary_fit_data_imported.mat and the upstream density estimator.
    """
    water, organic, *_ = bat_activity_coefficients(
        molar_mass_ratio=0.0938,
        organic_mole_fraction=np.array([0.9, 0.5, 0.1]),
        oxygen2carbon=1.16667,
        density=1469.1849580794367,
    )
    np.testing.assert_allclose(
        water,
        [0.0403303454851997, 0.288546529581348, 0.86813460026711],
        rtol=0,
        atol=2e-14,
    )
    np.testing.assert_allclose(
        organic,
        [0.897913722913258, 0.411838850232126, 0.0200162251692363],
        rtol=0,
        atol=2e-14,
    )


def test_miscible_figure_9_reference():
    """Check three compositions from the authors' archived mid-O:C curve."""
    water, organic, *_ = bat_activity_coefficients(
        molar_mass_ratio=0.09008,
        organic_mole_fraction=np.array([0.9, 0.5, 0.1]),
        oxygen2carbon=0.5,
        density=1243.4933333333333,
    )
    np.testing.assert_allclose(
        water, [0.0771364279, 0.4581165422, 0.9302584435], rtol=0, atol=3e-6
    )
    np.testing.assert_allclose(
        organic, [0.8988582008, 0.4564539978, 0.0787672383], rtol=0, atol=2e-6
    )
