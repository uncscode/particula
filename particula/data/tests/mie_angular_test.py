"""Test initialize"""

import numpy as np
import pytest

from particula.data.process import mie_angular


def test_discretize_scattering_angles():
    """Test that discretize_scattering_angles returns arrays with correct
    shapes and values."""
    # Define inputs
    m_sphere = 1.5  # Example refractive index
    wavelength = 532  # Example wavelength in nm
    diameter = 100  # Example diameter in nm
    min_angle = 0  # Minimum angle in degrees
    max_angle = 180  # Maximum angle in degrees
    angular_resolution = 45  # Angular resolution in degrees

    # Call the function
    measure, parallel, perpendicular, unpolarized = \
        mie_angular.discretize_scattering_angles(
            m_sphere=m_sphere,
            wavelength=wavelength,
            diameter=diameter,
            min_angle=min_angle,
            max_angle=max_angle,
            angular_resolution=angular_resolution
        )
    exp_measure = np.array([0.0, 0.78539816, 1.57079633,
                            2.35619449, 3.14159265])
    exp_parallel = np.array(
        [0.00432451, 0.00412889, 0.00368662, 0.00328417, 0.00312842])
    exp_perpendicular = np.array(
        [4.32451363e-03, 2.11356450e-03, 1.10332321e-06,
         1.60080223e-03, 3.12841559e-03])
    exp_unpolarized = np.array(
        [0.00432451, 0.00312123, 0.00184386, 0.00244249, 0.00312842])

    # Check that the output arrays have values close to the expected values
    assert np.allclose(
        measure, exp_measure, atol=1e-6), "Measure not match values"
    assert np.allclose(
        parallel, exp_parallel, atol=1e-6), "parellel not match values"
    assert np.allclose(
        perpendicular, exp_perpendicular, atol=1e-6
        ), "perpendicular not match values"
    assert np.allclose(
        unpolarized, exp_unpolarized, atol=1e-6
        ), "unpolarized not match expected values"


def test_calculate_scattering_angles():
    """Test scattering angles calculation for a position inside the sphere,
    not at the edges."""
    # Define inputs not at the edge of the sphere
    z_position = 5.0
    integrate_sphere_diameter_cm = 20.0
    tube_diameter_cm = 5.0

    alpha, beta = mie_angular.calculate_scattering_angles(
        z_position, integrate_sphere_diameter_cm, tube_diameter_cm)
    exp_alpha = 0.4636476090008061
    exp_beta = 0.16514867741462683
    # Check that alpha and beta are floats and within the expected range
    assert isinstance(alpha, float), "Alpha is not a float"
    assert isinstance(beta, float), "Beta is not a float"
    # check values
    assert alpha == pytest.approx(
        exp_alpha, abs=1e-8
    ), "Alpha value compared"
    assert beta == pytest.approx(
        exp_beta, abs=1e-8
    ), "Beta value compared"


def test_calculate_scattering_angles_outside_sphere():
    """Test scattering angles calculation for a position outside the sphere."""
    # Define inputs for a position outside the sphere
    z_position_outside = 11.0  # Outside the sphere's radius
    integrate_sphere_diameter_cm = 20.0
    tube_diameter_cm = 5.0

    # Calculate expected values for alpha and beta for the given position
    # These values need to be calculated based on the specific geometry
    exp_alpha_outside = np.arctan(
        tube_diameter_cm / 2 /
        (z_position_outside - integrate_sphere_diameter_cm / 2))
    exp_beta_outside = np.arctan(
        tube_diameter_cm / 2 /
        (z_position_outside + integrate_sphere_diameter_cm / 2))

    alpha_outside, beta_outside = mie_angular.calculate_scattering_angles(
        z_position_outside, integrate_sphere_diameter_cm, tube_diameter_cm)

    # Check that alpha and beta are floats and match the expected values
    assert isinstance(alpha_outside, float), "Alpha (outside) is not a float"
    assert isinstance(beta_outside, float), "Beta (outside) is not a float"
    assert alpha_outside == pytest.approx(
        exp_alpha_outside, abs=1e-8), "Alpha (outside) value mismatch"
    assert beta_outside == pytest.approx(
        exp_beta_outside, abs=1e-8), "Beta (outside) value mismatch"


def test_calculate_scattering_angles_at_edge():
    """Test scattering angles calculation for a position at the
    edge of the sphere."""
    # Define inputs for a position at the edge of the sphere
    z_position_edge = 10.0  # Exactly at the sphere's edge
    integrate_sphere_diameter_cm = 20.0
    tube_diameter_cm = 5.0

    # At the edge, alpha and beta are expected to be pi/2
    exp_alpha_edge = np.pi / 2
    exp_beta_edge = np.arctan(
        tube_diameter_cm / 2 /
        abs(integrate_sphere_diameter_cm / 2 + z_position_edge))

    alpha_edge, beta_edge = mie_angular.calculate_scattering_angles(
        z_position_edge, integrate_sphere_diameter_cm, tube_diameter_cm)

    # Check that alpha and beta are floats and match the expected values of
    # pi/2
    assert isinstance(alpha_edge, float), "Alpha (edge) is not a float"
    assert isinstance(beta_edge, float), "Beta (edge) is not a float"
    assert alpha_edge == pytest.approx(
        exp_alpha_edge, abs=1e-8), "Alpha (edge) value mismatch"
    assert beta_edge == pytest.approx(
        exp_beta_edge, abs=1e-8), "Beta (edge) value mismatch"


def test_assign_scattering_thetas_outside_below():
    """Test for z_position outside and below the integrating sphere."""
    alpha = np.pi / 4  # 45 degrees in radians
    beta = np.pi / 6   # 30 degrees in radians
    q_mie = 1.5
    z_position = -10.5  # Outside and below
    integrate_sphere_diameter_cm = 20.0  # Sphere diameter

    expected_qsca_ideal = 0
    expected_theta1 = 0.7853981633974483
    expected_theta2 = beta

    theta1, theta2, qsca_ideal = mie_angular.assign_scattering_thetas(
        alpha, beta, q_mie, z_position, integrate_sphere_diameter_cm)

    assert theta1 == pytest.approx(
        expected_theta1), "Incorrect theta1 for outside and below the sphere"
    assert theta2 == pytest.approx(
        expected_theta2), "Incorrect theta2 for outside and below the sphere"
    assert qsca_ideal == pytest.approx(
        expected_qsca_ideal), "Incorrect qsca_ideal for outside the sphere"


def test_assign_scattering_thetas_outside_above():
    """Test for z_position outside and above the integrating sphere."""
    alpha = np.pi / 4
    beta = np.pi / 6
    q_mie = 1.5
    z_position = 10.5  # Outside and above
    integrate_sphere_diameter_cm = 20.0

    expected_qsca_ideal = 0
    expected_theta1 = 2.356194490192345
    expected_theta2 = 2.6179938779914944

    theta1, theta2, qsca_ideal = mie_angular.assign_scattering_thetas(
        alpha, beta, q_mie, z_position, integrate_sphere_diameter_cm)

    assert theta1 == pytest.approx(
        expected_theta1), "Incorrect theta1 for position outside the sphere"
    assert theta2 == pytest.approx(
        expected_theta2), "Incorrect theta2 for position outside the sphere"
    assert qsca_ideal == pytest.approx(
        expected_qsca_ideal), "Incorrect qsca_ideal for outside the sphere"


def test_assign_scattering_thetas_inside():
    """Test for z_position inside the integrating sphere."""
    alpha = np.pi / 4
    beta = np.pi / 6
    q_mie = 1.5
    z_position = 0  # Inside
    integrate_sphere_diameter_cm = 20.0

    expected_qsca_ideal = q_mie
    expected_theta1 = alpha
    expected_theta2 = np.pi - beta

    theta1, theta2, qsca_ideal = mie_angular.assign_scattering_thetas(
        alpha, beta, q_mie, z_position, integrate_sphere_diameter_cm)

    assert theta1 == pytest.approx(
        expected_theta1), "Incorrect theta1 for position inside the sphere"
    assert theta2 == pytest.approx(
        expected_theta2), "Incorrect theta2 for position inside the sphere"
    assert qsca_ideal == pytest.approx(
        expected_qsca_ideal), "Incorrect qsca_ideal for position inside sphere"
