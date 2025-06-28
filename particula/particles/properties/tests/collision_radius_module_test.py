"""Test for collision_radius_module.py module."""

import numpy as np

from particula.particles.properties.collision_radius_module import (
    get_collision_radius_mg1988,
    get_collision_radius_mzg2002,
    get_collision_radius_sr1992,
    get_collision_radius_tt2012,
    get_collision_radius_wq2022_rg,
    get_collision_radius_wq2022_rg_df,
    get_collision_radius_wq2022_rg_df_k0,
    get_collision_radius_wq2022_rg_df_k0_a13,
)


def test_mulholland_1988():
    """Test the mulholland_1988 function."""
    assert get_collision_radius_mg1988(1.5) == 1.5
    assert get_collision_radius_mg1988(0.0) == 0.0
    assert get_collision_radius_mg1988(2.5) == 2.5
    test_array = np.array([[1.5, 0.0], [1.5, 2.0]])
    np.testing.assert_array_equal(
        get_collision_radius_mg1988(test_array), test_array
    )


def test_rogak_flagan_1992():
    """Test the rogak_flagan_1992 function."""
    result = get_collision_radius_sr1992(1.5, 2.5)
    expected = np.sqrt((2.5 + 2) / 3) * 1.5
    assert np.isclose(result, expected)


def test_zurita_gotor_2002():
    """Test the zurita_gotor_2002 function."""
    result = get_collision_radius_mzg2002(1.5, 1.2)
    expected = 1.037 * (1.2**0.077) * 1.5
    assert np.isclose(result, expected)


def test_thajudeen_2012():
    """Test the thajudeen_2012 function."""
    result = get_collision_radius_tt2012(2.5, 100, 1.5, 0.1)
    alpha1 = 0.253 * 2.5**2 - 1.209 * 2.5 + 1.433
    alpha2 = -0.218 * 2.5**2 + 0.964 * 2.5 - 0.180
    phi = 1 / (alpha1 * np.log(100) + alpha2)
    radius_s_i = phi * 1.5
    radius_s_ii = (0.1 * (1.203 - 0.4315 / 2.5) / 2) * (
        4 * radius_s_i / 0.1
    ) ** (0.8806 + 0.3497 / 2.5)
    expected = radius_s_ii / 2
    assert np.isclose(result, expected)


def test_qian_2022_rg():
    """Test the qian_2022_rg function."""
    result = get_collision_radius_wq2022_rg(1.5, 0.1)
    expected = (0.973 * (1.5 / 0.1) + 0.441) * 0.1
    assert np.isclose(result, expected)


def test_qian_2022_rg_df():
    """Test the qian_2022_rg_df function."""
    result = get_collision_radius_wq2022_rg_df(2.5, 1.5, 0.1)
    expected = (0.882 * (2.5**0.223) * (1.5 / 0.1) + 0.387) * 0.1
    assert np.isclose(result, expected)


def test_qian_2022_rg_df_k0():
    """Test the qian_2022_rg_df_k0 function."""
    result = get_collision_radius_wq2022_rg_df_k0(2.5, 1.2, 1.5, 0.1)
    expected = (
        0.777 * (2.5**0.479) * (1.2**0.000970) * (1.5 / 0.1)
        + 0.267 * 1.2
        + -0.0790
    ) * 0.1
    assert np.isclose(result, expected)


def test_qian_2022_rg_df_k0_a13():
    """Test the qian_2022_rg_df_k0_a13 function."""
    result = get_collision_radius_wq2022_rg_df_k0_a13(2.5, 1.2, 0.5, 1.5, 0.1)
    expected = (
        0.876 * (2.5**0.363) * (1.2**-0.105) * (1.5 / 0.1)
        + 0.421 * 1.2
        + -0.0360 * 0.5
        + -0.227
    ) * 0.1
    assert np.isclose(result, expected)
