import numpy as np
from particula.particles.representation import get_sorted_bins_by_radius

def test_sorted_vector():
    radius = np.array([1., 2., 3.])
    dist   = np.array([10., 20., 30.])
    conc   = np.array([100, 200, 300])
    charge = np.array([1, 2, 3])

    d2, c2, q2 = get_sorted_bins_by_radius(radius, dist, conc, charge)

    np.testing.assert_array_equal(d2, dist)
    np.testing.assert_array_equal(c2, conc)
    np.testing.assert_array_equal(q2, charge)

def test_unsorted_vector():
    radius = np.array([2., 1., 3.])
    dist   = np.array([20., 10., 30.])
    conc   = np.array([200, 100, 300])
    charge = np.array([2, 1, 3])

    d2, c2, q2 = get_sorted_bins_by_radius(radius, dist, conc, charge)

    np.testing.assert_array_equal(d2,  np.array([10., 20., 30.]))
    np.testing.assert_array_equal(c2,  np.array([100, 200, 300]))
    np.testing.assert_array_equal(q2,  np.array([1, 2, 3]))

def test_unsorted_matrix():
    radius = np.array([2., 1., 3.])
    dist   = np.array([[20., 21.],
                       [10., 11.],
                       [30., 31.]])
    conc   = np.array([200, 100, 300])
    charge = np.array([2, 1, 3])

    d2, c2, q2 = get_sorted_bins_by_radius(radius, dist, conc, charge)

    np.testing.assert_array_equal(d2,  np.array([[10., 11.],
                                                 [20., 21.],
                                                 [30., 31.]]))
    np.testing.assert_array_equal(c2,  np.array([100, 200, 300]))
    np.testing.assert_array_equal(q2,  np.array([1, 2, 3]))
