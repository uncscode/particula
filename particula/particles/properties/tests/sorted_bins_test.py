"""Unit tests for particula.particles.representation.get_sorted_bins_by_radius.

The tests cover:
• Already-sorted 1-D inputs (function must return inputs unchanged).
• Unsorted 1-D inputs (function must reorder accompanying arrays).
• Unsorted inputs with a 2-D distribution matrix (rows must be reordered
  consistently with the radii).
• Unsorted inputs with a 2-D distribution matrix and an all-zero charge array.
"""

import numpy as np

from particula.particles.representation import get_sorted_bins_by_radius


def test_sorted_vector():
    """Already-sorted 1-D input should remain unchanged."""
    radius = np.array([1.0, 2.0, 3.0])
    dist = np.array([10.0, 20.0, 30.0])
    conc = np.array([100, 200, 300])
    charge = np.array([1, 2, 3])

    d2, c2, q2 = get_sorted_bins_by_radius(radius, dist, conc, charge)

    np.testing.assert_array_equal(d2, dist)
    np.testing.assert_array_equal(c2, conc)
    np.testing.assert_array_equal(q2, charge)


def test_unsorted_vector():
    """Unsorted 1-D input must be reordered into ascending radii."""
    radius = np.array([2.0, 1.0, 3.0])
    dist = np.array([20.0, 10.0, 30.0])
    conc = np.array([200, 100, 300])
    charge = np.array([2, 1, 3])

    d2, c2, q2 = get_sorted_bins_by_radius(radius, dist, conc, charge)

    np.testing.assert_array_equal(d2, np.array([10.0, 20.0, 30.0]))
    np.testing.assert_array_equal(c2, np.array([100, 200, 300]))
    np.testing.assert_array_equal(q2, np.array([1, 2, 3]))


def test_unsorted_matrix():
    """Unsorted radii with 2-D distribution; rows must be reordered."""
    radius = np.array([2.0, 1.0, 3.0])
    dist = np.array([[20.0, 21.0], [10.0, 11.0], [30.0, 31.0]])
    conc = np.array([200, 100, 300])
    charge = np.array([2, 1, 3])

    d2, c2, q2 = get_sorted_bins_by_radius(radius, dist, conc, charge)

    np.testing.assert_array_equal(
        d2, np.array([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]])
    )
    np.testing.assert_array_equal(c2, np.array([100, 200, 300]))
    np.testing.assert_array_equal(q2, np.array([1, 2, 3]))


def test_unsorted_matrix_zero_charge():
    """Unsorted radii with 2-D distribution and zero-charge array."""
    radius = np.array([3.0, 1.0, 2.0])
    dist = np.array([[30.0, 31.0], [10.0, 11.0], [20.0, 21.0]])
    conc = np.array([300, 100, 200])
    charge = np.array([0])  # all zeros

    d2, c2, q2 = get_sorted_bins_by_radius(radius, dist, conc, charge)

    np.testing.assert_array_equal(
        d2, np.array([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]])
    )
    np.testing.assert_array_equal(c2, np.array([100, 200, 300]))
    np.testing.assert_array_equal(q2, charge)  # still all zeros
