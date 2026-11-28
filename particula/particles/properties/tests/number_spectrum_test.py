"""Independent boundary, inventory and measure checks on explicit diameter grids."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from particula.particles import (
    ParticleData,
    get_bin_number_density,
    get_binned_number_spectrum,
    get_number_per_decade,
    get_particle_number_spectrum,
    get_rebinned_number_spectrum,
)


def test_issue_example_and_neighbors():
    """Check exact endpoints and their immediate floating-point neighbors."""
    edges = np.array([1, 2, 4]) * 1e-6
    sizes = np.array([0.5, 1, 2, 4, 5]) * 1e-6
    bins, low, high = get_binned_number_spectrum(sizes, [1, 2, 3, 5, 7], edges)
    assert_array_equal(bins, [2, 8])
    assert (low, high) == (1, 7)
    sizes = np.stack(
        [np.nextafter(edges, 0), edges, np.nextafter(edges, np.inf)]
    )
    weights = np.arange(9).reshape(3, 3) + 1
    result = get_binned_number_spectrum(sizes, weights, edges)
    assert_array_equal(result.bin_number_density, [[2, 3], [4, 11], [7, 8]])
    assert_array_equal(result.below_range, [1, 0, 0])
    assert_array_equal(result.above_range, [0, 0, 9])


def test_arbitrary_axes_and_independent_bins():
    """Check every row against a direct sum with disparate weight scales."""
    d = np.array([12, 1, 8, 2, 1, 0, 0.1])
    w = np.array([1, 1e22, 3, 2, 1e22, 0, 4])
    shape = (2, 3, 7)
    result = get_binned_number_spectrum(
        np.broadcast_to(d, shape), np.broadcast_to(w, shape), [1, 2, 8]
    )
    for j, mask in enumerate([(1 <= d) & (d < 2), (2 <= d) & (d <= 8)]):
        assert_array_equal(
            result.bin_number_density[..., j], np.full(shape[:2], w[mask].sum())
        )
    assert_array_equal(result.below_range, np.full(shape[:2], 4))
    assert_array_equal(result.above_range, np.ones(shape[:2]))
    assert_array_equal(d, [12, 1, 8, 2, 1, 0, 0.1])


@pytest.mark.parametrize("shape", [(0,), (2, 0), (0, 3)])
def test_empty_rows(shape):
    """Keep the bin and batch dimensions when a slot or batch axis is empty."""
    result = get_binned_number_spectrum(
        np.zeros(shape), np.zeros(shape), [1, 2]
    )
    assert_array_equal(result.bin_number_density, np.zeros((*shape[:-1], 1)))
    assert_array_equal(result.below_range, np.zeros(shape[:-1]))
    assert_array_equal(result.above_range, np.zeros(shape[:-1]))


def test_width_conversion_and_close_edges():
    """Test unequal log widths and a pair of adjacent float64 boundaries."""
    edges = np.array([1, 2, 8]) * 1e-6
    density = np.array([[3, 3], [0, 5]])
    integrated = np.log10(2) * np.array([[3, 6], [0, 10]])
    assert_allclose(
        get_bin_number_density(density, edges), integrated, rtol=1e-14
    )
    assert_allclose(
        get_number_per_decade(integrated, edges), density, rtol=1e-14
    )
    a = 1e-9
    b = np.nextafter(a, np.inf)
    assert get_bin_number_density([1], [a, b])[0] > 0


@pytest.mark.parametrize(
    "edges", [[], [1], [0, 1], [1, 1], [3, 2], [1, np.inf], [[1, 2]], [1, 2j]]
)
def test_invalid_edges(edges):
    """Reject grids without positive, finite, strictly increasing edges."""
    with pytest.raises(ValueError):
        get_binned_number_spectrum([1], [1], edges)


@pytest.mark.parametrize(
    ("sizes", "weights"),
    [
        ([0], [1]),
        ([-1], [0]),
        ([np.nan], [0]),
        ([1], [np.inf]),
        ([1], [-1]),
        ([1], [2j]),
        ([1, 2], [1]),
        (1, 1),
    ],
)
def test_invalid_slots(sizes, weights):
    """Reject inconsistent shapes and unphysical point weights or diameters."""
    with pytest.raises(ValueError):
        get_binned_number_spectrum(sizes, weights, [1, 2])


def test_population_measures():
    """Check native discrete and resolved inventories and invalid slot states."""

    def population(volumes, kind):
        radius = np.array([0.75e-6, 1.5e-6, 0])
        material_density = np.array([1000.0, 1500.0])
        mass = 4 * np.pi / 3 * radius[:, None] ** 3 * material_density / 2
        counts = np.array([2.0, 8.0, 0.0]) * np.asarray(volumes)[:, None]
        return ParticleData(
            np.tile(mass, (len(volumes), 1, 1)),
            counts,
            np.zeros_like(counts),
            material_density,
            np.array(volumes),
            distribution_type=kind,
        )

    for p in (
        population([1.0], "discrete"),
        population([0.5, 2.0], "particle_resolved"),
    ):
        result = get_particle_number_spectrum(p, np.array([1, 2, 4]) * 1e-6)
        assert_array_equal(
            result.bin_number_density, np.tile([2, 8], (p.n_boxes, 1))
        )
        assert_array_equal(
            result.bin_number_density.sum(axis=-1)
            + result.below_range
            + result.above_range,
            p.number_density,
        )
    p.masses[0, -1] = 1
    with pytest.raises(ValueError, match="slot"):
        get_particle_number_spectrum(p, [1, 2])
    p.distribution_type = "discrete"
    p.volume[:] = 1
    get_particle_number_spectrum(p, [1, 2])
    for kind in (None, "continuous_pdf"):
        p.distribution_type = kind
        with pytest.raises(ValueError, match="tagged"):
            get_particle_number_spectrum(p, [1, 2])


@pytest.mark.parametrize(
    "target",
    [[1, 2, 8, 16], [2, 4, 8], [0.25, 0.5], [20, 30], [0.25, 32], [1.5, 3, 12]],
)
def test_rebin_against_cumulative_number(target):
    """Compare log-overlap redistribution with an independent cumulative sum."""
    edges = [1, 2, 8, 16]
    counts = np.array([[1.0, 8.0, 3.0], [2.0, 0.0, 4.0]])
    bins, low, high = get_rebinned_number_spectrum(counts, edges, target)
    for i, row in enumerate(counts):
        cdf = np.interp(
            np.log10(target), np.log10(edges), np.r_[0, np.cumsum(row)]
        )
        assert_allclose(bins[i], np.diff(cdf), rtol=1e-14)
        assert_allclose(
            [low[i], high[i]], [cdf[0], row.sum() - cdf[-1]], rtol=1e-14
        )
    assert_allclose(
        bins.sum(axis=-1) + low + high, counts.sum(axis=-1), rtol=1e-14
    )


def test_overflow_channel_and_single_bin():
    """Assign exact edge values according to the overflow channel convention."""
    d = [0.5, 1, 2, 3]
    for include, expected in ((True, [2, 8, 1]), (False, [2, 3, 6])):
        bins, low, high = get_binned_number_spectrum(
            d, [2, 3, 5, 1], [1, 2], include_upper_edge=include
        )
        assert_array_equal([low, bins[0], high], expected)


@pytest.mark.parametrize(
    "func", [get_bin_number_density, get_number_per_decade]
)
@pytest.mark.parametrize("values", [1, [2, 3], [-2], [np.nan], [1j]])
def test_invalid_density_conversion(func, values):
    """Reject invalid bin measures in both conversion directions."""
    with pytest.raises(ValueError):
        func(values, [1, 2])
