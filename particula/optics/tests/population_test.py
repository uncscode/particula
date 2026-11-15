"""Population measure, optical inventory and path-length invariants."""

import numpy as np
import pytest
from particula.optics.column import get_column_optics, get_mass_specific_opacity
from particula.optics.cross_sections import OpticalCrossSections
from particula.optics.population import (
    get_population_mueller_matrix,
    get_population_optics,
    get_population_weights,
)


def test_resolved_discrete_equivalence():
    """Unequal counts and physical volumes normalize once."""
    count = np.array([[2.0, 4.0, 6.0], [3.0, 6.0, 9.0]])
    volume = np.array([2.0, 3.0])
    resolved = get_population_weights(count, "particle_resolved", volume=volume)
    discrete = get_population_weights([[1.0, 2.0, 3.0]] * 2, "discrete")
    np.testing.assert_array_equal(resolved, discrete)
    cross = OpticalCrossSections(
        np.array([2.0, 4.0, 8.0]),
        np.array([1.0, 3.0, 6.0]),
        np.array([1.0, 1.0, 2.0]),
        np.array([0.1, 0.2, 0.3]),
        np.array([2.0, 2.5, 2.0]),
    )
    result = get_population_optics(cross, [0.0, 0.5, 1.0], resolved)
    np.testing.assert_allclose(result.extinction, [34.0, 34.0])
    np.testing.assert_allclose(result.scattering, [25.0, 25.0])
    np.testing.assert_allclose(result.absorption, [9.0, 9.0])
    np.testing.assert_allclose(result.asymmetry, [21 / 25, 21 / 25])
    np.testing.assert_allclose(result.single_scattering_albedo, 25 / 34)
    assert np.all(result.extinction_valid & result.scattering_valid)


def test_pdf_product_quadrature():
    """Nonuniform radius quadrature integrates the complete varying product."""
    radius = np.array([1.0, 2.0, 5.0, 9.0]) * 1e-6
    pdf = np.array([[2.0, 3.0, 1.0, 7.0], [4.0, 1.0, 5.0, 2.0]]) * 1e12
    cross = np.array([0.1, 0.8, 2.0, 3.0]) * 1e-12
    weights = get_population_weights(pdf, "continuous_pdf", radius_grid=radius)
    actual = np.sum(weights * cross, axis=-1)
    expected = np.trapezoid(pdf * cross, radius, axis=-1)
    np.testing.assert_allclose(actual, expected, rtol=1e-15)


def test_spectral_and_empty_reductions():
    """Explicit particle axes preserve boxes and wavelength dimensions."""
    area = np.ones((2, 3, 4))
    cross = OpticalCrossSections(area, area, np.zeros_like(area), area, area)
    result = get_population_optics(cross, 0.0, np.arange(4.0)[None, None, :])
    np.testing.assert_array_equal(result.extinction, np.full((2, 3), 6.0))
    empty = np.empty((2, 0))
    result = get_population_optics(
        OpticalCrossSections(*([empty] * 5)), 0.0, empty
    )
    assert np.all(result.extinction == 0)
    assert not np.any(result.scattering_valid | result.extinction_valid)


def test_incoherent_mueller_sum():
    """Signed polarization terms sum by density, with no amplitude coherence."""
    matrix = np.zeros((2, 3, 4, 4))
    matrix[0, :, 0, 0], matrix[1, :, 0, 0] = 2.0, 5.0
    matrix[0, :, 0, 1], matrix[1, :, 0, 1] = -1.0, 2.0
    bulk = get_population_mueller_matrix(matrix, [3.0, 7.0])
    np.testing.assert_array_equal(bulk[:, 0, 0], [41.0] * 3)
    np.testing.assert_array_equal(bulk[:, 0, 1], [11.0] * 3)


def test_column_and_opacity():
    """Optical depths add; direct transmissions multiply along a path."""
    column = get_column_optics([[0.2, 0.3], [0.1, 0.4]], [2.0, 3.0])
    np.testing.assert_allclose(column.optical_depth, [1.3, 1.4])
    np.testing.assert_allclose(
        column.transmission,
        np.prod(np.exp(-column.layer_optical_depth), axis=-1),
    )
    scalar = get_column_optics(0.2, 3.0)
    np.testing.assert_allclose(scalar.optical_depth, 0.6)
    empty = get_column_optics([], [])
    assert empty.transmission == 1.0
    opacity, valid = get_mass_specific_opacity([0.0, 0.2], [0.0, 0.01])
    np.testing.assert_allclose(opacity, [0.0, 20.0])
    np.testing.assert_array_equal(valid, [False, True])


@pytest.mark.parametrize(
    "kind,kwargs",
    [
        ("bad", {}),
        ("discrete", {"volume": 1.0}),
        ("particle_resolved", {}),
        ("particle_resolved", {"volume": 0.0}),
        ("continuous_pdf", {"radius_grid": [2.0, 1.0]}),
        ("continuous_pdf", {"radius_grid": [1.0]}),
    ],
)
def test_invalid_measures(kind, kwargs):
    """Reject ambiguous weighting and invalid physical metadata."""
    with pytest.raises(ValueError):
        get_population_weights([1.0, 2.0], kind, **kwargs)


def test_invalid_geometry_and_opacity():
    """Invalid geometry or absent mass never silently generates opacity."""
    with pytest.raises(ValueError):
        get_column_optics(0.1, -1.0)
    with pytest.raises(ValueError):
        get_column_optics(np.inf, 1.0)
    with pytest.raises(ValueError):
        get_mass_specific_opacity(1.0, 0.0)


def test_saved_pymiescatt_populations():
    """Compare discrete and continuous distribution sums with saved outputs."""
    import json
    from pathlib import Path

    from particula.optics import get_cross_sections, get_mie_efficiencies

    data = json.loads(
        (Path(__file__).parent / "data/pymiescatt_populations.json").read_text()
    )
    for case in data["cases"]:
        radius = np.array(case["diameter_nm"]) * 0.5e-9
        m = complex(*case["index"])
        eff = get_mie_efficiencies(radius, 500e-9, m)
        cross = get_cross_sections(eff, radius)
        if case["SMPS"]:
            weights = get_population_weights(
                np.array(case["number"]) * 1e6, "discrete"
            )
        else:
            # Reference dn/dd is per cm³ per nm diameter; convert to
            # dn/dr per m³ per metre radius (including dd/dr=2).
            weights = get_population_weights(
                np.array(case["number"]) * 2e15,
                "continuous_pdf",
                radius_grid=radius,
            )
        result = get_population_optics(cross, eff.asymmetry, weights)
        raw = case["raw_outputs"]
        for value, index in zip(result[:5], [0, 1, 2, 5, 4], strict=True):
            np.testing.assert_allclose(
                value, raw[index] * 1e-6, rtol=1e-6, atol=1e-12
            )
        np.testing.assert_allclose(result.asymmetry, raw[3], rtol=1e-6)
