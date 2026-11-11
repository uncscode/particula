"""Independent numerical and ownership checks for native initializers."""

import numpy as np
import pytest
from particula.particles import (
    create_lognormal_binned_particle_data,
    create_particle_data_from_radii,
)


@pytest.mark.parametrize("kind", ["discrete", "continuous_pdf"])
def test_radius_mass_density_and_owned_arrays(kind):
    """Radius conversion and product quadrature agree with scalar arithmetic."""
    radii = np.array([1.0, 2.0, 4.0])
    weights = np.array([1.0, 2.0, 1.0])
    charge = np.array([0.0, -1.0, 2.0])
    data = create_particle_data_from_radii(
        radii=radii,
        concentration=weights,
        distribution_type=kind,
        density=1.0,
        density_units="g/cm^3",
        radius_units="nm",
        charge=charge,
    )
    masses = 4 / 3 * np.pi * 1000 * np.array([1e-27, 8e-27, 64e-27])
    np.testing.assert_allclose(
        data.masses[0, :, 0], masses, rtol=1e-12, atol=1e-40
    )
    if kind == "discrete":
        number = 4.0
        mass = masses[0] + 2 * masses[1] + masses[2]
    else:
        number = 4.5e-9
        mass = (masses[0] + 2 * masses[1]) * 0.5e-9 + (
            2 * masses[1] + masses[2]
        ) * 1e-9
    np.testing.assert_allclose(
        data.number_density, [number], rtol=1e-12, atol=0
    )
    np.testing.assert_allclose(
        data.species_mass_density, [[mass]], rtol=1e-12, atol=1e-40
    )
    for value in (data.masses, data.concentration, data.charge, data.density):
        for source in (radii, weights, charge):
            assert not np.shares_memory(value, source)
    radii[:] = 9
    weights[:] = 9
    charge[:] = 9
    np.testing.assert_array_equal(data.concentration, [[1.0, 2.0, 1.0]])
    np.testing.assert_array_equal(data.charge, [[0.0, -1.0, 2.0]])


@pytest.mark.parametrize("kind", ["discrete", "continuous_pdf"])
def test_lognormal_defaults_number_and_mass(kind):
    """Both presets preserve total density using their distinct quadrature."""
    data = create_lognormal_binned_particle_data(distribution_type=kind)
    assert data.masses.shape == (1, 250, 1)
    np.testing.assert_allclose(
        data.number_density, [1.1e10], rtol=1e-12, atol=0
    )
    np.testing.assert_array_equal(data.volume, [1.0])
    np.testing.assert_array_equal(data.charge, np.zeros((1, 250)))
    radius = np.logspace(-9, -4, 250)
    mass = 4 / 3 * np.pi * radius**3 * 1000
    density = data.concentration[0]
    expected = (
        sum(mass * density)
        if kind == "discrete"
        else sum(
            (mass[:-1] * density[:-1] + mass[1:] * density[1:])
            * np.diff(radius)
            / 2
        )
    )
    np.testing.assert_allclose(
        data.species_mass_density, [[expected]], rtol=1e-12, atol=1e-30
    )


@pytest.mark.parametrize("kind", ["discrete", "continuous_pdf"])
@pytest.mark.parametrize("volume", [0.25, 4.0])
def test_nonunit_binned_volume_rejects(kind, volume):
    """Non-unit-volume binned examples are invalid under delivered M1."""
    with pytest.raises(ValueError, match="exactly 1"):
        create_particle_data_from_radii(
            radii=[1e-9, 2e-9],
            concentration=0,
            distribution_type=kind,
            volume=volume,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"radii": [0.0, 1.0]},
        {"radii": [np.nan, 1.0]},
        {"density": [1000.0, 1200.0]},
        {"density": -1.0},
        {"charge": [1.0, 2.0, 3.0]},
        {"concentration": [-1.0, 1.0]},
        {"distribution_type": "pmf"},
        {"distribution_type": "particle_resolved"},
        {"concentration_units": "count"},
        {"radii": [2.0, 1.0], "distribution_type": "continuous_pdf"},
    ],
)
def test_invalid_radius_inputs_do_not_mutate(kwargs):
    """Wrong shape, kind or physical values reject without caller mutation."""
    args = dict(
        radii=np.array([1.0, 2.0]),
        concentration=np.array([1.0, 2.0]),
        distribution_type="discrete",
    )
    args.update(kwargs)
    saved = {k: np.array(v, copy=True) for k, v in args.items()}
    with pytest.raises(ValueError):
        create_particle_data_from_radii(**args)
    for k, v in saved.items():
        np.testing.assert_equal(np.asarray(args[k]), v)


def test_scalar_singleton_units_and_empty_discrete():
    """Singleton forms preserve values and explicit unit conversions."""
    data = create_particle_data_from_radii(
        radii=[[100.0, 200.0]],
        radius_units="nm",
        concentration=[[1.0, 2.0]],
        concentration_units="1/cm^3",
        distribution_type="discrete",
        density=[[1000.0, 1000.0]],
        volume=1000.0,
        volume_units="L",
    )
    np.testing.assert_allclose(
        data.concentration, [[1e6, 2e6]], rtol=1e-12, atol=0
    )
    empty = create_particle_data_from_radii(
        radii=[], concentration=[], distribution_type="discrete"
    )
    assert empty.masses.shape == (1, 0, 1)
    np.testing.assert_array_equal(empty.number_density, [0.0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"geometric_standard_deviation": [1.0, 1.4]},
        {"mode": [1.0]},
        {"number_concentration": [-1.0, 1.0]},
        {"radius_bins": [2.0, 1.0]},
        {"radius_bins": [1.0]},
        {"mode": [np.inf, 1.0]},
    ],
)
def test_invalid_lognormal_inputs(kwargs):
    """Invalid distributions fail explicitly instead of returning silent zeros."""
    with pytest.raises(ValueError):
        create_lognormal_binned_particle_data(
            distribution_type="discrete", **kwargs
        )


@pytest.mark.parametrize("kind", ["discrete", "continuous_pdf"])
def test_lognormal_explicit_unit_equivalence(kind):
    """Node, mode and modal number conversions retain the same population."""
    nodes = np.geomspace(1.0, 1000.0, 40)
    converted = create_lognormal_binned_particle_data(
        distribution_type=kind,
        radius_bins=nodes,
        radius_units="nm",
        mode=[100.0, 200.0],
        geometric_standard_deviation=[1.2, 1.4],
        number_concentration=[10.0, 20.0],
        number_concentration_units="1/cm^3",
        density=1.0,
        density_units="g/cm^3",
    )
    si = create_lognormal_binned_particle_data(
        distribution_type=kind,
        radius_bins=nodes * 1e-9,
        mode=np.array([100.0, 200.0]) * 1e-9,
        geometric_standard_deviation=[1.2, 1.4],
        number_concentration=[1e7, 2e7],
    )
    np.testing.assert_allclose(
        converted.masses, si.masses, rtol=1e-12, atol=1e-30
    )
    np.testing.assert_allclose(
        converted.concentration, si.concentration, rtol=1e-12, atol=0
    )
    assert not np.shares_memory(converted.concentration, si.concentration)


def test_zero_pdf_retains_valid_grid():
    """A zero density still has explicit coordinates and zero inventories."""
    data = create_particle_data_from_radii(
        radii=[1e-9, 2e-9, 4e-9],
        concentration=0,
        distribution_type="continuous_pdf",
    )
    np.testing.assert_array_equal(data.number_density, [0.0])
    np.testing.assert_array_equal(data.species_mass_density, [[0.0]])
    np.testing.assert_array_equal(data.radius_grid, [1e-9, 2e-9, 4e-9])


def test_si_volume_rule_stays_exact():
    """Unit-conversion roundoff handling does not relax direct SI admission."""
    with pytest.raises(ValueError, match="exactly 1"):
        create_particle_data_from_radii(
            radii=[1e-9],
            concentration=1,
            distribution_type="discrete",
            volume=np.nextafter(1.0, 2.0),
        )
