"""Independent count, density and sampling contracts for native initialization."""

import numpy as np
import particula.util.convert_units as convert_units
import pytest
from particula.particles import (
    create_lognormal_resolved_particle_data,
    create_resolved_particle_data,
)
from particula.particles import particle_data_initialization as initialization


@pytest.mark.parametrize("volume", [0.25, 1.0, 4.0])
@pytest.mark.parametrize("units", ["count", "1/m^3", "1/cm^3"])
def test_counts_and_density_inventory(monkeypatch, volume, units):
    """Fractional weights preserve each species and normalize exactly once."""
    if units == "1/cm^3":
        pytest.importorskip("pint")
    else:
        monkeypatch.setattr(convert_units, "unit_registry", None)
    masses = np.array([[0.5e-18, 1.5e-18], [2e-18, 3e-18], [0.0, 0.0]])
    weights = np.array([8.0, 0.5, 0.0])
    data = create_resolved_particle_data(
        masses=masses,
        density=[1000.0, 1200.0],
        concentration=weights,
        concentration_units=units,
        volume=volume,
    )
    factor = (
        1.0
        if units == "count"
        else volume * (1e6 if units == "1/cm^3" else 1.0)
    )
    np.testing.assert_allclose(
        data.concentration, [weights * factor], rtol=1e-12, atol=0
    )
    np.testing.assert_allclose(
        data.number_density, [8.5 * factor / volume], rtol=1e-12, atol=0
    )
    np.testing.assert_allclose(
        data.species_mass_inventory,
        [[5e-18 * factor, 13.5e-18 * factor]],
        rtol=1e-12,
        atol=1e-30,
    )
    np.testing.assert_allclose(
        data.species_mass_density,
        [[5e-18 * factor / volume, 13.5e-18 * factor / volume]],
        rtol=1e-12,
        atol=1e-30,
    )
    assert not np.shares_memory(data.masses, masses)
    assert not np.shares_memory(data.concentration, weights)


def test_multibox_units_and_owned_broadcasts():
    """Each box uses its own volume while density stays per species."""
    pytest.importorskip("pint")
    masses = np.ones((2, 3, 2)) * 1e-9
    data = create_resolved_particle_data(
        masses=masses,
        mass_units="ng",
        density=[[1.0, 1.2]],
        density_units="g/cm^3",
        concentration=[2.0, 3.0, 0.0],
        concentration_units="1/m^3",
        charge=[[1.0], [-1.0], [0.0]],
        volume=[250.0, 4000.0],
        volume_units="L",
    )
    np.testing.assert_allclose(
        data.masses, np.full((2, 3, 2), 1e-21), rtol=1e-12, atol=1e-33
    )
    np.testing.assert_allclose(
        data.concentration,
        [[0.5, 0.75, 0.0], [8.0, 12.0, 0.0]],
        rtol=1e-12,
        atol=0,
    )
    np.testing.assert_array_equal(
        data.charge, [[1.0, -1.0, 0.0], [1.0, -1.0, 0.0]]
    )
    data.charge[0, 0] = 5
    assert (
        data.charge[1, 0] == 1
    )  # Broadcast results are independently writable.


@pytest.mark.parametrize("masses", [2e-18, [2e-18], [[2e-18]], [[[2e-18]]]])
def test_singleton_mass_forms(masses):
    """Scalar and explicit singleton ranks all retain a single species/slot."""
    data = create_resolved_particle_data(masses=masses, density=1000.0)
    assert data.masses.shape == (1, 1, 1)
    np.testing.assert_array_equal(data.concentration, [[1.0]])


def test_redundant_density_charge_and_empty_population():
    """Redundant matrices are accepted only when reduction is lossless."""
    data = create_resolved_particle_data(
        masses=np.ones((3, 2)) * 1e-18,
        density=[[1000.0, 1200.0]] * 3,
        charge=[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
    )
    np.testing.assert_array_equal(data.density, [1000.0, 1200.0])
    np.testing.assert_array_equal(data.charge, [[1.0, 2.0, 3.0]])
    empty = create_resolved_particle_data(
        masses=np.zeros((0, 2)), density=[1000.0, 1200.0], volume=0.25
    )
    assert empty.masses.shape == (1, 0, 2)
    np.testing.assert_array_equal(empty.species_mass_inventory, [[0.0, 0.0]])


@pytest.mark.parametrize(
    ("charge", "expected"),
    [
        (None, [0, 0, 0]),
        (np.zeros((3, 1)), [0, 0, 0]),
        (np.array([[1, 2, 3]]), [1, 2, 3]),
        (np.array([[1], [2], [3]]), [1, 2, 3]),
    ],
)
def test_native_builder_successor_charge_shapes(charge, expected):
    """Lossless singleton charge forms preserve every particle lane."""
    data = create_resolved_particle_data(
        masses=np.array([[1.0], [2.0], [3.0]]),
        density=np.full((3, 1), 1000.0),
        charge=charge,
        volume=1.0,
    )
    np.testing.assert_array_equal(data.density, [1000.0])
    np.testing.assert_array_equal(data.charge, [expected])
    np.testing.assert_array_equal(data.concentration, [[1.0, 1.0, 1.0]])


def test_native_empty_population_with_array_charge():
    """Empty resolved slots retain empty charge and explicit material density."""
    data = create_resolved_particle_data(
        masses=np.zeros((0, 1)),
        density=[1000.0],
        charge=np.zeros((0, 1)),
        volume=1.0,
    )
    assert data.masses.shape == (1, 0, 1)
    assert data.total_mass.size == 0
    assert data.charge.shape == (1, 0)
    np.testing.assert_array_equal(data.density, [1000.0])


def test_empty_population_requires_material_density():
    """An empty population cannot establish positive species density metadata."""
    masses = np.zeros((0, 2))
    density = np.zeros((0, 2))
    with pytest.raises(ValueError, match="density"):
        create_resolved_particle_data(masses=masses, density=density)
    assert masses.shape == (0, 2)
    assert density.shape == (0, 2)


@pytest.mark.parametrize("density", [1000.0, [1000.0], [900.0, 1000.0]])
@pytest.mark.parametrize("charge", [2.0, [1.0, 2.0, 3.0]])
def test_native_density_forms_and_empty_charge_rejection(density, charge):
    """Normalize explicit material/charge lanes without inventing missing data."""
    expected_density = np.atleast_1d(density)
    masses = np.ones((3, expected_density.size))
    data = create_resolved_particle_data(
        masses=masses,
        density=density,
        charge=charge,
    )
    np.testing.assert_array_equal(data.density, expected_density)
    np.testing.assert_array_equal(data.charge[0], np.broadcast_to(charge, (3,)))
    with pytest.raises(ValueError, match="charge"):
        create_resolved_particle_data(
            masses=masses,
            density=density,
            charge=np.array([]),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"density": [[1000.0, 1200.0], [900.0, 1200.0], [1000.0, 1200.0]]},
        {"charge": [[1.0, 2.0], [3.0, 3.0], [4.0, 4.0]]},
        {"density": [1000.0, 1200.0, 1300.0]},
        {"charge": [1.0, 2.0]},
        {"concentration": [-1.0, 1.0, 0.0]},
        {"concentration_units": "1/m^4"},
        {"concentration_units": "1/m^3"},
        {"volume": 0.0},
        {"volume": np.inf},
        {"masses": np.ones((0, 3, 2))},
        {"masses": np.ones((1, 3, 0))},
    ],
)
def test_invalid_inputs_are_read_only(kwargs):
    """No invalid shape is repaired by dropping particle/species values."""
    args = dict(
        masses=np.ones((3, 2)) * 1e-18, density=np.array([1000.0, 1200.0])
    )
    args.update(kwargs)
    saved = {key: np.array(value, copy=True) for key, value in args.items()}
    with pytest.raises(ValueError):
        create_resolved_particle_data(**args)
    for key, value in saved.items():
        np.testing.assert_equal(np.asarray(args[key]), value)


@pytest.mark.parametrize("volume", [0.25, 1.0, 4.0])
def test_controlled_samples_have_unit_counts(monkeypatch, volume):
    """Modal densities determine mixture selection, never slot weight scaling."""
    monkeypatch.setattr(convert_units, "unit_registry", None)
    radii = np.array([1e-7, 2e-7, 3e-7])

    def sample(**kwargs):
        assert kwargs["number_of_samples"] == 3
        np.testing.assert_array_equal(
            kwargs["number_of_particles"], [1e10, 1e9]
        )
        return radii

    monkeypatch.setattr(
        initialization, "get_lognormal_sample_distribution", sample
    )
    data = create_lognormal_resolved_particle_data(
        particle_resolved_count=3, volume=volume
    )
    expected = 4 / 3 * np.pi * 1000 * np.array([1e-21, 8e-21, 27e-21])
    np.testing.assert_allclose(
        data.masses[0, :, 0], expected, rtol=1e-12, atol=1e-30
    )
    np.testing.assert_array_equal(data.concentration, [[1.0, 1.0, 1.0]])
    np.testing.assert_allclose(
        data.number_density, [3 / volume], rtol=1e-12, atol=0
    )
    assert not np.shares_memory(data.masses, radii)


def test_sample_defaults_and_existing_random_state():
    """Use exactly the existing sampler without installing a new RNG policy."""
    saved = np.random.get_state()
    try:
        np.random.seed(123)
        expected = initialization.get_lognormal_sample_distribution(
            mode=np.array([100e-9, 1e-6]),
            geometric_standard_deviation=np.array([1.2, 1.4]),
            number_of_particles=np.array([1e10, 1e9]),
            number_of_samples=20_000,
        )
        np.random.seed(123)
        data = create_lognormal_resolved_particle_data()
        np.testing.assert_allclose(
            data.masses[0, :, 0],
            4 / 3 * np.pi * expected**3 * 1000,
            rtol=1e-12,
            atol=1e-30,
        )
        assert data.masses.shape == (1, 20_000, 1)
        np.testing.assert_array_equal(data.number_density, [20_000.0])
    finally:
        np.random.set_state(saved)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"particle_resolved_count": 0},
        {"particle_resolved_count": True},
        {"particle_resolved_count": 2.5},
        {"volume": 0.0},
        {"charge": [1.0, 2.0]},
        {"density": [1000.0, 1200.0]},
    ],
)
def test_sample_rejection_precedes_rng(monkeypatch, kwargs):
    """Invalid deterministic inputs never consume random samples."""

    def fail(**kwargs):
        raise AssertionError("sampler must not run")

    monkeypatch.setattr(
        initialization, "get_lognormal_sample_distribution", fail
    )
    with pytest.raises(ValueError):
        create_lognormal_resolved_particle_data(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mass_units": "ng"},
        {"density_units": "g/cm^3"},
        {"volume_units": "L"},
        {"concentration_units": "1/cm^3"},
    ],
)
def test_non_si_resolved_requires_pint(monkeypatch, kwargs):
    """Optional unit conversions fail clearly without mutating caller data."""
    monkeypatch.setattr(convert_units, "unit_registry", None)
    masses = np.array([1e-18, 2e-18])
    weights = np.array([1.0, 0.5])
    with pytest.raises(ImportError, match="Install pint"):
        create_resolved_particle_data(
            masses=masses, density=1000.0, concentration=weights, **kwargs
        )
    np.testing.assert_array_equal(masses, [1e-18, 2e-18])
    np.testing.assert_array_equal(weights, [1.0, 0.5])


def test_sample_missing_conversion_dependency_precedes_rng(monkeypatch):
    """Non-SI sample inputs require Pint before any random state is consumed."""
    monkeypatch.setattr(convert_units, "unit_registry", None)

    def fail(**kwargs):
        raise AssertionError("sampler must not run")

    monkeypatch.setattr(
        initialization, "get_lognormal_sample_distribution", fail
    )
    with pytest.raises(ImportError, match="Install pint"):
        create_lognormal_resolved_particle_data(
            particle_resolved_count=3, volume=1000.0, volume_units="L"
        )


@pytest.mark.parametrize("guard_sampler", [False, True])
def test_overflowing_modal_total_rejects_before_sampling(
    monkeypatch, guard_sampler
):
    """Finite individual weights cannot overflow their total after RNG use."""
    if guard_sampler:

        def fail(**kwargs):
            raise AssertionError("sampler must not run")

        monkeypatch.setattr(
            initialization, "get_lognormal_sample_distribution", fail
        )
    numbers = np.array([1e308, 1e308])
    before = np.random.get_state()
    try:
        with pytest.raises(
            ValueError, match="total modal number concentration"
        ):
            create_lognormal_resolved_particle_data(
                particle_resolved_count=4, number_concentration=numbers
            )
        after = np.random.get_state()
        assert before[0] == after[0]
        np.testing.assert_array_equal(before[1], after[1])
        assert before[2:] == after[2:]
        np.testing.assert_array_equal(numbers, [1e308, 1e308])
    finally:
        np.random.set_state(before)


def test_large_finite_modal_total_remains_supported():
    """Large valid mixture weights still produce the requested unit counts."""
    before = np.random.get_state()
    try:
        np.random.seed(123)
        data = create_lognormal_resolved_particle_data(
            particle_resolved_count=4, number_concentration=[4e307, 4e307]
        )
        assert data.masses.shape == (1, 4, 1)
        np.testing.assert_array_equal(data.concentration, [[1.0] * 4])
        assert np.all(np.isfinite(data.masses))
    finally:
        np.random.set_state(before)
