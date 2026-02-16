"""Tests for ParticleRepresentation facade behavior."""

import warnings

import numpy as np
import pytest
from particula.particles.activity_strategies import ActivityIdealMass
from particula.particles.distribution_strategies import (
    MassBasedMovingBin,
    ParticleResolvedSpeciatedMass,
    RadiiBasedMovingBin,
    SpeciatedMassMovingBin,
)
from particula.particles.particle_data import ParticleData, to_representation
from particula.particles.representation import (
    _DEPRECATION_MESSAGE,
    ParticleRepresentation,
    _should_warn_deprecation,
    _warn_deprecated,
)
from particula.particles.surface_strategies import SurfaceStrategyVolume


def _make_representation(
    strategy,
    distribution,
    density,
    concentration,
    charge,
    volume=1.0,
) -> ParticleRepresentation:
    return ParticleRepresentation(
        strategy=strategy,
        activity=ActivityIdealMass(),
        surface=SurfaceStrategyVolume(),
        distribution=distribution,
        density=density,
        concentration=concentration,
        charge=charge,
        volume=volume,
    )


def test_init_deprecation_warning():
    """Constructor emits a deprecation warning."""
    with pytest.warns(DeprecationWarning):
        _make_representation(
            strategy=RadiiBasedMovingBin(),
            distribution=np.array([1.0, 2.0]),
            density=np.array([1000.0]),
            concentration=np.array([1.0, 2.0]),
            charge=np.array([0.0, 0.0]),
        )


def test_warn_helpers_follow_filters():
    """Deprecation helpers respect warning filters."""
    assert _should_warn_deprecation()
    with pytest.warns(DeprecationWarning):
        _warn_deprecated(stacklevel=1)

    with warnings.catch_warnings(record=True) as warning_info:
        warnings.filterwarnings("error", category=DeprecationWarning)
        assert not _should_warn_deprecation()
        _warn_deprecated(stacklevel=1)
    assert warning_info == []


def test_init_deprecation_warning_message():
    """Deprecation warning includes migration guidance."""
    with pytest.warns(DeprecationWarning, match="ParticleData") as warning_info:
        _make_representation(
            strategy=RadiiBasedMovingBin(),
            distribution=np.array([1.0, 2.0]),
            density=np.array([1000.0]),
            concentration=np.array([1.0, 2.0]),
            charge=np.array([0.0, 0.0]),
        )
    assert _DEPRECATION_MESSAGE in str(warning_info[0].message)
    assert "migration/particle-data.md" in str(warning_info[0].message)


def test_distribution_setter_deprecation_warning():
    """Distribution setter emits a deprecation warning."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        rep = _make_representation(
            strategy=RadiiBasedMovingBin(),
            distribution=np.array([1.0, 2.0]),
            density=np.array([1000.0]),
            concentration=np.array([1.0, 2.0]),
            charge=np.array([0.0, 0.0]),
        )
    with pytest.warns(DeprecationWarning):
        rep.distribution = np.array([2.0, 3.0])


@pytest.mark.parametrize(
    "method_name",
    ["add_mass", "add_concentration", "collide_pairs"],
)
def test_mutation_deprecation_warnings(method_name):
    """Mutation helpers emit deprecation warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        if method_name == "collide_pairs":
            rep = _make_representation(
                strategy=ParticleResolvedSpeciatedMass(),
                distribution=np.array([[1.0, 2.0]]),
                density=np.array([1000.0, 1200.0]),
                concentration=np.array([1.0]),
                charge=np.array([0.0]),
            )
        else:
            rep = _make_representation(
                strategy=SpeciatedMassMovingBin(),
                distribution=np.array([[1.0, 2.0]]),
                density=np.array([1000.0, 1200.0]),
                concentration=np.array([1.0]),
                charge=np.array([0.0]),
            )
    with pytest.warns(DeprecationWarning):
        if method_name == "add_mass":
            rep.add_mass(np.array([[0.1, 0.2]]))
        elif method_name == "add_concentration":
            rep.add_concentration(
                added_concentration=np.array([1.0]),
                added_distribution=np.array([[1.0, 2.0]]),
                added_charge=np.array([0.0]),
            )
        else:
            rep.collide_pairs(np.array([[0, 0]], dtype=np.int64))


def test_data_property_returns_particle_data():
    """Data property exposes ParticleData."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        rep = _make_representation(
            strategy=RadiiBasedMovingBin(),
            distribution=np.array([1.0, 2.0]),
            density=np.array([1000.0]),
            concentration=np.array([1.0, 2.0]),
            charge=np.array([0.0, 0.0]),
        )
    assert isinstance(rep.data, ParticleData)
    assert rep.data.masses.shape[0] == 1


def test_from_data_no_deprecation_warning():
    """from_data constructs facade without warnings."""
    data = ParticleData(
        masses=np.array([[[1.0], [2.0]]]),
        concentration=np.array([[1.0, 2.0]]),
        charge=np.array([[0.0, 0.0]]),
        density=np.array([1000.0]),
        volume=np.array([1.0]),
    )
    rep = ParticleRepresentation.from_data(
        data,
        strategy=RadiiBasedMovingBin(),
        activity=ActivityIdealMass(),
        surface=SurfaceStrategyVolume(),
        distribution=np.array([1.0, 2.0]),
        charge=np.array([0.0, 0.0]),
    )
    assert rep.data is data
    assert rep.get_charge() is not None


def test_from_data_sets_charge_none():
    """from_data preserves charge=None behavior."""
    data = ParticleData(
        masses=np.array([[[1.0], [2.0]]]),
        concentration=np.array([[1.0, 2.0]]),
        charge=np.array([[0.0, 0.0]]),
        density=np.array([1000.0]),
        volume=np.array([1.0]),
    )
    rep = ParticleRepresentation.from_data(
        data,
        strategy=RadiiBasedMovingBin(),
        activity=ActivityIdealMass(),
        surface=SurfaceStrategyVolume(),
        distribution=np.array([1.0, 2.0]),
        charge=None,
    )
    assert rep.get_charge() is None


def test_facade_delegation_getters():
    """Facade getters align with ParticleData values."""
    data = ParticleData(
        masses=np.array([[[1.0], [2.0]]]),
        concentration=np.array([[2.0, 4.0]]),
        charge=np.array([[0.5, 1.0]]),
        density=np.array([1000.0]),
        volume=np.array([2.0]),
    )
    rep = ParticleRepresentation.from_data(
        data,
        strategy=MassBasedMovingBin(),
        activity=ActivityIdealMass(),
        surface=SurfaceStrategyVolume(),
        distribution=np.array([1.0, 2.0]),
        charge=np.array([0.5, 1.0]),
    )
    np.testing.assert_allclose(rep.get_radius(), data.radii[0])
    np.testing.assert_allclose(rep.get_mass(), data.total_mass[0])
    np.testing.assert_allclose(rep.get_species_mass(), data.masses[0][:, 0])
    np.testing.assert_allclose(
        rep.get_concentration(), data.concentration[0] / data.volume[0]
    )
    charge = rep.get_charge()
    assert charge is not None
    charge_array = np.asarray(charge, dtype=np.float64)
    np.testing.assert_allclose(charge_array, data.charge[0])
    np.testing.assert_allclose(
        rep.get_effective_density(), data.effective_density[0]
    )


def test_facade_concentration_volume_scaling():
    """Concentration scales by volume for facade getter."""
    data = ParticleData(
        masses=np.array([[[1.0], [2.0]]]),
        concentration=np.array([[2.0, 4.0]]),
        charge=np.array([[0.0, 0.0]]),
        density=np.array([1000.0]),
        volume=np.array([0.5]),
    )
    rep = ParticleRepresentation.from_data(
        data,
        strategy=MassBasedMovingBin(),
        activity=ActivityIdealMass(),
        surface=SurfaceStrategyVolume(),
        distribution=np.array([1.0, 2.0]),
        charge=np.array([0.0, 0.0]),
    )
    np.testing.assert_allclose(rep.get_concentration(), np.array([4.0, 8.0]))


def test_add_mass_updates_internal_data():
    """add_mass updates ParticleData masses."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        rep = _make_representation(
            strategy=RadiiBasedMovingBin(),
            distribution=np.array([1.0, 2.0]),
            density=np.array([1000.0]),
            concentration=np.array([1.0, 1.0]),
            charge=np.array([0.0, 0.0]),
        )
        before = rep.data.masses.copy()
    with pytest.warns(DeprecationWarning):
        rep.add_mass(np.array([0.1, 0.2]))
    assert not np.array_equal(before, rep.data.masses)


def test_add_concentration_updates_internal_data():
    """add_concentration updates ParticleData concentration."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        rep = _make_representation(
            strategy=RadiiBasedMovingBin(),
            distribution=np.array([1.0, 2.0]),
            density=np.array([1000.0]),
            concentration=np.array([1.0, 1.0]),
            charge=np.array([0.0, 0.0]),
        )
    with pytest.warns(DeprecationWarning):
        rep.add_concentration(
            added_concentration=np.array([1.0, 1.0]),
            added_distribution=np.array([1.0, 2.0]),
        )
    np.testing.assert_allclose(rep.data.concentration[0], np.array([2.0, 2.0]))


def test_init_shape_mismatch_raises_value_error():
    """Invalid shapes raise ValueError mirroring ParticleData."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        with pytest.raises(ValueError, match="concentration shape"):
            _make_representation(
                strategy=RadiiBasedMovingBin(),
                distribution=np.array([1.0, 2.0]),
                density=np.array([1000.0]),
                concentration=np.array([1.0, 2.0, 3.0]),
                charge=np.array([0.0, 0.0]),
            )


def test_charge_none_preserved():
    """Charge None stays None through mutation."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        rep = _make_representation(
            strategy=ParticleResolvedSpeciatedMass(),
            distribution=np.array([[1.0, 2.0]]),
            density=np.array([1000.0, 1200.0]),
            concentration=np.array([1.0]),
            charge=None,
        )
    with pytest.warns(DeprecationWarning):
        rep.add_concentration(
            added_concentration=np.array([1.0]),
            added_distribution=np.array([[1.0, 2.0]]),
        )
    assert rep.get_charge() is None


def test_from_data_round_trip_with_converter():
    """to_representation returns facade matching source data."""
    data = ParticleData(
        masses=np.array([[[1.0, 2.0]]]),
        concentration=np.array([[1.0]]),
        charge=np.array([[0.0]]),
        density=np.array([1000.0, 1200.0]),
        volume=np.array([1.0]),
    )
    with pytest.warns(DeprecationWarning):
        rep = to_representation(
            data,
            strategy=SpeciatedMassMovingBin(),
            activity=ActivityIdealMass(),
            surface=SurfaceStrategyVolume(),
        )
    np.testing.assert_allclose(rep.data.masses, data.masses)
