"""Tests for ParticleRepresentation facade behavior."""

import logging

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


@pytest.fixture()
def _enable_particula_log_capture(caplog):
    """Allow caplog to capture the 'particula' logger (propagate=False)."""
    particula_logger = logging.getLogger("particula")
    old_propagate = particula_logger.propagate
    particula_logger.propagate = True
    caplog.set_level(logging.INFO, logger="particula")
    yield caplog
    particula_logger.propagate = old_propagate


def test_init_deprecation_log(_enable_particula_log_capture):
    """Constructor logs a deprecation info message."""
    caplog = _enable_particula_log_capture
    _make_representation(
        strategy=RadiiBasedMovingBin(),
        distribution=np.array([1.0, 2.0]),
        density=np.array([1000.0]),
        concentration=np.array([1.0, 2.0]),
        charge=np.array([0.0, 0.0]),
    )
    assert _DEPRECATION_MESSAGE in caplog.text


def test_warn_deprecated_logs_message(_enable_particula_log_capture):
    """_warn_deprecated emits an info log message."""
    caplog = _enable_particula_log_capture
    _warn_deprecated(stacklevel=1)
    assert _DEPRECATION_MESSAGE in caplog.text


def test_init_deprecation_log_message(_enable_particula_log_capture):
    """Deprecation log includes migration guidance."""
    caplog = _enable_particula_log_capture
    _make_representation(
        strategy=RadiiBasedMovingBin(),
        distribution=np.array([1.0, 2.0]),
        density=np.array([1000.0]),
        concentration=np.array([1.0, 2.0]),
        charge=np.array([0.0, 0.0]),
    )
    assert "ParticleData" in caplog.text
    assert "migration/particle-data.md" in caplog.text


def test_distribution_setter_deprecation_log(
    _enable_particula_log_capture,
):
    """Distribution setter logs a deprecation message."""
    caplog = _enable_particula_log_capture
    rep = _make_representation(
        strategy=RadiiBasedMovingBin(),
        distribution=np.array([1.0, 2.0]),
        density=np.array([1000.0]),
        concentration=np.array([1.0, 2.0]),
        charge=np.array([0.0, 0.0]),
    )
    caplog.clear()
    rep.distribution = np.array([2.0, 3.0])
    assert _DEPRECATION_MESSAGE in caplog.text


@pytest.mark.parametrize(
    "method_name",
    ["add_mass", "add_concentration", "collide_pairs"],
)
def test_mutation_deprecation_logs(method_name, _enable_particula_log_capture):
    """Mutation helpers log deprecation messages."""
    caplog = _enable_particula_log_capture
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
    caplog.clear()
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
    assert _DEPRECATION_MESSAGE in caplog.text


def test_data_property_returns_particle_data():
    """Data property exposes ParticleData."""
    rep = _make_representation(
        strategy=RadiiBasedMovingBin(),
        distribution=np.array([1.0, 2.0]),
        density=np.array([1000.0]),
        concentration=np.array([1.0, 2.0]),
        charge=np.array([0.0, 0.0]),
    )
    assert isinstance(rep.data, ParticleData)
    assert rep.data.masses.shape[0] == 1


def test_from_data_no_deprecation_log(_enable_particula_log_capture):
    """from_data constructs facade without deprecation log."""
    caplog = _enable_particula_log_capture
    data = ParticleData(
        masses=np.array([[[1.0], [2.0]]]),
        concentration=np.array([[1.0, 2.0]]),
        charge=np.array([[0.0, 0.0]]),
        density=np.array([1000.0]),
        volume=np.array([1.0]),
    )
    caplog.clear()
    rep = ParticleRepresentation.from_data(
        data,
        strategy=RadiiBasedMovingBin(),
        activity=ActivityIdealMass(),
        surface=SurfaceStrategyVolume(),
        distribution=np.array([1.0, 2.0]),
        charge=np.array([0.0, 0.0]),
    )
    assert _DEPRECATION_MESSAGE not in caplog.text
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
        rep.get_concentration(),
        data.concentration[0] / data.volume[0],
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
    rep = _make_representation(
        strategy=RadiiBasedMovingBin(),
        distribution=np.array([1.0, 2.0]),
        density=np.array([1000.0]),
        concentration=np.array([1.0, 1.0]),
        charge=np.array([0.0, 0.0]),
    )
    before = rep.data.masses.copy()
    rep.add_mass(np.array([0.1, 0.2]))
    assert not np.array_equal(before, rep.data.masses)


def test_add_concentration_updates_internal_data():
    """add_concentration updates ParticleData concentration."""
    rep = _make_representation(
        strategy=RadiiBasedMovingBin(),
        distribution=np.array([1.0, 2.0]),
        density=np.array([1000.0]),
        concentration=np.array([1.0, 1.0]),
        charge=np.array([0.0, 0.0]),
    )
    rep.add_concentration(
        added_concentration=np.array([1.0, 1.0]),
        added_distribution=np.array([1.0, 2.0]),
    )
    np.testing.assert_allclose(rep.data.concentration[0], np.array([2.0, 2.0]))


def test_init_shape_mismatch_raises_value_error():
    """Invalid shapes raise ValueError mirroring ParticleData."""
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
    rep = _make_representation(
        strategy=ParticleResolvedSpeciatedMass(),
        distribution=np.array([[1.0, 2.0]]),
        density=np.array([1000.0, 1200.0]),
        concentration=np.array([1.0]),
        charge=None,
    )
    rep.add_concentration(
        added_concentration=np.array([1.0]),
        added_distribution=np.array([[1.0, 2.0]]),
    )
    assert rep.get_charge() is None


def test_from_data_round_trip_with_converter(
    _enable_particula_log_capture,
):
    """to_representation returns facade matching source data."""
    caplog = _enable_particula_log_capture
    data = ParticleData(
        masses=np.array([[[1.0, 2.0]]]),
        concentration=np.array([[1.0]]),
        charge=np.array([[0.0]]),
        density=np.array([1000.0, 1200.0]),
        volume=np.array([1.0]),
    )
    rep = to_representation(
        data,
        strategy=SpeciatedMassMovingBin(),
        activity=ActivityIdealMass(),
        surface=SurfaceStrategyVolume(),
    )
    assert _DEPRECATION_MESSAGE in caplog.text
    np.testing.assert_allclose(rep.data.masses, data.masses)
