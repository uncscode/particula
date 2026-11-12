"""Removed concrete modules stay absent while native APIs remain usable."""

import importlib
import importlib.util

import pytest
from particula import Aerosol, AerosolBuilder
from particula.tests.aerosol_test import native_triple


@pytest.mark.parametrize(
    "module_name",
    [
        "particula._legacy_aerosol",
        "particula.particles.change_particle_representation",
    ],
)
def test_removed_concrete_module(module_name):
    """Absence must name the removed module, not an unrelated dependency."""
    assert importlib.util.find_spec(module_name) is None
    with pytest.raises(ModuleNotFoundError) as caught:
        importlib.import_module(module_name)
    assert caught.value.name == module_name


def test_native_aggregate_survives_legacy_removal():
    """Public construction and builder retain the three native identities."""
    particles, gas, environment = native_triple()
    direct = Aerosol(particles=particles, gas=gas, environment=environment)
    built = (
        AerosolBuilder()
        .set_particles(particles)
        .set_gas(gas)
        .set_environment(environment)
        .build()
    )
    for aerosol in (direct, built):
        assert aerosol.particles is particles
        assert aerosol.gas is gas
        assert aerosol.environment is environment
        assert not hasattr(aerosol, "atmosphere")
