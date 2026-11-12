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
        "particula.particles.representation_factories",
    ],
)
def test_removed_concrete_module(module_name):
    """Absence must name the removed module, not an unrelated dependency."""
    assert importlib.util.find_spec(module_name) is None
    with pytest.raises(ModuleNotFoundError) as caught:
        importlib.import_module(module_name)
    assert caught.value.name == module_name


@pytest.mark.parametrize(
    "module_name",
    ["particula", "particula.particles", "particula.particles.particle_data"],
)
@pytest.mark.parametrize("name", ["from_representation", "to_representation"])
def test_removed_particle_bridge(module_name, name):
    """Existing modules import successfully but expose no facade bridges."""
    module = importlib.import_module(module_name)
    assert not hasattr(module, name)
    with pytest.raises(ImportError) as caught:
        exec(f"from {module_name} import {name}", {})  # noqa: S102
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


@pytest.mark.parametrize("module_name", ["particula", "particula.particles"])
def test_removed_particle_factory(module_name):
    """Particle construction no longer exports the obsolete facade factory."""
    module = importlib.import_module(module_name)
    assert not hasattr(module, "ParticleRepresentationFactory")
    with pytest.raises(ImportError) as caught:
        exec(  # noqa: S102
            f"from {module_name} import ParticleRepresentationFactory", {}
        )
    assert caught.value.name == module_name
