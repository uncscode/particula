"""Native aggregate construction and rejection contracts."""

import numpy as np
import pytest
from particula import Aerosol
from particula.gas import EnvironmentData, GasData
from particula.particles import ParticleData


def native_triple(boxes=1):
    """Return distinguishable particle and full-gas lanes."""
    return (
        ParticleData(
            masses=np.ones((boxes, 2, 2)) * 1e-18,
            concentration=np.ones((boxes, 2)),
            charge=np.zeros((boxes, 2)),
            density=np.array([1000.0, 1200.0]),
            volume=np.full(boxes, 0.25),
            distribution_type="particle_resolved",
        ),
        GasData(
            name=["water", "inert", "organic"],
            molar_mass=np.array([0.018, 0.028, 0.1]),
            concentration=np.zeros((boxes, 3)),
            partitioning=np.array([True, False, True]),
        ),
        EnvironmentData(
            temperature=np.full(boxes, 298.0),
            pressure=np.full(boxes, 101325.0),
            saturation_ratio=np.tile([0.7, 7.0, 1.3], (boxes, 1)),
        ),
    )


@pytest.mark.parametrize("boxes", [1, 2])
@pytest.mark.parametrize("partitioning", [True, False])
def test_native_identity_current_values(boxes, partitioning):
    """Wrapping preserves identities, full gas lanes and current values."""
    p, g, e = native_triple(boxes)
    if not partitioning:
        g.partitioning[:] = False
    a = Aerosol(particles=p, gas=g, environment=e)
    assert a.particles is p and a.gas is g and a.environment is e
    p.masses[:] *= 2
    assert a.particles.masses[0, 0, 0] == 2e-18
    assert str(a) == f"{p}\n{g}\n{e}"
    assert not hasattr(a, "atmosphere")
    np.testing.assert_array_equal(a.environment.saturation_ratio[:, 1], 7.0)


@pytest.mark.parametrize("field", ["particles", "gas", "environment"])
def test_wrong_container(field):
    """Every wrong top-level container rejects before publication."""
    args = dict(
        zip(("particles", "gas", "environment"), native_triple(), strict=True)
    )
    args[field] = object()
    with pytest.raises(TypeError):
        Aerosol(**args)


@pytest.mark.parametrize("failure", ["volume", "kind", "names", "boxes"])
def test_invalid_candidate_unchanged(failure):
    """Read-only validation cannot repair rejected candidate data."""
    p, g, e = native_triple()
    if failure == "volume":
        p.volume[:] = 0
    if failure == "kind":
        p.distribution_type = None
    if failure == "names":
        g.name[1] = g.name[0]
    if failure == "boxes":
        e.temperature = np.ones(2)
    before = [
        (
            obj,
            {
                k: v.copy() if hasattr(v, "copy") else v
                for k, v in vars(obj).items()
            },
        )
        for obj in (p, g, e)
    ]
    with pytest.raises(ValueError):
        Aerosol(particles=p, gas=g, environment=e)
    for obj, fields in before:
        for key, value in fields.items():
            np.testing.assert_equal(getattr(obj, key), value)
