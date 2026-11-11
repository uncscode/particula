"""Native aggregate builder identity and admission contracts."""

import numpy as np
import pytest
from particula import Aerosol, AerosolBuilder
from particula.tests.aerosol_replacement_test import assert_unchanged, snapshot
from particula.tests.aerosol_test import native_triple

NAMES = ("particles", "gas", "environment")


@pytest.mark.parametrize("boxes", [1, 2])
@pytest.mark.parametrize("partitioning", [True, False])
def test_fluent_identity_and_repeat_builds(boxes, partitioning):
    """Unequal species widths and full all-false gas survive repeated builds."""
    p, g, e = native_triple(boxes)
    if not partitioning:
        g.partitioning[:] = False
    saved = snapshot((p, g, e))
    builder = AerosolBuilder()
    assert builder.set_particles(p) is builder
    assert builder.set_gas(g) is builder
    assert builder.set_environment(e) is builder
    first, second = builder.build(), builder.build()
    assert first is not second
    for a in (first, second):
        assert isinstance(a, Aerosol)
        assert a.particles is p and a.gas is g and a.environment is e
    assert_unchanged(saved)
    p.masses[:] *= 2
    np.testing.assert_equal(second.particles.masses, first.particles.masses)


@pytest.mark.parametrize("missing", NAMES)
def test_missing_component(missing):
    """No component is synthesized or defaulted by the aggregate builder."""
    builder = AerosolBuilder()
    for name, value in zip(NAMES, native_triple(), strict=True):
        if name != missing:
            getattr(builder, f"set_{name}")(value)
    with pytest.raises(ValueError, match=missing):
        builder.build()


@pytest.mark.parametrize("failure", ["type", "names", "volume", "shape"])
def test_builder_and_constructor_agree_without_mutation(failure):
    """Build delegates admission rather than enforcing strategy/count rules."""
    p, g, e = native_triple()
    if failure == "names":
        g.name[1] = g.name[0]
    if failure == "volume":
        p.volume[:] = 0
    if failure == "shape":
        e.saturation_ratio = np.zeros((1, 2))
    saved = snapshot((p, g, e))
    args = dict(particles=p, gas=g, environment=e)
    if failure == "type":
        args["particles"] = object()
    builder = AerosolBuilder()
    for name, value in args.items():
        getattr(builder, f"set_{name}")(value)
    with pytest.raises((TypeError, ValueError)) as direct:
        Aerosol(**args)
    with pytest.raises(type(direct.value)) as built:
        builder.build()
    assert str(built.value) == str(direct.value)
    assert_unchanged(saved)


def test_set_parameters_and_revalidation():
    """The generic builder protocol retains current objects until build."""
    p, g, e = native_triple()
    builder = AerosolBuilder().set_parameters(
        dict(particles=p, gas=g, environment=e)
    )
    assert builder.build().gas is g
    g.name[0] = ""
    with pytest.raises(ValueError, match="nonblank"):
        builder.build()
