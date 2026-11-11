"""Atomic publication of complete native aerosol state."""

import numpy as np
import pytest
from particula import Aerosol
from particula.tests.aerosol_test import native_triple

NAMES = ("particles", "gas", "environment")


def snapshot(objects):
    """Preserve both field identity and values, including mutable names."""
    return [
        (
            obj,
            {
                key: (value, value.copy() if hasattr(value, "copy") else value)
                for key, value in vars(obj).items()
            },
        )
        for obj in objects
    ]


def assert_unchanged(saved):
    """Reject both reference swaps and payload writes."""
    for obj, fields in saved:
        for key, (identity, value) in fields.items():
            assert getattr(obj, key) is identity
            np.testing.assert_equal(getattr(obj, key), value)


@pytest.mark.parametrize("index", range(3))
@pytest.mark.parametrize("same", [True, False])
def test_individual_identity_and_peers(index, same):
    """All setters preserve selected identity and unaffected peers."""
    old = native_triple()
    a = Aerosol(**dict(zip(NAMES, old, strict=True)))
    candidate = old[index] if same else native_triple()[index]
    saved = snapshot((*old, candidate))
    setattr(a, NAMES[index], candidate)
    assert getattr(a, NAMES[index]) is candidate
    for j, name in enumerate(NAMES):
        if j != index:
            assert getattr(a, name) is old[j]
    assert_unchanged(saved)


@pytest.mark.parametrize("index", range(3))
def test_individual_layout_change_rejects_then_coordinated_succeeds(index):
    """Coordinated box changes never pass through an incompatible state."""
    old, candidates = native_triple(), native_triple(2)
    a = Aerosol(**dict(zip(NAMES, old, strict=True)))
    saved = snapshot((*old, *candidates))
    with pytest.raises(ValueError):
        setattr(a, NAMES[index], candidates[index])
    assert_unchanged(saved)
    assert all(getattr(a, k) is v for k, v in zip(NAMES, old, strict=True))
    assert a.replace_data(**dict(zip(NAMES, candidates, strict=True))) is None
    assert all(
        getattr(a, k) is v for k, v in zip(NAMES, candidates, strict=True)
    )
    assert_unchanged(saved)


@pytest.mark.parametrize(
    "failure",
    [
        "type",
        "particle_shape",
        "gas_names",
        "ratio",
        "volume",
        "kind",
        "dtype",
        "grid",
    ],
)
def test_coordinated_rejection_preserves_old_candidates_and_aliases(failure):
    """Failures at early and late admission stages never publish anything."""
    old = native_triple()
    a = Aerosol(**dict(zip(NAMES, old, strict=True)))
    p, g, e = native_triple()
    p.masses = old[0].masses  # Shared arrays must remain untouched too.
    if failure == "type":
        candidate = object()
    else:
        candidate = p
    if failure == "particle_shape":
        p.charge = np.zeros((1, 9))
    if failure == "gas_names":
        g.name[1] = g.name[0]
    if failure == "ratio":
        e.saturation_ratio = np.ones((1, 2))
    if failure == "volume":
        p.volume[:] = 0
    if failure == "kind":
        p.distribution_type = None
    if failure == "dtype":
        p.concentration = p.concentration.astype(np.float32)
    if failure == "grid":
        p.radius_grid = np.array([1.0, 2.0])
    saved = snapshot((*old, p, g, e))
    with pytest.raises((TypeError, ValueError)):
        a.replace_data(particles=candidate, gas=g, environment=e)
    assert_unchanged(saved)
    assert all(getattr(a, k) is v for k, v in zip(NAMES, old, strict=True))


@pytest.mark.parametrize("index", range(3))
def test_same_object_revalidates_mutated_peer(index):
    """A setter cannot bypass admission by assigning an existing object."""
    old = native_triple()
    a = Aerosol(**dict(zip(NAMES, old, strict=True)))
    old[0].volume[:] = 0
    saved = snapshot(old)
    with pytest.raises(ValueError):
        setattr(a, NAMES[index], old[index])
    assert_unchanged(saved)


def test_coordinated_species_change_and_no_external_rebind():
    """Gas/environment widths change together; retained bindings stay old."""
    old = native_triple()
    a = Aerosol(**dict(zip(NAMES, old, strict=True)))
    p, g, e = native_triple()
    g.name = ["new"]
    g.molar_mass = np.array([0.1])
    g.concentration = np.array([[3.0]])
    g.partitioning = np.array([False])
    e.saturation_ratio = np.array([[9.0]])
    with pytest.raises(ValueError):
        a.gas = g
    with pytest.raises(ValueError):
        a.environment = e
    binding = (a.particles, a.gas, a.environment)
    a.replace_data(particles=p, gas=g, environment=e)
    assert all(x is y for x, y in zip(binding, old, strict=True))
    assert a.gas.name == ["new"] and a.particles is p
