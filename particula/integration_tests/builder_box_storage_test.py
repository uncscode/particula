"""Exercise builder box contracts through mutable CPU slot activation."""

import numpy as np
import pytest
from particula.gas import GasDataBuilder
from particula.particles import ParticleDataBuilder
from particula.particles.slot_management import activate_slots


@pytest.mark.parametrize("builder", [ParticleDataBuilder, GasDataBuilder])
@pytest.mark.parametrize("count", [True, np.bool_(False), 0, -1, 1.5, 2.0])
def test_invalid_box_counts(builder, count):
    """Reject counts without truncating or accepting Boolean dimensions."""
    with pytest.raises(ValueError, match="positive integer"):
        builder().set_n_boxes(count)


def test_particle_broadcast_supports_independent_slot_activation():
    """Expanded fields are writable and boxes do not share mutable rows."""
    concentration = np.zeros(3)
    charge = np.zeros(3)
    volume = np.array([0.25])
    particles = (
        ParticleDataBuilder()
        .set_distribution_type("particle_resolved")
        .set_n_boxes(np.int64(2))
        .set_masses(np.zeros((2, 3, 1)))
        .set_density(np.array([1000.0]))
        .set_concentration(concentration, units="count")
        .set_charge(charge)
        .set_volume(volume)
        .build()
    )
    activated = activate_slots(
        particles,
        np.full((2, 1, 1), 1e-18),
        np.ones((2, 1)),
        np.zeros((2, 1)),
        np.array([1, 0], dtype=np.int32),
    )
    np.testing.assert_array_equal(activated, [1, 0])
    np.testing.assert_array_equal(particles.concentration[1], 0)
    particles.volume[0] = 0.5
    assert particles.volume[1] == 0.25
    for field, source in (
        (particles.concentration, concentration),
        (particles.charge, charge),
        (particles.volume, volume),
    ):
        assert field.flags.writeable
        assert not np.shares_memory(field, source)


def test_particle_matching_fields_retain_storage():
    """The broadcast fix does not change existing exact-shape ownership."""
    concentration = np.ones((2, 3))
    particles = (
        ParticleDataBuilder()
        .set_distribution_type("discrete")
        .set_masses(np.full((2, 3, 1), 1e-18))
        .set_density(np.array([1000.0]))
        .set_concentration(concentration)
        .build()
    )
    assert particles.concentration is concentration


@pytest.mark.parametrize("source_boxes, target_boxes", [(1, 3), (2, 2)])
def test_gas_matching_or_singleton_box_counts(source_boxes, target_boxes):
    """Accept exact matches and expand singleton input into writable rows."""
    gas = _gas_builder(source_boxes).set_n_boxes(target_boxes).build()
    assert gas.n_boxes == target_boxes
    assert gas.concentration.flags.writeable
    gas.concentration[0, 0] = 4.0
    assert gas.concentration[1, 0] == 1.0


def test_gas_box_count_mismatch_rejects_without_changing_input():
    """Explicit dimensions cannot silently disagree with supplied storage."""
    builder = _gas_builder(2).set_n_boxes(3)
    with pytest.raises(ValueError, match="box count"):
        builder.build()
    assert builder.set_n_boxes(2).build().n_boxes == 2


def _gas_builder(boxes):
    return (
        GasDataBuilder()
        .set_names(["water"])
        .set_molar_mass([0.018])
        .set_partitioning([True])
        .set_concentration(np.ones((boxes, 1)))
    )
