"""Regression tests for the read-only CPU structural alignment seam."""

from unittest.mock import patch

import numpy as np
import pytest
from particula.aerosol_validation import validate_aerosol_structure
from particula.gas.environment_data import EnvironmentData
from particula.gas.gas_data import GasData
from particula.particles.particle_data import ParticleData


def _containers(boxes=2, slots=2, species=1, mask=None):
    """Build distinct gas and particle species widths with ordered lanes."""
    if mask is None:
        mask = [True, False, True]
    particles = ParticleData(
        masses=np.ones((boxes, slots, species)),
        concentration=np.ones((boxes, slots)),
        charge=np.zeros((boxes, slots)),
        density=np.ones(species),
        volume=np.ones(boxes),
    )
    gas = GasData(
        name=["water", "inert", "acid"],
        molar_mass=np.array([0.018, 0.028, 0.098]),
        concentration=np.tile([1.0, 2.0, 3.0], (boxes, 1)),
        partitioning=np.array(mask),
    )
    environment = EnvironmentData(
        temperature=np.full(boxes, 298.0),
        pressure=np.full(boxes, 101325.0),
        saturation_ratio=np.tile([0.1, 0.2, 0.3], (boxes, 1)),
    )
    return particles, gas, environment


def _snapshot(containers):
    """Capture all raw field identities and contents for rejection checks."""
    fields = (
        ("masses", "concentration", "charge", "density", "volume"),
        ("name", "molar_mass", "concentration", "partitioning"),
        ("temperature", "pressure", "saturation_ratio"),
    )
    return [
        (
            container,
            field,
            value,
            value.copy() if hasattr(value, "copy") else value,
        )
        for container, names in zip(containers, fields, strict=True)
        for field in names
        for value in (getattr(container, field),)
    ]


def _unchanged(snapshot):
    """Assert fields and every lane remain unmodified by inspection."""
    for container, field, original, values in snapshot:
        assert getattr(container, field) is original
        if isinstance(original, np.ndarray):
            np.testing.assert_array_equal(original, values)
        else:
            assert original == values


@pytest.mark.parametrize(
    ("boxes", "slots", "species", "mask"),
    [
        (2, 2, 1, [True, False, True]),
        (2, 2, 4, [False, False, False]),
        (1, 0, 0, [False, False, False]),
    ],
)
def test_accepts_full_order_without_mutation(boxes, slots, species, mask):
    """Inactive gas lanes and empty particle axes still retain their data."""
    containers = _containers(boxes, slots, species, mask)
    snapshot = _snapshot(containers)
    assert validate_aerosol_structure(*containers) is None
    _unchanged(snapshot)
    assert containers[1].name == ["water", "inert", "acid"]
    np.testing.assert_array_equal(
        containers[2].saturation_ratio[0], [0.1, 0.2, 0.3]
    )


@pytest.mark.parametrize("index", [0, 1, 2])
def test_top_level_type_precedes_stored_fields(index):
    """Top-level type failures are TypeError even with malformed storage."""
    containers = list(_containers())
    containers[0].masses = np.array(0)
    containers[index] = object()
    with pytest.raises(
        TypeError, match=("particles", "gas", "environment")[index]
    ):
        validate_aerosol_structure(*containers)


@pytest.mark.parametrize(
    ("index", "field", "bad", "message"),
    [
        (0, "masses", np.array(1), "particles.masses"),
        (0, "masses", np.ones(2), "particles.masses"),
        (0, "masses", np.ones((0, 2, 1)), "box count"),
        (0, "masses", "not an array", "particles.masses"),
        (0, "masses", None, "particles.masses"),
        (0, "concentration", None, "particles.concentration"),
        (0, "charge", [1, 2], "particles.charge"),
        (0, "concentration", np.ones((2, 3)), "particles.concentration"),
        (0, "charge", np.ones(2), "particles.charge"),
        (0, "density", np.ones(2), "particles.density"),
        (0, "volume", np.ones(1), "particles.volume"),
        (1, "name", [], "gas.name"),
        (1, "name", None, "gas.name"),
        (1, "name", ["water", "", "acid"], "gas.name"),
        (1, "name", ["water", "water", "acid"], "gas.name"),
        (1, "name", ["water", "  ", "acid"], "gas.name"),
        (1, "name", ("water", "inert", "acid"), "gas.name"),
        (1, "name", ["water", 2, "acid"], "gas.name"),
        (1, "molar_mass", np.ones(2), "gas.molar_mass"),
        (1, "molar_mass", None, "gas.molar_mass"),
        (1, "partitioning", [True, False, True], "gas.partitioning"),
        (1, "concentration", np.ones(3), "gas.concentration"),
        (1, "concentration", np.ones((2, 2)), "gas.concentration"),
        (1, "partitioning", np.ones(2), "gas.partitioning"),
        (2, "temperature", np.array(300), "environment.temperature"),
        (2, "temperature", None, "environment.temperature"),
        (2, "pressure", [300, 301], "environment.pressure"),
        (2, "pressure", np.ones(1), "environment.pressure"),
        (
            2,
            "saturation_ratio",
            np.ones((2, 2)),
            "environment.saturation_ratio",
        ),
        (2, "saturation_ratio", np.ones(3), "environment.saturation_ratio"),
    ],
)
def test_rejects_malformed_stored_fields_without_mutation(
    index, field, bad, message
):
    """Each malformed raw field fails by name without changing any input."""
    containers = _containers()
    setattr(containers[index], field, bad)
    snapshot = _snapshot(containers)
    with pytest.raises(ValueError, match=message):
        validate_aerosol_structure(*containers)
    _unchanged(snapshot)


@pytest.mark.parametrize(
    ("index", "field", "bad", "message"),
    [
        (1, "concentration", np.ones((1, 3)), "gas.concentration box count"),
        (2, "temperature", np.ones(1), "environment.pressure"),
    ],
)
def test_cross_box_mismatch_and_local_order(index, field, bad, message):
    """Local environment consistency is checked before cross-box agreement."""
    containers = _containers()
    setattr(containers[index], field, bad)
    snapshot = _snapshot(containers)
    with pytest.raises(ValueError, match=message):
        validate_aerosol_structure(*containers)
    _unchanged(snapshot)


def test_environment_box_mismatch_after_local_validation():
    """A locally valid environment still needs the particle box axis."""
    containers = list(_containers())
    containers[2] = EnvironmentData([300.0], [101325.0], [[0.1, 0.2, 0.3]])
    snapshot = _snapshot(containers)
    with pytest.raises(ValueError, match="environment.temperature box count"):
        validate_aerosol_structure(*containers)
    _unchanged(snapshot)


def test_inspection_precedence_and_no_physical_or_copy_calls():
    """Structure wins over names; no representation or copy is requested."""
    containers = _containers(mask=[False, False, False])
    containers[0].masses[0, 0, 0] = np.nan
    containers[1].concentration[0, 0] = -1.0
    containers[2].temperature[0] = -1.0
    snapshot = _snapshot(containers)
    with (
        patch.object(
            ParticleData, "validate_representation", side_effect=AssertionError
        ),
        patch.object(ParticleData, "copy", side_effect=AssertionError),
        patch.object(GasData, "copy", side_effect=AssertionError),
        patch.object(EnvironmentData, "copy", side_effect=AssertionError),
        patch.object(ParticleData, "__post_init__", side_effect=AssertionError),
        patch.object(GasData, "__post_init__", side_effect=AssertionError),
        patch.object(
            EnvironmentData, "__post_init__", side_effect=AssertionError
        ),
    ):
        assert validate_aerosol_structure(*containers) is None
    _unchanged(snapshot)
    containers[1].name = []
    containers[0].charge = np.ones(1)
    with pytest.raises(ValueError, match="particles.charge"):
        validate_aerosol_structure(*containers)


@pytest.mark.parametrize("replacement", [None, 42])
@pytest.mark.parametrize("index", [0, 1, 2])
def test_wrong_container_type_names_offending_parameter(index, replacement):
    """Every wrong input type is rejected before stored fields are read."""
    containers = list(_containers())
    containers[index] = replacement
    with pytest.raises(
        TypeError, match=("particles", "gas", "environment")[index]
    ):
        validate_aerosol_structure(*containers)


@pytest.mark.parametrize(
    (
        "early_index",
        "early_field",
        "early_bad",
        "late_index",
        "late_field",
        "late_bad",
        "message",
    ),
    [
        (0, "masses", np.ones(2), 1, "name", [], "particles.masses"),
        (1, "name", [], 2, "temperature", np.ones(3), "gas.name"),
        (
            2,
            "pressure",
            np.ones(3),
            1,
            "concentration",
            np.ones((1, 3)),
            "environment.pressure",
        ),
    ],
)
def test_first_malformed_field_wins(
    early_index,
    early_field,
    early_bad,
    late_index,
    late_field,
    late_bad,
    message,
):
    """Container-local schema checks precede cross-container relationships."""
    containers = _containers()
    setattr(containers[early_index], early_field, early_bad)
    setattr(containers[late_index], late_field, late_bad)
    snapshot = _snapshot(containers)
    with pytest.raises(ValueError, match=message):
        validate_aerosol_structure(*containers)
    _unchanged(snapshot)


def test_exact_gas_names_and_permuted_ratio_are_structurally_valid():
    """The checker cannot infer ratio-producer provenance from equal widths."""
    containers = _containers()
    containers[1].name = ["water", "Water", " water "]
    containers[2].saturation_ratio[:, :] = containers[2].saturation_ratio[
        :, ::-1
    ]
    snapshot = _snapshot(containers)
    assert validate_aerosol_structure(*containers) is None
    _unchanged(snapshot)


def test_detached_gas_and_environment_copies_keep_all_lanes():
    """Copies retain ordered inactive lanes without sharing mutable storage."""
    for mask in ([True, False, True], [False, False, False]):
        _, gas, environment = _containers(mask=mask)
        gas_copy = gas.copy()
        environment_copy = environment.copy()
        for original, copied, fields in (
            (
                gas,
                gas_copy,
                ("name", "molar_mass", "partitioning", "concentration"),
            ),
            (
                environment,
                environment_copy,
                ("temperature", "pressure", "saturation_ratio"),
            ),
        ):
            for field in fields:
                source = getattr(original, field)
                target = getattr(copied, field)
                assert target is not source
                if isinstance(source, np.ndarray):
                    np.testing.assert_array_equal(target, source)
                    assert not np.shares_memory(target, source)
                else:
                    assert target == source
