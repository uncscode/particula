"""Tests for the direct-only fixed-shape communication-map preflight."""

import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.testing as npt
import pytest


def _communication() -> Any:
    """Import the Warp-dependent boundary only when Warp is installed."""
    pytest.importorskip("warp")
    import particula.execution.communication as communication

    return communication


def _configuration(
    communication: Any,
    *,
    form: Any = None,
    mode: Any = None,
    source: object | None = None,
    destination: object | None = None,
    enabled: object | None = None,
    rates: object | None = None,
    volumes: object | None = None,
    edge_capacity: int = 2,
) -> tuple[Any, Any]:
    """Build an independent canonical three-box two-edge configuration."""
    wp = pytest.importorskip("warp")
    form = (
        form
        if form is not None
        else communication.CommunicationMapForm.ONE_DIMENSIONAL
    )
    mode = (
        mode
        if mode is not None
        else communication.CommunicationTransportMode.GAS
    )
    source = (
        source
        if source is not None
        else wp.array([0, 1], dtype=wp.int32, device="cpu")
    )
    destination = (
        destination
        if destination is not None
        else wp.array([1, 2], dtype=wp.int32, device="cpu")
    )
    enabled = (
        enabled
        if enabled is not None
        else wp.array([1, 1], dtype=wp.int32, device="cpu")
    )
    rates = (
        rates
        if rates is not None
        else wp.array([1.0, 2.0], dtype=wp.float64, device="cpu")
    )
    map_data = communication.CommunicationMap(
        form, mode, edge_capacity, source, destination, enabled, rates
    )
    configuration = communication.CommunicationConfiguration(
        map_data,
        communication.PrescribedVolumeUpdate(volumes),
        (
            communication.CommunicationResourceShape(
                "edge_rates",
                wp.float64,
                communication.CommunicationShapeKind.E,
            ),
        ),
    )
    from particula.execution.gpu_session import ResidentDimensions

    return configuration, ResidentDimensions(3, 2, 1)


def test_carriers_are_frozen_and_validate_cheap_metadata() -> None:
    """Typed immutable declarations retain caller payload identities."""
    communication = _communication()
    wp = pytest.importorskip("warp")
    resource = communication.CommunicationResourceShape(
        "edges", wp.float64, communication.CommunicationShapeKind.E
    )
    assert resource.role == "edges"
    with pytest.raises(FrozenInstanceError):
        resource.role = "other"  # type: ignore[misc]
    with pytest.raises(ValueError, match="nonempty"):
        communication.CommunicationResourceShape(
            " ", wp.float64, communication.CommunicationShapeKind.E
        )
    with pytest.raises(TypeError, match="integral"):
        communication.CommunicationMap(
            communication.CommunicationMapForm.ONE_DIMENSIONAL,
            communication.CommunicationTransportMode.GAS,
            True,
            object(),
            object(),
            object(),
            object(),
        )


@pytest.mark.warp
def test_configuration_retains_all_caller_array_identities() -> None:
    """Frozen declaration carriers preserve every supplied Warp array."""
    communication = _communication()
    wp = pytest.importorskip("warp")
    volumes = wp.array([1.0, 2.0, 3.0], dtype=wp.float64, device="cpu")
    configuration, dimensions = _configuration(communication, volumes=volumes)

    result = communication.validate_communication_configuration(
        configuration, dimensions, wp.get_device("cpu")
    )

    map_data = result.communication_map
    assert map_data.source_boxes is configuration.communication_map.source_boxes
    assert (
        map_data.destination_boxes
        is configuration.communication_map.destination_boxes
    )
    assert map_data.enabled is configuration.communication_map.enabled
    assert map_data.rates is configuration.communication_map.rates
    assert result.prescribed_volume.final_volumes is volumes


def test_carrier_metadata_rejects_invalid_enum_nested_and_roles() -> None:
    """Carrier construction rejects malformed metadata before Warp preflight."""
    communication = _communication()
    wp = pytest.importorskip("warp")
    with pytest.raises(TypeError, match="shape_kind"):
        communication.CommunicationResourceShape("edges", wp.float64, "e")
    with pytest.raises(TypeError, match="map form"):
        communication.CommunicationMap(
            "one_dimensional",
            communication.CommunicationTransportMode.GAS,
            0,
            object(),
            object(),
            object(),
            object(),
        )
    with pytest.raises(ValueError, match="nonnegative"):
        communication.CommunicationMap(
            communication.CommunicationMapForm.ONE_DIMENSIONAL,
            communication.CommunicationTransportMode.GAS,
            -1,
            object(),
            object(),
            object(),
            object(),
        )
    map_data = communication.CommunicationMap(
        communication.CommunicationMapForm.ONE_DIMENSIONAL,
        communication.CommunicationTransportMode.GAS,
        0,
        object(),
        object(),
        object(),
        object(),
    )
    role = communication.CommunicationResourceShape(
        "edges", wp.float64, communication.CommunicationShapeKind.E
    )
    with pytest.raises(ValueError, match="roles must be unique"):
        communication.CommunicationConfiguration(
            map_data,
            communication.PrescribedVolumeUpdate(None),
            (role, role),
        )


@pytest.mark.warp
@pytest.mark.parametrize(
    ("dimensions", "device", "exc_type", "match"),
    [
        (
            object(),
            None,
            TypeError,
            "dimensions must be an exact ResidentDimensions",
        ),
        (
            None,
            object(),
            TypeError,
            "device must be non-null Warp device metadata",
        ),
    ],
)
def test_outer_metadata_rejection_order(
    dimensions: object,
    device: object,
    exc_type: type[Exception],
    match: str,
) -> None:
    """Outer metadata rejection is exact before any array preflight."""
    communication = _communication()
    wp = pytest.importorskip("warp")
    configuration, valid_dimensions = _configuration(communication)
    valid_device = wp.get_device("cpu")

    with pytest.raises(exc_type, match=match):
        communication.validate_communication_configuration(
            configuration,
            valid_dimensions if dimensions is None else dimensions,
            valid_device if device is None else device,
        )


@pytest.mark.warp
@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("configuration", "configuration must be an exact"),
        (
            "communication_map",
            "communication_map must be an exact CommunicationMap",
        ),
        (
            "prescribed_volume",
            "prescribed_volume must be an exact PrescribedVolumeUpdate",
        ),
    ],
)
def test_validate_metadata_rejects_bad_outer_types(
    case: str,
    match: str,
) -> None:
    """Exact outer carriers are enforced before any array validation."""
    communication = _communication()
    wp = pytest.importorskip("warp")
    configuration, dimensions = _configuration(communication)
    if case == "configuration":
        target: object = object()
    elif case == "communication_map":
        target = object()
        object.__setattr__(configuration, "communication_map", target)
    else:
        target = object()
        object.__setattr__(configuration, "prescribed_volume", target)

    with pytest.raises(TypeError, match=match):
        communication.validate_communication_configuration(
            target if case == "configuration" else configuration,
            dimensions,
            wp.get_device("cpu"),
        )


@pytest.mark.warp
@pytest.mark.parametrize(
    ("resource_shapes", "exc_type", "match"),
    [
        ((object(),), TypeError, "resource_shapes must contain exact"),
        (
            (
                lambda communication, wp: (
                    communication.CommunicationResourceShape(
                        "edge_rates",
                        wp.float64,
                        communication.CommunicationShapeKind.E,
                    ),
                    communication.CommunicationResourceShape(
                        "edge_rates",
                        wp.float64,
                        communication.CommunicationShapeKind.E,
                    ),
                )
            ),
            ValueError,
            "resource_shapes roles must be unique",
        ),
    ],
)
def test_validate_metadata_revalidates_resource_shapes(
    resource_shapes: object,
    exc_type: type[Exception],
    match: str,
) -> None:
    """Validator rechecks resource shape contents and role uniqueness."""
    communication = _communication()
    wp = pytest.importorskip("warp")
    configuration, dimensions = _configuration(communication)
    if callable(resource_shapes):
        shapes = resource_shapes(communication, wp)
    else:
        shapes = resource_shapes
    object.__setattr__(configuration, "resource_shapes", shapes)

    with pytest.raises(exc_type, match=match):
        communication.validate_communication_configuration(
            configuration, dimensions, wp.get_device("cpu")
        )


@pytest.mark.warp
@pytest.mark.parametrize(
    "form",
    [
        "one_dimensional",
        "arbitrary_pairs",
    ],
)
@pytest.mark.parametrize("mode", ["gas", "particles", "gas_and_particles"])
def test_valid_forms_and_modes_return_exact_configuration(
    form: str, mode: str
) -> None:
    """Every form and transport declaration accepts an independent valid map."""
    communication = _communication()
    wp = pytest.importorskip("warp")
    source = wp.array([0, 1], dtype=wp.int32, device="cpu")
    destination_values = [1, 2] if form == "one_dimensional" else [2, 0]
    destination = wp.array(destination_values, dtype=wp.int32, device="cpu")
    configuration, dimensions = _configuration(
        communication,
        form=communication.CommunicationMapForm(form),
        mode=communication.CommunicationTransportMode(mode),
        source=source,
        destination=destination,
        volumes=wp.array([1.0, 2.0, 3.0], dtype=wp.float64, device="cpu"),
    )

    result = communication.validate_communication_configuration(
        configuration, dimensions, wp.get_device("cpu")
    )

    assert result is configuration
    assert result.communication_map.source_boxes is source
    assert result.communication_map.destination_boxes is destination


@pytest.mark.warp
@pytest.mark.parametrize(
    "form",
    ["one_dimensional", "arbitrary_pairs"],
)
@pytest.mark.parametrize("mode", ["gas", "particles", "gas_and_particles"])
def test_reverse_edges_without_final_volumes_are_valid(
    form: str, mode: str
) -> None:
    """Reverse edges are valid and no-volume forms remain write-free."""
    communication = _communication()
    wp = pytest.importorskip("warp")
    configuration, dimensions = _configuration(
        communication,
        form=communication.CommunicationMapForm(form),
        mode=communication.CommunicationTransportMode(mode),
        source=wp.array([0, 1], dtype=wp.int32, device="cpu"),
        destination=wp.array([1, 0], dtype=wp.int32, device="cpu"),
    )

    result = communication.validate_communication_configuration(
        configuration, dimensions, wp.get_device("cpu")
    )

    assert result is configuration
    assert result.prescribed_volume.final_volumes is None


@pytest.mark.warp
@pytest.mark.parametrize(
    ("field", "values", "match"),
    [
        ("enabled", [2, 1], "enabled values"),
        ("rates", [np.nan, 1.0], "rates values"),
    ],
)
def test_domain_validation_precedes_topology(
    field: str, values: list[float], match: str
) -> None:
    """Payload domains reject before an otherwise-invalid enabled endpoint."""
    communication = _communication()
    wp = pytest.importorskip("warp")
    kwargs: dict[str, object] = {
        "source": wp.array([3, 1], dtype=wp.int32, device="cpu"),
    }
    dtype = wp.int32 if field == "enabled" else wp.float64
    kwargs[field] = wp.array(values, dtype=dtype, device="cpu")
    configuration, dimensions = _configuration(
        communication, **cast(Any, kwargs)
    )

    with pytest.raises(ValueError, match=match):
        communication.validate_communication_configuration(
            configuration, dimensions, wp.get_device("cpu")
        )


@pytest.mark.warp
@pytest.mark.parametrize(
    ("source_values", "destination_values", "match"),
    [
        ([3, 1], [1, 2], "valid distinct topology"),
        ([0, 1], [0, 2], "valid distinct topology"),
        ([0, 0], [2, 2], "valid distinct topology"),
        ([0, 1], [2, 0], "valid distinct topology"),
    ],
)
def test_enabled_topology_rejections(
    source_values: list[int], destination_values: list[int], match: str
) -> None:
    """Bounds, self edges, duplicates, and one-dimensional gaps reject."""
    communication = _communication()
    wp = pytest.importorskip("warp")
    configuration, dimensions = _configuration(
        communication,
        source=wp.array(source_values, dtype=wp.int32, device="cpu"),
        destination=wp.array(destination_values, dtype=wp.int32, device="cpu"),
    )
    with pytest.raises(ValueError, match=match):
        communication.validate_communication_configuration(
            configuration, dimensions, wp.get_device("cpu")
        )


@pytest.mark.warp
def test_duplicate_directed_pair_and_disabled_padding() -> None:
    """Reject duplicate pairs but ignore disabled endpoint padding."""
    communication = _communication()
    wp = pytest.importorskip("warp")
    duplicate, dimensions = _configuration(
        communication,
        form=communication.CommunicationMapForm.ARBITRARY_PAIRS,
        source=wp.array([0, 0], dtype=wp.int32, device="cpu"),
        destination=wp.array([2, 2], dtype=wp.int32, device="cpu"),
    )
    with pytest.raises(ValueError, match="directed edge pairs"):
        communication.validate_communication_configuration(
            duplicate, dimensions, wp.get_device("cpu")
        )

    padded, dimensions = _configuration(
        communication,
        source=wp.array([-1, -1], dtype=wp.int32, device="cpu"),
        destination=wp.array([99, 99], dtype=wp.int32, device="cpu"),
        enabled=wp.array([0, 0], dtype=wp.int32, device="cpu"),
    )
    assert (
        communication.validate_communication_configuration(
            padded, dimensions, wp.get_device("cpu")
        )
        is padded
    )


@pytest.mark.warp
def test_sparse_duplicate_allocation_scales_with_edges_not_boxes() -> None:
    """Late duplicate detection uses a bounded edge-sized private table."""
    communication = _communication()
    wp = pytest.importorskip("warp")
    from particula.execution.gpu_session import ResidentDimensions

    configuration, dimensions = _configuration(
        communication,
        form=communication.CommunicationMapForm.ARBITRARY_PAIRS,
        edge_capacity=4,
        source=wp.array([0, 1, 2, 0], dtype=wp.int32, device="cpu"),
        destination=wp.array([1, 2, 3, 1], dtype=wp.int32, device="cpu"),
        enabled=wp.array([1, 1, 1, 1], dtype=wp.int32, device="cpu"),
        rates=wp.array([1.0, 1.0, 1.0, 1.0], dtype=wp.float64, device="cpu"),
    )
    dimensions = ResidentDimensions(4, 2, 1)
    table_sizes: list[int] = []
    original_full = communication.wp.full

    def tracked_full(*args: object, **kwargs: object) -> Any:
        shape = args[0] if args else kwargs["shape"]
        table_sizes.append(int(cast(Any, shape)))
        return original_full(*args, **kwargs)

    communication.wp.full = tracked_full  # type: ignore[assignment]
    try:
        with pytest.raises(ValueError, match="directed edge pairs"):
            communication.validate_communication_configuration(
                configuration, dimensions, wp.get_device("cpu")
            )
    finally:
        communication.wp.full = original_full  # type: ignore[assignment]

    assert table_sizes == [communication._duplicate_scratch_size(4)]


@pytest.mark.warp
def test_schema_alias_and_empty_preflight() -> None:
    """Arrays need independent storage, while valid empty maps are no-ops."""
    communication = _communication()
    wp = pytest.importorskip("warp")
    shared = wp.array([0, 1], dtype=wp.int32, device="cpu")
    configuration, dimensions = _configuration(communication, source=shared)
    object.__setattr__(
        configuration.communication_map, "destination_boxes", shared
    )
    with pytest.raises(ValueError, match="must not alias"):
        communication.validate_communication_configuration(
            configuration, dimensions, wp.get_device("cpu")
        )

    empty_map = communication.CommunicationMap(
        communication.CommunicationMapForm.ONE_DIMENSIONAL,
        communication.CommunicationTransportMode.GAS,
        0,
        wp.empty(0, dtype=wp.int32, device="cpu"),
        wp.empty(0, dtype=wp.int32, device="cpu"),
        wp.empty(0, dtype=wp.int32, device="cpu"),
        wp.empty(0, dtype=wp.float64, device="cpu"),
    )
    empty = communication.CommunicationConfiguration(
        empty_map, communication.PrescribedVolumeUpdate(None), ()
    )
    from particula.execution.gpu_session import ResidentDimensions

    assert (
        communication.validate_communication_configuration(
            empty, ResidentDimensions(0, 0, 0), wp.get_device("cpu")
        )
        is empty
    )


@pytest.mark.warp
def test_final_volumes_alias_map_storage_rejects_independently() -> None:
    """Volume storage must not overlap a distinct map array."""
    communication = _communication()
    wp = pytest.importorskip("warp")
    rates = wp.array([1.0, 2.0], dtype=wp.float64, device="cpu")
    overlapping_volumes = wp.array(
        ptr=rates.ptr,
        capacity=24,
        dtype=wp.float64,
        shape=(3,),
        strides=(8,),
        device="cpu",
        copy=False,
    )
    configuration, dimensions = _configuration(
        communication, rates=rates, volumes=overlapping_volumes
    )

    with pytest.raises(ValueError, match="must not alias"):
        communication.validate_communication_configuration(
            configuration, dimensions, wp.get_device("cpu")
        )


@pytest.mark.warp
def test_all_disabled_and_empty_maps_validate_full_rate_volume_domains() -> (
    None
):
    """No-op maps still reject invalid applicable rate and volume domains."""
    communication = _communication()
    wp = pytest.importorskip("warp")
    disabled, dimensions = _configuration(
        communication,
        enabled=wp.array([0, 0], dtype=wp.int32, device="cpu"),
        rates=wp.array([np.nan, 1.0], dtype=wp.float64, device="cpu"),
    )
    with pytest.raises(ValueError, match="rates values"):
        communication.validate_communication_configuration(
            disabled, dimensions, wp.get_device("cpu")
        )

    empty_map = communication.CommunicationMap(
        communication.CommunicationMapForm.ONE_DIMENSIONAL,
        communication.CommunicationTransportMode.GAS,
        0,
        wp.empty(0, dtype=wp.int32, device="cpu"),
        wp.empty(0, dtype=wp.int32, device="cpu"),
        wp.empty(0, dtype=wp.int32, device="cpu"),
        wp.empty(0, dtype=wp.float64, device="cpu"),
    )
    empty = communication.CommunicationConfiguration(
        empty_map,
        communication.PrescribedVolumeUpdate(
            wp.array([0.0], dtype=wp.float64, device="cpu")
        ),
        (),
    )
    from particula.execution.gpu_session import ResidentDimensions

    with pytest.raises(ValueError, match="final_volumes values"):
        communication.validate_communication_configuration(
            empty, ResidentDimensions(1, 0, 0), wp.get_device("cpu")
        )


@pytest.mark.warp
@pytest.mark.parametrize(
    ("value", "exc_type", "match"),
    [
        (True, TypeError, "dimensions.n_boxes must be an integral"),
        (-1, ValueError, "dimensions.n_boxes must be nonnegative"),
        (1.5, TypeError, "dimensions.n_boxes must be an integral"),
    ],
)
def test_mutated_dimensions_n_boxes_reject_before_array_preflight(
    value: object, exc_type: type[Exception], match: str
) -> None:
    """Frozen-dataclass bypasses cannot invalidate preflight dimensions."""
    communication = _communication()
    wp = pytest.importorskip("warp")
    configuration, dimensions = _configuration(communication)
    object.__setattr__(dimensions, "n_boxes", value)

    with pytest.raises(exc_type, match=match):
        communication.validate_communication_configuration(
            configuration, dimensions, wp.get_device("cpu")
        )


@pytest.mark.warp
def test_duplicate_sort_uses_enabled_edges_not_disabled_capacity() -> None:
    """Collision-prone valid keys and disabled padding use compact scratch."""
    communication = _communication()
    wp = pytest.importorskip("warp")
    from particula.execution.gpu_session import ResidentDimensions

    capacity = 64
    source = wp.array(
        [0, 1] + [-1] * (capacity - 2), dtype=wp.int32, device="cpu"
    )
    destination = wp.array(
        [8, 7] + [-1] * (capacity - 2), dtype=wp.int32, device="cpu"
    )
    enabled = wp.array(
        [1, 1] + [0] * (capacity - 2), dtype=wp.int32, device="cpu"
    )
    rates = wp.array([1.0] * capacity, dtype=wp.float64, device="cpu")
    configuration, _ = _configuration(
        communication,
        form=communication.CommunicationMapForm.ARBITRARY_PAIRS,
        edge_capacity=capacity,
        source=source,
        destination=destination,
        enabled=enabled,
        rates=rates,
    )
    sizes: list[int] = []
    original_full = communication.wp.full

    def tracked_full(*args: object, **kwargs: object) -> Any:
        sizes.append(cast(int, args[0] if args else kwargs["shape"]))
        return original_full(*args, **kwargs)

    communication.wp.full = tracked_full  # type: ignore[assignment]
    try:
        assert (
            communication.validate_communication_configuration(
                configuration, ResidentDimensions(9, 0, 0), wp.get_device("cpu")
            )
            is configuration
        )
    finally:
        communication.wp.full = original_full  # type: ignore[assignment]

    assert sizes == [communication._duplicate_scratch_size(2)]


@pytest.mark.warp
@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("source_boxes", object(), "source_boxes must be a Warp array"),
        (
            "source_boxes",
            "rank",
            "source_boxes must have rank 1",
        ),
        (
            "source_boxes",
            "shape",
            "source_boxes must have shape (2,)",
        ),
        (
            "source_boxes",
            "dtype",
            "source_boxes must have dtype",
        ),
        (
            "source_boxes",
            "null_pointer",
            "source_boxes must have a valid pointer",
        ),
        (
            "source_boxes",
            "contiguity",
            "source_boxes must be contiguous",
        ),
        (
            "source_boxes",
            "alignment",
            "source_boxes pointer must be 4-byte aligned",
        ),
        (
            "source_boxes",
            "capacity",
            "source_boxes must have sufficient integral storage capacity",
        ),
    ],
)
def test_source_schema_preflight_rejects_before_payload_scans(
    field: str,
    value: object,
    match: str,
) -> None:
    """Source schema failures precede all payload scans and writer activity."""
    communication = _communication()
    wp = pytest.importorskip("warp")
    replacement: object = value
    if value == "rank":
        replacement = wp.array([[0, 1]], dtype=wp.int32, device="cpu")
    elif value == "shape":
        replacement = wp.array([0], dtype=wp.int32, device="cpu")
    elif value == "dtype":
        replacement = wp.array([0.0, 1.0], dtype=wp.float64, device="cpu")
    elif value == "null_pointer":
        replacement = wp.array(
            ptr=0,
            capacity=8,
            dtype=wp.int32,
            shape=(2,),
            strides=(4,),
            device="cpu",
            copy=False,
        )
    elif value == "contiguity":
        replacement = wp.array(
            ptr=8,
            capacity=8,
            dtype=wp.int32,
            shape=(2,),
            strides=(8,),
            device="cpu",
            copy=False,
        )
    elif value == "alignment":
        replacement = wp.array(
            ptr=4,
            capacity=8,
            dtype=wp.int32,
            shape=(2,),
            strides=(4,),
            device="cpu",
            copy=False,
        )
    elif value == "capacity":
        replacement = wp.array(
            ptr=8,
            capacity=7,
            dtype=wp.int32,
            shape=(2,),
            strides=(4,),
            device="cpu",
            copy=False,
        )
    configuration, dimensions = _configuration(communication)
    object.__setattr__(configuration.communication_map, field, replacement)

    with pytest.raises(ValueError, match=match):
        communication.validate_communication_configuration(
            configuration, dimensions, wp.get_device("cpu")
        )


@pytest.mark.warp
@pytest.mark.parametrize(
    ("field", "replacement", "match"),
    [
        (
            "destination_boxes",
            "rank",
            "destination_boxes must have rank 1",
        ),
        (
            "enabled",
            "dtype",
            "enabled must have dtype",
        ),
        (
            "rates",
            "dtype",
            "rates must have dtype",
        ),
    ],
)
def test_later_schema_failures_reject_before_payload_scans(
    field: str,
    replacement: str,
    match: str,
) -> None:
    """Later field schema failures reject in the documented validation order."""
    communication = _communication()
    wp = pytest.importorskip("warp")
    configuration, dimensions = _configuration(communication)
    if field == "destination_boxes":
        value = wp.array([[0, 1]], dtype=wp.int32, device="cpu")
    elif field == "enabled":
        value = wp.array([0.0, 1.0], dtype=wp.float64, device="cpu")
    else:
        value = wp.array([1, 2], dtype=wp.int32, device="cpu")
    object.__setattr__(configuration.communication_map, field, value)

    with pytest.raises(ValueError, match=match):
        communication.validate_communication_configuration(
            configuration, dimensions, wp.get_device("cpu")
        )


@pytest.mark.warp
def test_rejected_validation_preserves_payload_identity_and_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rejected preflight leaves every caller-owned payload unchanged."""
    communication = _communication()
    wp = pytest.importorskip("warp")
    configuration, dimensions = _configuration(
        communication,
        source=wp.array([3, 1], dtype=wp.int32, device="cpu"),
    )
    map_data = configuration.communication_map
    payloads = (
        map_data.source_boxes,
        map_data.destination_boxes,
        map_data.enabled,
        map_data.rates,
    )
    snapshots = [payload.numpy().copy() for payload in payloads]
    monkeypatch.setattr(
        communication.wp,
        "copy",
        lambda *_args, **_kwargs: pytest.fail("P1 must not copy payloads"),
    )

    with pytest.raises(ValueError, match="valid distinct topology"):
        communication.validate_communication_configuration(
            configuration, dimensions, wp.get_device("cpu")
        )

    for payload, snapshot in zip(payloads, snapshots, strict=True):
        npt.assert_array_equal(payload.numpy(), snapshot)


@pytest.mark.warp
@pytest.mark.parametrize(
    ("value", "match"),
    [
        ("rank", "final_volumes must have rank 1"),
        ("shape", "final_volumes must have shape (3,)"),
        ("dtype", "final_volumes must have dtype"),
        ("contiguity", "final_volumes must be contiguous"),
        (
            "null_pointer",
            "final_volumes must have a valid pointer",
        ),
        (
            "alignment",
            "final_volumes pointer must be 8-byte aligned",
        ),
        (
            "capacity",
            "final_volumes must have sufficient integral storage capacity",
        ),
    ],
)
def test_final_volume_schema_rejects_before_topology(
    value: str,
    match: str,
) -> None:
    """Final volume schema failures reject before any topology check."""
    communication = _communication()
    wp = pytest.importorskip("warp")
    configuration, dimensions = _configuration(
        communication,
        source=wp.array([3, 1], dtype=wp.int32, device="cpu"),
    )
    if value == "rank":
        replacement = wp.array(
            [[1.0, 2.0, 3.0]], dtype=wp.float64, device="cpu"
        )
    elif value == "shape":
        replacement = wp.array([1.0], dtype=wp.float64, device="cpu")
    elif value == "dtype":
        replacement = wp.array([1, 2, 3], dtype=wp.int32, device="cpu")
    elif value == "contiguity":
        replacement = wp.array(
            ptr=8,
            capacity=8,
            dtype=wp.float64,
            shape=(3,),
            strides=(16,),
            device="cpu",
            copy=False,
        )
    elif value == "null_pointer":
        replacement = wp.array(
            ptr=0,
            capacity=24,
            dtype=wp.float64,
            shape=(3,),
            strides=(8,),
            device="cpu",
            copy=False,
        )
    elif value == "alignment":
        replacement = wp.array(
            ptr=4,
            capacity=24,
            dtype=wp.float64,
            shape=(3,),
            strides=(8,),
            device="cpu",
            copy=False,
        )
    else:
        replacement = wp.array(
            ptr=8,
            capacity=23,
            dtype=wp.float64,
            shape=(3,),
            strides=(8,),
            device="cpu",
            copy=False,
        )
    object.__setattr__(
        configuration.prescribed_volume, "final_volumes", replacement
    )

    with pytest.raises(ValueError, match=match):
        communication.validate_communication_configuration(
            configuration, dimensions, wp.get_device("cpu")
        )


@pytest.mark.warp
def test_source_schema_precedence_over_later_volume_schema() -> None:
    """Earlier array schema failures surface before later field failures."""
    communication = _communication()
    wp = pytest.importorskip("warp")
    configuration, dimensions = _configuration(communication)
    object.__setattr__(
        configuration.communication_map,
        "source_boxes",
        wp.array([[0, 1]], dtype=wp.int32, device="cpu"),
    )
    object.__setattr__(
        configuration.prescribed_volume,
        "final_volumes",
        wp.array([1, 2, 3], dtype=wp.int32, device="cpu"),
    )

    with pytest.raises(ValueError, match="source_boxes must have rank 1"):
        communication.validate_communication_configuration(
            configuration, dimensions, wp.get_device("cpu")
        )


@pytest.mark.warp
def test_source_device_schema_rejects_before_payload_scans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Array device mismatches reject before any payload scan begins."""
    communication = _communication()
    wp = pytest.importorskip("warp")

    class _FakeArray:
        def __init__(self) -> None:
            self.shape = (2,)
            self.dtype = wp.int32
            self.device = object()
            self.strides = (4,)
            self.ptr = 8
            self.capacity = 8

    configuration, dimensions = _configuration(communication)
    fake = _FakeArray()
    monkeypatch.setattr(communication, "_is_warp_array", lambda _value: True)
    object.__setattr__(configuration.communication_map, "source_boxes", fake)

    with pytest.raises(
        ValueError, match="source_boxes device must match device"
    ):
        communication.validate_communication_configuration(
            configuration, dimensions, wp.get_device("cpu")
        )


@pytest.mark.warp
@pytest.mark.parametrize("volume", [[0.0, 2.0, 3.0], [np.nan, 2.0, 3.0]])
def test_volume_domain_is_scanned_before_topology(
    volume: list[float],
) -> None:
    """Invalid prescribed volumes reject before invalid enabled endpoints."""
    communication = _communication()
    wp = pytest.importorskip("warp")
    configuration, dimensions = _configuration(
        communication,
        source=wp.array([3, 1], dtype=wp.int32, device="cpu"),
        volumes=wp.array(volume, dtype=wp.float64, device="cpu"),
    )

    with pytest.raises(ValueError, match="final_volumes values"):
        communication.validate_communication_configuration(
            configuration, dimensions, wp.get_device("cpu")
        )


@pytest.mark.warp
def test_validation_is_read_only_for_all_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Successful preflight neither copies nor changes caller-owned payloads."""
    communication = _communication()
    wp = pytest.importorskip("warp")
    volumes = wp.array([1.0, 2.0, 3.0], dtype=wp.float64, device="cpu")
    configuration, dimensions = _configuration(communication, volumes=volumes)
    map_data = configuration.communication_map
    payloads = (
        map_data.source_boxes,
        map_data.destination_boxes,
        map_data.enabled,
        map_data.rates,
        volumes,
    )
    snapshots = [payload.numpy().copy() for payload in payloads]
    monkeypatch.setattr(
        communication.wp,
        "copy",
        lambda *_args, **_kwargs: pytest.fail("P1 must not copy payloads"),
    )

    assert (
        communication.validate_communication_configuration(
            configuration, dimensions, wp.get_device("cpu")
        )
        is configuration
    )
    for payload, snapshot in zip(payloads, snapshots, strict=True):
        npt.assert_array_equal(payload.numpy(), snapshot)


@pytest.mark.warp
def test_private_range_and_duplicate_helpers_cover_edge_cases() -> None:
    """Private range overlap and sort sizing preserve bounded-map invariants."""
    communication = _communication()
    assert communication._overlaps((4, 8), (8, 12)) is False
    assert communication._overlaps((4, 9), (8, 12)) is True
    assert communication._overlaps(None, (8, 12)) is False
    assert communication._duplicate_scratch_size(0) == 1
    assert communication._duplicate_scratch_size(3) == 4


def test_communication_remains_unexported_and_has_no_overdraw_input() -> None:
    """Package import neither imports this concrete module nor claims P3 work."""
    root = Path(__file__).parents[3]
    environment = os.environ | {"PYTHONPATH": str(root)}
    script = """
import sys
import particula.execution as execution
assert 'CommunicationConfiguration' not in execution.__all__
assert 'particula.execution.communication' not in sys.modules
assert 'warp' not in sys.modules
"""
    completed = subprocess.run(  # noqa: S603 -- fixed interpreter and script
        [sys.executable, "-Werror", "-c", script],
        cwd=root,
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    communication = _communication()
    validator_docstring = (
        communication.validate_communication_configuration.__doc__
    )
    assert "source inventory" in validator_docstring
