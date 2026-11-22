"""Event-order, migration, capacity and restart controls in physical metres."""

from dataclasses import replace

import numpy as np
import pytest
from particula.dynamics.parcel_cluster import ParcelCluster
from particula.dynamics.parcel_motion import (
    MotionState,
    Particle,
    advance_motion,
)
from particula.dynamics.tests.parcel_fixtures import (  # noqa: F401
    parcel_device_fixture,
    parcel_trace_fixture,
)


def column(trace, shape=(1, 1, 2)):
    """Unit-width grid with one distinct ID per motion cell."""
    return ParcelCluster.regular(
        {
            str(i): replace(trace, parcel_id=str(i))
            for i in range(np.prod(shape))
        },
        shape=shape,
        widths=(1.0, 1.0, 1.0),
    )


def tracer(name, cell, position):
    """A dilute resolved tracer in the creeping-flow regime."""
    return Particle(
        name, cell, position, 1e-6, 1000.0, 3.0, (2e-15, 2.2e-15), -1.0
    )


def test_moving_frame_crossing_restart_and_capacity(trace, device, tmp_path):
    """Cross at relative speed and reject a full target transactionally."""
    cluster = column(trace, (3, 1, 1))
    initial = MotionState(0.0, (tracer("A", "0", (0.5, 0.5, 0.5)),))
    opts = dict(
        air_velocity=(2.0, 0.0, 0.0),
        mesh_velocity=(1.0, 0.0, 0.0),
        capacity=1,
        device=device,
    )
    at_face = advance_motion(cluster, initial, duration=0.5, **opts)
    assert at_face.particles[0].cell == "1"
    assert at_face.events[0].time == 0.5
    at_face.save(tmp_path / "motion.json")
    restored = MotionState.load(tmp_path / "motion.json")
    full = advance_motion(cluster, initial, duration=2.0, **opts)
    continued = advance_motion(cluster, restored, duration=1.5, **opts)
    assert full.events == continued.events
    np.testing.assert_allclose(
        full.particles[0].position, continued.particles[0].position, atol=2e-15
    )
    assert full.particles[0].position[0] == 4.5
    assert [e.destination for e in full.events] == ["1", "2"]
    occupied = replace(
        initial,
        particles=(*initial.particles, tracer("B", "1", (1.5, 0.5, 0.5))),
    )
    with pytest.raises(ValueError, match="capacity"):
        advance_motion(
            cluster,
            occupied,
            duration=0.75,
            **{
                **opts,
                "air_velocity": {
                    "0": (2.0, 0.0, 0.0),
                    "1": (1.0, 0.0, 0.0),
                    "2": (1.0, 0.0, 0.0),
                },
            },
        )
    assert occupied.particles[0].cell == "0" and not occupied.events
    with pytest.raises(ValueError, match="geometry"):
        advance_motion(
            cluster,
            restored,
            duration=0.5,
            **{**opts, "mesh_velocity": (0.0, 0.0, 0.0)},
        )
    with pytest.raises(ValueError, match="gradients"):
        advance_motion(cluster, initial, duration=0.5, gradient=True, **opts)


@pytest.mark.parametrize("mode", ["open", "deposit", "periodic", "retain"])
def test_boundary_ledgers(trace, device, mode):
    """Retain or account for every represented particle at the boundary."""
    cluster = column(trace, (2, 1, 1))
    p = tracer("A", "1", (1.5, 0.5, 0.5))
    out = advance_motion(
        cluster,
        MotionState(0.0, (p,)),
        duration=0.75,
        air_velocity=(1.0, 0.0, 0.0),
        capacity=1,
        device=device,
        boundaries={("1", 0, 1): mode},
    )
    (event,) = out.events
    assert event.kind == mode and event.time == 0.5
    assert (event.record.count, event.record.masses, event.record.charge) == (
        p.count,
        p.masses,
        p.charge,
    )
    if mode in ("open", "deposit"):
        assert not out.particles and event.destination is None
    else:
        assert out.particles[0].position[0] == (
            0.25 if mode == "periodic" else 2.0
        )


def test_corner_crosses_adjacent_faces(trace, device):
    """Resolve simultaneous corner events through oriented shared faces."""
    cluster = column(trace, (2, 2, 1))
    p = tracer("a", "0", (0.5, 0.5, 0.5))
    end = advance_motion(
        cluster,
        MotionState(0.0, (p,)),
        duration=0.5,
        air_velocity=(1.0, 1.0, 0.0),
        capacity=1,
        device=device,
    )
    assert [(e.axis, e.source, e.destination) for e in end.events] == [
        (0, "0", "2"),
        (1, "2", "3"),
    ]


def test_retained_contact_restart_and_separating_geometry(trace, device):
    """Keep a retaining contact once and require explicit new shared faces."""
    c = column(trace, (2, 1, 1))
    p = tracer("A", "1", (1.5, 0.5, 0.5))
    opts = dict(air_velocity=(1.0, 0, 0), capacity=1, device=device)
    contact = advance_motion(c, MotionState(0.0, (p,)), duration=0.5, **opts)
    later = advance_motion(c, contact, duration=0.25, **opts)
    assert len(later.events) == len(contact.events) == 1
    boxes = {"0": ((0, 0, 0), (1, 1, 1)), "1": ((1.1, 0, 0), (2.1, 1, 1))}
    separated = ParcelCluster.custom(c.traces, centers=c.centers, bounds=boxes)
    assert not separated.faces
    rebound = later.rebind_geometry(separated)
    end = advance_motion(separated, rebound, duration=0.25, **opts)
    assert end.particles[0].position[0] == 2.1


@pytest.mark.parametrize(
    ("destination_wind", "boundary", "final_x", "events"),
    [
        (-1.0, "retain", 0.75, ["retain", "cross"]),
        (1.0, "retain", 1.0, ["retain", "cross"]),
        (1.0, "open", None, ["retain", "cross", "open"]),
    ],
)
def test_sliding_contact_across_cells(
    trace, device, destination_wind, boundary, final_x, events
):
    """A sliding particle obeys the velocity and wall policy of its new cell."""
    cluster = column(trace, (1, 2, 1))
    start = MotionState(0.0, (tracer("A", "0", (0.75, 0.5, 0.5)),))
    opts = dict(
        air_velocity={"0": (1.0, 1.0, 0.0), "1": (destination_wind, 1.0, 0.0)},
        boundaries={("1", 0, 1): boundary},
        capacity=1,
        device=device,
    )
    full = advance_motion(cluster, start, duration=0.75, **opts)
    at_face = advance_motion(cluster, start, duration=0.5, **opts)
    resumed = advance_motion(cluster, at_face, duration=0.25, **opts)
    assert [e.kind for e in full.events] == events
    assert full.events == resumed.events
    assert full.retained == resumed.retained
    if final_x is None:
        assert full.particles == resumed.particles == ()
        assert full.events[-1].time == 0.5
    else:
        np.testing.assert_allclose(
            full.particles[0].position,
            resumed.particles[0].position,
            atol=1e-15,
        )
        assert full.particles[0].position[:2] == (final_x, 1.25)
