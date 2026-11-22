"""Parcel geometry, conservative transfer and smooth flow sensitivities."""

from dataclasses import replace

import numpy as np
import pytest
from particula.dynamics.parcel_cluster import (
    Connection,
    ParcelCluster,
    run_cluster,
)
from particula.dynamics.parcel_replay import OrganicForcing, replay
from particula.dynamics.tests.parcel_fixtures import (  # noqa: F401
    parcel_device_fixture,
    parcel_trace_fixture,
)


@pytest.mark.parametrize(
    "shape,faces",
    [((1, 1, 4), 3), ((6, 1, 1), 5), ((4, 3, 1), 17), ((2, 2, 2), 12)],
)
def test_regular_and_independent_histories(trace, shape, faces):
    """Regular geometry reproduces independent per-parcel trajectories."""
    traces = {
        str(i): replace(
            trace,
            parcel_id=str(i),
            temperature=trace.temperature + i * trace.time * 0.01,
        )
        for i in range(np.prod(shape))
    }
    cluster = ParcelCluster.regular(traces, shape=shape, widths=(1.0, 0.5, 2.0))
    assert len(cluster.faces) == faces
    r = run_cluster(cluster, max_step=0.05)
    for key, t in traces.items():
        np.testing.assert_array_equal(
            r.parcels[key].wet_radius, replay(t, max_step=0.05).wet_radius
        )


def test_extensive_transfer_and_capacity(trace, device):
    """Conserve binned material across unequal volumes and multiple exits."""
    donors = replace(
        trace,
        parcel_id="A",
        organic=OrganicForcing("soa", 1e-8),
        dry_density=1350.0,
    )
    empty = replace(
        donors,
        parcel_id="B",
        volume=donors.volume * 2,
        initial=replace(donors.initial, number=np.zeros(2)),
        organic=OrganicForcing("soa"),
    )
    traces = {"A": donors, "B": empty, "C": replace(empty, parcel_id="C")}
    centers = {k: (i, 0, 0) for i, k in enumerate(traces)}
    flow = 0.05 * trace.volume[0]
    edges = [
        Connection("A", r, flow, particles=True, water=True) for r in ("B", "C")
    ]

    def run(data, edges):
        return run_cluster(
            ParcelCluster.custom(data, centers=centers, connections=edges),
            max_step=0.25,
            device=device,
        )

    result = run(traces, edges)
    fraction = 0.25 * flow / donors.volume[0]
    for name in ("B", "C"):
        p = result.parcels[name]
        np.testing.assert_allclose(
            p.number[1], fraction * trace.initial.number, rtol=1e-14
        )
        assert p.organic_gas[1] / p.volume[1] == pytest.approx(
            1e-8 * fraction / empty.volume[1], rel=1e-14, abs=0
        )
        np.testing.assert_allclose(
            p.dry_mass[1],
            result.inventories_transferred["A", name]["dry_mass"][1],
            rtol=1e-14,
        )
    np.testing.assert_allclose(
        result.totals["number"], sum(trace.initial.number), rtol=1e-14
    )
    np.testing.assert_allclose(
        result.totals["organic_gas"], 1e-8, rtol=1e-14, atol=0
    )
    shuffled = run(dict(reversed(traces.items())), edges)
    for k in traces:
        np.testing.assert_array_equal(
            result.parcels[k].bin_water, shuffled.parcels[k].bin_water
        )
    with pytest.raises(ValueError, match="aggregate"):
        run(traces, [replace(e, flow=3 * trace.volume[0]) for e in edges])
    with pytest.raises(ValueError, match="cover"):
        run({**traces, "B": replace(empty, time=empty.time * 2)}, edges)


def test_flow_ad_passive_and_uptake(trace, device):
    """Compare flow adjoints with a donor-product oracle and uptake reruns."""
    traces = {
        k: replace(
            trace,
            parcel_id=k,
            volume=np.ones_like(trace.time),
            organic=OrganicForcing("org", mass, uptake),
        )
        for k, mass, uptake in [("A", 1e-8, 0), ("B", 0, 0.3)]
    }
    centers = {"A": (0, 0, 0), "B": (0, 0, 1)}

    def run(rate, observation=None):
        return run_cluster(
            ParcelCluster.custom(
                traces,
                centers=centers,
                connections=[Connection("A", "B", rate)],
            ),
            max_step=0.125,
            device=device,
            gradient=("A", "B", "B", observation) if observation else None,
        )

    ad = run(0.1, "particle").flow_gradient
    assert ad > 0
    for delta in (0.0001, 0.00003):
        endpoints = [
            run(0.1 + sign * delta).parcels["B"].organic_particle[-1]
            for sign in (-1, 1)
        ]
        assert ad == pytest.approx(
            (endpoints[1] - endpoints[0]) / (2 * delta), rel=2e-7, abs=0
        )
    traces["B"] = replace(traces["B"], organic=OrganicForcing("org"))
    assert run(0.1, "gas").flow_gradient == pytest.approx(
        1e-8 * (1 - 0.1 * 0.125) ** 7, rel=2e-14, abs=0
    )


def test_contacts_and_distinct_native_schedules(trace):
    """Respect declared contact windows and independent solver schedules."""
    t = replace(trace, organic=OrganicForcing(initial_gas=1.0))
    rough = replace(t.sample_every(2), parcel_id="coarse")
    traces = {x.parcel_id: x for x in (t, rough)}
    centers = {"ascent": (0, 0, 0), "coarse": (1, 0, 0)}
    disconnected = ParcelCluster.custom(traces, centers=centers)
    run = run_cluster(disconnected, max_step=0.19)
    for k, p in run.parcels.items():
        np.testing.assert_array_equal(
            p.wet_radius, replay(traces[k], max_step=0.19).wet_radius
        )
    edge = Connection(
        "ascent", "coarse", 0.01, active_intervals=((0.1, 0.3), (0.7, 0.9))
    )
    connected = ParcelCluster.custom(
        traces, centers=centers, connections=[edge]
    )
    report = run_cluster(connected, max_step=0.19)
    flux = report.transfers["ascent", "coarse"]
    assert flux[report.time == 0.3] == flux[report.time == 0.7]
    assert flux[report.time == 0.7] < flux[-1]
    with pytest.raises(ValueError, match="fixed connectivity"):
        run_cluster(
            connected, max_step=0.19, gradient=("ascent", "coarse", "coarse")
        )


def test_cluster_humidity_derivative(trace, device):
    """Differentiate a common humidity ramp at a named parcel observation."""
    traces = {
        k: replace(trace, parcel_id=k, dry_density=1400.0) for k in ("A", "B")
    }
    cluster = ParcelCluster.custom(
        traces,
        centers={"A": (0, 0, 0), "B": (1, 0, 0)},
        connections=[Connection("A", "B", 0.04, particles=True)],
    )

    def run(a, gradient=None):
        return run_cluster(
            cluster,
            max_step=0.05,
            device=device,
            humidity_amplitude=a,
            gradient=gradient,
        )

    grad = run(0.0, ("humidity", "B")).parcels["B"].humidity_gradient
    h = 3e-6
    fd = (run(h).parcels["B"].liquid[-1] - run(-h).parcels["B"].liquid[-1]) / (
        2 * h
    )
    assert grad == pytest.approx(fd, rel=2e-6, abs=0)
