"""Independent physical controls for spatial providers, fixed grids and replay."""

from dataclasses import replace
from importlib.util import find_spec

import numpy as np
import pytest
import warp as wp
from particula.coupling.flow_factory import FlowFactory, UniformFlowBuilder
from particula.coupling.flow_replay import GridReplay
from particula.coupling.flows import (
    DirectionalMixing,
    StabilityMixing,
    UniformFlow,
)
from particula.coupling.grid import FlowGrid
from particula.coupling.grid_inventory import InventoryLayout, Uptake
from particula.coupling.grid_run import GridRun, HostTransport, Release
from particula.coupling.prescribed import Sounding
from particula.dynamics.parcel import ParcelInitialState
from particula.dynamics.parcel_motion import Particle
from particula.gas.gas_data import GasData
from particula.particles.particle_data import ParticleData

GAS = InventoryLayout(("dust-vapor",))
WIND = UniformFlow((1.0, 0.0, 0.0), 290.0, 100000.0, 0.008)
PERIODIC = ("periodic", "periodic", *["closed"] * 4)
OPEN = ("open", "open", *["closed"] * 4)
PARAMS = dict(velocity=(1, 0, 0), temperature=290, pressure=100000, vapor=0.008)


@pytest.fixture(params=["cpu", "cuda:0"])
def device(request):
    """Require actual CUDA when running the qualified device matrix."""
    if request.param.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("NVIDIA CUDA unavailable")
    return request.param


def record(grid, flow=WIND, times=(0, 1), mix=None, device="cpu"):
    """Generate accepted forcing using the selected provider."""
    return GridReplay.record(
        grid, flow, times, mix, device=device, provenance="analytical test"
    )


def test_factory_and_trajectory(tmp_path):
    """Check all construction routes, path replay and capability rejection."""
    direct = UniformFlow(**PARAMS)
    for flow in (
        direct,
        UniformFlowBuilder().set_parameters(PARAMS).build(),
        FlowFactory().get_strategy("uniform", PARAMS),
    ):
        np.testing.assert_allclose(
            flow.sample([[0, 1, 2]], 0)[0], [1, 0, 0, 290, 1e5, 0.008]
        )
    template = ParcelInitialState(
        np.array([5e-8]), np.array([1e7]), 0.5, 1e5, 290, 0.005
    )
    track, trace = flow.parcel_trace(
        "air", [0, 0, 2], [0, 1, 2], template, provenance="uniform"
    )
    np.testing.assert_allclose(track.xyz[:, 0], [0, 1, 2])
    path = tmp_path / "air.csv"
    trace.save(path)
    trajectory = FlowFactory().get_strategy("trajectory_replay", {"path": path})
    np.testing.assert_array_equal(
        trajectory.trajectory("air").volume, trace.volume
    )
    grid = FlowGrid((2, 1, 1), ((0, 2), (0, 1), (0, 1)))
    with pytest.raises(TypeError):
        record(grid, trajectory)
    with pytest.raises(ValueError):
        trajectory.trajectory("other")
    with pytest.raises(ValueError):
        FlowFactory().get_strategy("uniform", {"vapor": 0})


def test_uniform_pressure_conversion():
    """Non-SI input converts through Pint or identifies the missing extra."""
    params = dict(PARAMS, pressure=1000, pressure_units="hPa")
    factory = FlowFactory()
    if find_spec("pint") is None:
        with pytest.raises(ImportError, match="Install pint"):
            factory.get_strategy("uniform", params)
    else:
        flow = factory.get_strategy("uniform", params)
        np.testing.assert_allclose(flow.sample([[0, 0, 0]], 0)[0, 4], 1e5)


def test_periodic_translation_and_numerical_spread(device):
    """Donor-cell displacement and known artificial variance under refinement."""
    errors = []
    for dx in (1.0, 0.5):
        n = int(64 / dx)
        grid = FlowGrid((n, 1, 1), ((0, 64), (0, 2), (0, 3)), PERIODIC)
        a = np.zeros((n, 1))
        a[int(58 / dx)] = 2
        flow = record(grid, times=(0, 9), device=device)
        run = GridRun(flow, GAS, a, device=device)
        run.advance()
        mass = run.snapshot()["state"][:, 0]
        displacement = (
            grid.centers[:, 0] - grid.centers[int(58 / dx), 0] + 32
        ) % 64 - 32
        mean = mass @ displacement / 2
        var = mass @ (displacement - mean) ** 2 / 2
        sub = np.ceil(9 / dx / 0.8)
        np.testing.assert_allclose(
            [mass.sum(), mean, var],
            [2, 9, 9 * dx * (1 - 9 / dx / sub)],
            atol=2e-12,
        )
        errors.append(var)
    assert errors[1] < errors[0]


def test_diffusion_puff(device):
    """A narrow impulse has exact second moments 2 K t in all active axes."""
    grid = FlowGrid((23, 21, 19), ((-11.5, 11.5), (-10.5, 10.5), (-9.5, 9.5)))
    k = np.array([0.24, 0.16, 0.10])
    f = record(
        grid,
        replace(WIND, velocity=(0, 0, 0)),
        times=(0, 2),
        mix=DirectionalMixing(k),
        device=device,
    )
    a = np.zeros((len(grid.volume), 1))
    a[len(a) // 2] = 3
    r = GridRun(f, GAS, a, device=device)
    r.advance()
    mass = r.snapshot()["state"][:, 0]
    assert mass.min() >= 0
    np.testing.assert_allclose(mass.sum(), 3)
    np.testing.assert_allclose(mass @ grid.centers, 0, atol=1e-13)
    np.testing.assert_allclose(mass @ grid.centers**2 / 3, 4 * k, rtol=1e-12)


def test_native_and_host_contract(device):
    """Native extensive conversion and host ownership with post-transport uptake."""
    grid = FlowGrid((2, 1, 1), ((0, 4), (0, 3), (0, 1)), PERIODIC)
    f = record(grid, times=(0, 1, 2), device=device)
    layout = InventoryLayout(("A", "B"), ("coarse",), ("dry", "condensed"))
    gas = GasData(
        ["A", "B"],
        np.array([0.1, 0.2]),
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([True, False]),
    )
    particles = ParticleData(
        np.array([[[2.0, 0.0]], [[1.0, 2.0]]]),
        np.array([[5.0], [7.0]]),
        np.zeros((2, 1)),
        np.ones(2),
        np.ones(2),
        distribution_type="discrete",
    )
    a = layout.pack(grid, gas, particles)
    np.testing.assert_array_equal(a[0], [6, 12, 30, 60, 0])
    g, p = layout.unpack(grid, a, gas, particles)
    np.testing.assert_array_equal(g.concentration, gas.concentration)
    np.testing.assert_array_equal(p.masses, particles.masses)
    sink = Uptake("A", "coarse", "condensed", np.log(2))
    run = GridRun(f, layout, a, owner="host", uptake=sink, device=device)
    zeros = np.zeros_like(a)
    with pytest.raises(ValueError):
        run.advance()
    with pytest.raises(ValueError):
        run.advance(HostTransport(0, 0.5, a, zeros, zeros))
    assert run.index == 0
    host = HostTransport(0, 1, a[::-1].copy(), zeros, zeros)
    run.advance(host)
    end = run.snapshot()["state"]
    np.testing.assert_allclose(end[:, 0], host.inventories[:, 0] / 2)
    np.testing.assert_allclose(end[:, 4], host.inventories[:, 4] + end[:, 0])
    np.testing.assert_array_equal(end[:, 1:4], host.inventories[:, 1:4])
    local = GridRun(f, layout, a, device=device)
    with pytest.raises(ValueError):
        local.advance(host)
    local.advance()
    np.testing.assert_allclose(local.budget()["state"], a.sum(axis=0))
    invalid = a.copy()
    invalid[0, 2] = 0
    with pytest.raises(ValueError):
        GridRun(f, layout, invalid)


def test_short_intervals_and_replay(tmp_path, device):
    """A short host advance records its actual time and paired volume exactly."""

    class ShortFlow(UniformFlow):
        def interval(self, grid, start, end, mixing, *, device="cpu"):
            return super().interval(
                grid, start, min(start + 0.5, end), mixing, device=device
            )

    grid = FlowGrid((2, 1, 1), ((0, 2), (0, 1), (0, 1)), OPEN)
    forcing = record(grid, ShortFlow(**PARAMS), device=device)
    np.testing.assert_array_equal(forcing.times, [0, 0.5, 1])
    path = tmp_path / "grid.npz"
    forcing.save(path)
    replay = FlowFactory().get_strategy("grid_replay", {"path": path})
    assert replay.identity == forcing.identity
    replayed = record(grid, replay)
    initial = np.array([[0.0], [1.0]])
    runs = [
        GridRun(f, GAS, initial, inlet=2.0, device=device)
        for f in (forcing, replayed)
    ]
    for r in runs:
        r.advance()
        r.advance()
    for k, v in runs[0].snapshot().items():
        np.testing.assert_array_equal(v, runs[1].snapshot()[k])
    budget = runs[0].budget()
    np.testing.assert_allclose(
        budget["state"], 1 + budget["imports"] - budget["exports"]
    )
    assert budget["imports"][0] == 2
    with pytest.raises(ValueError):
        replay.interval(grid, 0, 0.1)
    bad = forcing.frames[0].transfer.copy()
    bad[0, 1] += 1
    with pytest.raises(ValueError):
        GridReplay(grid, [replace(forcing.frames[0], transfer=bad)], "invalid")


def test_deposition_restart_and_escape(tmp_path, device):
    """Restart on either side of a pickup conserves mass, ID, charge and fate."""
    grid = FlowGrid(
        (4, 1, 2),
        ((0, 4), (0, 1), (0, 2)),
        ("open", "open", "closed", "closed", "deposit", "closed"),
    )
    f = record(grid, times=(0, 1, 2, 4), device=device)
    grain = Particle(
        "sand", "0", (0.25, 0.5, 0.5), 75e-6, 2600, 2, (0.03, 0.01), -2
    )
    dust = Particle(
        "dust", "7", (3.5, 0.5, 1.2), 5e-6, 2600, 3, (0.001, 0.002), 1
    )
    releases = [Release(grain, 1.0, 1), Release(dust, 0.0, -1)]

    def make():
        return GridRun(
            f,
            GAS,
            np.ones((8, 1)),
            releases=releases,
            bed=[0.1, 0.05],
            device=device,
        )

    reference = make()
    for _ in f.frames:
        reference.advance()
    for stop in (1, 2):
        r = make()
        for _ in range(stop):
            r.advance()
        p = tmp_path / "state.npz"
        r.checkpoint(p)
        r = make()
        r.restore(p)
        while r.index < len(f.frames):
            r.advance()
        for k, v in reference.snapshot().items():
            np.testing.assert_array_equal(v, r.snapshot()[k])
        s = r.snapshot()
        np.testing.assert_allclose(s["event_time"], [1.5, 0.5])
        np.testing.assert_allclose(
            s["world"], [[0.75, 0.5, 0], [4.0, 0.5, 1.2]]
        )
        np.testing.assert_array_equal(s["status"], [2, 3])
        b = r.budget()
        np.testing.assert_allclose(
            b["bed_mass"] + b["deposited"][2:], [0.1, 0.05]
        )
        np.testing.assert_allclose(b["escaped"], [3, 3, 0.003, 0.006])


@pytest.mark.parametrize("plane", ["xz", "xy"])
def test_extruded_grid(plane, device):
    """A slice and uniform extrusion produce identical concentrations."""
    dims = ([9, 1, 8], [9, 3, 8]) if plane == "xz" else ([9, 8, 1], [9, 8, 3])
    axis = 1 if plane == "xz" else 2
    results = []
    for shape in dims:
        bounds = ((0, 9), (0, 8), (0, 8))
        grid = FlowGrid(tuple(shape), bounds, PERIODIC)
        a = (1 + np.sin(grid.centers[:, 0]) ** 2) * grid.volume
        r = GridRun(
            record(
                grid, mix=DirectionalMixing((0.1, 0.02, 0.03)), device=device
            ),
            GAS,
            a[:, None],
            device=device,
        )
        r.advance()
        results.append(
            (r.snapshot()["state"][:, 0] / grid.volume).reshape(shape)
        )
    np.testing.assert_allclose(results[1], np.repeat(results[0], 3, axis=axis))


# Briggs open-country Table A-1 evaluated at 100, 305 and 505 metres.
WIDTHS = {
    "A": ((21.89082, 66.09955, 108.39669), (20.0, 61.0, 101.0)),
    "B": ((15.92060, 48.07240, 78.83396), (12.0, 36.6, 60.6)),
    "C": ((10.94541, 33.04978, 54.19835), (7.92118, 23.68820, 38.50239)),
    "D": ((7.96030, 24.03620, 39.41698), (5.59503, 15.15817, 22.85572)),
    "E": ((5.97022, 18.02715, 29.56273), (2.91262, 8.38296, 13.15675)),
    "F": ((3.98015, 12.01810, 19.70849), (1.55340, 4.47091, 7.01693)),
}


@pytest.mark.parametrize("stability", list(WIDTHS))
@pytest.mark.parametrize("axis", [1, 2])
def test_briggs_steady_grid(stability, axis, device):
    """At steady state both marginal widths approach tabulated rural curves."""
    inlet_sigma, first, last = WIDTHS[stability][axis - 1]
    # A six-sigma outer boundary isolates spreading from domain truncation.
    bounds = [(100, 510), (0, 1), (0, 1)]
    bounds[axis] = (-6 * last, 6 * last) if axis == 1 else (0, 6 * last)
    shape = [41, 1, 1]
    shape[axis] = 100 if axis == 1 else 50
    grid = FlowGrid(tuple(shape), tuple(bounds), OPEN)
    mix = StabilityMixing(stability, 0)
    wind = replace(WIND, velocity=(5.0, 0.0, 0.0))
    f = record(grid, wind, (0, 220), mix, device=device)
    cross = grid.centers[:, axis]
    inlet = np.exp(-0.5 * (cross / inlet_sigma) ** 2)[:, None, None]
    initial = np.zeros((len(cross), 1))
    r = GridRun(f, GAS, initial, inlet=inlet, device=device)
    r.advance()
    mass = r.snapshot()["state"][:, 0].reshape(41, -1)
    coords = cross.reshape(41, -1)
    # dx=10 m gives a half-cell (~5 m) longitudinal bias at steady state.
    for station, target in [(20, first), (40, last)]:
        width = np.sqrt(
            mass[station] @ coords[station] ** 2 / mass[station].sum()
        )
        assert width == pytest.approx(target, rel=0.022)


def test_dune_refinement(device):
    """Conserve a uniform tracer and converge to the continuous dune track."""
    sounding = Sounding(
        [0, 100], [290, 289], [1e5, 99000], [0.008, 0.006], [4, 4], [0, 0]
    )
    factory = FlowFactory()
    assert factory.get_strategy("sounding", {"sounding": sounding}).sample(
        [[0, 0, 50]], 0, device=device
    )[0, 3] == pytest.approx(289.5)
    flow = factory.get_strategy(
        "dune", dict(sounding=sounding, height=8, width=15, lid=100)
    )
    start = (-20.0, 0.5, float(flow.dune.ground(-20) + 2))
    expected = flow.dune.track(start, [0, 12], settling=0.4)
    assert expected.deposition_time is not None
    errors = []
    for n in (12, 96):
        edges = np.linspace(-30, 30, n + 1)
        grid = FlowGrid(
            (n, 1, 8),
            ((-30, 30), (0, 1), (0, 100)),
            ("open", "open", "closed", "closed", "deposit", "closed"),
            tuple(flow.dune.ground(edges)),
        )
        f = record(grid, flow, (0, 12), device=device)
        np.testing.assert_allclose(
            f.frames[0].transfer.sum(axis=1), 0, atol=2e-12
        )
        i = np.searchsorted(edges, start[0], side="right") - 1
        p = Particle("dust", str(i * 8), start, 1e-5, 2600, 1, (0.001,), 0)
        run = GridRun(
            f,
            GAS,
            grid.volume[:, None],
            inlet=1,
            releases=[Release(p, 0.4)],
            device=device,
        )
        run.advance()
        s = run.snapshot()
        np.testing.assert_allclose(
            s["state"][:, 0] / grid.volume, 1, atol=2e-14
        )
        assert s["status"][0] == 2
        np.testing.assert_allclose(
            s["world"][0, 2], grid.ground(s["world"][0, 0]), atol=1e-13
        )
        errors.append(abs(s["world"][0, 0] - expected.xyz[-1, 0]))
    assert errors[1] < errors[0]
    assert errors[1] < 60 / 96


def test_full_array_cpu_cuda_parity():
    """Compare replay, every lane, particle state and ledgers at common times."""
    if not wp.is_cuda_available():
        pytest.skip("requires NVIDIA GPU")
    grid = FlowGrid(
        (9, 5, 4),
        ((0, 9), (0, 5), (0, 4)),
        ("open", "open", "periodic", "periodic", "deposit", "closed"),
    )
    mixing = DirectionalMixing((0.02, 0.03, 0.05))
    flow = UniformFlow((1.0, 0.5, 0.0), 290.0, 1e5, 0.008)
    layout = InventoryLayout(("organic",), ("fine",), ("dry", "organic"))
    a = np.zeros((len(grid.volume), 4))
    a[7] = [1e-9, 1e4, 1e-11, 0]
    p = Particle("cross", "0", (0.4, 0.7, 0.8), 1e-5, 2000, 3, (1e-12,), -1)
    runs = []
    for dev in ("cpu", "cuda:0"):
        f = record(grid, flow, (0, 0.7, 2, 5), mixing, device=dev)
        runs.append(
            GridRun(
                f,
                layout,
                a,
                uptake=Uptake("organic", "fine", "organic", 0.2),
                releases=[Release(p, 0.3, 1)],
                device=dev,
            )
        )
    for _ in range(3):
        for run in runs:
            run.advance()
        left, right = [run.snapshot() for run in runs]
        for key in (
            "state",
            "imports",
            "exports",
            "world",
            "cells",
            "status",
            "event_time",
        ):
            np.testing.assert_allclose(
                left[key], right[key], rtol=2e-12, atol=0
            )
        np.testing.assert_array_equal(left["ids"], right["ids"])
        budgets = [r.budget() for r in runs]
        for key, value in budgets[0].items():
            np.testing.assert_allclose(
                value, budgets[1][key], rtol=2e-12, atol=0
            )


def test_provider_reuses_buffer_and_host_error():
    """Copy accepted host buffers before requesting the next time."""
    g = FlowGrid((2, 1, 1), ((0, 2), (0, 1), (0, 1)))
    template = record(g).frames[0]
    buffer = template.air.copy()

    class Borrowed:
        def interval(self, grid, start, end, mixing, *, device):
            buffer[:, 3] = 290 + start
            return replace(template, start=start, end=end, air=buffer)

    f = record(g, Borrowed(), (0, 1, 2))
    assert f.frames[0].air[0, 3] == 290
    assert f.frames[1].air[0, 3] == 291
    a = np.ones((2, 1))
    run = GridRun(f, GAS, a, owner="host")
    with pytest.raises(ValueError, match="boundary transfers"):
        run.advance(HostTransport(0, 1, 2 * a, 0 * a, 0 * a))
    assert run.index == 0
    np.testing.assert_array_equal(run.snapshot()["state"], a)


def test_live_step_replay_and_resume(tmp_path, device):
    """Bind before the next host time is known; restart from matching history."""
    from particula.coupling.flow_replay import LiveGrid

    grid = FlowGrid((4, 1, 1), ((0, 4), (0, 1), (0, 1)), PERIODIC)
    initial = np.eye(4, 1)
    live = LiveGrid(grid, 0, "test host")
    r = GridRun(live, GAS, initial, device=device)
    for start, end in [(0.0, 0.3), (0.3, 1.0)]:
        f = WIND.interval(grid, start, end, None, device=device)
        r.advance_interval(f)
        if start == 0:
            p = tmp_path / "live.npz"
            r.checkpoint(p)
            binding = LiveGrid(grid, 0, "test host", history=live.recording())
            r = GridRun(binding, GAS, initial, device=device)
            r.restore(p)
            live = binding
    archive = live.recording()
    offline = GridRun(archive, GAS, initial, device=device)
    for _ in archive.frames:
        offline.advance()
    np.testing.assert_array_equal(
        offline.snapshot()["state"], r.snapshot()["state"]
    )
    assert live.times == [0.0, 0.3, 1.0]
    with pytest.raises(ValueError):
        r.advance_interval(f)
    external = GridRun(
        LiveGrid(grid, 0, "host transport"),
        GAS,
        initial,
        owner="host",
        device=device,
    )
    f = archive.frames[0]
    transported = initial[::-1].copy()
    zero = initial * 0
    external.advance_interval(f, HostTransport(0, 0.3, transported, zero, zero))
    np.testing.assert_array_equal(external.snapshot()["state"], transported)
