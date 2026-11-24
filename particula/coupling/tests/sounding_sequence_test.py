"""Analytical controls for time-dependent grid and parcel forcing."""

import numpy as np
import pytest
import warp as wp
from particula.coupling.flow_factory import FlowFactory
from particula.coupling.flow_replay import GridReplay
from particula.coupling.grid import FlowGrid
from particula.coupling.grid_inventory import InventoryLayout
from particula.coupling.grid_run import GridRun
from particula.coupling.prescribed import Sounding
from particula.coupling.sounding_sequence import SoundingSequence
from particula.dynamics.parcel import ParcelInitialState
from particula.dynamics.parcel_replay import ParcelTrace, replay


def profile(z=(0, 1), u=1, v=0, q=0.006):
    """Build one SI column with linear temperature and pressure gradients."""
    levels = np.asarray(z, dtype=float)
    ones = np.ones_like(levels)
    return Sounding(
        levels,
        290 - levels / 100,
        1e5 - 12 * levels,
        q * ones,
        u * ones,
        v * ones,
    )


def changing(times, profiles):
    """Build a changing wind from a documented synthetic atmosphere."""
    seq = SoundingSequence(times, profiles, "experiment start", "test datum")
    return FlowFactory().get_strategy("sounding_sequence", {"sequence": seq})


@pytest.fixture(params=["cpu", "cuda:0"])
def device(request):
    """Exercise actual native device calculations when available."""
    if request.param != "cpu" and not wp.is_cuda_available():
        pytest.skip("NVIDIA CUDA unavailable")
    return request.param


def test_vertical_then_time_interpolation(device):
    """Preserve bends in both profiles and restrict to the shared domain."""
    a = profile((0, 2, 4), u=(1, 5, 1))
    b = profile((1, 3, 5), u=(6, 2, 8), v=2, q=0.009)
    flow = changing((0, 4), (a, b))
    z = np.array([1, 2, 2.5, 3, 4])
    xyz = np.column_stack((0 * z, 0 * z, z))
    for t in (0, 1, 4):
        fraction = t / 4
        u = (1 - fraction) * np.interp(
            z, a.height, a.east
        ) + fraction * np.interp(z, b.height, b.east)
        sampled = flow.sample(xyz, t, device=device)
        np.testing.assert_allclose(sampled[:, 0], u)
        np.testing.assert_allclose(sampled[:, 1], 2 * fraction)
        np.testing.assert_allclose(sampled[:, 5], 0.006 + 0.003 * fraction)
    for x, t in (([0, 0, 0], 0), ([0, 0, 5], 4), ([0, 0, 2], 5)):
        with pytest.raises(ValueError):
            flow.sample([x], t, device=device)


def test_reversal_transport_and_restart(tmp_path, device):
    """Two opposing triangular transfers use the donor at the actual time."""
    grid = FlowGrid(
        (3, 1, 1),
        ((0, 3), (0, 1), (0, 1)),
        ("periodic", "periodic", *("closed",) * 4),
    )
    flow = changing((0, 2, 3), (profile(), profile(u=-1), profile(u=-1)))
    record = GridReplay.record(
        grid, flow, (0, 3), device=device, provenance=flow.sequence.provenance
    )
    np.testing.assert_array_equal(record.times, [0, 1, 2, 3])
    np.testing.assert_allclose(
        [f.transfer[1, 1] for f in record.frames], [0.5, -0.5, -1]
    )
    path = tmp_path / "forcing.npz"
    record.save(path)

    def run():
        return GridRun(
            GridReplay.load(path),
            InventoryLayout(("tracer",)),
            [[1.0], [0.0], [0.0]],
            device=device,
        )

    direct = run()
    expected = ([0.5, 0.5, 0.0], [0.5, 0.25, 0.25])
    for i, row in enumerate(expected):
        direct.advance()
        np.testing.assert_allclose(direct.snapshot()["state"][:, 0], row)
        path_ck = tmp_path / f"state-{i}.npz"
        direct.checkpoint(path_ck)
        resumed = run()
        resumed.restore(path_ck)
        for _ in record.frames[resumed.index :]:
            resumed.advance()
        if i == 0:
            final = resumed.snapshot()["state"]
        else:
            np.testing.assert_array_equal(resumed.snapshot()["state"], final)
    direct.advance()
    np.testing.assert_array_equal(direct.snapshot()["state"], final)


def test_parcel_replay_and_steady_dune(tmp_path, device):
    """Recover the integrated rotating wind and sampled wet-growth forcing."""
    flow = changing(
        (0, 2, 4), (profile(u=2), profile(u=0, v=2, q=0.01), profile(u=-2))
    )
    aerosol = ParcelInitialState(
        np.array([7e-8]), np.array([1e7]), 0.5, 1e5, 290, 0.006
    )
    track, trace = flow.parcel_trace(
        "parcel", [0, 0, 0.5], [0, 1, 3, 4], aerosol, provenance="rotating wind"
    )
    np.testing.assert_allclose(
        track.xyz[:, :2], [[0, 0], [1.5, 0.5], [2, 2], [1.5, 3.5], [0, 4]]
    )
    np.testing.assert_allclose(trace.vapor, [0.006, 0.008, 0.010, 0.008, 0.006])
    trace.save(tmp_path / "parcel.csv")
    loaded = ParcelTrace.load(tmp_path / "parcel.csv")
    direct = replay(trace, max_step=0.01, device=device)
    cpu = replay(trace, max_step=0.01, device="cpu")
    np.testing.assert_allclose(direct.wet_radius, cpu.wet_radius, rtol=1e-10)
    np.testing.assert_allclose(
        direct.water_restored, cpu.water_restored, rtol=1e-10, atol=1e-14
    )
    assert np.max(direct.wet_radius) > 1.1 * direct.wet_radius[0, 0]
    for i in (1, 3):
        before = replay(loaded, max_step=0.01, stop_index=i, device=device)
        resumed = replay(
            loaded, max_step=0.01, checkpoint=before.checkpoint, device=device
        )
        np.testing.assert_allclose(
            resumed.wet_radius[-1], direct.wet_radius[-1], rtol=1e-13
        )
    seq = changing((0, 4), (profile((0, 10)),) * 2).sequence

    def dune(p):
        return FlowFactory().get_strategy(
            "dune", dict(sounding=p, height=2, width=4, lid=10)
        )

    base, changing_dune = dune(seq.profiles[0]), dune(seq)
    args = ("parcel", [-2, 0, 4], [0, 1, 4], aerosol)
    baseline, _ = base.parcel_trace(*args, provenance="steady")
    tracked, _ = changing_dune.parcel_trace(*args, provenance="sequence")
    np.testing.assert_allclose(tracked.xyz, baseline.xyz, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(
        base.sample(tracked.xyz, 1, device=device),
        changing_dune.sample(tracked.xyz, 1, device=device),
    )


def test_invalid_sounding_sequences():
    """Fail before a step on invalid times, coverage or provenance."""
    p = profile()
    for times in ((1, 1), (1, 0), (0, np.nan)):
        with pytest.raises(ValueError):
            changing(times, (p, p))
    with pytest.raises(ValueError):
        changing((0, 1), (p, profile((2, 3))))
    with pytest.raises(ValueError):
        SoundingSequence((0, 1), (p, p), "", "source")


def test_csv_order_and_column_units(tmp_path):
    """Read a long-form table only when both times and heights are ordered."""
    table = tmp_path / "sounding.csv"
    header = (
        "time_s,height_m,temperature_K,pressure_Pa,vapor_kg_per_kg_dry_air,"
        "east_m_s,north_m_s"
    )

    def load(h, times):
        rows = [f"{t},{z},290,100000,.006,1,2" for t, z in times]
        table.write_text("\n".join([h, *rows]) + "\n")
        return SoundingSequence.from_csv(
            table, time_origin="release", provenance="test profile"
        )

    rows = [(0, 0), (0, 2), (1, 0), (1, 2)]
    assert load(header, rows).times == (0, 1)
    for h, values in (
        (header.replace("pressure_Pa", "pressure_hPa"), rows),
        (header, rows[2:] + rows[:2]),
        (header, [rows[1], rows[0], *rows[2:]]),
    ):
        with pytest.raises(ValueError):
            load(h, values)


def test_nonsteady_dune_air_and_uniform_grid(device):
    """Changing streamline winds maintain cell divergence and uniform tracer."""
    seq = changing(
        (0, 2, 4),
        (
            profile((0, 10), u=2),
            profile((0, 10), u=-1, q=0.007),
            profile((0, 10), u=1, q=0.008),
        ),
    ).sequence
    flow = FlowFactory().get_strategy(
        "dune", dict(sounding=seq, height=2, width=4, lid=10)
    )
    edges = np.linspace(-10, 10, 9)
    bed = tuple(2 * np.exp(-0.5 * (edges / 4) ** 2))
    grid = FlowGrid(
        (8, 1, 5),
        ((-10, 10), (0, 1), (0, 10)),
        ("open", "open", *("closed",) * 4),
        bed,
    )
    forcing = GridReplay.record(
        grid, flow, [0, 4], device=device, provenance=seq.provenance
    )
    np.testing.assert_allclose(forcing.times, [0, 4 / 3, 2, 3, 4])
    run = GridRun(
        forcing,
        InventoryLayout(("tracer",)),
        grid.volume[:, None],
        inlet=1.0,
        device=device,
    )
    for frame in forcing.frames:
        np.testing.assert_allclose(frame.transfer.sum(axis=1), 0, atol=1e-13)
        run.advance()
        np.testing.assert_allclose(run.snapshot()["state"][:, 0], grid.volume)


def test_changing_dune_with_emission_and_dispersion(tmp_path, device):
    """A gust, turbulent jumps and restart share one source and particle clock."""
    from particula.coupling.flows import DirectionalMixing
    from particula.coupling.particle_dispersion import ParticleDispersion
    from particula.coupling.surface_emission import (
        BedPatch,
        DustClass,
        SurfaceEmission,
    )

    sequence = changing(
        (0, 2, 4), (profile((0, 8)), profile((0, 8), u=-2), profile((0, 8)))
    ).sequence
    flow = FlowFactory().get_strategy(
        "dune", dict(sounding=sequence, height=1.5, width=3, lid=8)
    )
    x = np.linspace(-6, 6, 7)
    grid = FlowGrid(
        (6, 2, 4),
        ((-6, 6), (-2, 2), (0, 8)),
        ("open", "open", "periodic", "periodic", "deposit", "closed"),
        tuple(1.5 * np.exp(-0.5 * (x / 3) ** 2)),
    )
    history = GridReplay.record(
        grid,
        flow,
        [0, 4],
        DirectionalMixing((0.3, 0.2, 0.1)),
        device=device,
        provenance=sequence.provenance,
    )
    dust = DustClass("fine", 0.05, 1e-7, 4e-6, 2600.0, (1.0,), slip=0.05)
    patch = BedPatch(
        "crest", int(np.ravel_multi_index((3, 0, 0), grid.shape)), 1.0, (2e-5,)
    )
    surface = SurfaceEmission(
        history,
        flow,
        [patch],
        [dust],
        ("mineral",),
        drag_coefficient=0.01,
        reference_height=0.5,
        ejection_height=0.25,
        representation="resolved",
        device=device,
    )

    def fresh():
        return GridRun(
            history,
            InventoryLayout(("gas",)),
            np.zeros((len(grid.volume), 1)),
            surface=surface,
            dispersion=ParticleDispersion(seed=13, max_step=0.1),
            device=device,
        )

    a = fresh()
    a.advance()
    path = tmp_path / "gust.npz"
    a.checkpoint(path)
    b = fresh()
    b.restore(path)
    for _ in history.frames[a.index :]:
        a.advance()
        b.advance()
    for k, v in a.snapshot().items():
        np.testing.assert_array_equal(b.snapshot()[k], v)
    budget = b.budget()
    assert np.sum(surface.emitted) > 0
    kg = sum(budget[k][-1] for k in ("airborne", "deposited", "escaped"))
    np.testing.assert_allclose(
        budget["surface_bed_kg"][0, 0] + kg,
        patch.available[0],
        rtol=1e-13,
        atol=1e-18,
    )
