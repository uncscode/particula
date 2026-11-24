"""Independent puff, first-passage, well-mixed and restart controls."""

from math import erfc

import numpy as np
import pytest
import warp as wp
from particula.coupling.flow_factory import FlowFactory
from particula.coupling.flow_replay import GridReplay
from particula.coupling.flows import DirectionalMixing, UniformFlow
from particula.coupling.grid import FlowGrid
from particula.coupling.grid_inventory import InventoryLayout
from particula.coupling.grid_run import GridRun, Release
from particula.coupling.particle_dispersion import ParticleDispersion
from particula.coupling.prescribed import Sounding
from particula.dynamics.parcel_motion import Particle

LAYOUT = InventoryLayout(("tracer",))
STILL = UniformFlow((0, 0, 0), 290, 100000, 0.005)


@pytest.fixture(params=["cpu", "cuda:0"])
def device(request):
    """Run each science check on the specified execution backend."""
    if request.param.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("NVIDIA CUDA unavailable")
    return request.param


def _forcing(grid, flow, times, k, device):
    return GridReplay.record(
        grid, flow, times, k, device=device, provenance="dispersion control"
    )


def _records(points, cells):
    return [
        Release(Particle(str(i), str(cell), p, 1e-6, 2600, 2, (0.25,), -1), 0)
        for i, (p, cell) in enumerate(zip(points, cells, strict=True))
    ]


def _at_center(grid, n):
    cell = len(grid.volume) // 2
    return _records([grid.centers[cell]] * n, [cell] * n)


def _runner(f, releases, config, device):
    return GridRun(
        f,
        LAYOUT,
        np.zeros((len(f.grid.volume), 1)),
        releases=releases,
        dispersion=config,
        device=device,
    )


@pytest.mark.parametrize("seed", [1431, 7781])
def test_gaussian_puff_moments(seed, device):
    """For N Gaussian draws test mean and unbiased variance at five SE."""
    grid = FlowGrid((13, 1, 13), ((-13, 13), (-1, 1), (-13, 13)))
    k = np.array([0.25, 200, 0.5])
    wind = UniformFlow((0.5, 0, 0), 290, 100000, 0.005)
    f = _forcing(grid, wind, [0, 1], DirectionalMixing(k), device)
    n = 2048
    run = _runner(f, _at_center(grid, n), ParticleDispersion(seed, 0.2), device)
    run.advance()
    x = run.snapshot()["world"]
    assert np.all(x[:, 1] == 0)
    axes = [0, 2]
    variance = 2 * k[axes]
    assert np.all(
        abs(x[:, axes].mean(0) - [0.5, 0]) < 5 * np.sqrt(variance / n)
    )
    assert np.all(
        abs(x[:, axes].var(0, ddof=1) / variance - 1) < 5 * np.sqrt(2 / (n - 1))
    )
    np.testing.assert_array_equal(
        run.budget()["airborne"], [2 * n, -2 * n, n / 2]
    )


def test_absorbing_first_passage_convergence(device):
    """Time refinement recovers a no-drift diffusing point's bed-hit law."""
    n = 3072
    grid = FlowGrid(
        (1, 1, 4),
        ((0, 1), (0, 1), (0, 8)),
        ("closed",) * 4 + ("deposit", "closed"),
    )
    f = _forcing(grid, STILL, [0, 1], DirectionalMixing((0, 0, 0.5)), device)
    releases = _records([(0.5, 0.5, 1)] * n, [0] * n)
    exact = erfc(1 / np.sqrt(2))
    se = np.sqrt(exact * (1 - exact) / n)
    errors = []
    for step in (1.0, 1 / 256):
        r = _runner(f, releases, ParticleDispersion(1948, step), device)
        r.advance()
        state = r.snapshot()
        hits = state["status"] == 2
        np.testing.assert_array_equal(state["world"][hits, 2], 0)
        assert np.all(
            (state["event_time"][hits] >= 0) & (state["event_time"][hits] <= 1)
        )
        budget = r.budget()
        np.testing.assert_array_equal(
            budget["airborne"] + budget["deposited"], [2 * n, -2 * n, n / 2]
        )
        errors.append(abs(np.mean(hits) - exact))
    # Missed substep excursions cause O(sqrt(dt)) monitoring bias.
    assert errors[1] < errors[0] / 2
    assert errors[1] < 3 * se + np.sqrt(1 / 256) * 0.5


def test_seeded_restart_reordering_and_zero_k(tmp_path, device):
    """Verify periodicity and exact counter recovery on the selected device."""
    grid = FlowGrid(
        (8, 1, 1), ((0, 8), (0, 1), (0, 1)), ("periodic",) * 2 + ("closed",) * 4
    )
    releases = _at_center(grid, 128)
    stoch = ParticleDispersion(123, 0.03)
    wind = UniformFlow((4, 0, 0), 290, 1e5, 0.005)
    f = _forcing(grid, wind, [0, 1, 2], DirectionalMixing((0.5, 0, 0)), device)
    r = _runner(f, releases, stoch, device)
    r.advance()
    ckpt = tmp_path / "turbulence.npz"
    r.checkpoint(ckpt)
    r.advance()
    s = r.snapshot()
    restored = _runner(f, releases, stoch, device)
    restored.restore(ckpt)
    restored.advance()
    for key, actual in restored.snapshot().items():
        np.testing.assert_array_equal(actual, s[key])
    with pytest.raises(ValueError, match="configuration"):
        _runner(f, releases, ParticleDispersion(124, 0.03), device).restore(
            ckpt
        )
    reverse = _runner(f, releases[::-1], stoch, device)
    reverse.advance()
    reverse.advance()
    np.testing.assert_array_equal(reverse.snapshot()["world"][::-1], s["world"])
    assert np.all((s["world"][:, 0] >= 0) & (s["world"][:, 0] <= 8))
    assert np.unique(s["cells"]).size > 4
    f = _forcing(grid, wind, [0, 2], DirectionalMixing(), device)
    outcomes = []
    for config in (None, stoch):
        r = _runner(f, releases, config, device)
        r.advance()
        outcomes.append(r.snapshot()["world"])
    np.testing.assert_array_equal(*outcomes)


class _TerrainMixing:
    def coefficient(self, flow, grid, ref, axis):
        """Set D_eta = Kz/J² constant while allowing variable horizontal K."""
        if axis == 0:
            return 0.06 * (1 + 0.8 * np.sin(np.pi * ref[:, 0] / 4))
        return (
            0.5 * grid.jacobian(ref[:, 0]) ** 2
            if axis == 2
            else np.zeros(len(ref))
        )


def _dune(nz, device):
    sounding = Sounding(
        [0, 4], [290, 290], [1e5, 1e5], [0.005, 0.005], [0, 0], [0, 0]
    )
    flow = FlowFactory().get_strategy(
        "dune", dict(sounding=sounding, height=2, width=1, lid=4)
    )
    e = np.linspace(-4, 4, 9)
    grid = FlowGrid(
        (8, 1, nz), ((-4, 4), (0, 1), (0, 4)), bed=tuple(flow.dune.ground(e))
    )
    return _forcing(grid, flow, [0, 3], _TerrainMixing(), device)


def _amplitude(steps, g, mode, full):
    mass = g.volume * (1 + 0.5 * mode)
    for _ in range(steps):
        flux = mass[:, None] * full / steps
        mass -= flux.sum(axis=1)
        for side, neighbors in enumerate(g.neighbors.T):
            valid = neighbors >= 0
            np.add.at(mass, neighbors[valid], flux[valid, side])
    return mass @ mode / (0.5 * (g.volume @ mode**2))


def test_dune_equilibrium_and_refinement(device):
    """Check detailed balance and decay of an independent vertical cosine mode."""
    n = 4096
    config = ParticleDispersion(90381, 0.01)
    errors = []
    for nz in (2, 8):
        f = _dune(nz, device)
        g = f.grid
        # Varying J and K are both present. At equilibrium number per cell
        # is proportional to physical volume. Stratify initial cell counts.
        weight = g.volume / g.volume.sum()
        ids = np.searchsorted(weight.cumsum(), (np.arange(n) + 0.5) / n)
        rng = np.random.default_rng(0)
        points = g.coordinates[ids] + (rng.random((n, 3)) - 0.5) * g.spacing
        # Sample x uniformly in *physical volume* inside each sloping cell.
        ix = np.unravel_index(ids, g.shape)[0]
        j0, j1 = g.jacobian(g.edges[0][ix]), g.jacobian(g.edges[0][ix + 1])
        q = rng.random(n)
        p = q * (j0 + j1) / (j0 + np.sqrt(j0 * j0 * (1 - q) + j1 * j1 * q))
        points[:, 0] = g.edges[0][ix] + p * g.spacing[0]
        r = _runner(f, _records(g.world(points), ids), config, device)
        r.advance()
        actual = np.bincount(r.snapshot()["cells"], minlength=len(weight))
        assert np.all(
            abs(actual - n * weight) < 5 * np.sqrt(n * weight * (1 - weight))
        )
        assert np.ptp(g.volume) > 0
        # A cosine in eta obeys dA/dt=-0.5*(pi/4)^2*A. Apply the identical
        # jump operator in expectation to remove sampling error from refinement.
        _, full = config.coefficients(g, f.frames[0])
        mode = np.cos(np.pi * g.coordinates[:, 2] / 4)
        exact = np.exp(-0.5 * (np.pi / 4) ** 2 * 3)

        fine_time = _amplitude(300, g, mode, full)
        # Start/end correlation under a uniform concentration estimates the
        # cosine attenuation. Independent bounded samples |mode_i mode_j|<=1
        # have SE no larger than 1/sqrt(n), before normalization.
        sampled = (
            mode[ids] @ mode[r.snapshot()["cells"]] / (mode[ids] @ mode[ids])
        )
        assert abs(sampled - fine_time) < 5 / (
            np.sqrt(n) * np.mean(mode[ids] ** 2)
        )

        errors.append(abs(fine_time - exact))
        eigen = 0.5 * (2 * np.sin(np.pi / (2 * nz)) / g.spacing[2]) ** 2
        assert abs(fine_time - np.exp(-3 * eigen)) < abs(
            _amplitude(75, g, mode, full) - np.exp(-3 * eigen)
        )
    assert errors[1] < errors[0] / 8


@pytest.mark.parametrize(
    "seed,step", [(-1, 1), (2**64, 1), (4, 0), (3, float("nan"))]
)
def test_invalid_settings(seed, step):
    """Reject invalid seeds and unresolved integration steps."""
    with pytest.raises(ValueError):
        ParticleDispersion(seed, step)


def test_reflect_or_escape(device):
    """Compare a reflected coordinate to its folded free-space counterpart."""
    bounds = ((-0.5, 0.5), (0, 1), (0, 1))
    n = 256
    records = _records([(0, 0.5, 0.5)] * n, [1] * n)
    states = []
    for box, mode in ((100, "closed"), (1, "closed"), (1, "open")):
        grid = FlowGrid(
            (3, 1, 1),
            ((-box / 2, box / 2), *bounds[1:]),
            (mode, mode, *("closed",) * 4),
        )
        # Field-conductance CFL must not alter the draw schedule as dx changes.
        f = _forcing(
            grid, STILL, [0, 0.04], DirectionalMixing((1, 0, 0)), device
        )
        r = _runner(f, records, ParticleDispersion(437, 1), device)
        r.advance()
        states.append(r.snapshot())
        budget = r.budget()
        np.testing.assert_array_equal(
            budget["airborne"] + budget["escaped"], [2 * n, -2 * n, n / 2]
        )
    free, ref, opened = states
    x = free["world"][:, 0]
    folded = 0.5 - abs((x + 0.5) % 2 - 1)
    np.testing.assert_allclose(ref["world"][:, 0], folded, atol=1e-14, rtol=0)
    exiting = abs(x) > 0.5
    assert exiting.any()
    np.testing.assert_array_equal(opened["status"] == 3, exiting)
    np.testing.assert_allclose(
        opened["event_time"][exiting], 0.04 * 0.5 / abs(x[exiting])
    )


def test_same_seed_device_agreement():
    """Compare CPU and native CUDA paths and boundary budgets for one seed."""
    if not wp.is_cuda_available():
        pytest.skip("NVIDIA CUDA unavailable")
    grid = FlowGrid(
        (7, 1, 7),
        ((-3.5, 3.5), (-0.5, 0.5), (0, 7)),
        ("periodic", "periodic", "closed", "closed", "deposit", "closed"),
    )
    wind = UniformFlow((1.0, 0, 0), 290, 1e5, 0.005)
    releases = _records([(0, 0, 0.5)] * 256, [21] * 256)
    runs = []
    for device in ("cpu", "cuda:0"):
        f = _forcing(
            grid, wind, [0, 1, 2], DirectionalMixing((0.25, 0, 0.25)), device
        )
        r = _runner(f, releases, ParticleDispersion(678, 0.1), device)
        r.advance()
        r.advance()
        runs.append(r)
    cpu, gpu = [r.snapshot() for r in runs]
    np.testing.assert_array_equal(cpu["status"], gpu["status"])
    np.testing.assert_allclose(cpu["world"], gpu["world"], atol=1e-13, rtol=0)
    for fate in ("airborne", "deposited", "escaped"):
        np.testing.assert_array_equal(
            runs[0].budget()[fate], runs[1].budget()[fate]
        )


def test_constant_centers_variable_face_mixing(device):
    """A host can supply zero center K with mixing across one shared face."""
    from dataclasses import replace

    grid = FlowGrid((3, 1, 1), ((0, 3), (0, 1), (0, 1)))
    base = _forcing(grid, STILL, [0, 1], DirectionalMixing(), device)
    exchange = np.zeros((3, 6))
    exchange[1, 1] = exchange[2, 0] = 0.4
    replay = GridReplay(
        grid, [replace(base.frames[0], mixing=exchange)], "face exchange"
    )
    n = 1024
    run = _runner(
        replay, _at_center(grid, n), ParticleDispersion(123, 1), device
    )
    run.advance()
    cells = run.snapshot()["cells"]
    assert (cells != 0).all()
    assert abs(np.mean(cells == 2) - 0.4) < 5 * np.sqrt(0.4 * 0.6 / n)
