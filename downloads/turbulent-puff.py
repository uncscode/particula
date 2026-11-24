# %% [markdown]
# # How wide is the same puff on a grid and as individual particles?
#
# A puff can widen even when the wind is perfectly uniform. Can we distinguish
# turbulent spread from broadening introduced by the grid? Release one
# microgram of an inert tracer at a point. One experiment carries its mass in
# fixed cells. Another divides that same mass among equal-weight particles.
# Compare their centers and variances with a Gaussian puff, then change the
# boundary to see why a timestep that predicts the width can still miss
# deposition. A final control checks constant concentration over a dune.
# [Derive the transport and first-passage laws](../../../theory/dynamics/prescribed-transport.md).
#
# ## Prescribe the experiment
#
# Use SI units throughout. The grid and particle runs receive the same eddy
# diffusivity and a horizontal wind of 0.6 m/s. A cell or particle carries
# inventory in kilograms. The particle radius and density define identical
# tracer records with zero prescribed slip. Each record's represented count
# sets its share of the microgram.

# %%
from math import erfc

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style as mplstyle
from particula import plotting
from particula.coupling.flow_factory import FlowFactory
from particula.coupling.flow_replay import GridReplay
from particula.coupling.flows import DirectionalMixing, UniformFlow
from particula.coupling.grid import FlowGrid
from particula.coupling.grid_inventory import InventoryLayout
from particula.coupling.grid_run import GridRun, Release
from particula.coupling.particle_dispersion import ParticleDispersion
from particula.coupling.prescribed import Sounding
from particula.dynamics.parcel_motion import Particle

LAYOUT = InventoryLayout(("inert tracer",))
MASS = 1e-9
SPEED = 0.6
TIMES = np.linspace(0, 4, 9)
SEED = 517
CLOSED = ("closed",) * 6


def _records(points, cells):
    grain_mass = 4 * np.pi / 3 * (1e-6) ** 3 * 2600
    count = MASS / (len(cells) * grain_mass)
    return [
        Release(
            Particle(
                f"p-{i}", str(cell), p, 1e-6, 2600, count, (grain_mass,), 0
            ),
            slip=0,
        )
        for i, (p, cell) in enumerate(zip(points, cells, strict=True))
    ]


def _forcing(grid, flow, t, k, device):
    return GridReplay.record(
        grid,
        flow,
        t,
        DirectionalMixing(k),
        device=device,
        provenance="synthetic puff control",
    )


# %% [markdown]
# ## Measure spread along the wind
#
# Center the release at the origin in a domain extending at least 16 m in
# both directions. At 4 s the nearest wall is over eight theoretical standard
# deviations from the center. For the grid, weight each cell center by its
# inventory. Each particle carries equal mass, so its coordinates have equal
# weights. Standard errors for independent Gaussian draws scale as
# $\sigma/\sqrt{N}$ for the center and $\sigma^2\sqrt{2/(N-1)}$
# for an unbiased variance.


# %%
def _puff(dx, n, k, device):
    half_cells = round(16 / dx)
    edge = (half_cells + 0.5) * dx
    grid = FlowGrid(
        (2 * half_cells + 1, 1, 1), ((-edge, edge), (-0.5, 0.5), (-0.5, 0.5))
    )
    flow = UniformFlow((SPEED, 0, 0), 290, 1e5, 0.005)
    air = _forcing(grid, flow, TIMES, (k, 0, 0), device)
    field = np.zeros((len(grid.volume), 1))
    releases = []
    if n:
        releases = _records([(0, 0, 0)] * n, [half_cells] * n)
    else:
        field[half_cells] = MASS
    run = GridRun(
        air,
        LAYOUT,
        field,
        releases=releases,
        device=device,
        dispersion=ParticleDispersion(SEED, 0.1) if n else None,
    )
    means, variances = [0.0], [0.0]
    for _ in air.frames:
        run.advance()
        obs = run.snapshot()
        if n:
            x = obs["world"][:, 0]
            means.append(float(x.mean()))
            variances.append(float(x.var(ddof=1)))
            assert (obs["status"] == 1).all()
        else:
            mass = obs["state"][:, 0]
            x = grid.centers[:, 0]
            np.testing.assert_allclose(mass.sum(), MASS, rtol=1e-12, atol=0)
            center = np.average(x, weights=mass)
            means.append(float(center))
            variances.append(float(np.average((x - center) ** 2, weights=mass)))
    return dict(dx_m=dx, n=n, K_m2_s=k, mean_m=means, variance_m2=variances)


# %% [markdown]
# ## Let the bed absorb particles
#
# Start 1 m above a flat bed in still air with vertical diffusivity 0.5 m²/s.
# The top boundary at 8 m is seven diffusion lengths away after 1 s.
# The continuum hit fraction by that time is
# $\mathrm{erfc}[z_0/\sqrt{4Kt}]$. A chord between successive sampled
# positions can miss a particle that crosses the bed and returns. Repeat the
# calculation with progressively shorter steps and count the recorded hits.


# %%
def _bed(device):
    grid = FlowGrid(
        (1, 1, 4), ((0, 1), (0, 1), (0, 8)), (*CLOSED[:4], "deposit", "closed")
    )
    air = _forcing(
        grid,
        UniformFlow((0, 0, 0), 290, 1e5, 0.005),
        [0, 1],
        (0, 0, 0.5),
        device,
    )
    n = 4096
    records = _records([(0.5, 0.5, 1)] * n, [0] * n)
    step_s = [1.0, 1 / 16, 1 / 64, 1 / 256]
    hits = []
    for step in step_s:
        run = GridRun(
            air,
            LAYOUT,
            np.zeros((4, 1)),
            releases=records,
            device=device,
            dispersion=ParticleDispersion(SEED, step),
        )
        run.advance()
        ledger = run.budget()
        np.testing.assert_allclose(
            ledger["deposited"][2] + ledger["airborne"][2],
            MASS,
            rtol=1e-11,
            atol=0,
        )
        hits.append(float(ledger["deposited"][2] / MASS))
    return dict(
        n=n, step_s=step_s, deposited_fraction=hits, exact=erfc(1 / np.sqrt(2))
    )


# %% [markdown]
# ## Mix a uniform suspension over a dune
#
# The lid stays level. Cells over the crest contain less air than cells at
# the flanks. Stratify the launch counts by physical cell volume and sample
# the horizontal coordinate within each sloping cell using the integrated
# column height. Measure the column mass per cubic meter before and after
# three seconds of mixing. The expected concentration is constant across x
# even though the expected count per column changes.


# %%
def _dune(device):
    edges = np.linspace(-4, 4, 9)
    sounding = Sounding(
        [0, 4], [290, 290], [1e5, 1e5], [0.005, 0.005], [0, 0], [0, 0]
    )
    flow = FlowFactory().get_strategy(
        "dune", dict(sounding=sounding, height=2, width=1, lid=4)
    )
    grid = FlowGrid(
        (8, 1, 4), ((-4, 4), (0, 1), (0, 4)), bed=tuple(flow.dune.ground(edges))
    )
    air = _forcing(grid, flow, [0, 3], (0.15, 0, 0.5), device)
    n = 4096
    weights = grid.volume / grid.volume.sum()
    cells = np.searchsorted(np.cumsum(weights), (np.arange(n) + 0.5) / n)
    rng = np.random.default_rng(SEED)
    ref = grid.coordinates[cells] + (rng.random((n, 3)) - 0.5) * grid.spacing
    col = np.unravel_index(cells, grid.shape)[0]
    q = rng.random(n)
    a, b = grid.jacobian(edges[col]), grid.jacobian(edges[col + 1])
    ref[:, 0] = edges[col] + grid.spacing[0] * q * (a + b) / (
        a + np.sqrt(a * a * (1 - q) + b * b * q)
    )
    records = _records(grid.world(ref), cells)
    run = GridRun(
        air,
        LAYOUT,
        np.zeros((len(weights), 1)),
        releases=records,
        dispersion=ParticleDispersion(SEED, 0.025),
        device=device,
    )
    run.advance()
    end = run.snapshot()
    assert (end["status"] == 1).all()
    end_col = np.unravel_index(end["cells"], grid.shape)[0]
    probability = weights.reshape(grid.shape).sum(axis=(1, 2))
    return dict(
        n=n,
        x_m=((edges[1:] + edges[:-1]) / 2).tolist(),
        initial=(np.bincount(col) / n / probability).tolist(),
        final=(np.bincount(end_col, minlength=8) / n / probability).tolist(),
        se=np.sqrt((1 - probability) / (n * probability)).tolist(),
    )


# %% [markdown]
# ## Compare the controls
#
# Run three grid spacings and two particle counts. The two runs with zero
# diffusivity isolate numerical spreading under pure advection. Save the
# measured moments and boundary budgets so each figure can be drawn again
# without storing thousands of trajectories.


# %%
def simulate(device="cpu"):
    """Return the puff, absorbing-bed and mapped-dune diagnostics."""
    return dict(
        device=device,
        precision="float64",
        seed=SEED,
        mass_kg=MASS,
        times_s=TIMES.tolist(),
        wind_m_s=SPEED,
        diffusivity_m2_s=0.25,
        particle_max_step_s=0.1,
        grid=[_puff(dx, 0, 0.25, device) for dx in (1.0, 0.5, 0.25)],
        particles=[_puff(0.5, n, 0.25, device) for n in (256, 4096)],
        zero=[_puff(1, 0, 0, device), _puff(0.5, 256, 0, device)],
        bed=_bed(device),
        dune=_dune(device),
    )


# %% [markdown]
# The upper panels compare the puff center and variance against time. The
# shaded variance band spans two sampling standard errors for 4,096 particles.
# The lower panels test contact with an absorbing bed and concentration over
# sloping cells. The bed bars span two binomial standard errors. The dune
# bars use twice the equilibrium standard error for independent multinomial
# draws as a reference scale. Stratifying the launch counts by cell changes
# the sampling variance of the final dune concentrations.


# %%
@mplstyle.context("particula.default")
def plot_results(report):
    """Plot compact saved diagnostics and independent analytical references."""
    fig, axes = plotting.subplots(2, 2, size="wide")
    mean, var, bed_ax, dune_ax = axes.flat
    t = np.asarray(report["times_s"])
    width = 2 * report["diffusivity_m2_s"] * t
    for case, ls in zip(report["grid"], ["-", "--", ":"], strict=True):
        label = f"Grid {case['dx_m']:g} m"
        mean.plot(t, case["mean_m"], ls, label=label)
        var.plot(t, case["variance_m2"], ls, label=label)
    for case, marker in zip(report["particles"], ["s", "o"], strict=True):
        label = f"{case['n']:,} particles"
        mean.plot(t, case["mean_m"], marker, ms=4, mfc="none", label=label)
        var.plot(t, case["variance_m2"], marker, ms=4, mfc="none", label=label)
    mean.plot(
        t, t * report["wind_m_s"], color="k", lw=1, ls="-.", label="Exact"
    )
    var.plot(t, width, color="k", lw=1, ls="-.", label="Exact")
    n = report["particles"][-1]["n"]
    sigma = width * np.sqrt(2 / (n - 1))
    var.fill_between(
        t, width - 2 * sigma, width + 2 * sigma, color="grey", alpha=0.2
    )
    for case, ls in zip(report["zero"], ["--", ":"], strict=True):
        label = "Particles" if case["n"] else "Grid 1 m"
        var.plot(
            t, case["variance_m2"], ls, color="0.45", label=f"{label}, K=0"
        )
    mean.set(xlabel="Time (s)", ylabel="Puff center (m)")
    var.set(xlabel="Time (s)", ylabel="Puff variance (m²)")
    mean.legend(fontsize=8)
    var.legend(fontsize=8, ncol=2)
    bed = report["bed"]
    p = np.array(bed["deposited_fraction"])
    bed_ax.errorbar(
        bed["step_s"], p, 2 * np.sqrt(p * (1 - p) / bed["n"]), fmt="o-"
    )
    bed_ax.axhline(bed["exact"], ls="--", color="k", label="First-passage law")
    bed_ax.set(
        xscale="log",
        xlabel="Maximum substep (s)",
        ylabel="Deposited fraction at 1 s",
    )
    bed_ax.legend(fontsize=8)
    dune = report["dune"]
    dune_ax.plot(dune["x_m"], dune["initial"], "k--", label="Initial")
    dune_ax.errorbar(
        dune["x_m"],
        dune["final"],
        2 * np.array(dune["se"]),
        fmt="o",
        label="After 3 s",
    )
    dune_ax.set(
        xlabel="Distance along dune (m)",
        ylabel="Column concentration / domain mean",
    )
    dune_ax.legend(fontsize=8)
    plotting.fit(fig)
    return fig


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda", action="store_true", help="run on the NVIDIA GPU"
    )
    device = "cuda:0" if parser.parse_args().cuda else "cpu"
    plot_results(simulate(device))
    plt.show()


# %% [markdown]
# ## What changes the answer?
#
# After 4 s the exact center is 2.4 m and the variance is 2.0 m². The
# 4,096-particle estimate is 1.995 m². The three grids give 3.68, 2.84
# and 2.50 m² as the spacing decreases. At the bed, reducing the maximum
# substep from 1 s to 1/256 s raises the deposited fraction from 0.165
# to 0.298, closer to the first-passage prediction of 0.317.
#
# At coarse resolution, the grid variance grows even in the zero-diffusivity
# control. The particle estimates fluctuate around the continuum variance.
# Increasing particle count reduces sampling uncertainty. Refining the grid
# reduces numerical spreading. Shortening the particle substep addresses a
# different error at the absorbing bed, where a missed crossing can change
# the eventual fate. The dune control checks mixing per physical volume.
# A field prediction would also require measured wind and a diffusivity
# appropriate to the site, averaging interval and atmospheric stability.
