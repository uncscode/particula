# %% [markdown]
# # Does CUDA wall loss reproduce spherical deposition theory?
#
# **Yes, for the tested conditions:** Particula's CUDA particle-removal
# simulation agrees with the theoretical survival prediction within sampling
# uncertainty. At 1 µm and wall mixing coefficient 3.6 s⁻¹, the original study
# gave 68.546% ± 0.266 percentage points versus the theoretical 68.414%.
# The interval is 95% Monte Carlo uncertainty on the ensemble mean.
# **No experimental measurements are compared here.**
#
# Crump and Seinfeld's [1981 paper](https://doi.org/10.1016/0021-8502(81)90036-7)
# couples diffusion and gravitational settling at a spherical wall. We evaluate
# its surface integral independently, then compare Particula's CPU coefficient
# and CUDA survival ensembles. The independent integral still uses Particula's
# diffusion coefficient and settling velocity, so it does not independently
# validate those transport properties. A simpler sum of the limiting rates
# illustrates why coupling matters: it predicts only 58.242% survival.
#
# Read [native particle data](../../../guide/data-model/index.md) and
# [prepared CUDA simulations](../../../guide/gpu/prepared.md) first.
# The full run requires CUDA, takes about two minutes on an RTX 5060, and never
# falls back to CPU. Setup, step scheduling, quadrature and plotting use CPU;
# particle-removal kernels execute on CUDA.
#
# ## Specify a synthetic chamber decay
#
# The chamber radius is 0.30 m and particle density is 1000 kg/m³, as in the
# paper's Fig. 2. We choose 298.15 K, 101325 Pa and one hour of decay. The
# paper's kₑ=36 s⁻¹ is supplemented by 0.36 and 3.6 s⁻¹ sensitivity cases.
# kₑ is the near-wall mixing coefficient in Dₑ=kₑy², not a bulk diffusivity.
# Particles remain fixed in size and neutral; no evaporation, coagulation,
# ventilation, electrostatics or inertia is modeled. Water density is a
# transport assumption, not a complete water-droplet experiment.
#
# With x=πv/(2√(kₑD)), Eq. 17 gives
# β=6√(kₑD)D₁(x)/(πR)+3v/(4R). Our additive baseline sets D₁(x)=1.
# The independent angular integral below instead uses Eq. 13 from the
# [author copy, PDF page 39](https://thesis.caltech.edu/3500/3/crump-jg_1983.pdf#page=39).
# The exponential exp(−βt) converts rate into expected surviving fraction.
#
# The first block computes both references on 121- and 241-point diameter
# grids and checks the CPU coefficient to 0.5% relative tolerance. It then
# builds unit-count particle populations for independent boxes. Retired slots
# retain raw weights, so surviving particles are counted by nonzero mass.
# The remaining mass fraction must match survival to 1e-12.
# %%
import time

import matplotlib.pyplot as plt
import numpy as np
import particula as par
from matplotlib import style as mplstyle
from particula import plotting
from particula.dynamics.properties.wall_loss_coefficient import (
    get_spherical_wall_loss_coefficient_via_system_state as beta_cpu,
)
from particula.particles.properties.diffusion_coefficient import (
    get_diffusion_coefficient_via_system_state,
)
from particula.particles.properties.settling_velocity import (
    get_particle_settling_velocity_via_system_state,
)
from scipy.integrate import quad
from scipy.stats import t

R = 0.30
T = 298.15
P = 101325.0
RHO = 1000.0
DURATION = 3600.0


def coefficients(d, ke):
    radius = np.asarray(d) / 2
    diffusion = get_diffusion_coefficient_via_system_state(radius, T, P)
    v = get_particle_settling_velocity_via_system_state(radius, RHO, T, P)
    a = 2 * np.sqrt(ke * diffusion) / np.pi

    # Surface-angle quadrature, independent of the implementation's Debye integral.
    def integral(vi, ai):
        def flux(mu):
            z = vi * mu / ai
            return ai if abs(z) < 1e-10 else vi * mu / np.expm1(z)

        return 3 / (2 * R) * quad(flux, -1, 1, epsabs=1e-15, epsrel=1e-11)[0]

    exact = np.array(
        [integral(vi, ai) for vi, ai in zip(np.atleast_1d(v), np.atleast_1d(a))]
    )
    add = 3 * a / R + 3 * v / (4 * R)
    cpu = beta_cpu(ke, radius, RHO, T, P, R)
    return exact, np.atleast_1d(add), np.atleast_1d(cpu)


def reference():
    rows = []
    for grid in [121, 241]:
        for ke in [0.36, 3.6, 36.0]:
            ds = np.geomspace(0.03e-6, 3e-6, grid)
            exact, add, cpu = coefficients(ds, ke)
            for d, b, a, c in zip(ds, exact, add, cpu):
                rows.append(
                    dict(
                        grid=grid,
                        ke_s_1=ke,
                        diameter_m=d,
                        beta_reference_s_1=b,
                        beta_additive_s_1=a,
                        beta_particula_s_1=c,
                        survival_reference=np.exp(-b * DURATION),
                        survival_additive=np.exp(-a * DURATION),
                    )
                )
    error = max(
        abs(r["beta_particula_s_1"] / r["beta_reference_s_1"] - 1) for r in rows
    )
    assert error < 0.005, error
    return rows, float(error)


def run(label, ke, n, reps, dt, diameters, seed):
    import warp as wp

    start = time.monotonic()
    ds = np.repeat(diameters, reps)
    boxes = len(ds)
    mass = 4 * np.pi / 3 * (ds / 2) ** 3 * RHO
    particles = par.particles.ParticleData(
        masses=np.broadcast_to(mass[:, None, None], (boxes, n, 1)).copy(),
        concentration=np.ones((boxes, n)),
        charge=np.zeros((boxes, n)),
        density=np.array([RHO]),
        volume=np.full(boxes, 4 * np.pi * R**3 / 3),
        distribution_type="particle_resolved",
    )
    gas = par.gas.GasData(
        name=["inert"],
        molar_mass=np.array([0.029]),
        concentration=np.zeros((boxes, 1)),
        partitioning=np.array([False]),
    )
    env = par.gas.EnvironmentData(
        temperature=np.full(boxes, T),
        pressure=np.full(boxes, P),
        saturation_ratio=np.zeros((boxes, 1)),
    )
    aerosol = par.Aerosol(particles=particles, gas=gas, environment=env)
    process = par.dynamics.WallLoss(
        par.dynamics.SphericalWallLossStrategy(ke, R, "particle_resolved")
    )
    with process.prepare(
        aerosol,
        backend=par.Backend.WARP,
        device="cuda:0",
        time_step=dt,
        root_seed=seed,
    ) as sim:
        sim.run(steps=int(DURATION / dt))
        result = sim.snapshot()
    state = result.particles
    alive = state.masses[:, :, 0] > 0
    survival = alive.sum(axis=1) / n
    mass_ratio = (state.masses[:, :, 0] * state.concentration).sum(axis=1) / (
        mass * n
    )
    assert np.all(np.isfinite(state.masses)) and np.all(state.masses >= 0)
    residual = float(np.max(np.abs(mass_ratio - survival)))
    assert residual < 1e-12, residual
    exact, add, cpu = coefficients(ds, ke)
    rows = [
        dict(
            label=label,
            ke_s_1=ke,
            diameter_m=float(d),
            box=i,
            n=n,
            dt_s=dt,
            root_seed=seed,
            survival=float(s),
            reference=float(np.exp(-b * DURATION)),
            additive=float(np.exp(-a * DURATION)),
        )
        for i, (d, s, b, a) in enumerate(zip(ds, survival, exact, add))
    ]
    meta = dict(
        label=label,
        wall_seconds=time.monotonic() - start,
        boxes=boxes,
        particles_per_box=n,
        steps=int(DURATION / dt),
        device=str(wp.get_device("cuda:0")),
        device_name=wp.get_device("cuda:0").name,
        mass_number_residual=residual,
    )
    return rows, meta


# %% [markdown]
# ## Resolve sampling and numerical effects on CUDA
#
# A four-box pilot precedes three campaigns for each mixing coefficient:
# 16 boxes per diameter at 2048 particles and 30 s steps, a 15 s refinement,
# and an 8192-particle refinement. The diameters are 0.1, 0.3, 1 and 3 µm.
# Each campaign has a distinct root seed; box streams are independently keyed.
# Changing particle count changes the Monte Carlo sample, not this linear
# first-order survival probability. These are independent sampling comparisons,
# not paired random trajectories or additional chamber physics.
#
# All 36 means must lie within five binomial standard errors plus 0.002 of
# theory. Report Student-t 95% intervals from the 16 box fractions separately;
# those intervals do not cover uncertainty in physical assumptions. Save only
# compact curves, box fractions, checks and timings, never full particle states.
# %%
def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["label"], row["diameter_m"]), []).append(row)
    summaries = []
    for (label, diameter), group in groups.items():
        y = np.array([r["survival"] for r in group])
        ref = group[0]["reference"]
        mc_se = np.sqrt(ref * (1 - ref) / (group[0]["n"] * len(y)))
        assert abs(y.mean() - ref) < 5 * mc_se + 0.002
        summaries.append(
            dict(
                label=label,
                diameter_um=diameter * 1e6,
                ke=group[0]["ke_s_1"],
                mean=float(y.mean()),
                reference=ref,
                additive=group[0]["additive"],
                ci95_half=float(
                    t.ppf(0.975, len(y) - 1) * y.std(ddof=1) / np.sqrt(len(y))
                ),
            )
        )
    return summaries


def simulate():
    """Run the full reference, pilot, and CUDA refinement workload."""
    curves, error = reference()
    pilot_rows, pilot_meta = run("pilot", 3.6, 1024, 4, 30.0, [1e-6], 101)
    summarize(pilot_rows)
    rows = []
    metadata = [pilot_meta]
    for i, ke in enumerate([0.36, 3.6, 36.0]):
        for j, (tag, n, dt) in enumerate(
            [
                ("base", 2048, 30.0),
                ("half-step", 2048, 15.0),
                ("large-n", 8192, 30.0),
            ]
        ):
            values, meta = run(
                f"k{i}-{tag}",
                ke,
                n,
                16,
                dt,
                [0.1e-6, 0.3e-6, 1e-6, 3e-6],
                20261119 + i * 10 + j,
            )
            rows.extend(values)
            metadata.append(meta)
    summaries = summarize(rows)
    report = dict(
        curves=curves,
        summaries=summaries,
        boxes=rows,
        pilot=pilot_rows,
        runs=metadata,
        coefficient_error=error,
    )
    print(
        f"Completed {sum(m['boxes'] for m in metadata)} CUDA boxes; "
        f"CPU coefficient maximum relative error {error:.6g}"
    )
    for r in summaries:
        if r["label"] == "k1-large-n" and abs(r["diameter_um"] - 1) < 1e-12:
            print(
                f"1 µm: CUDA {100 * r['mean']:.3f}% ± "
                f"{100 * r['ci95_half']:.3f} pp (95% MC); "
                f"theory {100 * r['reference']:.3f}%; "
                f"additive {100 * r['additive']:.3f}%"
            )
    return report


# %% [markdown]
# ## Compare theory, CUDA and the additive approximation
#
# Solid curves below show the coupled surface integral; dashed curves add
# the limiting rates. Points show 8192-particle CUDA ensembles with 95% Monte
# Carlo intervals, often smaller than the symbols. Everything is synthetic.
# The figure is generated from the saved report; replotting needs no GPU.
# %%
@mplstyle.context("particula.default")
def plot_results(report):
    """Plot retained numerical predictions without advancing a simulation."""
    fig, axes = plotting.subplots(1, 3, sharey=True)
    for ax, ke in zip(axes, [0.36, 3.6, 36.0]):
        sub = [
            r
            for r in report["curves"]
            if r["ke_s_1"] == ke and r["grid"] == 241
        ]
        ds = [r["diameter_m"] * 1e6 for r in sub]
        ax.semilogx(
            ds,
            [r["survival_reference"] for r in sub],
            label="Coupled theory",
            color=plotting.COLORS["teal"],
        )
        ax.semilogx(
            ds,
            [r["survival_additive"] for r in sub],
            label="Additive limits",
            color=plotting.COLORS["rust"],
            ls="--",
        )
        rr = [
            r
            for r in report["summaries"]
            if r["ke"] == ke and "large-n" in r["label"]
        ]
        ax.errorbar(
            [r["diameter_um"] for r in rr],
            [r["mean"] for r in rr],
            yerr=[r["ci95_half"] for r in rr],
            fmt="o",
            color="black",
            ms=4,
            capsize=3,
            label="CUDA mean, 95% MC interval",
        )
        ax.set_title(f"Wall mixing\n$k_e$ = {ke:g} s⁻¹")
        ax.set_xlabel("Particle diameter (µm)")
        ax.grid(alpha=0.2)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Surviving fraction after 1 hour")
    plotting.legend(axes[1], loc="lower left")
    fig.suptitle(
        "CUDA wall loss agrees with spherical deposition theory", fontsize=12
    )
    plotting.fit(fig)
    return fig


if __name__ == "__main__":
    result = simulate()
    figure = plot_results(result)
    plt.show()

# %% [markdown]
# ## Agreement with theory is established; agreement with measurements is open
#
# The original 580-box study agreed with theory in all 36 ensemble comparisons.
# At 1 µm and kₑ=3.6 s⁻¹, CUDA and theory differed by 0.132 percentage points,
# inside the ±0.266-point sampling interval. This supports implementation of
# the published theory under the stated transport inputs. It is not an
# experimental validation of Particula or an independent test of D and v.
#
# The additive approximation underpredicts survival by 10.172 points in that
# case. Across the scanned diameters its largest difference is 13.361 points
# at kₑ=36 s⁻¹. Original timestep and population refinement shifts were at most
# 0.504 and 0.359 points, respectively, including independent sampling noise.
# See the [validation record](../../../contribute/validation/gpu/spherical-wall-loss.md)
# for the full evidence and provenance. Current saved outputs above record
# the deliberate publication rerun separately from the original study.
#
# For a real chamber, measure size-selected decay near 0.3, 1 and 3 µm while
# independently characterizing mixing, charge and ventilation. Check whether
# inertia invalidates the coarse-particle assumptions before attributing a
# discrepancy to software. Fitting a different mixing coefficient at every
# diameter could hide the coupling effect. That experimental comparison remains
# a proposed next investigation.
#
# Continue with the [wall-loss guide](../../../guide/processes/wall-loss.md)
# or [spherical strategy example](spherical_wall_loss_strategy.py).
