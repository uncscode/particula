# %% [markdown]
# # Can wet aerosol release glyoxal while it forms an imidazole?
#
# Galloway et al. exposed wet ammonium sulfate seed to glyoxal in the dark,
# then diluted the chamber. Replay the vapor in their Figure 6 and compare two
# particles with the same initial composition. Both can absorb and release
# glyoxal. Only one can turn it into 1H-imidazole-2-carboxaldehyde (IC).
# The gas observations drive the calculation. Particle measurements test it.
# [Chamber study, experiment 3](https://doi.org/10.5194/acp-9-3331-2009)
#
# ## Recover the measurements and start with wet seed
#
# Table 1 reports 293 K, 56% RH, a calculated 37% liquid water fraction and
# 87 µm³ of wet seed per cm³ of air. We use this water fraction by volume,
# additive volumes, 1 atm, and a representative wet radius of 100 nm. Sulfate
# normalizes the particle measurements, cancelling proportional seed losses.
# Radius is an assumption. The number of represented particles follows from
# the measured volume. Figure 6 gas markers specify the forcing in ppbv.

# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import particula as par
from particula.dynamics.chemistry import ReactionMechanism
from particula.dynamics.chemistry.process import ChemistrySolver
from particula.dynamics.chemistry.resolved import ResolvedChemistry
from particula.dynamics.chemistry.resolved_adapter import (
    ResolvedChemistryAdapter,
)
from particula.dynamics.condensation.species_mapping import (
    CondensationSpeciesMap,
)
from scipy.optimize import brentq

TEMPERATURE = 293.0
PRESSURE = 101325.0
R = 8.31446261815324
DILUTION = 710.04  # minutes from the black line in Figure 6
END = 1540  # last full ten-minute interval observed by all three series
VOLUME = 4 * np.pi * (100e-9) ** 3 / 3
COUNT = 87e-12 / VOLUME  # wet seed volume per m³ of chamber air
LANES = ("dissolved", "bound", "salt", "acid", "water", "IC")
DENSITY = np.array([1690.0, 1690.0, 1770.0, 1830.0, 1000.0, 1400.0])
# Dry-equivalent molecular masses; use the same atomic weights in all budgets.
ELEMENTS = np.array(
    [
        [2, 2, 0, 2, 0],
        [2, 2, 0, 2, 0],
        [0, 8, 2, 4, 1],
        [0, 2, 0, 4, 1],
        [0, 2, 0, 1, 0],
        [4, 4, 2, 1, 0],
    ]
)
MW = ELEMENTS @ (np.array([12.011, 1.008, 14.007, 15.999, 32.06]) / 1000)
SULFATE_MW = (32.06 + 4 * 15.999) / 1000
# Yu et al. Table S2: measured IC production in units of 1e-8 M/s at
# 1.5 M glyoxal and 2 M ammonium, except 3.5 which uses the 1 M glyoxal row.
YU = {
    1.9: (0.067, 1.5),
    2.8: (0.791, 1.5),
    3.5: (4.54, 1.0),
    4.0: (13.2, 1.5),
    4.5: (49.1, 1.5),
}


def inputs():
    """Locate the same data in the checkout, notebook kernel or download."""
    starts = [Path.cwd()]
    if "__file__" in globals():
        starts.append(Path(__file__).resolve().parent)
    for start in starts:
        for parent in (start, *start.parents):
            path = parent / "docs/data/glyoxal-chamber"
            if (path / "mechanism.json").is_file():
                return path
    raise FileNotFoundError(
        "Extract the bundle including docs/data/glyoxal-chamber"
    )


def observations():
    return {
        name: np.loadtxt(inputs() / (name + ".csv"), delimiter=",", skiprows=1)
        for name in ("gas_ppbv", "organic_per_sulfate", "marker68_per_sulfate")
    }


def initial_state(ppbv, nitrogen=True):
    mass = np.zeros(6)
    mass[2] = VOLUME * 0.63 * DENSITY[2]
    mass[4] = VOLUME * 0.37 * DENSITY[4]
    if not nitrogen:
        # Preserve sulfate equivalents and water in the chemical knockout.
        mass[3] = mass[2] * MW[3] / MW[2]
        mass[2] = 0
    gas = par.gas.GasData(
        ["glyoxal"], MW[:1].copy(), np.array([[gas_kg(ppbv)]]), np.array([True])
    )
    particles = par.particles.ParticleData(
        masses=mass.reshape(1, 1, -1),
        density=DENSITY.copy(),
        concentration=np.array([[COUNT]]),
        charge=np.zeros((1, 1)),
        volume=np.ones(1),
        distribution_type="particle_resolved",
    )
    env = par.gas.EnvironmentData(
        temperature=np.array([TEMPERATURE]),
        pressure=np.array([PRESSURE]),
        saturation_ratio=np.zeros((1, 1)),
    )
    return par.Aerosol(particles=particles, gas=gas, environment=env)


def gas_kg(ppbv):
    return ppbv * 1e-9 * PRESSURE * MW[0] / (R * TEMPERATURE)


# %% [markdown]
# ## Give condensation and chemistry distinct material transfers
#
# Only kinetic condensation exchanges vapor with the dissolved pool. On this
# seed the initial salt molality exceeds 12 mol/kg water. Following Kampf et
# al. and the implementation in Knote et al., cap salting-in at 12, assign a
# bound/dissolved equilibrium ratio of 0.5 and use relaxation times of 44,000
# and 47,000 s. Divide total Henry capacity between the pools once. Dissolved
# masses count glyoxal equivalents, including rapid hydration implicitly.
# [Partitioning measurements](https://doi.org/10.1021/es400083d) ·
# [Two-pool parameterization, equations 1–4](https://doi.org/10.5194/acp-14-6213-2014)
#
# Convert the water-volume Henry law into an effective pressure for the native
# mole-fraction activity strategy at the beginning of each outer step. As a
# result the equilibrium surface pressure is dissolved moles divided by water
# volume and Henry capacity. Match the native initial condensation flux to the
# measured monomer relaxation time by solving for an effective accommodation
# coefficient. This coefficient includes condensed-phase resistance. Use a flat
# interface limit for this bulk-solution equilibrium comparison.
#
# The second pathway uses finite ammonium sulfate. Per mole of IC, two glyoxal
# and one salt formula unit produce IC, three waters and one sulfuric acid
# equivalent. The acid lane accounts for the sulfate and protons left when two
# ammonias are consumed. Prescribed pH controls the rate. Acid equivalents
# do not specify aqueous speciation. At each pH, derive the conditional rate
# from the published initial rate at 2 M ammonium, carry its activity
# coefficient as fixed, and extrapolate with the measured second orders in
# glyoxal and ammonium. Two ammoniums per salt and M-to-SI conversion enter
# the coefficient. This extrapolation to concentrated aerosol is uncertain.
# [Product kinetics and initial rates](https://doi.org/10.1021/es200989n)

# %%
SALT_MOLAL = 0.63 * DENSITY[2] / (MW[2] * 0.37 * DENSITY[4])
H_TOTAL = 4.19e5 * 10 ** (0.24 * min(12.0, SALT_MOLAL))  # M/atm
H_DISSOLVED = H_TOTAL / 1.5


def product_k(ph):
    rate, gly = YU[ph]
    # At 2 M NH4 the concentration of salt formula units is 1 M.
    return rate * 1e-8 / gly**2 / 1000**3


def chemistry(state, ph, enabled):
    record = json.loads((inputs() / "mechanism.json").read_text())
    record["reactions"][-1]["rate"]["k"] = product_k(ph) if enabled else 0.0
    mechanism = ReactionMechanism.from_dict(record)
    bound = mechanism.bind_resolved(
        state, {"particle:" + s: i for i, s in enumerate(LANES)}
    )
    return ResolvedChemistry(bound, ChemistrySolver(rtol=1e-7, atol=1e-12))


def condensation(state, alpha):
    mass = state.particles.masses[0, 0]
    water_liters = 1000 * mass[4] / DENSITY[4]
    reference_pa = PRESSURE * (mass / MW).sum() / (water_liters * H_DISSOLVED)
    return par.dynamics.CondensationIsothermal(
        molar_mass=MW[:1].copy(),
        accommodation_coefficient=alpha,
        activity_strategy=par.particles.ActivityIdealMolar(MW.copy()),
        vapor_pressure_strategy=par.gas.ConstantVaporPressureStrategy(
            reference_pa
        ),
        surface_strategy=par.particles.SurfaceStrategyVolume(
            surface_tension=1e-12, density=1000.0
        ),
        species_map=CondensationSpeciesMap(("glyoxal",), ((0, 0),)),
    )


def flux(state, alpha):
    return condensation(state, alpha).mass_transfer_rate(
        state.particles, state.gas, TEMPERATURE, PRESSURE
    )[0, 0]


def exchange_coefficient(state, ppbv):
    water_liters = state.particles.masses[0, 0, 4]  # water density = 1 kg/L
    expected = H_DISSOLVED * ppbv * 1e-9 * water_liters * MW[0] / 44000
    return brentq(lambda a: flux(state, a) / expected - 1, 1e-12, 1, xtol=1e-15)


def ledger(state):
    """Total kg and moles of C, H, N, O and S per m³ of air."""
    mass = state.particles.masses[0, 0] * COUNT
    gas = state.gas.concentration[0, 0]
    return np.r_[
        mass.sum() + gas, (mass / MW) @ ELEMENTS + gas / MW[0] * ELEMENTS[0]
    ]


# %% [markdown]
# ## Replay the chamber and close the forced budget
#
# Use linear interpolation between gas measurements. Prescribe the midpoint
# concentration for each interval, then let the finite gas and particles
# evolve together. Record the difference at each prescription as an external
# gas exchange. Reprepare the CUDA sequence at each interval to refresh the
# effective Henry pressure using the current composition. Every chemistry and
# condensation step on that backend runs on the GPU. Observation and forcing
# explicitly cross the device boundary.
#
# Report organic mass per sulfate mass and IC per initial sulfate. Use the
# same two 20-minute windows for measured and simulated organic decline, one
# immediately preceding dilution and one at the end. Retain the raw marker
# scatter and compare its direction with modeled product accumulation.


# %%
def decline(trace):
    t, y = np.transpose(trace)
    initial = np.median(y[(t >= DILUTION - 20) & (t <= DILUTION)])
    final = np.median(y[(t >= END - 20) & (t <= END)])
    return float(100 * (1 - final / initial))


def replay(gas, device, dt, ph=3.5, enabled=True, nitrogen=True, end=END):
    state = initial_state(np.interp(0, *gas.T), nitrogen)
    alpha = exchange_coefficient(state, np.interp(0, *gas.T))
    reaction = chemistry(state, ph, enabled)
    registry = par.ProcessRegistry()
    registry.register(ResolvedChemistry, ResolvedChemistryAdapter())
    baseline = ledger(state)
    sulfate = (state.particles.masses[0, 0, 2:4] / MW[2:4]).sum() * SULFATE_MW
    gas_unit = np.r_[1.0, ELEMENTS[0] / MW[0]]
    forced = 0.0
    supplied = removed = evaporated = 0.0
    worst = 0.0
    rows = []
    for step in range(round(end * 60 / dt) + 1):
        minutes = step * dt / 60
        mass = state.particles.masses[0, 0]
        if np.any(mass < 0) or state.gas.concentration.min() < 0:
            raise ValueError("Negative gas or particle inventory")
        if minutes % 5 == 0:
            rows.append(
                [
                    minutes,
                    mass[[0, 1, 5]].sum() / sulfate,
                    mass[5] / sulfate,
                    mass[0] / sulfate,
                    mass[1] / sulfate,
                    forced,
                    supplied,
                    removed,
                    evaporated,
                ]
            )
        balance = ledger(state) - baseline - forced * gas_unit
        scale = np.maximum(np.abs(baseline) + np.abs(forced * gas_unit), 1e-30)
        worst = max(worst, float(np.max(np.abs(balance) / scale)))
        if step == round(end * 60 / dt):
            break
        target = gas_kg(np.interp(minutes + dt / 120, *gas.T))
        delta = target - state.gas.concentration[0, 0]
        supplied += max(delta, 0.0)
        removed += max(-delta, 0.0)
        forced += delta
        state.gas.concentration[0, 0] = target
        transfer = par.dynamics.MassCondensation(condensation(state, alpha))
        if device == "numpy":
            transfer.execute(state, dt)
            reaction.execute(state, dt)
        else:
            with (transfer | reaction).prepare(
                state, device=device, time_step=dt, process_registry=registry
            ) as sim:
                sim.step()
                state = sim.snapshot()
        evaporated += max(state.gas.concentration[0, 0] - target, 0.0)
    if worst > 2e-10:
        raise AssertionError(f"Relative forced inventory error {worst}")
    return np.array(rows), worst, alpha


def independent_checks():
    """Test the Henry zero, evaporation and a held-out Yu laboratory rate."""
    state = initial_state(100.0)
    mass = state.particles.masses[0, 0]
    alpha = exchange_coefficient(state, 100.0)
    mass[0] = H_DISSOLVED * 100e-9 * mass[4] * MW[0]
    equilibrium_flux = flux(state, alpha)
    state.gas.concentration[0, 0] *= 0.5
    negative_flux = flux(state, alpha)
    assert negative_flux < 0 and abs(equilibrium_flux / negative_flux) < 1e-9
    before = mass[0]
    par.dynamics.MassCondensation(condensation(state, alpha)).execute(
        state, 60.0
    )
    assert state.particles.masses[0, 0, 0] < before

    # Independent initial-rate row: 1.5 M glyoxal and 1 M salt (2 M NH4).
    mass = state.particles.masses[0, 0]
    mass[:] = 0
    mass[[0, 2]] = np.array([1500.0, 1000.0]) * MW[[0, 2]] * VOLUME
    mass[4] = (VOLUME - (mass / DENSITY).sum()) * DENSITY[4]
    seconds = 0.1
    chemistry(state, 3.5, True).execute(state, seconds)
    rate = state.particles.masses[0, 0, 5] / (MW[5] * VOLUME * 1000 * seconds)
    assert abs(rate - 9.74e-8) <= 0.90e-8
    return {
        "IC_M_per_s": float(rate),
        "Henry_flux_kg_s": float(equilibrium_flux),
        "half_vapor_flux_kg_s": float(negative_flux),
    }


def simulate():
    """Qualify a replay using the same observation windows on both devices."""
    obs = observations()
    gas = obs["gas_ppbv"]
    ot, oy = obs["organic_per_sulfate"].T
    measured = decline(obs["organic_per_sulfate"])
    pre = np.median(oy[(ot >= DILUTION - 20) & (ot <= DILUTION)])
    post = np.median(oy[(ot >= END - 20) & (ot <= END)])
    # Conservative center-location allowance in each window, propagated to %.
    dy = 0.15 * 3.5 / (179.135986 - 77.244982)
    resolution = 100 * ((post + dy) / (pre - dy) - post / pre)
    report = {
        "observed": {k: v.tolist() for k, v in obs.items()},
        "measured_decline_pct": measured,
        "digitizing_pp": float(resolution),
        "checks": independent_checks(),
        "runs": {},
        "pH_sensitivity": {},
    }
    for device in ("numpy", "cuda:0"):
        backend = {}
        for name in ("reversible", "IC"):
            previous = None
            for dt in (300, 150):
                rows, err, alpha = replay(gas, device, dt, enabled=name == "IC")
                predicted = decline(
                    np.column_stack((ot, np.interp(ot, rows[:, 0], rows[:, 1])))
                )
                if (
                    previous is not None
                    and abs(predicted - previous) >= resolution
                ):
                    raise AssertionError(
                        f"{device} {name}: refine timestep further"
                    )
                if name == "reversible":
                    assert np.all(rows[:, 2] == 0)
                backend[f"{name}_{dt}"] = {
                    "decline_pct": predicted,
                    "balance": err,
                    "alpha": alpha,
                    "history": rows.tolist(),
                }
                previous = predicted
        zero, _, _ = replay(gas, device, 300, nitrogen=False, end=20)
        assert np.all(zero[:, 2] == 0) and zero[-1, 3] > 0
        report["runs"][device] = backend
    for ph in YU:
        if ph == 3.5:
            history = np.array(report["runs"]["numpy"]["IC_150"]["history"])
        else:
            history, _, _ = replay(gas, "numpy", 150, ph=ph)
        report["pH_sensitivity"][str(ph)] = float(history[-1, 2])
    for name in ("reversible_150", "IC_150"):
        cpu = np.array(report["runs"]["numpy"][name]["history"])
        gpu = np.array(report["runs"]["cuda:0"][name]["history"])
        diff = abs(
            report["runs"]["numpy"][name]["decline_pct"]
            - report["runs"]["cuda:0"][name]["decline_pct"]
        )
        assert diff < resolution
        assert np.max(np.abs(cpu[:, 1:3] - gpu[:, 1:3])) < 0.01
    return report


# %% [markdown]
# ## Compare the release and the persistent product
#
# The organic-to-sulfate panel compares dry organic mass with sulfate ion
# mass. Retain all digitized observations from seed injection onward. The
# chamber instrument did not supply a calibrated IC mass. Normalize the
# measured fragment and modeled IC separately to their medians just before
# dilution so that only the direction and timing can be compared.


# %%
def plot_results(report):
    obs = {k: np.array(v) for k, v in report["observed"].items()}
    fig, axes = plt.subplots(
        3, 1, figsize=(8, 8.5), sharex=True, layout="constrained"
    )

    def markers(ax, series, label):
        t, y = series.T
        ax.scatter(
            t / 60,
            y,
            s=4,
            color="#626569",
            alpha=0.5,
            label=label,
            rasterized=True,
        )

    markers(axes[0], obs["gas_ppbv"], "Figure 6 · prescribed")
    markers(axes[1], obs["organic_per_sulfate"], "Figure 6 · observed")
    for key, label, color in (
        ("reversible_150", "Reversible exchange", "#19717b"),
        ("IC_150", "Exchange and IC formation", "#a04c22"),
    ):
        h = np.array(report["runs"]["numpy"][key]["history"])
        axes[1].plot(h[:, 0] / 60, h[:, 1], label=label, color=color, lw=2)
    marker = obs["marker68_per_sulfate"].copy()
    t, y = marker.T
    marker[:, 1] /= np.median(y[(t >= DILUTION - 20) & (t <= DILUTION)])
    markers(axes[2], marker, "Figure 6 · m/z 68")
    t, p = h[:, 0], h[:, 2]
    axes[2].plot(
        t / 60,
        p / np.median(p[(t >= DILUTION - 20) & (t <= DILUTION)]),
        color="#a04c22",
        label="Modeled IC",
    )
    for ax, label in zip(
        axes,
        (
            "Gas glyoxal (ppbv)",
            "Organic / sulfate (kg/kg)",
            "Relative to pre-dilution",
        ),
        strict=True,
    ):
        ax.axvline(DILUTION / 60, color="#48494b", ls="--", lw=1)
        ax.set_ylabel(label)
        ax.spines[["right", "top"]].set_visible(False)
        ax.legend(frameon=False, loc="best", fontsize=8)
    axes[0].set_title(
        "Glyoxal in a dark chamber with wet ammonium sulfate", loc="left"
    )
    axes[0].text(
        DILUTION / 60 + 0.25,
        0.08,
        "Dilution",
        transform=axes[0].get_xaxis_transform(),
    )
    axes[-1].set(xlabel="Time since seed injection (h)", xlim=(0, END / 60))
    return fig


def summarize(report):
    print(
        f"Organic/sulfate decline over matched windows (positive = loss): "
        f"measured {report['measured_decline_pct']:.2f}%."
    )
    for device, runs in report["runs"].items():
        for key in ("reversible_150", "IC_150"):
            run = runs[key]
            print(
                f"{device}, {key}: {run['decline_pct']:.2f}% decline; "
                f"maximum relative inventory error {run['balance']:.2g}."
            )
    print(
        "Final IC / sulfate (kg/kg), prescribed pH:", report["pH_sensitivity"]
    )
    print(
        f"Digitization allowance: {report['digitizing_pp']:.3f} percentage points."
    )


# %% [markdown]
# ## What the comparison tells us
#
# Compare the signs of the printed declines. The chamber loses particle
# organics after dilution while the measured nitrogen-containing marker
# persists. A timescale inferred from a separate sulfate study can leave the
# simulated particles still filling when dilution starts. The comparison
# tests that prediction directly. The IC signal is especially sensitive to
# the prescribed pH and ammonium activity. A measurement of particle acidity
# together with a calibrated product abundance would test that extrapolation.

# %%
if __name__ == "__main__":
    results = simulate()
    summarize(results)
    plot_results(results)
    plt.show()
