# %% [markdown]
# # Water and organic exchange in a closed box
#
# Can one droplet gain water while releasing a dissolved organic? At 298 K,
# the answer depends on two independent vapor pressure differences. Here we
# place water 2% above its droplet equilibrium pressure and 1,4-butanediol 20%
# below its own, and repeat with the offsets reversed. Each vapor has a finite
# reservoir. Vanillic acid and sodium chloride stay in the particle and affect
# both equilibrium pressures. From the repository root, install the model
# and plotting dependencies with
# `python -m pip install ./packages/particula-aiomfac ".[aiomfac,plot]"`, then run
# `python docs/studies/examples/thermodynamics/activity/aiomfac_closed_box.py`.
# The complete solver comparison requires an NVIDIA GPU.
#
# ## Define one mixture and two gas reservoirs
#
# The mixture and neutral activities come from the 298 K row of pinned
# AIOMFAC-web v3.15 example 0008. We use the independent Fortran outputs
# only to initialize the reservoirs and check the first flux. The temperature,
# density, surface tension, accommodation and pure vapor pressures are fixed
# benchmark inputs. In particular, 2 Pa for butanediol is prescribed for this
# numerical experiment. The subsequent integration recalculates activities.
# [Pinned mixture input](https://github.com/andizuend/AIOMFAC/blob/24041a1b0db090de1285008814a5285b4b9b746b/Examples_Input/input_0008.txt).
# %%
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
from particula.aerosol import Aerosol
from particula.dynamics.condensation.condensation_strategies import (
    CondensationIsothermal,
)
from particula.dynamics.condensation.species_mapping import (
    CondensationSpeciesMap,
)
from particula.gas import GasData
from particula.gas.environment_data import EnvironmentData
from particula.gas.properties.mean_free_path import get_molecule_mean_free_path
from particula.gas.vapor_pressure_strategies import (
    ConstantVaporPressureStrategy,
)
from particula.particles import ParticleData
from particula.particles.activity_factories import ActivityFactory
from particula.particles.surface_strategies import SurfaceStrategyVolume
from particula.util.constants import GAS_CONSTANT
from scipy.integrate import solve_ivp

GROUPS = {
    "water": {16: 1},
    "vanillic_acid": {9: 3, 10: 2, 17: 1, 24: 1, 137: 1},
    "butanediol": {142: 2, 150: 2, 153: 2},
    "NaCl": {202: 1, 242: 1},
}
FRACTIONS = dict(zip(GROUPS, (0.977, 0.001, 0.002, 0.020), strict=True))
VAPORS = ("water", "butanediol")
FORTRAN_ACTIVITY = np.array([0.9878567852129456, 0.0011898857970537608])
PURE_PRESSURE = np.array([3167.0, 2.0])  # Pa
MOLAR_MASS = np.array([0.01801528, 0.090121])  # kg/mol
DIFFUSIVITY = np.array([2.5e-5, 8.0e-6])  # m²/s
TEMPERATURE, PRESSURE = 298.0, 101325.0  # K, Pa
DENSITY, SIGMA = 1000.0, 0.072  # kg/m³, N/m
TIMES = np.linspace(0.0, 0.004, 9)  # s


@dataclass
class Box:
    """Native state and ordered transfer indices for one isothermal box."""

    aerosol: Aerosol
    process: CondensationIsothermal
    sources: list
    targets: list
    equilibrium: np.ndarray


def make_box(direction=1, permuted=False, *, table=None):
    """Prepare native state with gas and material maps in explicit orders."""
    from particula_aiomfac import MixtureSpec

    materials = list(GROUPS)
    gases = list(VAPORS)
    if permuted:
        materials = ["NaCl", "inert", "butanediol", "water", "vanillic_acid"]
        gases = ["butanediol", "inert", "water"]
    targets = [materials.index(k) for k in VAPORS]
    sources = [gases.index(k) for k in VAPORS]
    material_mass = (
        np.array([FRACTIONS.get(k, 0.20) for k in materials]) * 5e-15
    )
    masses = material_mass[None, :].copy()
    if permuted:
        second = material_mass.copy()
        second[materials.index("NaCl")] *= 2
        masses = np.stack((material_mass, second, np.zeros_like(second)))
    counts = np.where(masses.sum(axis=1) > 0, 1e10, 0.0)
    p = ParticleData(
        masses=masses[None, :, :],
        concentration=counts[None, :],
        charge=np.zeros((1, len(counts))),
        density=np.full(len(materials), DENSITY),
        volume=np.ones(1),
        distribution_type="particle_resolved",
    )
    kelvin = np.exp(
        2
        * SIGMA
        * MOLAR_MASS
        / (DENSITY * GAS_CONSTANT * TEMPERATURE * p.radii[0, 0])
    )
    equilibrium = FORTRAN_ACTIVITY * PURE_PRESSURE * kelvin
    eps = direction * np.array([0.02, -0.20])
    gas_mass = np.zeros(len(gases))
    gas_mass[sources] = (
        (1 + eps) * equilibrium * MOLAR_MASS / (GAS_CONSTANT * TEMPERATURE)
    )
    mw = np.full(len(gases), 0.029)
    diffusivity = np.full(len(gases), 1e-5)
    pure = np.zeros(len(gases))
    mw[sources], diffusivity[sources], pure[sources] = (
        MOLAR_MASS,
        DIFFUSIVITY,
        PURE_PRESSURE,
    )
    gas = GasData(
        name=gases,
        concentration=gas_mass[None, :],
        molar_mass=mw,
        partitioning=np.isin(gases, VAPORS),
    )
    environment = EnvironmentData(
        temperature=np.array([TEMPERATURE]),
        pressure=np.array([PRESSURE]),
        saturation_ratio=np.zeros((1, len(gases))),
    )
    provider = ActivityFactory().get_strategy(
        "aiomfac",
        {
            **(
                {"evaluation": "hybrid", "table": table}
                if table is not None
                else {}
            ),
            "mixture": MixtureSpec(GROUPS),
            "material_mapping": {
                i: k for i, k in enumerate(materials) if k in GROUPS
            },
            "inert_indices": [
                i for i, k in enumerate(materials) if k == "inert"
            ],
        },
    )
    process = CondensationIsothermal(
        molar_mass=mw,
        diffusion_coefficient=diffusivity,
        accommodation_coefficient=1.0,
        vapor_pressure_strategy=ConstantVaporPressureStrategy(pure),
        activity_strategy=provider,
        surface_strategy=SurfaceStrategyVolume(
            density=DENSITY, surface_tension=SIGMA
        ),
        species_map=CondensationSpeciesMap(
            tuple(gases), tuple(zip(sources, targets, strict=True))
        ),
        update_gases=True,
    )
    return Box(
        Aerosol(particles=p, gas=gas, environment=environment),
        process,
        sources,
        targets,
        equilibrium,
    )


# %% [markdown]
# ## Check the first pressure and flux
#
# For each vapor, multiply its pure vapor pressure by the stored neutral
# activity and the species-specific Kelvin factor. Compute the expected
# diffusion-limited mass rate from that pressure difference and the
# Fuchs–Sutugin transition correction at unit accommodation. The resulting
# signs distinguish condensation from evaporation. In the permuted layout,
# the first slot still contains the reference mixture plus an inert material.
# %%
def first_flux(box):
    """Return independent pressure and flux errors on the reference slot."""
    p, g = box.aerosol.particles, box.aerosol.gas
    rate = box.process.mass_transfer_rate(p, g, TEMPERATURE, PRESSURE)[
        0, box.targets
    ]
    radius = np.maximum(p.radii[0], 1e-10)
    surface = box.process._native_pressure_terms(
        p, g, TEMPERATURE, radius, box.sources, box.targets
    )[1][0, box.targets]
    gas_pressure = (
        g.concentration[0, box.sources]
        * GAS_CONSTANT
        * TEMPERATURE
        / MOLAR_MASS
    )
    kn = (
        get_molecule_mean_free_path(
            molar_mass=MOLAR_MASS, temperature=TEMPERATURE, pressure=PRESSURE
        )
        / radius[0]
    )
    correction = 0.75 * (1 + kn) / (kn**2 + 1.283 * kn + 0.75)
    expected = (
        4
        * np.pi
        * radius[0]
        * DIFFUSIVITY
        * correction
        * (gas_pressure - box.equilibrium)
        * MOLAR_MASS
        / (GAS_CONSTANT * TEMPERATURE)
    )
    np.testing.assert_allclose(surface, box.equilibrium, rtol=2e-12, atol=0)
    np.testing.assert_allclose(rate, expected, rtol=1e-9, atol=0)
    return {
        "equilibrium_pressure_Pa": box.equilibrium.tolist(),
        "pressure_error_Pa": (surface - box.equilibrium).tolist(),
        "flux_error_kg_s": (rate - expected).tolist(),
        "flux_kg_s": rate.tolist(),
    }


# %% [markdown]
# ## Integrate and audit both reservoirs
#
# The independent time integrator advances only the exchanging per-particle
# masses. At each Radau trial it reconstructs gas from each species' fixed
# inventory and calls the native CPU rate. In the production comparisons
# Particula owns both updates. Observations record gas and particle kg/m³,
# wet radius, dimensionless activities and instantaneous kg/s per droplet.
# %%
def sample(box, aerosol):
    p, g = aerosol.particles, aerosol.gas
    active = p.concentration[0] > 0
    rate = box.process.mass_transfer_rate(p, g, TEMPERATURE, PRESSURE)[
        :, box.targets
    ]
    activity = box.process.activity_strategy.neutral_activities(
        p.masses[0], TEMPERATURE, box.targets, active
    )
    return dict(
        gas=g.concentration[0, box.sources].copy(),
        particle=p.species_mass_density[0, box.targets].copy(),
        flux=rate.copy(),
        activity=activity.copy(),
        radius=p.radii[0].copy(),
        mass=p.masses[0].copy(),
    )


def radau(box, rtol=2e-10):
    p, g = box.aerosol.particles, box.aerosol.gas
    active = np.flatnonzero(p.concentration[0] > 0)
    lanes = np.ix_(active, box.targets)
    start = p.masses[0][lanes].copy()
    counts = p.concentration[0, active] / p.volume[0]
    gas0 = g.concentration[0, box.sources].copy()

    def unpack(y):
        value = y.reshape(start.shape) * start
        p.masses[0][lanes] = value
        g.concentration[0, box.sources] = gas0 - counts @ (value - start)

    def rhs(_, y):
        unpack(y)
        rate = box.process.mass_transfer_rate(p, g, TEMPERATURE, PRESSURE)[
            lanes
        ]
        return (rate / start).ravel()

    solution = solve_ivp(
        rhs,
        (0, TIMES[-1]),
        np.ones(start.size),
        t_eval=TIMES,
        method="Radau",
        rtol=rtol,
        atol=1e-13,
    )
    if not solution.success:
        raise ArithmeticError(solution.message)
    records = []
    for y in solution.y.T:
        unpack(y)
        records.append(sample(box, box.aerosol))
    return records


def native(box, steps_per_sample):
    result = [sample(box, box.aerosol)]
    p, g = box.aerosol.particles, box.aerosol.gas
    for interval in np.diff(TIMES):
        for _ in range(steps_per_sample):
            box.process.step(
                p,
                g,
                TEMPERATURE,
                PRESSURE,
                time_step=interval / steps_per_sample,
            )
        result.append(sample(box, box.aerosol))
    return result


def gpu_run(box, solver, steps_per_sample=8):
    """Observe exactly the same physical output times across CUDA solvers."""
    import warp as wp
    from particula.dynamics import MassCondensation
    from particula.execution.gpu_condensation_ode import GPUCondensationODE
    from particula.execution.gpu_resident_condensation import (
        GPUResidentCondensation,
    )

    device = wp.get_device("cuda:0")
    result = [sample(box, box.aerosol)]
    interval_times = []
    start = perf_counter()
    if solver == "fixed":
        with MassCondensation(box.process).prepare(
            box.aerosol,
            device=device.alias,
            time_step=TIMES[1] / steps_per_sample,
        ) as owner:
            owner.capture()
            wp.synchronize_device(device)
            setup = perf_counter() - start
            for _ in TIMES[1:]:
                start = perf_counter()
                owner.run(steps_per_sample)
                wp.synchronize_device(device)
                interval_times.append(perf_counter() - start)
                result.append(sample(box, owner.snapshot()))
            stats = dict(
                accepted=owner.completed_steps,
                rejected=0,
                activity_evaluations=5 * owner.completed_steps,
            )
            provider = owner._bindings["condensation"].prepared.provider
            if getattr(provider, "hybrid", False):
                stats["lookup_counts"] = provider.counts.numpy().tolist()
    else:
        driver, method = solver.split("-")
        cls = {"host": GPUCondensationODE, "resident": GPUResidentCondensation}[
            driver
        ]
        owner = cls(
            box.aerosol,
            box.process,
            device=device.alias,
            method=method,
            rtol=1e-7,
            particle_atol=1e-29,
            gas_atol=1e-16,
            first_step=1e-5,
            max_step=TIMES[1],
        )
        wp.synchronize_device(device)
        setup = perf_counter() - start
        # Graph warmup uses the rate workspace at trial compositions. Sample
        # the production kernel explicitly at the accepted initial state.
        owner._rate(owner.time, owner._state, owner._rates[0])
        initial_rate = owner._mass_rate.numpy()[0][:, box.targets]
        np.testing.assert_allclose(
            initial_rate, result[0]["flux"], rtol=1e-8, atol=1e-27
        )
        for bound in TIMES[1:]:
            start = perf_counter()
            while owner.time < bound:
                owner.step(float(bound))
            wp.synchronize_device(device)
            interval_times.append(perf_counter() - start)
            result.append(sample(box, owner.snapshot()))
        stats = asdict(owner.statistics)
        if getattr(owner._provider, "hybrid", False):
            stats["lookup_counts"] = owner._provider.counts.numpy().tolist()
        stats["activity_evaluations"] = stats["rate_calls"]
        stats["control_readbacks"] = owner.control_readbacks
        stats["initial_rate_downloads"] = 1
        stats["initial_flux_error_kg_s"] = (
            initial_rate - result[0]["flux"]
        ).tolist()
    stats.update(
        setup_s=setup,
        advance_s=sum(interval_times),
        first_interval_s=interval_times[0],
        warmed_advance_s=sum(interval_times[1:]),
        warmed_physical_duration_s=TIMES[-1] - TIMES[1],
        explicit_snapshots=len(result) - 1,
        device=device.name,
        warp_version=wp.__version__,
        process_peak_pool_bytes=wp.get_mempool_used_mem_high(device),
    )
    return result, stats


def qualify_restart(method, device="cuda:0", *, table=None):
    """Force reversed gas offsets between intervals and resume an archived cycle."""
    from particula.dynamics import Dilution, DilutionStrategy
    from particula.execution.gpu_coupled import (
        GPUCondensationControls,
        GPUCoupledSimulation,
    )
    from particula.execution.gpu_coupled_restart import (
        load_coupled_checkpoint,
        resume_coupled,
        save_coupled_checkpoint,
    )

    box = make_box(table=table)
    interval = 1e-6
    # Zero removal preserves the finite box budget across the event grid.
    events = (Dilution(DilutionStrategy(0.0)),)
    controls = GPUCondensationControls(
        method=method,
        rtol=1e-8,
        particle_atol=1e-30,
        gas_atol=1e-16,
        first_step=interval,
        max_step=interval,
    )
    with GPUCoupledSimulation(
        box.aerosol,
        events,
        box.process,
        time_step=interval,
        device=device,
        controls=controls,
        resident=True,
    ) as owner:
        owner.capture_events()
        owner.step()
        before = owner.snapshot()
        prior = sample(box, before)
        changed_gas = make_box(-1).aerosol.gas.concentration
        owner.update_gas(changed_gas)
        snapshot = owner.snapshot()
        changed = sample(box, snapshot)
        np.testing.assert_array_equal(
            np.sign(changed["flux"]), -np.sign(prior["flux"])
        )
        with TemporaryDirectory() as tmp:
            archive = Path(tmp) / "water-organic.checkpoint"
            save_coupled_checkpoint(owner.checkpoint(), archive)
            saved = load_coupled_checkpoint(archive)
        try:
            invalid = resume_coupled(
                saved, events, make_box(permuted=True).process, device=device
            )
        except ValueError as error:
            if "condensation physics differs" not in str(error):
                raise
        else:
            invalid.close()
            raise AssertionError(
                "Restart accepted incompatible exchange mapping"
            )
        with resume_coupled(
            saved, events, box.process, device=device
        ) as resumed:
            resumed.capture_events()
            owner.step()
            resumed.step()
            actual = owner.snapshot()
            continued = resumed.snapshot()
            np.testing.assert_array_equal(
                actual.particles.masses, continued.particles.masses
            )
            np.testing.assert_array_equal(
                actual.gas.concentration, continued.gas.concentration
            )
            if (
                owner.time != resumed.time
                or owner.statistics != resumed.statistics
            ):
                raise AssertionError(
                    "Different accepted clock or work after restart"
                )
            average_rate = (
                actual.particles.masses[0, 0, box.targets]
                - snapshot.particles.masses[0, 0, box.targets]
            ) / interval
            np.testing.assert_allclose(
                average_rate, changed["flux"][0], rtol=2e-4, atol=0.0
            )
            return dict(
                method=method,
                continued_to_s=owner.time,
                post_event_relative_flux_error=(
                    average_rate / changed["flux"][0] - 1
                ).tolist(),
                statistics=asdict(owner.statistics),
                coupling_downloads=owner.coupling_downloads,
                coupling_uploads=owner.coupling_uploads,
            )


def compare(records, reference, targets):
    """Compare independent trajectories and per-species physical ownership."""
    initial = records[0]
    total = initial["gas"] + initial["particle"]
    scale = np.maximum(total, 1e-30)  # Absolute floor in kg/m³.
    dry = np.delete(np.arange(initial["mass"].shape[-1]), np.sort(targets))
    residual = phase_error = radius_error = 0.0
    for got, want in zip(records, reference, strict=True):
        if any(not np.isfinite(value).all() for value in got.values()):
            raise AssertionError("Nonfinite state in closed-box trajectory")
        residual = max(
            residual,
            float(np.max(abs(got["particle"] + got["gas"] - total) / scale)),
        )
        for phase in ("particle", "gas"):
            phase_error = max(
                phase_error,
                float(np.max(abs(got[phase] - want[phase]) / scale)),
            )
        radius_error = max(
            radius_error,
            float(
                np.max(
                    abs(got["radius"] - want["radius"])
                    / np.maximum(want["radius"], 1e-30)
                )
            ),
        )
        np.testing.assert_array_equal(
            got["mass"][:, dry], initial["mass"][:, dry]
        )
    profile_errors = {}
    for quantity, floor in (
        ("activity", 1e-12),
        ("flux", 1e-26),
        ("mass", 1e-30),
        ("gas", 1e-30),
        ("particle", 1e-30),
    ):
        actual = np.stack([r[quantity] for r in records])
        oracle = np.stack([r[quantity] for r in reference])
        # Per slot and per species, normalize to its own peak over time.
        norm = np.maximum(np.max(np.abs(oracle), axis=0), floor)
        profile_errors[quantity] = float(np.max(np.abs(actual - oracle) / norm))
    if any(
        error > (5e-3 if quantity == "flux" else 1e-3)
        for quantity, error in profile_errors.items()
    ):
        raise AssertionError(f"Trajectory differs from Radau: {profile_errors}")
    if residual > 1e-10 or phase_error > 1e-3 or radius_error > 1e-3:
        raise AssertionError(
            f"budget={residual}, phase={phase_error}, radius={radius_error}"
        )
    return dict(
        budget=residual,
        phase=phase_error,
        radius=radius_error,
        profiles=profile_errors,
    )


def simulate(*, cuda=True, permuted=False):
    report = {"time_s": TIMES.tolist(), "reservoirs": []}
    for direction in (1, -1):
        box = make_box(direction, permuted)
        errors = first_flux(box)
        ref = radau(box)
        metrics = {}
        for n in (4, 8):
            records = native(make_box(direction, permuted), n)
            metrics[f"cpu-fixed-{n}"] = compare(records, ref, box.targets)
        if metrics["cpu-fixed-8"]["radius"] >= metrics["cpu-fixed-4"]["radius"]:
            raise AssertionError("Wet radius failed timestep refinement")
        if cuda:
            for solver in (
                "fixed",
                "host-rk23",
                "host-ros3",
                "resident-rk23",
                "resident-ros3",
            ):
                records, work = gpu_run(make_box(direction, permuted), solver)
                metrics[solver] = {**compare(records, ref, box.targets), **work}
        report["reservoirs"].append(
            dict(
                direction=direction,
                initial=errors,
                metrics=metrics,
                reference={
                    key: np.array([r[key] for r in ref]).tolist()
                    for key in ("gas", "particle", "radius", "activity", "flux")
                },
            )
        )
    if cuda:
        report["restart"] = [
            qualify_restart(method) for method in ("rk23", "ros3")
        ]
    return report


# %% [markdown]
# ## Follow both fluxes and inventories
#
# Each column follows one vapor along the CPU Radau reference trajectory. Positive flux adds particle mass. The lower
# row displays changes in the gas and particle reservoirs in the same units,
# so opposite curves indicate conservation. Blue and orange distinguish the
# two initial pressure offsets. Markers repeat the offset encoding. The printed
# summary lists maximum budget and trajectory errors over the integration.
# The saved data also contain each process-wide Warp memory-pool high water
# mark (bytes), adaptive scalar control readbacks and explicitly requested full
# state snapshots. Timed advances exclude those snapshots. The warmed time
# omits the first observation interval, including first replay overhead. A fixed step makes
# five activity evaluations: four flux substeps and a final accepted-state check.
# %%
def plot_results(report):
    """Draw per-droplet fluxes and finite-reservoir inventory changes."""
    import matplotlib.pyplot as plt

    t = np.array(report["time_s"]) * 1e3
    fig, axes = plt.subplots(
        2, 2, figsize=(9, 6), sharex=True, layout="constrained"
    )
    for column, label in enumerate(("Water", "1,4-butanediol")):
        for entry, color, marker in zip(
            report["reservoirs"],
            ("tab:blue", "tab:orange"),
            ("o", "s"),
            strict=True,
        ):
            data = entry["reference"]
            flux = np.asarray(data["flux"])[:, 0, column] * 1e15
            sign = "condensing" if flux[0] > 0 else "evaporating"
            axes[0, column].plot(
                t,
                flux,
                color=color,
                marker=marker,
                markersize=4,
                markevery=2,
                label=f"Initially {sign}",
            )
            for phase, style in (("particle", "-"), ("gas", "--")):
                values = np.asarray(data[phase])[:, column]
                axes[1, column].plot(
                    t,
                    (values - values[0]) * 1e9,
                    ls=style,
                    marker=marker,
                    markersize=3,
                    markevery=2,
                    color=color,
                    label=phase if color == "tab:blue" else None,
                )
        for ax in axes[:, column]:
            ax.axhline(0, color="0.6", linewidth=0.6)
            ax.legend(fontsize=8)
        axes[0, column].set_title(label)
        axes[1, column].set_xlabel("Elapsed time (ms)")
    axes[0, 0].set_ylabel("Mass flux per droplet (pg/s)")
    axes[1, 0].set_ylabel("Inventory change (µg/m³)")
    return fig


def run_example():
    report = simulate()
    for case in report["reservoirs"]:
        print("Reservoir offsets", case["direction"], "× [+2%, −20%]")
        print(
            "Initial pressure errors (Pa)", case["initial"]["pressure_error_Pa"]
        )
        print("Solver              max budget     max inventory    max radius")
        for solver, m in case["metrics"].items():
            print(
                f"{solver:18s}  {m['budget']:10.3e}     {m['phase']:10.3e}     {m['radius']:10.3e}"
            )
    plot_results(report)
    return report


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    run_example()
    plt.show()


# %% [markdown]
# With the water reservoir 2% above equilibrium, each reference droplet gains
# water at 143 pg/s initially while releasing butanediol at 0.00181 pg/s.
# The two lower curves for each pressure offset should mirror each other as
# mass leaves one reservoir and appears in the other. The Fortran activities
# constrain the starting pressure only. Tracking the later radius and both
# phase inventories tests the numerical treatment of the changing mixture.
# Varying the salt loading at fixed organic abundance would distinguish the
# electrolyte contribution to the two fluxes.
