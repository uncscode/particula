# %% [markdown]
# # Water and organic uptake in a rising parcel
#
# A warm parcel carries two droplet classes and finite reservoirs of water
# vapor and 1,4-butanediol. Both gases can change the droplet composition and
# release latent heat. We lift the parcel 150 m twice. In the second ascent
# we disable organic exchange while keeping its initial condensed mass and
# vapor loading. Comparing the two ascents isolates this organic transfer.
# We then start a separate parcel at 82% relative humidity and follow it
# 1,500 m through cloud formation, tracking water saturation and activation
# of each nucleus class. Both ascents are controlled numerical experiments.
# From the repository root, install both local projects with
# `python -m pip install ./packages/particula-aiomfac ".[aiomfac,plot]"`, then run
# `python docs/studies/examples/thermodynamics/parcel-experiments/aiomfac_multivapor_parcel.py`.
# The full example runs on a CUDA device with conditional graph support.
# [Pinned analytical mixture](https://github.com/andizuend/AIOMFAC/blob/24041a1b0db090de1285008814a5285b4b9b746b/Examples_Input/input_0008.txt)
#
# ## Define the wet particles and gas reservoirs
#
# Each class starts at the recorded water/vanillic acid/butanediol/NaCl
# mass fractions. Initial wet radii are 1 and 2.5 µm. The two classes
# represent 50 million and 20 million droplets in one kg of dry air. The benchmark prescribes additive component volumes with
# densities of 1000 kg/m³, surface tension of 0.072 N/m, unit accommodation,
# and continuum diffusivities of 2.11e-5 and 7e-6 m²/s at 273.15 K and
# 101325 Pa. These diffusivities scale with temperature and inverse pressure.
#
# The organic reference pressure is 0.73 Pa at 298.4 K from Toktonov,
# Table B4 (transpiration data from 298.4 to 349.3 K). We continue its
# measured vaporization enthalpy of 79.01 kJ/mol as a constant over this
# short ascent, including a small extrapolation below the first data point.
# The water pressure is prescribed at 3169 Pa at 298.15 K, with constant
# latent heat 2.5 MJ/kg. The short ascent stays near 297–298 K. The
# 1,500 m ascent extends the constant-property and vapor-pressure laws to
# 289.6 K, about 9 K below the lowest measured butanediol pressure.
# The calculation keeps one aqueous liquid phase and extrapolates a
# subcooled pure-liquid organic reference below its fusion temperature
# (NIST reports a fusion enthalpy at 293.58 K).
# [Pressure measurements](https://rosdok.uni-rostock.de/file/rosdok_disshab_0000000219/rosdok_derivate_0000003844/Dissertation_Toktonov_2009.pdf#page=122)
# and [NIST phase-change data](https://webbook.nist.gov/cgi/cbook.cgi?ID=C110634&Mask=4).
#
# The water vapor starts at 99.5% and the organic at 120% of equilibrium
# over the larger class. These offsets refer to the activity- and curvature-
# corrected surface pressures. An ideal gas mixture converts the two prescribed
# partial pressures into kg vapor per kg dry air.

# %%
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
from particula.dynamics.multivapor_parcel import (
    MultiVaporParcelInitialState,
    ParcelVapor,
    multivapor_trajectory,
)
from particula.dynamics.parcel import ParcelRK23, ParcelROS3, run_parcel
from particula.execution.multivapor_parcel import (
    ParcelReplayCheckpoint,
    ResidentMultiVaporParcel,
)
from particula.particles.activity_factories import ActivityFactory
from particula.util.constants import GAS_CONSTANT, MOLECULAR_WEIGHT_AIR
from particula_aiomfac import MixtureSpec


def initial_parcel():
    """Construct a reproducible wet two-class parcel with finite reservoirs."""
    groups = {
        "water": {16: 1},
        "vanillic_acid": {9: 3, 10: 2, 17: 1, 24: 1, 137: 1},
        "butanediol": {142: 2, 150: 2, 153: 2},
        "NaCl": {202: 1, 242: 1},
    }
    names = tuple(groups)
    density = np.full(4, 1000.0)
    fractions = np.array([0.977, 0.001, 0.002, 0.020])
    radius = np.array([1.0, 2.5]) * 1e-6
    masses = (4 * np.pi / 3 * radius**3 / (fractions @ (1 / density)))[
        :, None
    ] * fractions
    gases = (
        ParcelVapor("water", 0, 0.01801528, 3169.0, 298.15, 2.5e6, 2.11e-5),
        ParcelVapor(
            "butanediol", 2, 0.090121, 0.73, 298.4, 79010 / 0.090121, 7e-6
        ),
    )
    strategy = ActivityFactory().get_strategy(
        "aiomfac",
        {
            "mixture": MixtureSpec(groups),
            "material_mapping": dict(enumerate(names)),
        },
    )
    temperature = 298.15
    pressure = 101325.0
    targets = [v.material_index for v in gases]
    activity = strategy.neutral_activities(
        masses, temperature, targets, np.ones(2, bool)
    )[-1]
    gas_r = GAS_CONSTANT / np.array([v.molar_mass for v in gases])
    saturation = np.array(
        [
            v.reference_pressure
            * np.exp(
                v.latent_heat
                / gas_r[j]
                * (1 / v.reference_temperature - 1 / temperature)
            )
            for j, v in enumerate(gases)
        ]
    )
    kelvin = np.exp(
        2 * 0.072 / (gas_r * temperature * density[targets] * radius[-1])
    )
    partial = np.array([0.995, 1.2]) * saturation * kelvin * activity
    mixing = (
        (GAS_CONSTANT / MOLECULAR_WEIGHT_AIR)
        / gas_r
        * partial
        / (pressure - sum(partial))
    )
    return MultiVaporParcelInitialState(
        masses=masses,
        number=np.array([5e7, 2e7]),
        vapor=mixing,
        material_names=names,
        density=density,
        vapors=gases,
        activity_strategy=strategy,
        pressure=pressure,
        temperature=temperature,
    )


# %% [markdown]
# ## Integrate and check the inventories
#
# The hydrostatic parcel expands as it rises at 1 m/s. At every numerical
# trial we use the current temperature and wet composition for activities.
# Diffusive fluxes share a surface heat balance and warm or cool the air
# through their summed latent heat. Check the independently evolved gas
# and particle reservoirs for each vapor. Reconstruct expansion work from
# sampled specific volume and pressure, and the latent heat from the change
# in each condensed inventory. The resulting temperature residual checks
# the first law independently of the integrator's error estimate.
#
# CUDA advances the same parcel with RK23 and ROS3 at two tolerances. Both
# solvers choose their trial steps inside a device graph. At 75 m we archive
# the accepted state, recapture a fresh graph, and compare every remaining
# sample with its uninterrupted counterpart. Only observation and restart
# boundaries transfer state to the host.


# %%
def first_law_error(history, stride=1):
    """Temperature-equivalent residual of independent output quadrature (K)."""
    v = history.volume[::stride] / history.initial.dry_air_mass
    work = np.r_[
        0.0,
        np.cumsum(0.5 * (v[:-1] + v[1:]) * np.diff(history.pressure[::stride])),
    ]
    latent = np.array([s.latent_heat for s in history.initial.vapors])
    return (
        history.temperature[::stride]
        - history.temperature[0]
        - (work + (history.liquid[::stride] - history.liquid[0]) @ latent)
        / 1005.0
    )


def _summary(history):
    inventory = history.vapor + history.liquid
    residual = np.abs(inventory - inventory[0]) / np.maximum(
        inventory[0], 1e-30
    )
    if np.max(residual) > 1e-10:
        raise RuntimeError("multi-vapor parcel failed a species budget")
    return {
        "height_m": history.height.tolist(),
        "temperature_K": history.temperature.tolist(),
        "supersaturation_pct": (100 * history.supersaturation).tolist(),
        "water_g_kg": (1e3 * history.vapor[:, 0]).tolist(),
        "organic_ug_kg": (1e9 * history.vapor[:, 1]).tolist(),
        "radius_um": (1e6 * history.wet_radius).tolist(),
        "max_relative_inventory_error": np.max(residual, axis=0).tolist(),
        "max_first_law_error_K": float(np.max(abs(first_law_error(history)))),
    }


def cuda_comparison(
    reference, control, device, restart, return_trajectory=False
):
    """Run device-controlled segments, comparing each output with Radau."""
    import warp as wp

    initial = reference.initial

    def prepare():
        return ResidentMultiVaporParcel(
            initial, device=device, adaptive=control, timestep=1.0, updraft=1.0
        )

    begin = perf_counter()
    owner = prepare()
    setup_s = perf_counter() - begin
    initial_state = (
        initial.pressure,
        initial.temperature,
        initial.vapor,
        initial.masses,
    )
    samples = [initial_state]
    duplicate = None
    for i, stop in enumerate(reference.time[1:], start=1):
        sample = owner.advance(stop)
        samples.append(sample)
        if duplicate is not None:
            resumed = duplicate.advance(stop)
            for x, y in zip(sample, resumed, strict=True):
                np.testing.assert_array_equal(x, y)
        if restart and i == len(reference.time) // 2:
            with TemporaryDirectory() as directory:
                archive = Path(directory) / "parcel-restart.zip"
                owner.checkpoint().save(archive)
                duplicate = prepare()
                duplicate.restore(ParcelReplayCheckpoint.load(archive))
    if duplicate is not None:
        assert owner.last_counts == duplicate.last_counts
    p, t, q, m = zip(*samples, strict=True)
    hist = multivapor_trajectory(initial, reference.time, p, t, q, m, 1.0)
    total = reference.vapor[0] + reference.liquid[0]
    errors = dict(
        radius=float(
            np.max(np.abs(hist.wet_radius / reference.wet_radius - 1))
        ),
        vapor=np.max(
            np.abs(hist.vapor - reference.vapor) / total, axis=0
        ).tolist(),
        condensed=np.max(
            np.abs(hist.liquid - reference.liquid) / total, axis=0
        ).tolist(),
        temperature_K=float(
            np.max(np.abs(hist.temperature - reference.temperature))
        ),
        supersaturation=float(
            np.max(np.abs(hist.supersaturation - reference.supersaturation))
        ),
    )
    if (
        max(errors["vapor"] + errors["condensed"] + [errors["radius"]]) > 1e-3
        or errors["temperature_K"] > 0.01
        or errors["supersaturation"] > 1e-4
    ):
        raise RuntimeError(
            f"device trajectory exceeded its reference tolerance: {errors}"
        )
    budgets = _summary(hist)
    if budgets["max_first_law_error_K"] > 0.01:
        raise RuntimeError("device trajectory exceeded the first-law tolerance")
    metrics = dict(
        errors=errors,
        rtol=control.rtol,
        steps=owner.last_counts[0],
        rejections=owner.last_counts[1],
        activity_evaluations=owner.last_counts[4],
        warmed_replay_s=owner.replay_seconds,
        setup_s=setup_s,
        segment_observation_readbacks=owner.boundary_readbacks,
        parcel_workspace_bytes=owner.scratch_bytes,
        process_device_pool_peak_bytes=wp.get_mempool_used_mem_high(
            owner.device
        ),
        device=owner.device.name,
        warp=wp.__version__,
        restart_compared=restart,
        max_relative_inventory_error=budgets["max_relative_inventory_error"],
        max_first_law_error_K=budgets["max_first_law_error_K"],
    )
    if getattr(owner.binding, "hybrid", False):
        metrics["lookup_counts"] = owner.binding.counts.numpy().tolist()
    return (metrics, hist) if return_trajectory else metrics


# %% [markdown]
# ## Follow initially subsaturated haze through 1,500 m
#
# Start at 82% RH with 3 µg of gas-phase butanediol per kg of dry air.
# The 0.32 and 0.48 µm nonvolatile cores contain vanillic acid and NaCl
# in a 1:20 mass ratio. Solve for the condensed water and butanediol in
# each size class so both vapors are initially in equilibrium. The closed
# parcel rises at 1 m/s with its two finite vapor inventories.
#
# At each sampled height vary only the trial water mass, keeping the
# current temperature and each nonwater mass fixed. The maximum of the
# activity- and curvature-dependent equilibrium water saturation ratio
# gives the class critical radius and supersaturation. Organic uptake
# therefore changes this maximum during the ascent. Mark saturation at the
# first ambient RH crossing of 100%. Mark class activation at the upward
# crossing of its critical radius with ambient supersaturation above the
# critical value and continuing water uptake and radius growth over 10 m.
# [Köhler maxima and wet activation](https://amt.copernicus.org/articles/7/2227/2014/)


# %%
def initial_haze():
    """Equilibrate water and organic reservoirs over two nonvolatile cores."""
    from scipy.optimize import root

    state = initial_parcel()
    temperature = state.temperature
    radii = np.array([0.32, 0.48]) * 1e-6
    core_fraction = np.array([0.0, 1.0, 0.0, 20.0]) / 21.0
    volume = 4 * np.pi / 3 * radii**3
    masses = (volume / (core_fraction @ (1 / state.density)))[
        :, None
    ] * core_fraction
    gas_r = GAS_CONSTANT / np.array([v.molar_mass for v in state.vapors])
    rd = GAS_CONSTANT / MOLECULAR_WEIGHT_AIR
    psat = np.array(
        [
            v.reference_pressure
            * np.exp(
                v.latent_heat
                / r
                * (1 / v.reference_temperature - 1 / temperature)
            )
            for v, r in zip(state.vapors, gas_r, strict=True)
        ]
    )
    water_pressure = 0.82 * psat[0]
    organic_mixing = 3e-9
    organic_pressure = (
        (state.pressure - water_pressure)
        * organic_mixing
        * gas_r[1]
        / (rd + organic_mixing * gas_r[1])
    )
    partial = np.array([water_pressure, organic_pressure])
    vapor = rd / gas_r * partial / (state.pressure - partial.sum())
    targets = [v.material_index for v in state.vapors]

    def residual(log_amount, core, core_mass):
        trial = core.copy()
        trial[targets] = core_mass * np.exp(log_amount)
        radius = np.cbrt(3 * (trial @ (1 / state.density)) / (4 * np.pi))
        activity = state.activity_strategy.neutral_activities(
            trial[None, :], temperature, targets, [True]
        )[0]
        return np.log(psat * activity / partial) + 2 * state.surface_tension / (
            gas_r * temperature * state.density[targets] * radius
        )

    for core in masses:
        core_mass = core.sum()

        sol = root(residual, [1.0, -9.0], args=(core, core_mass))
        if (
            not sol.success
            or np.max(np.abs(residual(sol.x, core, core_mass))) > 1e-9
        ):
            raise RuntimeError("initial haze did not equilibrate both vapors")
        core[targets] = core_mass * np.exp(sol.x)
    return replace(state, masses=masses, vapor=vapor)


def critical_water_curve(state, masses, temperature, grid_points=65):
    """Locate the global Köhler maximum for each sampled mixture in a batch."""
    # Flatten the (height, class) axes; keep every instantaneous organic mass.
    shape = masses.shape[:-1]
    width = masses.shape[-1]
    rows = masses.reshape(-1, width).copy()
    rows[:, 0] = 0
    nonwater_volume = rows @ (1 / state.density)
    dry_radius = np.cbrt(3 * nonwater_volume / (4 * np.pi))
    tk = np.broadcast_to(temperature[:, None], shape).ravel()
    curvature = (
        2
        * state.surface_tension
        * state.vapors[0].molar_mass
        / (GAS_CONSTANT * tk * state.density[0])
    )

    def log_equilibrium(log_ratio):
        ratio = np.exp(log_ratio)
        trial = np.repeat(rows, ratio.shape[1], axis=0)
        trial[:, 0] = (
            nonwater_volume[:, None]
            * np.expm1(3 * log_ratio)
            * state.density[0]
        ).ravel()
        temps = np.repeat(tk, ratio.shape[1])
        water = state.activity_strategy.neutral_activities(
            trial, temps, [0], np.ones(len(temps), bool)
        )
        return np.log(water.reshape(ratio.shape)) + curvature[:, None] / (
            ratio * dry_radius[:, None]
        )

    left = np.full(len(tk), np.log(1.01))
    right = np.full(len(tk), np.log(256.0))
    ix = np.arange(len(tk))
    # A wide scan brackets the maximum; two local scans and centered
    # quadratic interpolation refine it in log-radius coordinates.
    for count in (grid_points, 11, 11):
        grid = left[:, None] + (right - left)[:, None] * np.linspace(
            0.0, 1.0, count
        )
        log_rh = log_equilibrium(grid)
        top = np.argmax(log_rh, axis=1)
        if np.any((top == 0) | (top == count - 1)):
            raise RuntimeError("Köhler maximum outside the sampled bracket")
        left, right = grid[ix, top - 1], grid[ix, top + 1]
    small, mid, large = (log_rh[ix, top + d] for d in (-1, 0, 1))
    offset = 0.5 * (small - large) / (small - 2 * mid + large)
    peak = grid[ix, top] + offset * (right - left) * 0.5
    if np.max(abs(offset)) > 1:
        raise RuntimeError("Köhler peak refinement left its bracket")
    critical_s = np.expm1(log_equilibrium(peak[:, None]).ravel())
    return (np.exp(peak) * dry_radius).reshape(shape), critical_s.reshape(shape)


def upward_crossing(height, value):
    """Interpolate the first negative-to-positive crossing in metres."""
    crossing = np.flatnonzero((value[:-1] < 0) & (value[1:] >= 0))
    if not crossing.size:
        raise RuntimeError("requested cloud threshold was never crossed")
    i = crossing[0]
    return float(
        height[i]
        - value[i] * (height[i + 1] - height[i]) / (value[i + 1] - value[i])
    )


def activation_events(history, index, grid_points=65):
    """Diagnose saturation, class crossings and continued growth on this grid."""
    z = history.height[index]
    supersaturation = history.supersaturation[index]
    radii = history.wet_radius[index]
    masses = history.masses[index]
    critical_r, critical_s = critical_water_curve(
        history.initial, masses, history.temperature[index], grid_points
    )
    starts = []
    for c in range(radii.shape[1]):
        zact = upward_crossing(z, radii[:, c] - critical_r[:, c])
        s_at, r_at, sc_at = (
            float(np.interp(zact, z, x))
            for x in (supersaturation, critical_r[:, c], critical_s[:, c])
        )
        after = (z >= zact) & (z <= zact + 10.0)
        # Sparse global samples are supplemented around each threshold below.
        run = np.flatnonzero(after)
        if (
            radii[0, c] >= critical_r[0, c]
            or s_at <= sc_at
            or run.size < 2
            or not np.all(np.diff(radii[run, c]) > 0)
            or not np.all(np.diff(masses[run, c, 0]) > 0)
            or not np.all(radii[z > zact, c] >= critical_r[z > zact, c])
        ):
            raise RuntimeError("class crossing lacks sustained growth")
        starts.append(
            dict(
                height_m=zact,
                critical_radius_um=r_at * 1e6,
                critical_supersaturation_pct=sc_at * 100,
                ambient_supersaturation_pct=s_at * 100,
            )
        )
    return starts, critical_r, critical_s


def validate_critical_peaks(history, index, events, rc, sc):
    """Recheck both class peaks independently with bounded scalar minimization."""
    from scipy.optimize import minimize_scalar

    state = history.initial
    radii_error = []
    excess_error = []

    def minus_log_ratio(
        log_multiplier, candidate, mixture, core_volume, temperature, curvature
    ):
        radius = candidate * np.exp(log_multiplier)
        mixture[0] = state.density[0] * (
            4 * np.pi / 3 * radius**3 - core_volume
        )
        water = state.activity_strategy.neutral_activities(
            mixture[None, :], temperature, [0], [True]
        )[0, 0]
        return -(np.log(water) + curvature / radius)

    for c, event in enumerate(events):
        pos = np.searchsorted(history.height[index], event["height_m"])
        row = index[pos]
        temperature = history.temperature[row]
        mixture = history.masses[row, c].copy()
        mixture[0] = 0
        core_volume = mixture @ (1 / state.density)
        curvature = (
            2
            * state.surface_tension
            * state.vapors[0].molar_mass
            / (GAS_CONSTANT * temperature * state.density[0])
        )
        candidate = rc[pos, c]

        args = (candidate, mixture, core_volume, temperature, curvature)
        trial = minimize_scalar(
            minus_log_ratio,
            args=args,
            bounds=(-0.1, 0.1),
            method="bounded",
            options={"xatol": 1e-9},
        )
        if (
            not trial.success
            or min(minus_log_ratio(-0.1, *args), minus_log_ratio(0.1, *args))
            <= trial.fun
        ):
            raise RuntimeError(
                "independent critical-curve maximum was unbracketed"
            )
        radii_error.append(float(abs(np.expm1(trial.x))))
        excess_error.append(float(abs(np.expm1(-trial.fun) - sc[pos, c])))
    if max(radii_error) > 1e-4 or max(excess_error) > 1e-9:
        raise RuntimeError("Köhler peak failed independent scalar refinement")
    return dict(
        max_relative_radius_error=max(radii_error),
        max_supersaturation_error=max(excess_error),
    )


def cloud_activation(device="cuda:0"):
    """Qualify the second ascent and save its numerical activation profile."""
    state = initial_haze()
    fine = run_parcel(state, updraft=1.0, timestep=0.5, steps=3000)
    z = fine.height
    coarse = np.arange(0, len(z), 10)  # 5 m; < 10 m growth window
    coarse_events, _, _ = activation_events(fine, coarse)
    sat = upward_crossing(z, fine.supersaturation)
    peak = np.argmax(fine.supersaturation)
    # Save the whole ascent at 5 m and each transition on a 0.5 m grid.
    anchors = [e["height_m"] for e in coarse_events] + [sat, z[peak]]
    dense = np.logical_or.reduce([abs(z - height) < 16 for height in anchors])
    index = np.union1d(coarse, np.flatnonzero(dense))
    one_m = index[index % 2 == 0]
    events, rc, sc = activation_events(fine, index, 97)
    peak_validation = validate_critical_peaks(fine, index, events, rc, sc)
    check, _, _ = activation_events(fine, one_m, 49)
    differences = [
        abs(e["height_m"] - c["height_m"])
        for e, c in zip(events, check, strict=True)
    ]
    if max(differences) > 0.1:
        raise RuntimeError("class activation height shifted under refinement")
    dry_mass = state.dry_air_mass
    total_w = fine.vapor[:, 0] + fine.liquid[:, 0]
    closure = np.max(
        abs((fine.vapor + fine.liquid) - (fine.vapor[0] + fine.liquid[0])),
        axis=0,
    ) / (fine.vapor[0] + fine.liquid[0])
    heat = float(np.max(abs(first_law_error(fine))))
    coarse_heat = float(np.max(abs(first_law_error(fine, stride=10))))
    if np.max(closure) > 1e-10 or not heat < min(0.01, coarse_heat):
        raise RuntimeError("long ascent failed its mass or heat budget")
    profile = dict(
        height_m=z[index].tolist(),
        temperature_K=fine.temperature[index].tolist(),
        ambient_supersaturation_pct=(
            100 * fine.supersaturation[index]
        ).tolist(),
        radius_um=(fine.wet_radius[index] * 1e6).tolist(),
        critical_radius_um=(rc * 1e6).tolist(),
        critical_supersaturation_pct=(100 * sc).tolist(),
        liquid_water_g_m3=(
            1e3 * fine.liquid[index, 0] * dry_mass / fine.volume[index]
        ).tolist(),
        organic_vapor_ug_kg=(1e9 * fine.vapor[index, 1]).tolist(),
        water_vapor_g_kg=(1e3 * fine.vapor[index, 0]).tolist(),
        class_events=events,
        critical_peak_validation=peak_validation,
        cloud_base_m=sat,
        peak_ambient_supersaturation_pct=float(
            100 * fine.supersaturation[peak]
        ),
        peak_height_m=float(z[peak]),
        initial_rh_pct=float(100 * (1 + fine.supersaturation[0])),
        terminal_activated_fraction=float(
            np.sum(state.number[fine.wet_radius[-1] > rc[-1]])
            / state.number.sum()
        ),
        class_counts=state.number.tolist(),
        refined_crossing_shift_m=differences,
        initial_total_water_kg_per_kg=float(total_w[0]),
        max_relative_inventory_error=closure.tolist(),
        max_first_law_error_K=heat,
        coarse_first_law_error_K=coarse_heat,
    )
    if device != "cpu":
        # Both methods are compared at the same 5 m samples, with the
        # resolved transition windows retained for the threshold comparisons.
        reference = multivapor_trajectory(
            state,
            fine.time[index],
            fine.pressure[index],
            fine.temperature[index],
            fine.vapor[index],
            fine.masses[index],
            1.0,
        )
        profile["cuda"] = {}
        for label, method in (("ROS3", ParcelROS3), ("RK23", ParcelRK23)):
            metrics, trajectory = cuda_comparison(
                reference,
                method(rtol=2e-6),
                device,
                restart=False,
                return_trajectory=True,
            )
            gpu_events, _, _ = activation_events(
                trajectory, np.arange(len(index))
            )
            shifts = [
                abs(e["height_m"] - c["height_m"])
                for e, c in zip(gpu_events, events, strict=True)
            ]
            if max(shifts) > 0.5:
                raise RuntimeError(
                    "device class crossing differs from the Radau result"
                )
            metrics.update(
                class_activation_height_m=[e["height_m"] for e in gpu_events],
                activation_shift_m=shifts,
                cloud_base_m=upward_crossing(
                    trajectory.height, trajectory.supersaturation
                ),
                peak_ambient_supersaturation_pct=float(
                    100 * trajectory.supersaturation.max()
                ),
            )
            profile["cuda"][label] = metrics
    return profile


def simulate(device="cuda:0"):
    """Run Radau controls and, on request, both real-device solvers at two tolerances."""
    state = initial_parcel()
    water_only = replace(
        state, vapors=(state.vapors[0], replace(state.vapors[1], enabled=False))
    )
    report = {}
    for label, parcel in (("two_vapors", state), ("water_only", water_only)):
        history = run_parcel(parcel, updraft=1.0, timestep=1.0, steps=150)
        report[label] = _summary(history)
        if label == "two_vapors":
            reference = history
        coarse_closure = float(
            np.max(np.abs(first_law_error(history, stride=5)))
        )
        if coarse_closure <= report[label]["max_first_law_error_K"]:
            raise RuntimeError(
                "first-law quadrature did not converge under refinement"
            )
        report[label]["coarse_first_law_error_K"] = coarse_closure
        if report[label]["max_first_law_error_K"] > 0.01:
            raise RuntimeError("first-law closure exceeded 0.01 K")
    if device != "cpu":
        report["cuda"] = {}
        for label, method in (("RK23", ParcelRK23), ("ROS3", ParcelROS3)):
            report["cuda"][label] = [
                cuda_comparison(
                    reference, method(rtol=tol), device, restart=refined
                )
                for tol, refined in ((2e-4, False), (2e-6, True))
            ]
    report["activation"] = cloud_activation(device)
    return report


# %% [markdown]
# ## Compare the ascent profiles
#
# Both calculations begin from the same wet particles. Compare the water
# and organic gas reservoirs first, then follow the temperature, water
# supersaturation, and radius of each class as the parcel gains altitude.
# The water-only control keeps its initial organic vapor mixing ratio.
#
# The second figure follows the longer parcel. Its vertical dotted lines
# mark saturation. The supersaturation panel zooms in near cloud base, and
# filled markers on the radius panel show each class crossing its evolving
# critical radius. Liquid water content counts condensed water per physical
# parcel volume.


# %%
def plot_results(report):
    """Show both temperature and phase-transfer histories against height."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        3, 2, figsize=(10, 9), sharex=True, layout="constrained"
    )
    fields = [
        ("temperature_K", "Temperature (K)"),
        ("supersaturation_pct", "Water supersaturation (%)"),
        ("water_g_kg", "Water vapor (g/kg dry air)"),
        ("organic_ug_kg", "Organic vapor (µg/kg dry air)"),
    ]
    for label, linestyle, color in (
        ("two_vapors", "-", "#176b80"),
        ("water_only", "--", "#ab5029"),
    ):
        data = report[label]
        plotted = dict(
            linestyle=linestyle, color=color, label=label.replace("_", " ")
        )
        for ax, (key, units) in zip(axes.flat, fields):
            ax.plot(data["height_m"], data[key], **plotted)
            ax.set_ylabel(units)
        radii = np.array(data["radius_um"])
        for i, ax in enumerate(axes[-1]):
            ax.plot(data["height_m"], radii[:, i], **plotted)
            ax.set_ylabel(f"Class {i + 1} wet radius (µm)")
            ax.set_xlabel("Height (m)")
    axes[0, 0].legend()
    return fig


def plot_activation(report):
    """Display thermodynamic saturation, class activation, and in-cloud growth."""
    import matplotlib.pyplot as plt

    data = report["activation"]
    height = np.asarray(data["height_m"])
    excess = np.asarray(data["ambient_supersaturation_pct"])
    critical = np.array(data["critical_supersaturation_pct"])
    radius, rc = (
        np.array(data["radius_um"]),
        np.array(data["critical_radius_um"]),
    )
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 9.5), layout="constrained")
    humidity, peak, drops, water, thermal, organic = axes.flat
    ambient_color = "#143d58"
    humidity.plot(height, 100 + excess, color=ambient_color)
    humidity.axhline(100, color=".45", ls=":", lw=1)
    humidity.set_ylabel("Water relative humidity (%)")
    peak.plot(height, excess, color=ambient_color, label="Ambient")
    for c, color in enumerate(("#16816c", "#b05222")):
        name = f"Class {c + 1}"
        peak.plot(
            height,
            critical[:, c],
            ls="--",
            color=color,
            label=f"{name} critical",
        )
        drops.plot(height, radius[:, c], color=color, label=f"{name} wet")
        drops.plot(
            height, rc[:, c], color=color, ls="--", label=f"{name} critical"
        )
        event = data["class_events"][c]
        drops.scatter(
            event["height_m"],
            event["critical_radius_um"],
            marker=("o", "s")[c],
            color=color,
            zorder=4,
        )
    peak.plot(
        data["peak_height_m"],
        data["peak_ambient_supersaturation_pct"],
        marker="o",
        color=ambient_color,
    )
    peak.set(
        xlim=(data["cloud_base_m"] - 2, data["peak_height_m"] + 38),
        ylim=(-0.09, 1.1 * data["peak_ambient_supersaturation_pct"]),
        ylabel="Water supersaturation (%)",
    )
    peak.legend(fontsize=8)
    drops.set_ylabel("Wet and critical radii (µm)")
    drops.legend(fontsize=8, ncol=2)
    water.plot(height, data["liquid_water_g_m3"], color=ambient_color)
    water.set_ylabel("Liquid water content (g/m³)")
    thermal.plot(height, data["temperature_K"], color=ambient_color)
    thermal.set_ylabel("Parcel temperature (K)")
    organic.plot(height, data["organic_vapor_ug_kg"], color=ambient_color)
    organic.set_ylabel("Butanediol vapor (µg/kg dry air)")
    for ax in axes.flat:
        ax.axvline(data["cloud_base_m"], color=".5", ls=":", lw=1)
        ax.set_xlabel("Height (m)")
        if ax is not peak:
            ax.set_xlim(0, height[-1])
    return fig


def run_example():
    """Run the ascent, print numerical closure and draw its six profiles."""
    import matplotlib.pyplot as plt

    report = simulate()
    for label in ("two_vapors", "water_only"):
        data = report[label]
        print(
            label,
            "maximum species budget errors",
            data["max_relative_inventory_error"],
            "maximum first-law residual (K)",
            data["max_first_law_error_K"],
        )
    for method, records in report["cuda"].items():
        print(method, "refined comparison with Radau", records[-1])
    cloud = report["activation"]
    print(
        "1,500 m ascent; initial RH (%)",
        cloud["initial_rh_pct"],
        "saturation height (m)",
        cloud["cloud_base_m"],
        "peak ambient S (%)",
        cloud["peak_ambient_supersaturation_pct"],
        "at (m)",
        cloud["peak_height_m"],
    )
    for cls, event in enumerate(cloud["class_events"], start=1):
        print("Class", cls, event)
    print(
        "Activated number fraction",
        cloud["terminal_activated_fraction"],
        "endpoint LWC (g/m³)",
        cloud["liquid_water_g_m3"][-1],
        "wet radii (µm)",
        cloud["radius_um"][-1],
        "temperature (K)",
        cloud["temperature_K"][-1],
    )
    for method, record in cloud["cuda"].items():
        print(
            "1,500 m",
            method,
            "maximum radius error",
            record["errors"]["radius"],
            "activation shifts (m)",
            record["activation_shift_m"],
        )
    plot_results(report)
    plot_activation(report)
    plt.show()
    return report


if __name__ == "__main__":
    run_example()

# %% [markdown]
# Organic vapor falls from 32.23 to 1.43 µg/kg dry air over 150 m.
# Water dominates the wet mass, so the two temperature and radius profiles
# nearly overlap. Allowing organic transfer warms the final parcel by
# 0.00030 K relative to the water-only run; the smaller and larger classes
# finish respectively 0.0069 µm larger and 0.0083 µm smaller. Changing the
# initial organic reservoir at fixed wet composition explores when organic
# uptake contributes appreciably to total growth.
#
# In the 1,500 m ascent the air first saturates at 400 m and reaches peak
# supersaturation of 0.430% at 422 m. The 0.32 and 0.48 µm cores cross
# their critical wet radii later, at 463 and 707 m. The larger core has a
# lower critical supersaturation and a larger critical radius; it spends
# longer growing to that radius. Both classes remain supercritical to the
# top, where liquid water content reaches 2.54 g/m³, the wet radii reach
# 20.44 and 20.74 µm, and the parcel temperature is 289.58 K. These heights
# depend on the prescribed loading, updraft and thermophysical laws.
