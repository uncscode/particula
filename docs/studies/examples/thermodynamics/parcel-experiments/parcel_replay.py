# %% [markdown]
# # Can a recorded ascent reproduce droplet growth and close two gas budgets?
#
# A closed parcel cools as it rises. Its droplets take up water and release
# latent heat, which feeds back on the supersaturation. Record the temperature,
# pressure, volume and vapor from a 100 s ascent, then ask how much of its
# droplet history can be recovered when those fields are prescribed. The
# radii and liquid water from the direct ascent are held aside for comparison.
#
# The three soluble classes have dry radii of 45, 70 and 110 nm. The parcel
# starts at 90 kPa and 288.15 K and contains 1 kg dry air. The original
# [closed-parcel calculation](../moist_parcel.py) gives the thermodynamics and
# [κ-Köhler growth law](../../../../theory/dynamics/condensation-equations.md).
# SciPy Radau generates our direct ascent at 1 m/s. We sample it every
# 0.05 s, regardless of its internal step sizes. The example download includes
# the forcing CSV and a companion JSON with provenance, initial state and direct
# comparison outputs. Run from the extracted directory or repository root.
# This example requires CUDA.

# %%
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import warp as wp
from matplotlib import style as mplstyle
from particula import plotting
from particula.dynamics.parcel import ParcelInitialState, run_parcel
from particula.dynamics.parcel_replay import OrganicForcing, ParcelTrace, replay

CADENCE = 0.05  # s
DURATION = 100.0  # s
DEVICE = "cuda:0"
GROWTH_STEPS = (0.025, 0.0125)  # s
PERTURBATIONS = (1e-5, 3e-6, 1e-6)  # fractional vapor perturbation at 100 s


def record_ascent(path):
    """Write the Particula-generated forcing table and its metadata."""
    initial = ParcelInitialState(
        dry_radius=1e-9 * np.array([45.0, 70.0, 110.0]),
        number=np.array([2.5e7, 5.0e7, 2.5e7]),
        kappa=0.5,
        temperature=288.15,
        pressure=90000.0,
        vapor=0.0118,
        dry_air_mass=1.0,
    )
    direct = run_parcel(
        initial,
        updraft=1.0,
        timestep=CADENCE,
        steps=round(DURATION / CADENCE),
        device="cpu",
    )
    trace = ParcelTrace.record(
        direct,
        parcel_id="ascent",
        provenance="Particula closed warm parcel, SciPy Radau, 1 m/s upward",
    )
    trace.save(path)
    return direct, ParcelTrace.load(path)


# %% [markdown]
# ## Separate forcing cadence from the growth timestep
#
# Compare all six combinations of record spacing (0.05, 0.10 and 0.20 s)
# and maximum growth step (0.025 and 0.0125 s) at the same observation times,
# spaced 0.20 s apart. For each class, measure the largest absolute radius
# error over the ascent. At each observation also compare liquid water and
# supersaturation. With vapor and thermodynamics imposed, supersaturation at
# the records checks units and timing. The radii and liquid water integrate
# the effects of the intervening forcing.
#
# In the closed calculation, vapor lost is liquid gained. During replay, the
# forcing can supply or remove water: the change in total water must equal
# the signed mass supplied by the external reservoir. Positive water exchange
# adds liquid to droplets. Positive restoration adds water to the parcel
# from the reservoir. Scale the residual by the initial water inventory to
# compare it with floating-point roundoff.


# %%
def replay_errors(result, direct):
    """Compare states at common records and close the external water ledger."""
    common = direct.time[::4]
    i = np.searchsorted(result.time, common)
    np.testing.assert_allclose(result.time[i], common, rtol=0, atol=1e-12)
    liquid = direct.liquid[::4] * direct.initial.dry_air_mass
    vapor = direct.vapor[::4] * direct.initial.dry_air_mass
    inventory = vapor + result.liquid[i]
    radius_error = 1e9 * (result.wet_radius[i] - direct.wet_radius[::4])
    residual = inventory - inventory[0] - result.water_restored[i]
    return {
        "class_max_radius_error_nm": np.max(
            np.abs(radius_error), axis=0
        ).tolist(),
        "max_liquid_error_kg": float(np.max(np.abs(result.liquid[i] - liquid))),
        "max_supersaturation_error": float(
            np.max(
                np.abs(result.supersaturation[i] - direct.supersaturation[::4])
            )
        ),
        "max_water_budget_residual_kg": float(np.max(np.abs(residual))),
        "water_budget_relative_residual": float(
            np.max(np.abs(residual)) / inventory[0]
        ),
        "terminal_water_exchange_kg": float(result.water_exchange[-1]),
        "terminal_water_restored_kg": float(result.water_restored[-1]),
    }


# %% [markdown]
# ## Add an organic vapor and repeat across a restart
#
# Keep water under trace control and introduce 10 µg of organic gas in the
# parcel. A constant source adds 0.1 µg/s throughout the run and a pulse
# adds 20 µg at the interval ending at 50 s. A first-order uptake rate of
# 0.02/s transfers organic mass from gas to particle phase. This idealized
# sink tests bookkeeping for a second species. At each time, gas plus
# particulate organic mass should equal the initial mass plus both sources.
# An isothermal expansion provides a separate control: at fixed gas inventory,
# doubling parcel volume halves the organic concentration. In that control
# dry-air mass and vapor mixing ratio are fixed while pressure halves.
# Compare restarts on either side of the pulse and repeat with reversed gas
# field order to check whether the pulse is counted exactly once.


# %%
def mixed_species(trace, device):
    """Return gas/particle ledgers and errors across both pulse boundaries."""
    initial_gas = 1e-8  # kg
    rate = 1e-10  # kg/s
    pulse = 2e-8  # kg
    index = int(np.searchsorted(trace.time, 50.0))
    assert trace.time[index] == 50.0
    totals = np.zeros(len(trace.time) - 1)
    totals[index - 1] = pulse
    mixed = replace(
        trace,
        organic=OrganicForcing(
            name="SOA",
            initial_gas=initial_gas,
            uptake_rate=0.02,
            rates=tuple(np.full(totals.size, rate)),
            totals=tuple(totals),
        ),
    )
    settings = dict(max_step=GROWTH_STEPS[0], device=device)
    endpoints = trace.sample_every(len(trace.time) - 1)
    expansion = replace(
        endpoints,
        time=np.array([0.0, 1.0]),
        height=np.repeat(trace.height[0], 2),
        temperature=np.repeat(trace.temperature[0], 2),
        pressure=trace.pressure[0] * np.array([1.0, 0.5]),
        volume=trace.volume[0] * np.array([1.0, 2.0]),
        vapor=np.repeat(trace.vapor[0], 2),
        organic=OrganicForcing(name="SOA", initial_gas=initial_gas),
        reference_radius=None,
        reference_liquid=None,
    )
    dilution = replay(expansion, species=("water", "SOA"), **settings)
    concentration = dilution.organic_gas / dilution.volume
    concentration_ratio = float(concentration[-1] / concentration[0])
    np.testing.assert_allclose(concentration_ratio, 0.5, rtol=1e-14, atol=0)
    fields = ("water", "SOA")
    complete = replay(mixed, species=fields, **settings)
    expected = initial_gas + rate * (complete.time - complete.time[0])
    expected += pulse * (complete.time >= 50.0)
    actual = complete.organic_gas + complete.organic_particle
    budget_error = float(np.max(np.abs(actual - expected)))
    restart_error = 0.0
    for cut in (index - 1, index):
        before = replay(mixed, species=fields, stop_index=cut, **settings)
        after = replay(
            mixed, species=fields, checkpoint=before.checkpoint, **settings
        )
        np.testing.assert_array_equal(after.time, complete.time[cut:])
        for name in ("organic_gas", "organic_particle", "organic_added"):
            difference = getattr(after, name) - getattr(complete, name)[cut:]
            restart_error = max(
                restart_error, float(np.max(np.abs(difference)))
            )
    reordered = replay(mixed, species=fields[::-1], **settings)
    np.testing.assert_allclose(
        reordered.organic_gas, complete.organic_gas, rtol=1e-13, atol=0
    )
    # Allow floating-point accumulation over all intervals, scaled to the
    # organic source inventory instead of the much larger vapor reservoir.
    tolerance = 16 * np.finfo(float).eps * len(trace.time) * expected[-1]
    assert max(budget_error, restart_error) <= tolerance
    select = np.unique(
        np.r_[np.arange(0, len(complete.time), 20), index - 1, index]
    )
    return {
        "time_s": complete.time[select].tolist(),
        "gas_ug": (1e9 * complete.organic_gas[select]).tolist(),
        "particle_ug": (1e9 * complete.organic_particle[select]).tolist(),
        "expected_ug": (1e9 * expected[select]).tolist(),
        "max_budget_residual_kg": budget_error,
        "max_restart_discrepancy_kg": restart_error,
        "doubling_volume_concentration_ratio": concentration_ratio,
        "terminal_gas_kg": float(complete.organic_gas[-1]),
        "terminal_particle_kg": float(complete.organic_particle[-1]),
    }


# %% [markdown]
# ## Differentiate a smooth change to the trace
#
# Scale vapor by $1+a t/(100\,\mathrm{s})$, keeping all initial wet radii
# fixed. Positive $a$ raises final humidity. The sensitivity is kilograms
# of liquid at 100 s per unit fractional perturbation, along the same
# record and step schedule. Repeat the full forward calculation on both
# sides of zero at three perturbation sizes, then halve the CPU growth step
# for an independent numerical refinement check.


# %%
def centered_gradient(trace, delta, step, device):
    """Return kg liquid per unit terminal fractional vapor perturbation."""
    ends = [
        replay(
            trace, max_step=step, device=device, humidity_amplitude=a
        ).liquid[-1]
        for a in (-delta, delta)
    ]
    return float((ends[1] - ends[0]) / (2 * delta))


def simulate():
    """Execute forward comparisons, mass checks, and a native CUDA gradient."""
    with TemporaryDirectory() as tmp:
        direct, regenerated = record_ascent(Path(tmp) / "ascent.csv")
    trace = ParcelTrace.load(Path("docs/data/parcel-replay/ascent.csv"))
    np.testing.assert_array_equal(trace.time, regenerated.time)
    # Verify the archived fields against the current independent CPU run.
    # Allow last-place changes in Radau across supported SciPy releases.
    for field in ("height", "pressure", "temperature", "volume", "vapor"):
        np.testing.assert_allclose(
            getattr(trace, field),
            getattr(regenerated, field),
            rtol=1e-9,
            atol=0,
        )
    errors = []
    for stride in (1, 2, 4):
        sampled = trace.sample_every(stride)
        for step in GROWTH_STEPS:
            result = replay(sampled, max_step=step, device=DEVICE)
            metrics = replay_errors(result, direct)
            metrics.update(
                forcing_interval_s=stride * CADENCE, max_growth_step_s=step
            )
            errors.append(metrics)
    fine = GROWTH_STEPS[-1]
    native = replay(trace, max_step=fine, device=DEVICE, gradient=True)
    baseline = replay_errors(native, direct)
    # The published closed-parcel comparison at 0.05 s resolves radii to
    # 11 nm and liquid to 3.8e-8 kg. Resolve the same scales on the finer
    # replay step, with explicit limits for prescribed-field identities.
    assert max(baseline["class_max_radius_error_nm"]) < 15.0
    assert baseline["max_liquid_error_kg"] < 6e-8
    assert baseline["max_supersaturation_error"] < 1e-12
    for row in errors:
        assert row["water_budget_relative_residual"] < 1e-11
    derivative = native.humidity_gradient
    assert derivative is not None and np.isfinite(derivative) and derivative > 0
    differences = [
        centered_gradient(trace, delta, fine, DEVICE) for delta in PERTURBATIONS
    ]
    cpu_refined = centered_gradient(trace, PERTURBATIONS[-1], fine / 2, "cpu")
    # Check the discrete derivative with the two finest centered increments.
    np.testing.assert_allclose(differences[-2:], derivative, rtol=1e-4, atol=0)
    # A thousandth of the response separates the physical signal from the
    # solver error required for a first-order prediction in this study.
    np.testing.assert_allclose(cpu_refined, derivative, rtol=1e-3, atol=0)
    mixed = {device: mixed_species(trace, device) for device in ("cpu", DEVICE)}
    for name in ("terminal_gas_kg", "terminal_particle_kg"):
        np.testing.assert_allclose(
            mixed["cpu"][name], mixed[DEVICE][name], rtol=1e-11
        )
    samples = np.arange(0, len(direct.time), 20)
    return {
        "device": wp.get_device(DEVICE).name,
        "precision": "float64",
        "duration_s": DURATION,
        "forcing_source": "Particula closed ascent, SciPy Radau",
        "time_s": direct.time[samples].tolist(),
        "direct_liquid_mg": (
            1e6 * direct.liquid[samples] * direct.initial.dry_air_mass
        ).tolist(),
        "replay_liquid_mg": (1e6 * native.liquid[samples]).tolist(),
        "dry_radius_nm": (1e9 * direct.initial.dry_radius).tolist(),
        "replays": errors,
        "mixed": mixed,
        "humidity_gradient_kg": float(derivative),
        "perturbations": list(PERTURBATIONS),
        "centered_gradients_kg": differences,
        "refined_cpu_gradient_kg": cpu_refined,
        "refined_cpu_relative_difference": float(
            abs(cpu_refined / derivative - 1)
        ),
    }


# %% [markdown]
# Read the time histories alongside the error curves. At a fixed growth
# step, differences between the three forcing cadences reflect interpolation
# of the prescribed thermodynamics. At fixed cadence, differences between
# growth steps reflect the numerical integration. The lower panels show how
# the 50 s addition passes into particle mass and how the numerical slope
# compares with reverse differentiation.


# %%
@mplstyle.context("particula.default")
def plot_results(report):
    """Plot water, maximum radius errors, organic budgets, and sensitivity."""
    fig, axes = plotting.subplots(2, 2)
    water, radius, organic, gradient = axes.flat
    water.plot(
        report["time_s"], report["direct_liquid_mg"], label="Direct ascent"
    )
    water.plot(
        report["time_s"][::5],
        report["replay_liquid_mg"][::5],
        "o",
        fillstyle="none",
        label="Replay",
    )
    water.set(xlabel="Time (s)", ylabel="Liquid water (mg)")
    plotting.legend(water)
    for step, marker in zip(GROWTH_STEPS, ("o", "s"), strict=True):
        rows = [r for r in report["replays"] if r["max_growth_step_s"] == step]
        radius.plot(
            [r["forcing_interval_s"] for r in rows],
            [max(r["class_max_radius_error_nm"]) for r in rows],
            marker=marker,
            label=f"Growth step ≤ {step:g} s",
        )
    radius.set(
        xlabel="Forcing interval (s)", ylabel="Largest wet-radius error (nm)"
    )
    radius.set_xticks([0.05, 0.10, 0.20])
    plotting.legend(radius)
    mass = report["mixed"][DEVICE]
    for field, label, style in (
        ("gas_ug", "Gas", "-"),
        ("particle_ug", "Particles", "--"),
        ("expected_ug", "Initial + sources", ":"),
    ):
        organic.plot(mass["time_s"], mass[field], style, label=label)
    organic.set(xlabel="Time (s)", ylabel="Organic mass (µg)")
    plotting.legend(organic)
    ad = report["humidity_gradient_kg"]
    delta = report["perturbations"]
    gradient.semilogx(
        delta,
        1e6 * (np.asarray(report["centered_gradients_kg"]) / ad - 1),
        "o",
        label="Centered CUDA",
    )
    gradient.axhline(0, color="0.45", linewidth=0.9, label="Reverse CUDA")
    gradient.axhline(
        1e6 * (report["refined_cpu_gradient_kg"] / ad - 1),
        linestyle="--",
        label="Refined CPU",
    )
    gradient.set(
        xlabel="Fractional vapor perturbation",
        ylabel="Difference from CUDA derivative (ppm)",
    )
    plotting.legend(gradient)
    plotting.fit(fig)
    return fig


# %% [markdown]
# The printed table retains a separate radius error for every dry class.
# The water-reservoir total shows how much water the replay needed to add
# or remove while enforcing the original vapor history.


# %%
def run_example():
    """Print physical residuals and gradient checks beside the figure."""
    report = simulate()
    for row in report["replays"]:
        print(
            f"Forcing {row['forcing_interval_s']:.3f} s, growth ≤ "
            f"{row['max_growth_step_s']:.4f} s: radius errors "
            f"{np.round(row['class_max_radius_error_nm'], 3)} nm; "
            f"water restored at 100 s {row['terminal_water_restored_kg']:+.3e} kg."
        )
    organic = report["mixed"][DEVICE]
    print(
        f"Maximum organic budget residual {organic['max_budget_residual_kg']:.3e} kg."
    )
    print(
        f"Maximum restart difference {organic['max_restart_discrepancy_kg']:.3e} kg."
    )
    print(
        f"CUDA humidity sensitivity {report['humidity_gradient_kg']:+.6g} kg/unit."
    )
    print(
        f"Refined CPU sensitivity {report['refined_cpu_gradient_kg']:+.6g} kg/unit."
    )
    plot_results(report)
    plt.show()
    return report


if __name__ == "__main__":
    run_example()

# %% [markdown]
# ## Interpreting a replay
#
# At the original 0.05 s cadence, halving the growth step from 0.025 to
# 0.0125 s reduces the largest wet-radius error from 2.37 to 1.18 nm.
# Keeping the finer step and spacing records four times farther apart
# increases the error to 1.23 nm. For this smooth ascent, resolving droplet
# growth matters more than adding forcing records within that range.
#
# On the fine native run, the maximum liquid-water difference is 9.8 µg in
# the parcel. At 100 s the external reservoir has removed a net 9.8 µg to
# maintain the original vapor history. The humidity derivative predicts
# about 0.347 mg more liquid for a small terminal vapor increase of ten
# parts per million of its recorded value.
#
# The organic ledger closes to within 2 × 10⁻²¹ kg. Both restart points
# carry the 20 µg pulse forward once. The volume-doubling control halves
# concentration while conserving organic gas mass. To explore a different
# aerosol population under the same history, vary the initial class counts
# and compare both the liquid water and the reservoir adjustment.
