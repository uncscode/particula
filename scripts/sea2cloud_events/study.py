"""Marine 1–10 nm birth cohorts, Brownian survival and inverse exposure."""

import csv
import itertools
from pathlib import Path

import numpy as np
import warp as wp
from particula.dynamics.coagulation.brownian_kernel import (
    get_brownian_kernel_via_system_state,
)
from particula.gas.properties.dynamic_viscosity import get_dynamic_viscosity
from particula.gas.properties.mean_free_path import get_molecule_mean_free_path
from particula.gpu.dynamics.coagulation_funcs import (
    brownian_diffusivity_wp,
    brownian_kernel_pair_wp,
    g_collection_term_wp,
    particle_mean_free_path_wp,
)
from particula.gpu.properties.particle_properties import (
    aerodynamic_mobility_wp,
    cunningham_slip_correction_wp,
    mean_thermal_speed_wp,
)
from particula.util.constants import BOLTZMANN_CONSTANT
from scipy.integrate import cumulative_trapezoid

TEMP = 273.15 + 14.2
PRESSURE = 101325.0
DENSITY = 1500.0
EXPONENT = 4.807
CAL_LIMIT = 2e7
J_LIMIT = 3.5e-37 * CAL_LIMIT**EXPONENT  # cm-3 s-1


@wp.func
def properties(
    radius: wp.float64,
    mass: wp.float64,
    temp: wp.float64,
    viscosity: wp.float64,
    free_path: wp.float64,
    kb: wp.float64,
):
    """Return diffusivity, collection length and thermal speed."""
    slip = cunningham_slip_correction_wp(free_path / radius)
    diff = brownian_diffusivity_wp(
        temp, aerodynamic_mobility_wp(radius, slip, viscosity), kb
    )
    speed = mean_thermal_speed_wp(mass, temp, kb)
    term = g_collection_term_wp(particle_mean_free_path_wp(diff, speed), radius)
    return diff, term, speed


@wp.kernel
def collision_rates(
    radius: wp.array(dtype=wp.float64),
    mass: wp.array(dtype=wp.float64),
    collector: wp.float64,
    collector_mass: wp.float64,
    temp: wp.float64,
    viscosity: wp.float64,
    free_path: wp.float64,
    kb: wp.float64,
    output: wp.array(dtype=wp.float64),
):
    """Evaluate Particula's neutral transition-regime collision rate on CUDA."""
    i = wp.tid()
    di, gi, vi = properties(radius[i], mass[i], temp, viscosity, free_path, kb)
    dj, gj, vj = properties(
        collector, collector_mass, temp, viscosity, free_path, kb
    )
    output[i] = brownian_kernel_pair_wp(
        radius[i], collector, di, dj, gi, gj, vi, vj, wp.float64(1.0)
    )


@wp.kernel
def occupancy(
    exposure: wp.array2d(dtype=wp.float64),
    age_step: wp.array(dtype=wp.float64),
    log_acid: wp.array(dtype=wp.float64),
    background: wp.array(dtype=wp.float64),
    power: wp.float64,
    j_limit: wp.float64,
    out: wp.array2d(dtype=wp.float64),
):
    """Accumulate one differentiable contribution for each case and age."""
    i, j = wp.tid()
    survival = wp.exp(-background[i] * exposure[i, j])
    out[i, j] = j_limit * wp.exp(power * log_acid[i]) * age_step[i] * survival


def rate_table(background_nm, diameter):
    """Compare a CUDA collision table with independent CPU kernels."""
    mu = float(get_dynamic_viscosity(TEMP))
    path = float(
        get_molecule_mean_free_path(
            temperature=TEMP, pressure=PRESSURE, dynamic_viscosity=mu
        )
    )
    r = diameter * 0.5e-9
    mass = 4 * np.pi / 3 * DENSITY * r**3
    collector = background_nm * 0.5e-9
    cmass = 4 * np.pi / 3 * DENSITY * collector**3

    def array(a):
        return wp.array(a, dtype=wp.float64, device="cuda:0")

    output = wp.empty(len(r), dtype=wp.float64, device="cuda:0")
    wp.launch(
        collision_rates,
        len(r),
        [
            array(r),
            array(mass),
            collector,
            cmass,
            TEMP,
            mu,
            path,
            BOLTZMANN_CONSTANT,
            output,
        ],
        device="cuda:0",
    )
    result = output.numpy()
    for i in np.linspace(0, len(r) - 1, 17).astype(int):
        cpu = get_brownian_kernel_via_system_state(
            np.array([r[i], collector]),
            np.array([mass[i], cmass]),
            TEMP,
            PRESSURE,
        )[0, 1]
        np.testing.assert_allclose(result[i], cpu, rtol=3e-7, atol=0)
    return result


def solve_grid(rate, diameter, collector, hours, growth, steps):
    """Propagate growing cohorts and account for loss, cutoff and dry mass."""
    duration = hours * 3600.0
    lifetime = min(duration, 9 * 3600 / growth) if growth else duration
    ages = np.linspace(0.0, lifetime, steps + 1)
    sizes = 1.0 + growth * ages / 3600
    # Kernel m3/s × 1e6 gives exposure per background particle/cm3.
    hazard = np.interp(sizes, diameter, rate) * 1e6
    cumulative = cumulative_trapezoid(hazard, ages, initial=0.0)
    middle = (ages[:-1] + ages[1:]) * 0.5
    exp_mid = (cumulative[:-1] + cumulative[1:]) * 0.5
    step = lifetime / steps
    survival = np.exp(-collector * exp_mid)
    remaining = np.sum(survival) * step
    exported = max(duration - lifetime, 0.0) * np.exp(
        -collector * cumulative[-1]
    )
    weight = (duration - middle) * survival * step
    loss_rate = (
        collector
        * np.interp(1.0 + growth * middle / 3600, diameter, rate)
        * 1e6
    )
    lost = np.sum(weight * loss_rate)
    # Single-particle mass in units of a dry 1 nm primary particle's mass.
    size = 1.0 + growth * middle / 3600.0
    gain = np.sum(weight * 3 * size**2 * growth / 3600)
    mass_final = np.sum(survival * size**3) * step
    mass_lost = np.sum(weight * loss_rate * size**3)
    mass_exit = exported * sizes[-1] ** 3
    budget_n = abs(duration - remaining - exported - lost) / duration
    budget_m = abs(duration + gain - mass_final - mass_lost - mass_exit) / (
        duration + gain
    )
    return exp_mid, step, remaining, exported, lost, budget_n, budget_m


def evaluate(controls, steps, derivatives=False, source_scale=1.0):
    """Compute number and optional CUDA gradients for each case."""
    diameter = np.linspace(1.0, 10.0, 2049)
    kernels = {
        b: rate_table(b, diameter) for b in sorted({x[3] for x in controls})
    }
    cache = {}
    exposure = []
    intervals = []
    estimates = []
    checks = []
    for hours, growth, bg, diam, acid in controls:
        key = hours, growth, bg, diam
        if key not in cache:
            cache[key] = solve_grid(
                kernels[diam], diameter, bg, hours, growth, steps
            )
        exp_mid, step, remaining, exported, lost, budget_n, budget_m = cache[
            key
        ]
        exposure.append(exp_mid)
        intervals.append(step)
        j = 3.5e-37 * acid**EXPONENT
        estimates.append(source_scale * j * remaining)
        checks.append((j * exported, j * lost, budget_n, budget_m))

    def device(a, grad=False):
        return wp.array(
            np.asarray(a), dtype=wp.float64, device="cuda:0", requires_grad=grad
        )

    acid_log = device([np.log(x[4] / CAL_LIMIT) for x in controls], derivatives)
    bg_device = device([x[2] for x in controls], derivatives)
    shape = (len(controls), steps)
    out = wp.zeros(
        shape,
        dtype=wp.float64,
        device="cuda:0",
        requires_grad=derivatives,
    )
    args = [
        device(exposure),
        device(intervals),
        acid_log,
        bg_device,
        EXPONENT,
        J_LIMIT * source_scale,
        out,
    ]
    if derivatives:
        with wp.Tape() as tape:
            wp.launch(
                occupancy,
                shape,
                inputs=args[:-1],
                outputs=[out],
                device="cuda:0",
            )
        tape.backward(grads={out: device(np.ones(shape))})
        derivatives = np.column_stack(
            (acid_log.grad.numpy(), bg_device.grad.numpy())
        )
    else:
        wp.launch(
            occupancy, shape, inputs=args[:-1], outputs=[out], device="cuda:0"
        )
        derivatives = None
    result = out.numpy().sum(axis=1)
    np.testing.assert_allclose(result, estimates, rtol=1e-12)
    return result, np.asarray(checks), derivatives


def simulate():
    """Compare bounded exposures with two Sea2Cloud event windows."""
    path = (
        Path(__file__).resolve().parents[2]
        / "docs/data/sea2cloud-events/events.csv"
    )
    with path.open() as stream:
        events = list(csv.DictReader(stream))
    conditions = list(
        itertools.product(
            (1.0, 3.0, 7.0, 12.0),
            (0.0, 0.5, 1.0, 2.0, 4.0),
            (0.0, 75.0, 150.0, 300.0),
            (50.0, 100.0, 200.0),
            (0.5e7, 1.0e7, 1.5e7, 2.0e7),
        )
    )
    fine, checks, gradients = evaluate(conditions, 1024, True)
    coarse, _, _ = evaluate(conditions, 512)
    np.testing.assert_allclose(coarse, fine, rtol=0.001, atol=1e-7)
    if np.any(fine < 0) or np.max(checks[:, 2:]) > 0.002:
        raise ValueError("Cohort inventory or positivity check failed")
    np.testing.assert_allclose(gradients[:, 0] / fine, EXPONENT, rtol=1e-10)
    if np.max(gradients[:, 1]) > 0:
        raise ValueError("Increased background increased survival")
    pivot = (7.0, 1.0, 150.0, 100.0, 1.5e7)
    verify = [
        tuple(list(pivot[:2]) + [pivot[2] + delta] + list(pivot[3:]))
        for delta in (-0.01, 0.01)
    ]
    fd = (np.diff(evaluate(verify, 1024)[0]) / 0.02).item()
    np.testing.assert_allclose(
        fd, gradients[conditions.index(pivot), 1], rtol=1e-7
    )
    for bg in (0.0, 75.0, 300.0):
        anchor = (7.0, 0.0, bg, 100.0, CAL_LIMIT)
        radius = np.array([0.5e-9, 50e-9])
        mass = 4 * np.pi / 3 * DENSITY * radius**3
        hazard = (
            bg
            * 1e6
            * get_brownian_kernel_via_system_state(
                radius, mass, TEMP, PRESSURE
            )[0, 1]
        )
        seconds = anchor[0] * 3600
        occupancy_time = (
            -np.expm1(-hazard * seconds) / hazard if hazard else seconds
        )
        np.testing.assert_allclose(
            fine[conditions.index(anchor)], J_LIMIT * occupancy_time, rtol=1e-4
        )
    rows = []
    for condition, n, stat, grad in zip(
        conditions, fine, checks, gradients, strict=True
    ):
        hours, growth, bg, diam, acid = condition
        j = 3.5e-37 * acid**EXPONENT
        rows.append(
            [
                hours,
                growth,
                bg,
                diam,
                acid,
                n,
                j * hours * 3600,
                stat[0],
                stat[1],
                grad[0] / n,
                bg * grad[1] / n,
            ]
        )
    # No-new-particle baseline from empty initial state; no imported population.
    zero_j = evaluate([pivot], 1024, source_scale=0.0)[0].item()
    if zero_j != 0.0:
        raise ArithmeticError("Zero acid must give zero source")
    return dict(
        events=events,
        rows=rows,
        columns=[
            "hours",
            "growth_nm_h",
            "background_cm3",
            "background_d_nm",
            "acid_molec_cm3",
            "number_cm3",
            "injected_cm3",
            "exported_cm3",
            "scavenged_cm3",
            "dlogn_dlogacid",
            "dlogn_dlogbackground",
        ],
        zero_source_cm3=zero_j,
        device=wp.get_device("cuda:0").name,
        max_number_budget=float(checks[:, 2].max()),
        max_mass_budget=float(checks[:, 3].max()),
        max_grid_change=float(np.max(abs(coarse / fine - 1))),
        background_gradient=fd,
    )
